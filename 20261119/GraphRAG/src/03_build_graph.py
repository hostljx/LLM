"""
写入 Neo4j 构建知识图谱

把上一步抽取的实体/关系写入 Neo4j，核心设计：

图模型（Schema）：
  (:Entity {name, type, description,            实体节点，name 唯一；
            aliases, aliases_text})              aliases 别名列表（实体链接用），
                                                 aliases_text 别名拼接串（全文索引用）
  (:Chunk  {id, text})                          原文块节点（来源追溯）
  (:Entity)-[:RELATES {relation, description,   实体间关系，统一用 RELATES 类型，
             weight, chunk_id}]->(:Entity)      具体关系词放在 relation 属性里
  (:Entity)-[:MENTIONED_IN]->(:Chunk)           实体在哪个块中被提到

去重合并策略：
- 实体用 MERGE 按 name 合并：多个 chunk 提到"曹操"时只建一个节点，
  描述用分号拼接累积（微软 GraphRAG 则是再用 LLM 汇总描述）
- 关系用 MERGE 按 (source, relation, target) 合并：重复出现时权重累加，
  出现次数越多的关系权重越大，社区检测时越"紧密"

输入：output/extraction.json
"""
import json
from config import OUTPUT_DIR, get_driver, run_cypher

def clear_graph():
    """清空创建的节点（Entity/Chunk/Community），保证脚本可重复运行。"""
    run_cypher("MATCH (n) WHERE n:Entity OR n:Chunk OR n:Community DETACH DELETE n")
    print("已清空旧图数据")


def create_constraints():
    """创建唯一性约束：既保证数据质量，又自动创建索引加速 MERGE 查找。"""
    run_cypher("CREATE CONSTRAINT entity_name IF NOT EXISTS "
               "FOR (e:Entity) REQUIRE e.name IS UNIQUE")
    run_cypher("CREATE CONSTRAINT chunk_id IF NOT EXISTS "
               "FOR (c:Chunk) REQUIRE c.id IS UNIQUE")
    print("唯一性约束已就绪")


def import_extraction(results: list[dict]):
    """把抽取结果逐块写入 Neo4j。"""
    chunks_json = json.loads((OUTPUT_DIR / "chunks.json").read_text(encoding="utf-8"))
    chunk_texts = {c["id"]: c["text"] for c in chunks_json}

    driver = get_driver()
    with driver.session() as session:
        # results是读取extraction.json文件得到的内容
        for item in results:
            chunk_id = item["chunk_id"]

            # 1. 创建 Chunk 节点（保存原文，便于溯源和 Local Search 引用）
            session.run(
                "MERGE (c:Chunk {id: $id}) SET c.text = $text",
                id=chunk_id, text=chunk_texts.get(chunk_id, ""),
            )

            # 2. 合并实体节点
            for ent in item["entities"]:
                if not ent.get("name"):          # 防御：跳过模型偶尔输出的空名实体
                    continue
                # 清洗别名：去空白、去掉与实体名相同的"伪别名"
                name = ent["name"].strip()
                aliases = [a.strip() for a in ent.get("aliases", [])
                           if isinstance(a, str) and a.strip() and a.strip() != name]
                session.run(
                    """
                    MERGE (e:Entity {name: $name})
                    ON CREATE SET e.type = $type, e.description = $desc,
                                  e.aliases = $aliases
                    ON MATCH  SET e.description =
                        // 已存在时拼接新描述（避免无限膨胀：超过 500 字不再追加）
                        CASE WHEN size(e.description) < 500 AND NOT e.description CONTAINS $desc
                             THEN e.description + '；' + $desc
                             ELSE e.description END,
                        // 别名取并集：不同 chunk 可能给出不同别名
                        e.aliases = [x IN coalesce(e.aliases, []) | x] +
                                    [x IN $aliases WHERE NOT x IN coalesce(e.aliases, [])]
                    // aliases_text：把别名拼成一个字符串，专供 Lucene 全文索引使用
                    SET e.aliases_text = reduce(s = '', x IN e.aliases | s + ' ' + x)
                    WITH e
                    MATCH (c:Chunk {id: $chunk_id})
                    MERGE (e)-[:MENTIONED_IN]->(c)
                    """,
                    name=name,
                    type=ent.get("type", "未知"),
                    desc=ent.get("description", ""),
                    aliases=aliases,
                    chunk_id=chunk_id,
                )

            # 3. 合并关系
            for rel in item["relations"]:
                src, tgt = rel.get("source", "").strip(), rel.get("target", "").strip()
                if not src or not tgt or src == tgt:   # 防御：跳过残缺/自环关系
                    continue
                session.run(
                    """
                    // 关系两端实体可能没出现在 entities 列表里，同样用 MERGE 兜底创建
                    MERGE (a:Entity {name: $src})
                        ON CREATE SET a.type = '未知', a.description = '', a.aliases = []
                    MERGE (b:Entity {name: $tgt})
                        ON CREATE SET b.type = '未知', b.description = '', b.aliases = []
                    // 按 (a, relation, b) 三元组合并；重复出现则权重累加
                    MERGE (a)-[r:RELATES {relation: $relation}]->(b)
                    ON CREATE SET r.description = $desc,
                                  r.weight = $strength,
                                  r.chunk_id = $chunk_id
                    ON MATCH  SET r.weight = r.weight + $strength
                    """,
                    src=src, tgt=tgt,
                    relation=rel.get("relation", "相关"),
                    desc=rel.get("description", ""),
                    strength=float(rel.get("strength", 5)),
                    chunk_id=chunk_id,
                )

def print_stats():
    """打印建图结果统计，验证导入是否成功。"""
    n = run_cypher("MATCH (e:Entity) RETURN count(e) AS c")[0]["c"]
    r = run_cypher("MATCH (:Entity)-[r:RELATES]->(:Entity) RETURN count(r) AS c")[0]["c"]
    print(f"建图完成：{n} 个实体节点，{r} 条关系")

    print("\n度数最高（关联最多）的 10 个实体：")
    top = run_cypher("""
        MATCH (e:Entity)
        RETURN e.name AS name, e.type AS type,
               COUNT { (e)-[:RELATES]-() } AS degree
        ORDER BY degree DESC LIMIT 10
    """)
    for row in top:
        print(f"  {row['name']}（{row['type']}）: {row['degree']} 条关系")


def main():
    results = json.loads((OUTPUT_DIR / "extraction.json").read_text(encoding="utf-8"))
    clear_graph()
    create_constraints()
    import_extraction(results)
    print_stats()


if __name__ == "__main__":
    main()