"""
实体链接索引（全文 + 向量）

解决 GraphRAG 的"节点入口"问题（Entity Linking）：
用户的自然语言问题 → 图谱中的起始实体节点。

主要为"四级漏斗"准备好两层基础设施：
  ② 全文检索层：Neo4j Lucene FULLTEXT INDEX
     - 索引 name / aliases_text / description 三个字段
     - 解决错别字、部分匹配、描述性提及（零模型调用）

  ③ 向量语义层：Neo4j 原生 VECTOR INDEX（5.13+ 内置，无需插件）
     - 用本地 bge-small-zh-v1.5 给每个实体计算 512 维向量
     - "实体卡片"= 名称 + 别名 + 类型 + 描述，语义信息最完整
     - 解决同义改写："七擒孟获的那个人" → 诸葛亮
"""

from config import EMBEDDING_DIM, embed_texts, get_driver, run_cypher

BATCH_SIZE = 64   # 每批编码/写入的实体数量

def create_fulltext_index():
    """② 全文检索层：创建 Lucene 全文索引"""
    run_cypher("""
        CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS
        FOR (e:Entity) ON EACH [e.name, e.aliases_text, e.description]
    """)
    print("[OK] 全文索引 entity_fulltext 已就绪（name / aliases_text / description）")

def create_vector_index():
    """③ 向量语义层：创建原生向量索引 """
    # cosine + 归一化向量 是文本检索的标准组合（bge 编码时已做归一化）
    run_cypher(f"""
        CREATE VECTOR INDEX entity_vector IF NOT EXISTS
        FOR (e:Entity) ON (e.embedding)
        OPTIONS {{indexConfig: {{
            `vector.dimensions`: {EMBEDDING_DIM},
            `vector.similarity_function`: 'cosine'
        }}}}
    """)
    print(f"[OK] 向量索引 entity_vector 已就绪（{EMBEDDING_DIM} 维, cosine）")


def build_entity_card(e: dict) -> str:
    """
    构造实体的"语义卡片"文本用于编码。
    卡片包含名称/别名/类型/描述——比只编码名称的语义信息丰富得多，
    这样"七擒孟获"这类描述性问题才能通过向量检索命中"诸葛亮"。
    """
    aliases = "、".join(e.get("aliases") or [])
    parts = [e["name"]]
    if aliases:
        parts.append(f"别名：{aliases}")
    parts.append(f"类型：{e['type']}")
    if e.get("description"):
        parts.append(f"描述：{e['description']}")
    return "。".join(parts)

def embed_all_entities():
    """给所有实体计算 embedding 并批量写回节点属性。"""
    entities = run_cypher("""
        MATCH (e:Entity)
        RETURN e.name AS name, e.type AS type,
               e.aliases AS aliases, e.description AS description
    """)
    print(f"开始为 {len(entities)} 个实体计算向量（本地 bge 模型, CPU）...")

    driver = get_driver()
    with driver.session() as session:
        # 分批处理：编码一批 → 写入一批
        for start in range(0, len(entities), BATCH_SIZE):
            batch = entities[start:start + BATCH_SIZE]
            cards = [build_entity_card(e) for e in batch]
            vectors = embed_texts(cards, is_query=False)   # 文档侧编码，不加查询前缀

            # UNWIND 批量写入：一次网络往返写一整批，生产环境的标准做法
            # db.create.setNodeVectorProperty 以紧凑格式存储向量（官方推荐）
            session.run(
                """
                UNWIND $rows AS row
                MATCH (e:Entity {name: row.name})
                CALL db.create.setNodeVectorProperty(e, 'embedding', row.vector)
                """,
                rows=[{"name": e["name"], "vector": v} for e, v in zip(batch, vectors)],
            )
            print(f"  [{min(start + BATCH_SIZE, len(entities))}/{len(entities)}] 已写入")


def verify():
    """自检：全文索引和向量索引各查一次，确认可用。"""
    # 全文索引测试：模糊搜"孔明"（"诸葛亮"的别名，靠 aliases_text 命中）
    ft = run_cypher("""
        CALL db.index.fulltext.queryNodes('entity_fulltext', $q, {limit: 3})
        YIELD node, score
        RETURN node.name AS name, round(score, 2) AS score
    """, q="孔明")
    print(f"\n[自检] 全文搜索'孔明' → {ft}")

    # 向量索引测试：语义搜"七擒孟获的人"
    qvec = embed_texts(["七擒孟获的人是谁"], is_query=True)[0]
    vt = run_cypher("""
        CALL db.index.vector.queryNodes('entity_vector', 3, $vec)
        YIELD node, score
        RETURN node.name AS name, round(score, 3) AS score
    """, vec=qvec)
    print(f"[自检] 向量搜索'七擒孟获的人是谁' → {vt}")


def main():
    create_fulltext_index()
    create_vector_index()
    embed_all_entities()
    verify()


if __name__ == "__main__":
    main()