"""
LLM 生成社区摘要

这是 GraphRAG 区别于传统 RAG 的核心一步：
对每个社区，把"社区内的实体 + 实体间的关系"整理成结构化文本，
交给 LLM 写一份主题摘要（社区报告）。

这些摘要 = 对整个知识库的分主题"预消化"，
Global Search 回答全局性问题时，靠的就是它们。

图模型新增：
  (:Community {id, title, summary, size})    社区节点
  (:Entity)-[:IN_COMMUNITY]->(:Community)    实体归属关系
"""
import json
from config import OUTPUT_DIR, chat, parse_json_safely, run_cypher

MIN_COMMUNITY_SIZE = 3  # 少于 3 个实体的社区信息量太小，不值得花钱做摘要
MAX_CONTEXT_CHARS = 3000  # 送给 LLM 的社区上下文上限，防止超长

SUMMARY_PROMPT = """你是知识图谱分析专家。下面是知识图谱中一个社区（联系紧密的实体群）的全部信息，
包括实体列表和实体间的关系。请生成这个社区的分析报告，以 json 格式输出：

{{
  "title": "社区标题（10字以内，概括社区主题，如'蜀汉核心集团'）",
  "summary": "社区综述（200字左右：这个社区围绕什么主题？核心成员是谁？内部有哪些关键关系和事件？）"
}}

社区信息：
{context}
"""

def fetch_communities() -> list[dict]:
    """从 Neo4j 取出每个社区的实体清单（含类型和描述）。"""
    return run_cypher("""
        MATCH (e:Entity)
        WHERE e.communityId IS NOT NULL
        WITH e.communityId AS cid,
             collect({name: e.name, type: e.type, description: e.description}) AS entities
        WHERE size(entities) >= $min_size
        RETURN cid, entities
        ORDER BY size(entities) DESC
    """, min_size=MIN_COMMUNITY_SIZE)

def fetch_inner_relations(cid: int) -> list[dict]:
    """取出社区内部（两端都属于该社区）的所有关系。"""
    return run_cypher("""
        MATCH (a:Entity)-[r:RELATES]->(b:Entity)
        WHERE a.communityId = $cid AND b.communityId = $cid
        RETURN a.name AS source, r.relation AS relation,
               r.description AS description, b.name AS target
        ORDER BY r.weight DESC
    """, cid=cid)


def build_context(entities: list[dict], relations: list[dict]) -> str:
    """把实体和关系拼成给 LLM 看的纯文本上下文。"""
    lines = ["【实体】"]
    for e in entities:
        lines.append(f"- {e['name']}（{e['type']}）：{e['description']}")
    lines.append("【关系】")
    for r in relations:
        lines.append(f"- {r['source']} --{r['relation']}--> {r['target']}：{r['description']}")
    context = "\n".join(lines)
    return context[:MAX_CONTEXT_CHARS]   # 截断保护


def summarize_community(cid: int, entities: list[dict]) -> dict | None:
    """对单个社区调用 LLM 生成 {title, summary}。"""
    relations = fetch_inner_relations(cid)
    context = build_context(entities, relations)
    # print("[社区实体和关系构建出来的上下文]：", context)
    raw = chat(SUMMARY_PROMPT.format(context=context), json_mode=True, temperature=0.5)
    data = parse_json_safely(raw)
    if data and data.get("title") and data.get("summary"):
        return data
    return None


def save_community(cid: int, title: str, summary: str, size: int):
    """把社区摘要存成 Community 节点，并连接其成员实体。"""
    run_cypher("""
        MERGE (c:Community {id: $cid})
        SET c.title = $title, c.summary = $summary, c.size = $size
        WITH c
        MATCH (e:Entity) WHERE e.communityId = $cid
        MERGE (e)-[:IN_COMMUNITY]->(c)
    """, cid=cid, title=title, summary=summary, size=size)


def main():
    communities = fetch_communities()
    print(f"需要生成摘要的社区（成员>={MIN_COMMUNITY_SIZE}）共 {len(communities)} 个")

    reports = []
    for i, comm in enumerate(communities, 1):
        cid, entities = comm["cid"], comm["entities"]
        data = summarize_community(cid, entities)
        if data is None:
            print(f"  [跳过] 社区 {cid} 摘要生成失败")
            continue
        save_community(cid, data["title"], data["summary"], len(entities))
        reports.append({"cid": cid, "size": len(entities), **data})
        print(f"  [{i}/{len(communities)}] 社区 {cid}（{len(entities)}个实体）→ 《{data['title']}》")

    # 同时落盘一份 JSON，方便直接查看
    out_file = OUTPUT_DIR / "community_reports.json"
    out_file.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n社区摘要已写入 Neo4j 并保存到 {out_file}")

    for rep in reports:
        print(f"\n=== 《{rep['title']}》(社区{rep['cid']}, {rep['size']}个实体) ===")
        print(rep["summary"])


if __name__ == "__main__":
    main()

