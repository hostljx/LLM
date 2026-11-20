"""
Global Search（全局检索问答）

适用问题：需要"通观全局"才能回答的主题型/汇总型问题
  例："这些故事里有哪几大势力集团？" "整个故事的主线是什么？"

传统 RAG 答不好这类问题：向量检索只能召回零散的文本片段，
拼不出全局图景。GraphRAG 的解法是 Map-Reduce：

  Map 阶段：  把问题分别和【每一份社区摘要】配对，让 LLM 从中
              提取与问题相关的要点，并打相关性分数（0-100）
  Reduce 阶段：汇总所有社区的高分要点，让 LLM 综合成最终答案

这正是微软 GraphRAG Global Search 的核心思想（官方还支持动态筛选社区层级等优化）。

"""
import sys

from config import chat, parse_json_safely, run_cypher

MAP_PROMPT = """下面是知识库中一个主题社区的摘要报告。请判断它是否有助于回答用户问题，
并提取相关要点，以 json 格式输出：

{{
  "points": [
    {{"point": "与问题相关的一条具体要点", "score": 85}}
  ]
}}

说明：score 为该要点对回答问题的重要性（0-100）；如果这份摘要与问题完全无关，输出 {{"points": []}}。

【社区摘要】
《{title}》
{summary}

【用户问题】
{question}
"""

REDUCE_PROMPT = """你是知识问答助手。下面是从知识库各主题社区中提取出的要点（已按重要性排序）。
请综合这些要点，条理清晰地回答用户问题。只使用给出的要点，不要编造。

【要点列表】
{points}

【用户问题】
{question}
"""


def map_phase(question: str, communities: list[dict]) -> list[dict]:
    """Map：逐个社区提取相关要点。返回 [{point, score, community}, ...]"""
    all_points = []
    for i, comm in enumerate(communities, 1):
        raw = chat(
            MAP_PROMPT.format(title=comm["title"], summary=comm["summary"], question=question),
            json_mode=True, temperature=0.3,
        )
        data = parse_json_safely(raw)
        points = data.get("points", []) if data else []
        # 只保留有效评分的要点
        for p in points:
            if isinstance(p, dict) and p.get("point") and isinstance(p.get("score"), (int, float)):
                all_points.append({**p, "community": comm["title"]})
        print(f"  [Map {i}/{len(communities)}] 《{comm['title']}》→ {len(points)} 条要点")
    return all_points


def reduce_phase(question: str, points: list[dict]) -> str:
    """Reduce：汇总高分要点，生成最终答案。"""
    # 按分数降序排列，重要的信息放前面（LLM 对开头的内容更敏感）
    points.sort(key=lambda p: p["score"], reverse=True)
    lines = [f"- [{p['score']}分][来自《{p['community']}》] {p['point']}" for p in points]
    return chat(
        REDUCE_PROMPT.format(points="\n".join(lines), question=question),
        temperature=0.7,
    )


def global_search(question: str) -> str:
    """完整的 Global Search 流程。"""
    # 取出第五步生成的全部社区摘要
    communities = run_cypher(
        "MATCH (c:Community) RETURN c.title AS title, c.summary AS summary "
        "ORDER BY c.size DESC"
    )
    if not communities:
        return "尚未生成社区摘要，请先运行 05_summarize.py"
    print(f"[Map 阶段] 扫描 {len(communities)} 个社区摘要...")

    points = map_phase(question, communities)
    if not points:
        return "所有社区摘要都与该问题无关，无法回答。"
    print(f"[Reduce 阶段] 汇总 {len(points)} 条要点生成最终答案...\n")

    return reduce_phase(question, points)


if __name__ == "__main__":
    q = "这些故事里有哪几大势力集团？各自的核心人物是谁？"
    print(f"问题：{q}\n")
    print("回答：\n" + global_search(q))