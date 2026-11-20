"""
Local Search（局部检索问答）

适用问题：围绕具体实体的细节型问题
  例："关羽和曹操是什么关系？" "赤壁之战谁赢了？"

流程：
  1. 实体链接   从问题定位图谱起始节点（★ 四级漏斗）
  2. 邻居扩展   取这些实体的 1 跳邻居关系（Cypher 图遍历）
  3. 上下文组装  实体描述 + 关系 + 所属社区摘要 + 原文 chunk
  4. LLM 生成   基于上下文回答，禁止编造

节点入口问题（Entity Linking）—— 四级漏斗：
  ①精确匹配层  实体名+别名子串匹配          零模型调用，覆盖大部分提问
  ②全文检索层  Lucene 全文索引模糊打分      解决错别字/部分匹配，零调用
  ③向量语义层  bge 向量 + Neo4j 向量索引    解决同义改写（"七擒孟获的人"→诸葛亮）
  ④LLM兜底层   把②③的候选 Top-N 交给 LLM 精选（不发全量实体，可扩展到百万级）
上一层命中即停（串行降级），逐层成本递增、覆盖面递增。


布置任务： 自习时间的时候，将四级漏斗的代码流程熟悉一下。

"""
import re
import sys

from config import chat, embed_texts, parse_json_safely, run_cypher

MAX_ENTITIES = 5     # 最多围绕几个实体展开
MAX_RELATIONS = 15   # 每个实体最多取多少条关系
MAX_CHUNKS = 2       # 每个实体最多附几段原文

FULLTEXT_TOP_K = 8           # 全文检索取的候选数
FULLTEXT_TRIGGER_SCORE = 8.0  # 触发阈值：榜首分数达到该值才信任全文层结果。
                              # 中文按"字"切词几乎任何问题都能匹到点什么，
                              # 榜首分低说明只是弱噪声匹配，应降级到向量层
FULLTEXT_REL_RATIO = 0.4      # 保留阈值：只留分数 >= 榜首*0.4 的结果
VECTOR_TOP_K = 8          # 向量检索取的候选数
VECTOR_MIN_SCORE = 0.80   # 向量相似度下限（Neo4j cosine 得分域 [0,1]）

ANSWER_PROMPT = """你是三国知识问答助手。请严格根据下面提供的知识图谱上下文回答用户问题。
规则：
1. 只使用上下文中的信息，不要编造上下文以外的内容
2. 回答要具体、有条理，可引用关系和原文佐证
3. 如果上下文不足以回答，直接说明"根据现有知识无法回答"

【知识图谱上下文】
{context}

【用户问题】
{question}
"""


# ============================================================
# 实体链接：四级漏斗
# ============================================================

def _level1_exact(question: str) -> list[str]:
    """①精确匹配层：实体名 + 别名 的子串匹配（毫秒级，零模型调用）。

    按名字长度降序匹配：先匹配"诸葛亮"再匹配"亮"，避免短名误伤。
    别名同样参与匹配："关公败走麦城" 通过别名"关公"命中实体"关羽"。
    """
    rows = run_cypher(
        "MATCH (e:Entity) RETURN e.name AS name, coalesce(e.aliases, []) AS aliases")
    # 建立 "可匹配词 -> 标准实体名" 映射（名字 + 所有别名）
    term_to_name = {}
    for r in rows:
        term_to_name[r["name"]] = r["name"]
        for alias in r["aliases"]:
            term_to_name.setdefault(alias, r["name"])

    matched = []
    for term in sorted(term_to_name, key=len, reverse=True):   # 长词优先
        if len(term) >= 2 and term in question:
            name = term_to_name[term]
            if name not in matched:
                matched.append(name)
    return matched[:MAX_ENTITIES]


def _escape_lucene(text: str) -> str:
    """转义 Lucene 查询语法的特殊字符，防止用户输入破坏查询。"""
    return re.sub(r'([+\-!(){}\[\]^"~*?:\\/]|&&|\|\|)', r"\\\1", text)


def _level2_fulltext(question: str) -> list[dict]:
    """②全文检索层：Lucene 模糊打分（零模型调用）。

    整句丢给全文索引，Lucene 对中文按字切词后做 OR 匹配 + TF-IDF 打分：
    问题里带错别字（"诸蔼亮"）或只出现部分名称时依然能召回。
    返回 [{name, score}]，供上层判断/做候选。
    """
    rows = run_cypher("""
        CALL db.index.fulltext.queryNodes('entity_fulltext', $q, {limit: $k})
        YIELD node, score
        RETURN node.name AS name, score
    """, q=_escape_lucene(question), k=FULLTEXT_TOP_K)
    if not rows:
        return []
    top = rows[0]["score"]
    # 触发条件：榜首分数必须足够高（强词法信号），否则整层放弃、降级到向量层。
    # 通过后按相对阈值保留与榜首接近的结果（错别字场景下正确实体常排第 2、3 位）
    if top < FULLTEXT_TRIGGER_SCORE:
        return []
    return [r for r in rows if r["score"] >= top * FULLTEXT_REL_RATIO]


def _level3_vector(question: str) -> list[dict]:
    """③向量语义层：本地 bge 编码问题 → Neo4j 向量索引 Top-K。

    解决"问题里根本没出现实体名"的语义改写场景：
    "七擒孟获的那个人是谁" 与实体卡片"诸葛亮。别名：孔明...描述：...七擒七纵孟获..."
    在向量空间中距离很近。
    """
    qvec = embed_texts([question], is_query=True)[0]    # 查询侧编码要加 bge 指令前缀
    rows = run_cypher("""
        CALL db.index.vector.queryNodes('entity_vector', $k, $vec)
        YIELD node, score
        RETURN node.name AS name, score
    """, k=VECTOR_TOP_K, vec=qvec)
    return [r for r in rows if r["score"] >= VECTOR_MIN_SCORE]


def _level4_llm(question: str, candidates: list[str]) -> list[str]:
    """④LLM 兜底层：把②③捞到的候选（而非全量实体！）交给 LLM 精选。

    注意：候选集是漏斗前几层收敛出来的 Top-N，
    无论图谱有多少实体，这里的 prompt 都是常数大小 → 可扩展。
    """
    if not candidates:
        # 极端兜底：前几层一个候选都没有，取度数最高的 30 个"核心实体"当候选
        rows = run_cypher("""
            MATCH (e:Entity)
            RETURN e.name AS name ORDER BY COUNT { (e)-[:RELATES]-() } DESC LIMIT 30
        """)
        candidates = [r["name"] for r in rows]

    raw = chat(
        f"候选实体列表：{'、'.join(candidates)}\n\n"
        f"用户问题：{question}\n\n"
        f"请从候选列表中挑出与问题最相关的至多 {MAX_ENTITIES} 个实体，"
        f'以 json 格式输出：{{"entities": ["实体1", "实体2"]}}。'
        f"如果都不相关，输出 {{\"entities\": []}}",
        json_mode=True,
    )
    data = parse_json_safely(raw)
    if data and isinstance(data.get("entities"), list):
        return [e for e in data["entities"] if e in candidates][:MAX_ENTITIES]
    return []


def find_entities_in_question(question: str) -> list[str]:
    """四级漏斗统一入口：上一层命中即返回，未命中降级到下一层。"""
    # ① 精确匹配（名字+别名）
    names = _level1_exact(question)
    if names:
        print(f"[实体链接] ①精确匹配层命中: {names}")
        return names

    # ② 全文模糊检索
    ft = _level2_fulltext(question)
    if ft:
        names = [r["name"] for r in ft][:MAX_ENTITIES]
        print(f"[实体链接] ②全文检索层命中: "
              f"{[(r['name'], round(r['score'], 2)) for r in ft]}")
        return names

    # ③ 向量语义检索
    vt = _level3_vector(question)
    if vt:
        names = [r["name"] for r in vt][:MAX_ENTITIES]
        print(f"[实体链接] ③向量语义层命中: "
              f"{[(r['name'], round(r['score'], 3)) for r in vt]}")
        return names

    # ④ LLM 兜底：合并三路候选（全文 Top15 + 向量 Top15 + 度数 Top20 核心实体），
    #   让 LLM 从中精选。候选池宁可大一点——LLM 挑选的准确率远高于阈值硬切
    loose = [r["name"] for r in run_cypher("""
        CALL db.index.fulltext.queryNodes('entity_fulltext', $q, {limit: 15})
        YIELD node RETURN node.name AS name
    """, q=_escape_lucene(question))]
    qvec = embed_texts([question], is_query=True)[0]
    loose += [r["name"] for r in run_cypher("""
        CALL db.index.vector.queryNodes('entity_vector', 15, $vec)
        YIELD node RETURN node.name AS name
    """, vec=qvec)]
    # 度数最高的核心实体兜底：开放性问题（"谁最勇猛"）往往落在核心人物上
    loose += [r["name"] for r in run_cypher("""
        MATCH (e:Entity)
        RETURN e.name AS name ORDER BY COUNT { (e)-[:RELATES]-() } DESC LIMIT 20
    """)]
    candidates = list(dict.fromkeys(loose))            # 去重保序
    names = _level4_llm(question, candidates)
    if names:
        print(f"[实体链接] ④LLM兜底层命中: {names}")
    return names


# ============================================================
# 上下文组装与回答（与四级漏斗解耦，逻辑不变）
# ============================================================

def build_context(entity_names: list[str]) -> str:
    """第 2、3 步：以实体为中心收集上下文。"""
    sections = []
    for name in entity_names:
        lines = []

        # (a) 实体自身信息 + 所属社区摘要
        info = run_cypher("""
            MATCH (e:Entity {name: $name})
            OPTIONAL MATCH (e)-[:IN_COMMUNITY]->(c:Community)
            RETURN e.type AS type, e.description AS desc,
                   c.title AS c_title, c.summary AS c_summary
        """, name=name)
        if not info:
            continue
        row = info[0]
        lines.append(f"### 实体：{name}（{row['type']}）")
        lines.append(f"描述：{row['desc']}")
        if row["c_title"]:
            lines.append(f"所属社区《{row['c_title']}》：{row['c_summary']}")

        # (b) 1 跳邻居关系（双向都要，按权重取最重要的若干条）
        rels = run_cypher("""
            MATCH (e:Entity {name: $name})-[r:RELATES]-(other:Entity)
            RETURN CASE WHEN startNode(r) = e THEN e.name ELSE other.name END AS source,
                   r.relation AS relation, r.description AS desc,
                   CASE WHEN startNode(r) = e THEN other.name ELSE e.name END AS target
            ORDER BY r.weight DESC LIMIT $limit
        """, name=name, limit=MAX_RELATIONS)
        if rels:
            lines.append("关联关系：")
            for r in rels:
                lines.append(f"- {r['source']} --{r['relation']}--> {r['target']}：{r['desc']}")

        # (c) 提到该实体的原文片段（证据溯源）
        chunks = run_cypher("""
            MATCH (e:Entity {name: $name})-[:MENTIONED_IN]->(c:Chunk)
            RETURN c.text AS text LIMIT $limit
        """, name=name, limit=MAX_CHUNKS)
        if chunks:
            lines.append("相关原文：")
            for c in chunks:
                lines.append(f"> {c['text'][:200]}...")

        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def local_search(question: str) -> str:
    """完整的 Local Search 流程。"""
    entities = find_entities_in_question(question)
    if not entities:
        return "抱歉，未能在知识图谱中定位到与问题相关的实体。"

    context = build_context(entities)
    print(f"[上下文长度] {len(context)} 字符\n")

    # 第 4 步：LLM 基于上下文生成回答
    return chat(ANSWER_PROMPT.format(context=context, question=question), temperature=0.7)


if __name__ == "__main__":
    q = "关羽和曹操是什么关系？"
    print(f"问题：{q}\n")
    print("回答：\n" + local_search(q))