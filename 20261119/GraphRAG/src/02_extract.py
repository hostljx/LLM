"""
实体和关系的抽取
"""
import json
import time
from config import OUTPUT_DIR, chat, parse_json_safely

# 限定实体类型：类型收敛能显著提升图谱质量（避免模型自由发挥出几十种类型）
ENTITY_TYPES = ["人物", "势力", "地点", "事件", "物品"]

# 系统提示词：定义模型的角色和总体要求
SYSTEM_PROMPT = "你是一个知识图谱构建专家，擅长从中文文本中精准抽取实体和关系。"

# 抽取提示词模板：
# - 明确输出 JSON 格式（JSON 模式要求 prompt 中必须出现 "json" 字样和格式说明）
# - 用一个 few-shot 示例锚定字段含义
# - 强调实体名用全称（"关羽"而不是"关公/云长"），这是后续实体对齐的关键
EXTRACT_PROMPT = """请从下面的文本中抽取实体和关系，以 json 格式输出。

要求：
1. 实体类型只能是：{entity_types}
2. 实体名必须使用最常用的标准全称（例如统一用"关羽"，不要用"关公""云长"）
3. aliases 列出该实体在文中或常识中的别名、字号、称号（没有则给空数组）。
   别名用于后续检索时的实体链接，请尽量完整
4. 关系是有方向的：source 是发出方，target 是接收方
5. strength 表示关系强度，1-10 的整数，关系越重要越紧密分数越高
6. description 用一句话概括，信息要具体

输出 json 格式示例：
{{
  "entities": [
    {{"name": "关羽", "type": "人物", "aliases": ["关公", "云长", "美髯公", "武圣"],
      "description": "刘备的结义兄弟，蜀汉五虎上将之首"}}
  ],
  "relations": [
    {{"source": "刘备", "target": "关羽", "relation": "结义兄弟", "description": "刘备与关羽在桃园结拜为异姓兄弟", "strength": 10}}
  ]
}}

待抽取文本：
{text}
"""

MAX_RETRIES = 3   # 单个 chunk 的最大重试次数

def extract_chunk(chunk: dict) -> dict | None:
    """对单个 chunk 调用 LLM 抽取，返回 {"entities": [...], "relations": [...]}。"""
    prompt = EXTRACT_PROMPT.format(entity_types="、".join(ENTITY_TYPES), text=chunk["text"])

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # json_mode=True 启用 JSON Output；temperature=0.3 让抽取结果尽量稳定
            raw = chat(prompt, system=SYSTEM_PROMPT, json_mode=True, temperature=0.3)
            data = parse_json_safely(raw)
            # 校验最基本的结构，缺字段视为失败
            if data and isinstance(data.get("entities"), list) and isinstance(data.get("relations"), list):
                return data
            print(f"  [警告] {chunk['id']} 第 {attempt} 次输出格式异常，重试...")
        except Exception as e:
            print(f"  [警告] {chunk['id']} 第 {attempt} 次调用失败: {e}，重试...")
            time.sleep(2 * attempt)   # 线性退避等待（2s → 4s），给 API 服务端恢复时间
    return None


def main():
    chunks = json.loads((OUTPUT_DIR / "chunks.json").read_text(encoding="utf-8"))
    print(f"开始抽取 {len(chunks)} 个 chunk（逐块调用 DeepSeek，请耐心等待）...")

    results = []
    for i, chunk in enumerate(chunks, 1):
        data = extract_chunk(chunk)
        if data is None:
            print(f"  [跳过] {chunk['id']} 多次重试仍失败")
            continue
        # 记录来源 chunk_id：知识图谱的每条信息都要能追溯到原文（provenance）
        results.append({
            "chunk_id": chunk["id"],
            "entities": data["entities"],
            "relations": data["relations"],
        })
        print(f"  [{i}/{len(chunks)}] {chunk['id']}: "
              f"实体 {len(data['entities'])} 个, 关系 {len(data['relations'])} 条")

    out_file = OUTPUT_DIR / "extraction.json"
    out_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # 汇总统计
    total_e = sum(len(r["entities"]) for r in results)
    total_r = sum(len(r["relations"]) for r in results)
    print(f"抽取完成：共 {total_e} 个实体提及、{total_r} 条关系，已保存到 {out_file}")
    print("（注意：不同 chunk 会重复提到同一实体，去重合并在下一步建图时完成）")
    return results


if __name__ == "__main__":
    main()

