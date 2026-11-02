# -*- coding: utf-8 -*-
"""压缩器：判断该不该压、把小模型生成的摘要写回 Redis。

核心思路是"滚动压缩"：
    旧摘要 + 已有事实 + 待压消息  -->  新摘要 + 新事实

最近 KEEP_TURNS 轮原文永远不压，保证当前对话连贯；
压过的消息用 XDEL 删掉，但游标 last_compacted_id 会记住压到哪了。
"""
import json

import config
import llm
import store

# 压缩质量几乎全看这个 prompt。要求越具体，小模型跑偏越少。
SUMMARY_SYSTEM = """你是会话压缩器。把「旧摘要」「已有事实」和「新增对话」合并成一份新的会话状态。

规则：
1. summary 不超过 200 字，只写：用户的目标、当前进展、已达成的结论。
2. facts 只抽取硬事实：编号、名称、数量、价格、日期、状态、路径，键名用简短英文。
3. 不写推理过程、寒暄、失败的尝试。
4. 被后续对话推翻的事实不要保留。
5. 只输出 JSON，不要任何解释。

输出格式：
{"summary": "……", "facts": {"order_id": "……", "amount": "……"}}"""


def should_compact(meta: dict):
    """判定是否需要压缩，需要则返回原因（用于打印），否则返回 None。"""
    if meta["tokens"] >= config.TOKEN_BUDGET:
        return f"token 超预算（{meta['tokens']} >= {config.TOKEN_BUDGET}）"
    if meta["turns"] and meta["turns"] % config.COMPACT_EVERY_N_TURNS == 0:
        return f"轮数达到阈值（{meta['turns']} 轮）"
    return None


def compact(r, sid: str, verbose: bool = True):
    """执行一次压缩。返回摘要信息 dict；没到压缩条件则返回 None。"""
    # 加锁：同一会话可能被并发触发，只让一个干活
    lock = store._key(sid, "lock")
    if not r.set(lock, "1", nx=True, px=config.LOCK_TTL_MS):
        if verbose:
            print("  [压缩] 已有压缩任务在跑，跳过")
        return None

    try:
        meta = store.get_meta(r, sid)
        pending = store.fetch_pending(r, sid)

        # 最近 KEEP_TURNS 轮（= KEEP_TURNS*2 条消息）留着不压
        keep = config.KEEP_TURNS * 2
        todo = pending[:-keep] if len(pending) > keep else []
        if len(todo) < config.MIN_BATCH:
            if verbose:
                print(f"  [压缩] 待压 {len(todo)} 条 < {config.MIN_BATCH}，不值得调模型")
            return None

        old_summary = r.get(store._key(sid, "summary")) or ""
        old_facts = r.hgetall(store._key(sid, "facts")) or {}

        convo = "\n".join(
            f"{'用户' if m.get('role') == 'user' else '助手'}: {m.get('c', '')}"
            for _, m in todo
        )
        user_prompt = (
            f"旧摘要：\n{old_summary or '（空）'}\n\n"
            f"已有事实：\n{json.dumps(old_facts, ensure_ascii=False)}\n\n"
            f"新增对话：\n{convo}"
        )

        summary, new_facts, engine = _run_digest(user_prompt, old_summary, old_facts, todo)

        before_tokens = meta["tokens"]
        retained = pending[len(todo):]
        retained_tokens = sum(store.estimate_tokens(m.get("c", "")) for _, m in retained)
        after_tokens = retained_tokens + store.estimate_tokens(summary)

        store.commit_compaction(
            r, sid,
            summary=summary,
            facts=new_facts,
            new_last_id=todo[-1][0],
            removed_ids=[mid for mid, _ in todo],
            retained_tokens=retained_tokens,
        )

        if verbose:
            print(f"  [压缩] 引擎={engine}  压掉 {len(todo)} 条  保留 {len(retained)} 条  "
                  f"摘要 {len(summary)} 字  事实 {len(new_facts)} 项")
            print(f"  [压缩] token: {before_tokens} -> {after_tokens}"
                  f"（压缩比约 {before_tokens / max(after_tokens, 1):.1f}:1）")

        return {"count": len(todo), "engine": engine, "summary": summary, "facts": new_facts}

    finally:
        r.delete(lock)


def _run_digest(user_prompt: str, old_summary: str, old_facts: dict, todo: list):
    """调小模型做摘要；小模型不可用时降级到规则拼接，保证流程不中断。

    返回 (summary, facts, engine_name)
    """
    try:
        raw = llm.small_model_chat(SUMMARY_SYSTEM, user_prompt, json_mode=True)
        data = json.loads(raw)
        summary = str(data.get("summary", "")).strip()
        if summary:
            facts = dict(old_facts)
            for k, v in (data.get("facts") or {}).items():
                key, val = str(k).strip(), str(v).strip()
                # 小模型偶尔会吐出空键值（如 order_id: ""），过滤掉
                if key and val:
                    facts[key] = val
            return summary, facts, f"ollama:{config.SMALL_MODEL}"
        print("  [压缩] 小模型返回空摘要，走规则降级")
    except Exception as exc:
        print(f"  [压缩] 小模型调用失败（{type(exc).__name__}: {exc}），走规则降级")

    return _rule_based(old_summary, old_facts, todo), dict(old_facts), "rule-fallback"


def _rule_based(old_summary: str, old_facts: dict, todo: list) -> str:
    """降级方案：把消息截断拼接。

    这不是真摘要，只是保证 ollama 挂了程序还能继续跑，
    并且仍然完成"删原文、留摘要"的流程，不至于上下文无限膨胀。
    """
    parts = [old_summary] if old_summary else []
    for _, m in todo:
        who = "用户" if m.get("role") == "user" else "助手"
        text = (m.get("c") or "").replace("\n", " ").strip()
        parts.append(f"{who}: {text[:60]}")
    return " / ".join(parts)[-200:]
