# -*- coding: utf-8 -*-
"""上下文组装：把 Redis 里的摘要、事实、最近原文拼成模型输入。

注意这里不再额外追加当前提问 —— 用户消息在写路径已经进 Redis 了，
最近 N 条原文里就包含它，再追加一次会重复。
"""
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import config
import store

BASE_SYSTEM = "你是一个耐心的助手，回答简洁，用中文作答。"


def build_messages(r, sid: str):
    """组装本轮喂给主模型的完整消息列表。

    结构：
        system  = 基础人设 + 历史摘要 + 关键事实
        history = 最近 KEEP_TURNS*2 条原文
    返回 (messages, info)，info 用于打印诊断信息。
    """
    summary, facts, recent = store.load_for_context(r, sid, config.KEEP_TURNS * 2)

    system_parts = [BASE_SYSTEM]
    if summary:
        system_parts.append(f"【历史摘要】\n{summary}")
    if facts:
        kv = "\n".join(f"- {k}: {v}" for k, v in facts.items())
        # 强调一句，防止模型在长对话里把金额、单号记串
        system_parts.append(f"【关键事实，回答时必须与此保持一致】\n{kv}")

    messages = [SystemMessage(content="\n\n".join(system_parts))]

    for _, m in recent:
        content = m.get("c", "")
        if m.get("role") == "user":
            messages.append(HumanMessage(content=content))
        else:
            messages.append(AIMessage(content=content))

    info = {
        "summary_len": len(summary),
        "facts_count": len(facts),
        "recent": len(recent),
        "total_msgs": len(messages),
    }
    return messages, info
