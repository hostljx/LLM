# -*- coding: utf-8 -*-
"""多轮对话主程序：Redis 存会话 + 小模型压缩摘要。

用法：
    python chat.py                 # 用假模型，不花积分，跑通全链路
    python chat.py --real          # 调真实 qwen3.8-max（会消耗积分）
    python chat.py --stats         # 只看当前会话在 Redis 里的状态
    python chat.py --reset         # 清空会话

对话中可用指令：
    /stats     查看 token、轮数、摘要、关键事实
    /compact   手动触发一次压缩
    /exit      退出
"""
import sys

import compressor
import config
import context
import llm
import store

SESSION_ID = "demo-001"


def print_stats(r, sid: str) -> None:
    """打印会话在 Redis 里的真实状态。"""
    s = store.dump_stats(r, sid)
    print("\n--- 会话状态 ---")
    print(f"消息条数    : {s['msgs']}")
    print(f"累计 token  : {s['tokens']}")
    print(f"轮数        : {s['turns']}")
    print(f"压缩次数    : {s['summary_ver']}")
    print(f"压缩游标    : {s['last_compacted_id']}")
    print(f"摘要        : {s['summary'] or '（尚未压缩）'}")
    print(f"关键事实    : {s['facts'] or '{}'}")
    print("----------------\n")


def main() -> None:
    real = "--real" in sys.argv           # 只有显式加 --real 才真调远程接口
    sid = SESSION_ID

    r = store.connect()

    if "--reset" in sys.argv:
        store.clear_session(r, sid)
        print(f"[chat] 已清空会话 {sid}")
        return

    if "--stats" in sys.argv:
        print_stats(r, sid)
        return

    # ---------- 启动信息 ----------
    model = llm.build_main_model() if real else llm.MockChatModel()
    main_desc = f"真实 {config.MAIN_MODEL}（会消耗积分）" if real else "Mock（不花积分）"

    ollama_ok = llm.is_ollama_alive()
    digest_desc = (f"本地 {config.SMALL_MODEL} @ ollama（0 积分）"
                   if ollama_ok else "ollama 未启动 → 将用规则降级")

    print(f"[chat] 主模型  : {main_desc}")
    print(f"[chat] 压缩模型: {digest_desc}")
    print(f"[chat] 会话 ID : {sid}")
    print("[chat] 指令：/stats 看状态  /compact 手动压缩  /exit 退出")

    # ---------- 对话循环 ----------
    while True:
        try:
            user_input = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[chat] 已退出")
            break

        if not user_input:
            continue
        if user_input in ("/exit", "/quit"):
            print("[chat] 已退出")
            break
        if user_input == "/stats":
            print_stats(r, sid)
            continue
        if user_input == "/compact":
            compressor.compact(r, sid)
            continue

        # 1) 写路径：消息进 Redis，并累加 token
        store.append(r, sid, "user", user_input, store.estimate_tokens(user_input))

        # 2) 判定：到阈值就压缩（真实项目里这一步应丢进后台队列，不阻塞用户）
        meta = store.get_meta(r, sid)
        reason = compressor.should_compact(meta)
        if reason:
            print(f"  [判定] 触发压缩：{reason}")
            compressor.compact(r, sid)

        # 3) 组装：摘要 + 事实 + 最近原文
        messages, info = context.build_messages(r, sid)
        print(f"  [上下文] 摘要 {info['summary_len']} 字 · 事实 {info['facts_count']} 项 · "
              f"原文 {info['recent']} 条 · 共 {info['total_msgs']} 条消息")

        # 4) 调用主模型
        reply = model.invoke(messages)
        text = reply.content if hasattr(reply, "content") else str(reply)
        print(f"\n助手: {text}")

        # 5) 回复也入库，下一轮才有完整上下文
        store.append(r, sid, "assistant", text, store.estimate_tokens(text))


if __name__ == "__main__":
    main()
