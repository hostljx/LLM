# -*- coding: utf-8 -*-
"""自检脚本：不花一分积分，自动跑完整链路并验证压缩是否生效。

灌一段模拟对话进 Redis，边走边触发压缩，最后把压缩前后对比打印出来。
作业演示 / 改完代码验证，用这个比手动聊天快得多。

用法：
    python selfcheck.py
"""
import compressor
import config
import context
import store

SID = "selfcheck-001"

# 一段带硬事实的模拟对话，正好用来检验小模型能不能把单号、金额抽进 facts
DIALOG = [
    ("user", "我想问下 Q3 的出差报销流程，我上周去上海出差了。"),
    ("assistant", "需要提交三样：行程单、发票、审批单。注意发票抬头必须是公司全称。"),
    ("user", "我发票开成个人抬头了，会有什么问题吗？"),
    ("assistant", "个人抬头财务不予受理，需要退回重开。"),
    ("user", "好，我重新开了。我的工单号是 BX-2026-1030，金额 4820 元。"),
    ("assistant", "已记录。工单 BX-2026-1030，金额 4820 元，提交后 5 个工作日到账。"),
    ("user", "到账是打到工资卡吗？"),
    ("assistant", "是的，默认打工资卡。要换卡需要在系统里提前修改。"),
    ("user", "顺便问下上海住宿标准是多少？"),
    ("assistant", "上海属于一类城市，住宿标准 600 元每晚。"),
    ("user", "我住了 4 晚，超了怎么办？"),
    ("assistant", "超出部分需要部门经理额外审批，你 4 晚一共超出 800 元。"),
    ("user", "知道了，帮我把这些都记下来。"),
    ("assistant", "已汇总：工单 BX-2026-1030，金额 4820 元，住宿 4 晚，超标 800 元待审批。"),
]


def main():
    r = store.connect()
    store.clear_session(r, SID)

    print(f"\n[自检] 开始灌入 {len(DIALOG)} 条消息，预算 {config.TOKEN_BUDGET} token\n")

    compact_count = 0
    for i, (role, text) in enumerate(DIALOG, 1):
        store.append(r, SID, role, text, store.estimate_tokens(text))
        meta = store.get_meta(r, SID)

        reason = compressor.should_compact(meta)
        if reason:
            print(f"  >> 第 {i} 条后触发压缩：{reason}")
            if compressor.compact(r, SID):
                compact_count += 1

    # ---------- 压缩结果 ----------
    print("\n" + "=" * 56)
    print("压缩后会话状态")
    print("=" * 56)
    s = store.dump_stats(r, SID)
    print(f"剩余原文条数 : {s['msgs']}（其余已并入摘要）")
    print(f"压缩次数     : {s['summary_ver']}")
    print(f"当前 token   : {s['tokens']}")
    print(f"\n摘要：\n{s['summary'] or '（未触发压缩）'}")
    print(f"\n关键事实：\n{s['facts'] or '{}'}")

    # ---------- 最终上下文结构 ----------
    messages, info = context.build_messages(r, SID)
    print("\n" + "=" * 56)
    print("下一轮实际发给主模型的内容")
    print("=" * 56)
    print(f"消息条数 {info['total_msgs']}（system 1 + 原文 {info['recent']}）")
    print(f"system 长度 {len(messages[0].content)} 字")
    print("-" * 56)
    print(messages[0].content[:600])
    print("-" * 56)
    for m in messages[1:]:
        print(f"[{m.type}] {m.content[:50]}")

    # ---------- 断言 ----------
    print("\n" + "=" * 56)
    ok = True
    if compact_count == 0 and len(DIALOG) * 40 > config.TOKEN_BUDGET:
        print("✗ 没有触发任何压缩，检查阈值配置")
        ok = False
    if s["summary_ver"] > 0 and not s["summary"]:
        print("✗ 压缩了但摘要为空")
        ok = False
    if s["summary_ver"] > 0 and s["msgs"] >= len(DIALOG):
        print("✗ 压缩了但原文没被清理")
        ok = False
    print("✓ 自检通过" if ok else "✗ 自检失败")
    print("=" * 56)


if __name__ == "__main__":
    main()
