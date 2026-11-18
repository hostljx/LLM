"""并发与重复投递压测（不需要钉钉 / LLM）。

用法（先启动 Redis）：

    .venv\\Scripts\\python.exe scripts\\stress_test.py

验证内容：
1. 同一个 msgId 并发投递 20 次，只入队 1 次（原子去重）
2. 50 条不同消息并发投递，全部入队
3. 消费组能正常读出并 ACK
4. 延迟重试队列能到期弹出并重新入队
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path

# 允许直接运行 scripts/stress_test.py，而不要求先 pip install -e .
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dingdingrobot.config import settings  # noqa: E402
from dingdingrobot.queue import RedisQueue, create_redis  # noqa: E402


def make_payload(msg_id: str) -> dict:
    return {
        "msg_id": msg_id,
        "conversation_id": "conv-selftest",
        "conversation_type": "1",
        "conversation_title": "",
        "sender_id": "user-1",
        "sender_staff_id": "staff-1",
        "sender_nick": "测试用户",
        "text": "现在几点？",
        "message_type": "text",
        "robot_code": "test-robot",
        "session_webhook": "",
        "session_webhook_expires_at": 0,
        "create_at": 0,
        "is_in_at_list": True,
    }


async def main() -> None:
    redis = create_redis(settings.redis_url)
    token = uuid.uuid4().hex[:8]
    prefix = f"{settings.key_prefix}:selftest:{token}"

    queue = RedisQueue(
        redis,
        stream=f"{prefix}:stream",
        group=f"{prefix}:group",
        dlq=f"{prefix}:dlq",
        retry_key=f"{prefix}:retry",
        dedup_ttl=300,
        maxlen=10000,
        key_prefix=prefix,
    )
    await queue.ensure_group()

    try:
        print("1) 同一个 msg_id 并发投递 20 次")
        results = await asyncio.gather(
            *(queue.enqueue(make_payload("dup-1")) for _ in range(20))
        )
        accepted = sum(results)
        print(f"   入队 {accepted} 次（期望 1）")
        assert accepted == 1, accepted

        print("2) 50 条不同消息并发投递")
        results = await asyncio.gather(
            *(queue.enqueue(make_payload(f"uniq-{i}")) for i in range(50))
        )
        accepted = sum(results)
        print(f"   入队 {accepted} 次（期望 50）")
        assert accepted == 50, accepted

        length = await redis.xlen(queue.stream)
        print(f"3) Stream 长度 {length}（期望 51）")
        assert length == 51, length

        print("4) 消费并 ACK")
        consumer = "selftest-worker"
        total = 0
        while True:
            entries = await queue.read(consumer, block_ms=1000, count=10)
            if not entries:
                break
            for entry in entries:
                await queue.ack(entry.entry_id)
                total += 1
        print(f"   消费 {total} 条（期望 51）")
        assert total == 51, total

        print("5) 延迟重试队列")
        await queue.retry(make_payload("retry-1"), delay_seconds=0.2)
        await asyncio.sleep(0.3)
        due = await queue.pop_due_retries(limit=10)
        print(f"   到期重试 {len(due)} 条（期望 1）")
        assert len(due) == 1, len(due)

        await queue.requeue(due[0])
        entries = await queue.read(consumer, block_ms=1000, count=10)
        print(f"   重新入队后可消费 {len(entries)} 条（期望 1）")
        assert len(entries) == 1, len(entries)
        for entry in entries:
            await queue.ack(entry.entry_id)

        print("\n全部通过")
    finally:
        keys = await redis.keys(f"{prefix}:*")
        if keys:
            await redis.delete(*keys)
        await redis.aclose()


if __name__ == "__main__":
    asyncio.run(main())
