"""Redis Stream 消息队列封装。

设计要点：

- 入队和去重在同一个 Lua 脚本里完成，原子操作：
  已见过 msgId -> 直接丢弃；没见过 -> XADD + 标记已见。
- 消费使用消费组 + 手动 XACK：处理成功才确认，worker 崩了消息还在 PEL，
  可以被 XAUTOCLAIM 换一个 worker 重做。
- 失败重试走独立的 ZSET 延迟队列，由 worker 的巡检循环搬回 Stream。
- 超过重试上限进死信 Stream，方便事后排查。
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import ResponseError

logger = logging.getLogger(__name__)

_ENQUEUE_LUA = """
-- KEYS[1] = 去重 key, KEYS[2] = stream
-- ARGV[1] = payload json, ARGV[2] = maxlen, ARGV[3] = dedup ttl 秒
if redis.call('EXISTS', KEYS[1]) == 1 then
    return 0
end
redis.call('XADD', KEYS[2], 'MAXLEN', '~', ARGV[2], '*', 'payload', ARGV[1])
redis.call('SET', KEYS[1], '1', 'EX', ARGV[3])
return 1
"""

_POP_DUE_RETRY_LUA = """
-- KEYS[1] = retry zset
-- ARGV[1] = now 时间戳, ARGV[2] = 最多取多少条
local items = redis.call('ZRANGEBYSCORE', KEYS[1], '-inf', ARGV[1], 'LIMIT', 0, tonumber(ARGV[2]))
if #items == 0 then
    return {}
end
redis.call('ZREM', KEYS[1], unpack(items))
return items
"""


@dataclass(slots=True)
class QueueEntry:
    entry_id: str
    payload: dict[str, Any]


def create_redis(url: str) -> Redis:
    """创建异步 Redis 客户端。decode_responses=True，拿到的都是 str。"""
    return Redis.from_url(
        url,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=30,
        health_check_interval=30,
    )


class RedisQueue:
    def __init__(
        self,
        redis: Redis,
        *,
        stream: str,
        group: str,
        dlq: str,
        retry_key: str,
        dedup_ttl: int = 86400,
        maxlen: int = 100000,
        key_prefix: str = "robot",
    ) -> None:
        self.redis = redis
        self.stream = stream
        self.group = group
        self.dlq = dlq
        self.retry_key = retry_key
        self.dedup_ttl = dedup_ttl
        self.maxlen = maxlen
        self.key_prefix = key_prefix
        self._enqueue_script = redis.register_script(_ENQUEUE_LUA)
        self._pop_retry_script = redis.register_script(_POP_DUE_RETRY_LUA)

    # ------------------------------------------------------------------
    # 消费组
    # ------------------------------------------------------------------
    async def ensure_group(self) -> None:
        try:
            await self.redis.xgroup_create(self.stream, self.group, id="0", mkstream=True)
            logger.info("已创建消费组 %s / %s", self.stream, self.group)
        except ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    # ------------------------------------------------------------------
    # 入队 / 去重
    # ------------------------------------------------------------------
    def _seen_key(self, msg_id: str) -> str:
        return f"{self.key_prefix}:seen:{msg_id}"

    async def enqueue(self, payload: dict[str, Any]) -> bool:
        """原子入队。返回 True 表示新消息，False 表示重复消息已丢弃。"""
        msg_id = str(payload.get("msg_id") or "").strip()
        if not msg_id:
            raise ValueError("payload 缺少 msg_id，无法去重")

        result = await self._enqueue_script(
            keys=[self._seen_key(msg_id), self.stream],
            args=[
                json.dumps(payload, ensure_ascii=False),
                self.maxlen,
                self.dedup_ttl,
            ],
        )
        return bool(int(result))

    # ------------------------------------------------------------------
    # 消费
    # ------------------------------------------------------------------
    async def read(
        self,
        consumer: str,
        *,
        block_ms: int | None = 5000,
        count: int = 1,
    ) -> list[QueueEntry]:
        response = await self.redis.xreadgroup(
            self.group,
            consumer,
            {self.stream: ">"},
            count=count,
            block=block_ms,
        )
        if not response:
            return []

        entries: list[QueueEntry] = []
        for _stream_name, items in response:
            for entry_id, fields in items:
                entries.append(QueueEntry(entry_id, self._decode(entry_id, fields)))
        return entries

    @staticmethod
    def _decode(entry_id: str, fields: dict[str, str]) -> dict[str, Any]:
        raw = fields.get("payload")
        if raw is None:
            return {"_error": f"消息 {entry_id} 缺少 payload 字段"}
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            return {"_error": f"消息 {entry_id} 的 payload 不是合法 JSON: {exc}"}
        if not isinstance(payload, dict):
            return {"_error": f"消息 {entry_id} 的 payload 不是 JSON 对象"}
        return payload

    async def ack(self, entry_id: str) -> None:
        await self.redis.xack(self.stream, self.group, entry_id)

    async def requeue(self, payload: dict[str, Any]) -> None:
        """把消息重新放回 Stream（不走去重，用于延迟重试/巡检回收）。"""
        await self.redis.xadd(
            self.stream,
            {"payload": json.dumps(payload, ensure_ascii=False)},
            maxlen=self.maxlen,
            approximate=True,
        )

    # ------------------------------------------------------------------
    # 延迟重试
    # ------------------------------------------------------------------
    async def retry(self, payload: dict[str, Any], delay_seconds: float) -> None:
        item = dict(payload)
        item["_retry_at"] = round(time.time() + max(0.0, delay_seconds), 3)
        member = json.dumps(item, ensure_ascii=False, sort_keys=True)
        await self.redis.zadd(self.retry_key, {member: item["_retry_at"]})
        logger.info(
            "消息延期重试 msg_id=%s delay=%.1fs",
            payload.get("msg_id"),
            delay_seconds,
        )

    async def pop_due_retries(self, limit: int = 20) -> list[dict[str, Any]]:
        members = await self._pop_retry_script(
            keys=[self.retry_key],
            args=[time.time(), limit],
        )
        result: list[dict[str, Any]] = []
        for member in members or []:
            try:
                item = json.loads(member)
            except json.JSONDecodeError:
                logger.warning("重试队列条目损坏，已丢弃: %s", str(member)[:200])
                continue
            item.pop("_retry_at", None)
            if isinstance(item, dict):
                result.append(item)
        return result

    # ------------------------------------------------------------------
    # 死信
    # ------------------------------------------------------------------
    async def dead_letter(self, payload: dict[str, Any], error: str) -> None:
        await self.redis.xadd(
            self.dlq,
            {
                "payload": json.dumps(payload, ensure_ascii=False),
                "error": str(error)[:1000],
                "time": str(int(time.time())),
            },
            maxlen=10000,
            approximate=True,
        )
        logger.error("消息进入死信队列 msg_id=%s error=%s", payload.get("msg_id"), error)

    # ------------------------------------------------------------------
    # 巡检：认领长时间未确认的消息
    # ------------------------------------------------------------------
    async def claim_stale(
        self,
        consumer: str,
        *,
        min_idle_ms: int,
        count: int = 10,
    ) -> list[QueueEntry]:
        response = await self.redis.xautoclaim(
            self.stream,
            self.group,
            consumer,
            min_idle_ms,
            start_id="0-0",
            count=count,
        )
        # redis-py 返回 [next_cursor, entries, deleted_ids]
        entries = response[1] if response else []
        return [
            QueueEntry(entry_id, self._decode(entry_id, fields))
            for entry_id, fields in entries or []
        ]

    async def pending_count(self) -> int:
        summary = await self.redis.xpending(self.stream, self.group)
        if isinstance(summary, dict):
            return int(summary.get("pending", 0))
        return 0

    async def close(self) -> None:
        await self.redis.aclose()
