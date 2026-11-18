"""幂等与分布式锁。

四类标记：

- processing:{msgId}   处理租约，防止同一条消息被多个 worker 同时处理
- done:{msgId}         处理完成，重复投递直接 ACK 跳过
- reply:{msgId}        回复中租约；replied:{msgId} 表示已经发成功
- answer:{msgId}       已生成的回复缓存，回复失败重试时不必重新调 LLM
- lock:conv:{cid}      同一会话串行处理，保证多轮顺序

锁的释放用 Lua 比对 token，避免误删别人续期后的锁。
"""

from __future__ import annotations

import uuid
from typing import Any

from redis.asyncio import Redis

_RELEASE_LUA = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('DEL', KEYS[1])
end
return 0
"""


class RedisLock:
    """带 token 的 Redis 分布式锁。"""

    def __init__(self, redis: Redis, key: str, ttl: int) -> None:
        self._redis = redis
        self.key = key
        self.ttl = ttl
        self._release_script = redis.register_script(_RELEASE_LUA)

    async def acquire(self) -> str | None:
        token = uuid.uuid4().hex
        acquired = await self._redis.set(self.key, token, nx=True, ex=self.ttl)
        return token if acquired else None

    async def release(self, token: str | None) -> None:
        if not token:
            return
        await self._release_script(keys=[self.key], args=[token])


class IdempotencyStore:
    def __init__(
        self,
        redis: Redis,
        *,
        prefix: str = "robot",
        done_ttl: int = 86400,
        processing_ttl: int = 300,
        reply_ttl: int = 120,
        answer_ttl: int = 3600,
        conversation_lock_ttl: int = 300,
    ) -> None:
        self._redis = redis
        self._prefix = prefix
        self._done_ttl = done_ttl
        self._processing_ttl = processing_ttl
        self._reply_ttl = reply_ttl
        self._answer_ttl = answer_ttl
        self._conversation_lock_ttl = conversation_lock_ttl

    def _key(self, kind: str, ident: str) -> str:
        return f"{self._prefix}:{kind}:{ident}"

    # ------------------------------------------------------------------
    # 处理完成标记
    # ------------------------------------------------------------------
    async def is_done(self, msg_id: str) -> bool:
        return bool(await self._redis.exists(self._key("done", msg_id)))

    async def mark_done(self, msg_id: str) -> None:
        await self._redis.set(self._key("done", msg_id), "1", ex=self._done_ttl)

    # ------------------------------------------------------------------
    # 处理租约
    # ------------------------------------------------------------------
    def processing_lock(self, msg_id: str) -> RedisLock:
        return RedisLock(self._redis, self._key("processing", msg_id), self._processing_ttl)

    # ------------------------------------------------------------------
    # 重试计数
    # ------------------------------------------------------------------
    async def incr_attempts(self, msg_id: str) -> int:
        key = self._key("attempts", msg_id)
        count = int(await self._redis.incr(key))
        if count == 1:
            await self._redis.expire(key, self._done_ttl)
        return count

    # ------------------------------------------------------------------
    # 回复幂等
    # ------------------------------------------------------------------
    async def is_replied(self, msg_id: str) -> bool:
        return bool(await self._redis.exists(self._key("replied", msg_id)))

    async def mark_replied(self, msg_id: str) -> None:
        await self._redis.set(self._key("replied", msg_id), "1", ex=self._done_ttl)

    def reply_lock(self, msg_id: str) -> RedisLock:
        return RedisLock(self._redis, self._key("reply", msg_id), self._reply_ttl)

    # ------------------------------------------------------------------
    # 回复内容缓存
    # ------------------------------------------------------------------
    async def get_answer(self, msg_id: str) -> str | None:
        return await self._redis.get(self._key("answer", msg_id))

    async def set_answer(self, msg_id: str, answer: str) -> None:
        await self._redis.set(self._key("answer", msg_id), answer, ex=self._answer_ttl)

    # ------------------------------------------------------------------
    # 会话锁
    # ------------------------------------------------------------------
    def conversation_lock(self, conversation_id: str) -> RedisLock:
        return RedisLock(
            self._redis,
            self._key("lock:conv", conversation_id),
            self._conversation_lock_ttl,
        )

    async def stats(self, msg_id: str) -> dict[str, Any]:
        """调试用：把某条消息的相关标记一次性取出来。"""
        keys = {
            "seen": self._key("seen", msg_id),
            "done": self._key("done", msg_id),
            "processing": self._key("processing", msg_id),
            "replied": self._key("replied", msg_id),
            "attempts": self._key("attempts", msg_id),
        }
        pipe = self._redis.pipeline()
        for key in keys.values():
            pipe.get(key)
        values = await pipe.execute()
        return dict(zip(keys, values))
