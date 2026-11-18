"""会话历史存储（Redis List）。

只存 user / assistant 的最终文本，不存中间的 tool 调用，避免历史膨胀。
同一个会话由 conversation lock 串行处理，因此顺序是可靠的。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from redis.asyncio import Redis

logger = logging.getLogger(__name__)


class ConversationMemory:
    def __init__(
        self,
        redis: Redis,
        *,
        prefix: str = "robot",
        limit: int = 20,
        ttl: int = 7 * 24 * 3600,
    ) -> None:
        self._redis = redis
        self._prefix = prefix
        self._limit = max(2, limit)
        self._ttl = ttl

    def _key(self, conversation_id: str) -> str:
        return f"{self._prefix}:history:{conversation_id}"

    async def load(self, conversation_id: str) -> list[dict[str, Any]]:
        if not conversation_id:
            return []

        raw_items = await self._redis.lrange(self._key(conversation_id), 0, -1)
        messages: list[dict[str, Any]] = []
        for raw in raw_items:
            try:
                item = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("会话历史条目损坏，已跳过: %s", str(raw)[:120])
                continue
            if (
                isinstance(item, dict)
                and item.get("role") in ("user", "assistant")
                and isinstance(item.get("content"), str)
            ):
                messages.append(
                    {"role": item["role"], "content": item["content"]}
                )
        return messages[-self._limit :]

    async def append(
        self,
        conversation_id: str,
        user_text: str,
        assistant_text: str,
    ) -> None:
        if not conversation_id:
            return

        key = self._key(conversation_id)
        pipe = self._redis.pipeline()
        pipe.rpush(
            key,
            json.dumps({"role": "user", "content": user_text}, ensure_ascii=False),
            json.dumps(
                {"role": "assistant", "content": assistant_text},
                ensure_ascii=False,
            ),
        )
        pipe.ltrim(key, -self._limit, -1)
        pipe.expire(key, self._ttl)
        await pipe.execute()

    async def clear(self, conversation_id: str) -> None:
        await self._redis.delete(self._key(conversation_id))
