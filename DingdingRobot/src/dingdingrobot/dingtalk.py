"""钉钉消息解析与回复。

回复策略（从快到稳）：

1. 优先用消息自带的 sessionWebhook 直接回复，带指数退避重试。
2. webhook 过期或多次失败，走钉钉 OpenAPI 主动发送：
   - 单聊 /v1.0/robot/oToMessages/batchSend
   - 群聊 /v1.0/robot/groupMessages/send
3. 回复前检查 / 占用 reply 租约，避免多个 worker 重复发同一条消息。
4. 长文本按字符数切片，避免超过钉钉单条消息限制。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, fields
from typing import Any
from uuid import uuid4

import httpx
from dingtalk_stream import ChatbotMessage
from redis.asyncio import Redis

from .config import Settings
from .dedup import IdempotencyStore

logger = logging.getLogger(__name__)

_AT_MENTION_RE = re.compile(r"^\s*(?:@[^\s@]+\s*)+")


class ReplyError(RuntimeError):
    """回复发送失败。"""


class ReplyInProgress(ReplyError):
    """另一个 worker 正在回复同一条消息。"""


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _strip_at_mention(text: str) -> str:
    """去掉群消息开头连续的 @某人（钉钉会把 @机器人 留在正文里）。"""
    return _AT_MENTION_RE.sub("", text, count=1)


@dataclass(slots=True)
class InboundMessage:
    msg_id: str = ""
    conversation_id: str = ""
    conversation_type: str = ""
    conversation_title: str = ""
    sender_id: str = ""
    sender_staff_id: str = ""
    sender_nick: str = ""
    text: str = ""
    message_type: str = "text"
    robot_code: str = ""
    session_webhook: str = ""
    session_webhook_expires_at: int = 0
    create_at: int = 0
    is_in_at_list: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InboundMessage":
        allowed = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in allowed})

    @classmethod
    def from_chatbot(cls, message: ChatbotMessage) -> "InboundMessage":
        text = (message.text.content if message.text else "") or ""
        text = _strip_at_mention(text).strip()
        return cls(
            msg_id=(message.message_id or "").strip() or uuid4().hex,
            conversation_id=message.conversation_id or "",
            conversation_type=str(message.conversation_type or ""),
            conversation_title=message.conversation_title or "",
            sender_id=message.sender_id or "",
            sender_staff_id=message.sender_staff_id or "",
            sender_nick=message.sender_nick or "",
            text=text,
            message_type=message.message_type or "text",
            robot_code=message.robot_code or "",
            session_webhook=message.session_webhook or "",
            session_webhook_expires_at=_to_int(message.session_webhook_expired_time),
            create_at=_to_int(message.create_at),
            is_in_at_list=bool(message.is_in_at_list),
        )


def split_text(text: str, limit: int) -> list[str]:
    """按字符数切片，优先在换行处断开。"""
    text = (text or "").strip()
    if not text:
        return []
    if limit <= 0:
        return [text]

    chunks: list[str] = []
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = limit
        chunk = text[:cut].strip()
        if chunk:
            chunks.append(chunk)
        text = text[cut:].strip()

    if text:
        chunks.append(text)
    return chunks


class DingTalkClient:
    def __init__(
        self,
        settings: Settings,
        redis: Redis,
        dedup: IdempotencyStore,
    ) -> None:
        self._settings = settings
        self._redis = redis
        self._dedup = dedup
        self._token_key = f"{settings.key_prefix}:dingtalk:token"
        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.dingtalk_timeout, connect=5.0),
            trust_env=settings.dingtalk_trust_env,
        )

    async def close(self) -> None:
        await self._http.aclose()

    async def reply(self, message: InboundMessage, text: str) -> None:
        """把回答发给用户。成功返回，失败抛 ReplyError。"""
        text = (text or "").strip()
        if not text:
            return

        if await self._dedup.is_replied(message.msg_id):
            logger.info("已回复过，跳过 msg_id=%s", message.msg_id)
            return

        lock = self._dedup.reply_lock(message.msg_id)
        token = await lock.acquire()
        if token is None:
            raise ReplyInProgress(f"消息 {message.msg_id} 正在被另一个 worker 回复")

        # 长文本分多条发送时记录进度：中途失败重试只补发剩下的，
        # 避免已经发出去的前几条重复发送。
        progress_key = f"{self._settings.key_prefix}:reply:chunk:{message.msg_id}"

        try:
            chunks = split_text(text, self._settings.dingtalk_reply_max_chars)
            start = _to_int(await self._redis.get(progress_key), 0)
            if start >= len(chunks):
                await self._dedup.mark_replied(message.msg_id)
                return

            for index in range(start, len(chunks)):
                chunk = chunks[index]
                sent = await self._send_webhook(message, chunk)
                if not sent:
                    logger.info(
                        "sessionWebhook 回复失败，改走主动发送 msg_id=%s",
                        message.msg_id,
                    )
                    await self._send_proactive(message, chunk)
                await self._redis.set(
                    progress_key,
                    index + 1,
                    ex=self._settings.done_ttl,
                )

            await self._dedup.mark_replied(message.msg_id)
            await self._redis.delete(progress_key)
        finally:
            await lock.release(token)

    # ------------------------------------------------------------------
    # sessionWebhook
    # ------------------------------------------------------------------
    async def _send_webhook(self, message: InboundMessage, text: str) -> bool:
        if not message.session_webhook:
            return False

        now_ms = int(time.time() * 1000)
        if (
            message.session_webhook_expires_at
            and message.session_webhook_expires_at - now_ms < 60_000
        ):
            logger.info("sessionWebhook 即将过期，直接走主动发送")
            return False

        payload: dict[str, Any] = {
            "msgtype": "text",
            "text": {"content": text},
        }
        if message.conversation_type == "2" and message.sender_staff_id:
            payload["at"] = {"atUserIds": [message.sender_staff_id]}

        retries = max(1, self._settings.dingtalk_reply_retries)
        for attempt in range(retries):
            try:
                response = await self._http.post(message.session_webhook, json=payload)
                if response.status_code == 200:
                    try:
                        data = response.json()
                    except ValueError:
                        logger.warning("webhook 返回非 JSON: %s", response.text[:200])
                    else:
                        if data.get("errcode", 0) == 0:
                            return True
                        logger.warning("webhook 返回业务错误: %s", data)
                else:
                    logger.warning(
                        "webhook HTTP %s: %s",
                        response.status_code,
                        response.text[:200],
                    )
            except httpx.HTTPError as exc:
                logger.warning("webhook 请求异常: %s", exc)

            await asyncio.sleep(0.5 * (2**attempt))

        return False

    # ------------------------------------------------------------------
    # OpenAPI 主动发送
    # ------------------------------------------------------------------
    async def _send_proactive(self, message: InboundMessage, text: str) -> None:
        robot_code = message.robot_code or self._settings.dingtalk_robot_code
        if not robot_code:
            raise ReplyError("缺少 robotCode，无法主动发送消息")

        access_token = await self._get_access_token()
        headers = {"x-acs-dingtalk-access-token": access_token}
        msg_param = json.dumps({"content": text}, ensure_ascii=False)

        if message.conversation_type == "1":
            if not message.sender_staff_id:
                raise ReplyError("单聊主动发送缺少 senderStaffId")
            url = (
                f"{self._settings.dingtalk_api_base}"
                "/v1.0/robot/oToMessages/batchSend"
            )
            body = {
                "robotCode": robot_code,
                "userIds": [message.sender_staff_id],
                "msgKey": "sampleText",
                "msgParam": msg_param,
            }
        else:
            if not message.conversation_id:
                raise ReplyError("群聊主动发送缺少 conversationId")
            url = (
                f"{self._settings.dingtalk_api_base}"
                "/v1.0/robot/groupMessages/send"
            )
            body = {
                "robotCode": robot_code,
                "openConversationId": message.conversation_id,
                "msgKey": "sampleText",
                "msgParam": msg_param,
            }

        response = await self._http.post(url, headers=headers, json=body)
        if response.status_code != 200:
            raise ReplyError(
                f"主动发送失败 HTTP {response.status_code}: {response.text[:500]}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise ReplyError(f"主动发送返回非 JSON: {response.text[:300]}") from exc

        code = data.get("code")
        if code not in (None, 0, "0"):
            raise ReplyError(f"主动发送失败: {data}")
        logger.info("主动发送成功 msg_id=%s", message.msg_id)

    async def _get_access_token(self) -> str:
        cached = await self._redis.get(self._token_key)
        if cached:
            return cached

        app_key = self._settings.dingtalk_app_key
        app_secret = self._settings.dingtalk_app_secret
        if not app_key or not app_secret:
            raise ReplyError("缺少 DINGTALK_APP_KEY / DINGTALK_APP_SECRET，无法主动发送")

        url = f"{self._settings.dingtalk_api_base}/v1.0/oauth2/accessToken"
        response = await self._http.post(
            url,
            json={"appKey": app_key, "appSecret": app_secret},
        )
        if response.status_code != 200:
            raise ReplyError(
                f"获取 access_token 失败 HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise ReplyError(f"access_token 返回非 JSON: {response.text[:300]}") from exc

        token = data.get("accessToken")
        if not token:
            raise ReplyError(f"获取 access_token 失败: {data}")

        expire_in = _to_int(data.get("expireIn"), 7200)
        await self._redis.set(
            self._token_key,
            token,
            ex=max(60, expire_in - 300),
        )
        return token
