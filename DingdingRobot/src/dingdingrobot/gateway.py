"""钉钉 Stream 接入层。

只做三件事：
1. 解析消息
2. 原子去重 + 入队 Redis Stream
3. 立刻 ACK（毫秒级，不在这里调 LLM）

入队失败时返回非 200 的 ACK，让钉钉重投，而不是静默丢消息。
"""

from __future__ import annotations

import asyncio
import json
import logging
from urllib.parse import quote_plus

import websockets
from dingtalk_stream import (
    AckMessage,
    CallbackHandler,
    CallbackMessage,
    ChatbotMessage,
    Credential,
    DingTalkStreamClient,
)

from .config import Settings
from .dingtalk import InboundMessage
from .queue import RedisQueue

logger = logging.getLogger(__name__)


class GatewayHandler(CallbackHandler):
    def __init__(self, queue: RedisQueue) -> None:
        super().__init__()
        self._queue = queue

    async def process(self, callback: CallbackMessage) -> tuple[int, str]:
        try:
            message = ChatbotMessage.from_dict(callback.data)
            inbound = InboundMessage.from_chatbot(message)
            accepted = await self._queue.enqueue(inbound.to_dict())
            if accepted:
                logger.info(
                    "已入队 msg_id=%s conversation=%s type=%s",
                    inbound.msg_id,
                    inbound.conversation_id,
                    inbound.conversation_type,
                )
            else:
                logger.info("重复消息，已丢弃 msg_id=%s", inbound.msg_id)
            return AckMessage.STATUS_OK, "ok"
        except Exception:
            logger.exception("消息入队失败，等待钉钉重投")
            return AckMessage.STATUS_SYSTEM_EXCEPTION, "enqueue failed"


def build_stream_client(settings: Settings, queue: RedisQueue) -> DingTalkStreamClient:
    settings.require_dingtalk()
    client = DingTalkStreamClient(
        Credential(settings.dingtalk_app_key, settings.dingtalk_app_secret)
    )
    client.register_callback_handler(ChatbotMessage.TOPIC, GatewayHandler(queue))
    return client


async def run_gateway(settings: Settings, queue: RedisQueue) -> None:
    """自己实现连接循环。

    不用 SDK 的 start()：它会把 CancelledError 吞掉然后继续重连，
    导致 Ctrl+C / 任务取消时停不下来。这里只复用它的
    open_connection / keepalive / background_task。
    """
    client = build_stream_client(settings, queue)
    client.pre_start()
    logger.info("gateway 启动，等待钉钉消息…")

    while True:
        try:
            try:
                connection = await asyncio.wait_for(
                    asyncio.to_thread(client.open_connection),
                    timeout=15,
                )
            except TimeoutError:
                logger.error("获取 Stream 连接超时，10 秒后重试")
                await asyncio.sleep(10)
                continue

            if not connection:
                logger.error("获取 Stream 连接失败，10 秒后重试")
                await asyncio.sleep(10)
                continue

            uri = (
                f'{connection["endpoint"]}?ticket='
                f'{quote_plus(connection["ticket"])}'
            )
            async with websockets.connect(uri) as websocket:
                client.websocket = websocket
                logger.info("已连接钉钉 Stream")

                keepalive = asyncio.create_task(client.keepalive(websocket))
                pending: set[asyncio.Task] = set()
                try:
                    async for raw_message in websocket:
                        message = json.loads(raw_message)
                        task = asyncio.create_task(
                            client.background_task(message)
                        )
                        pending.add(task)
                        task.add_done_callback(pending.discard)
                finally:
                    keepalive.cancel()
                    await asyncio.gather(keepalive, return_exceptions=True)
                    for task in list(pending):
                        task.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)

        except asyncio.CancelledError:
            logger.info("gateway 停止")
            raise
        except Exception:
            logger.exception("Stream 连接异常，3 秒后重连")
            await asyncio.sleep(3)
