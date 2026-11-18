"""消息消费 worker。

每个 worker 进程内起 N 个消费协程，共同竞争同一个消费组：

- 收到消息 -> 查 done 标记 -> 抢处理租约 -> 抢会话锁
- 调 LLM + 工具 -> 回复钉钉 -> 写会话历史 -> 标记完成 -> XACK
- 失败 -> 按次数延迟重试 -> 超过上限进死信并给用户兜底提示
- 后台巡检 -> 把延迟重试搬回 Stream、认领长时间未 ACK 的消息
"""

from __future__ import annotations

import asyncio
import logging
import time

from .config import Settings
from .dedup import IdempotencyStore, RedisLock
from .dingtalk import DingTalkClient, InboundMessage, ReplyInProgress
from .llm import LLMClient
from .memory import ConversationMemory
from .queue import QueueEntry, RedisQueue, create_redis

logger = logging.getLogger(__name__)


class Worker:
    def __init__(self, settings: Settings, consumer_name: str = "worker") -> None:
        self._settings = settings
        self._base_consumer = consumer_name
        self._redis = create_redis(settings.redis_url)
        self._queue = RedisQueue(
            self._redis,
            stream=settings.stream_key,
            group=settings.group_name,
            dlq=settings.dlq_key,
            retry_key=settings.retry_key,
            dedup_ttl=settings.dedup_ttl,
            maxlen=settings.queue_maxlen,
            key_prefix=settings.key_prefix,
        )
        self._dedup = IdempotencyStore(
            self._redis,
            prefix=settings.key_prefix,
            done_ttl=settings.done_ttl,
            processing_ttl=settings.processing_ttl,
            reply_ttl=settings.reply_ttl,
            answer_ttl=settings.answer_ttl,
            conversation_lock_ttl=settings.conversation_lock_ttl,
        )
        self._memory = ConversationMemory(
            self._redis,
            prefix=settings.key_prefix,
            limit=settings.history_limit,
            ttl=settings.history_ttl,
        )
        self._llm = LLMClient(settings)
        self._dingtalk = DingTalkClient(settings, self._redis, self._dedup)
        self._stop = asyncio.Event()

    async def run(self) -> None:
        tasks: list[asyncio.Task] = []
        try:
            await self._queue.ensure_group()
            for index in range(max(1, self._settings.max_concurrency)):
                consumer = f"{self._base_consumer}-{index}"
                tasks.append(
                    asyncio.create_task(
                        self._consume_loop(consumer),
                        name=f"consumer-{index}",
                    )
                )
            tasks.append(asyncio.create_task(self._retry_loop(), name="retry-loop"))
            tasks.append(asyncio.create_task(self._claim_loop(), name="claim-loop"))

            logger.info(
                "worker 启动: concurrency=%s consumer=%s",
                self._settings.max_concurrency,
                self._base_consumer,
            )
            await self._stop.wait()
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self.close()

    async def stop(self) -> None:
        self._stop.set()

    async def close(self) -> None:
        await self._llm.close()
        await self._dingtalk.close()
        await self._queue.close()

    # ------------------------------------------------------------------
    # 消费
    # ------------------------------------------------------------------
    async def _consume_loop(self, consumer: str) -> None:
        while not self._stop.is_set():
            try:
                entries = await self._queue.read(consumer, block_ms=5000, count=1)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("读取队列失败，2 秒后重试")
                await self._sleep(2)
                continue

            for entry in entries:
                try:
                    await self._handle_entry(entry)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("处理消息异常 entry_id=%s", entry.entry_id)

    async def _handle_entry(self, entry: QueueEntry) -> None:
        payload = entry.payload
        if payload.get("_error"):
            await self._queue.dead_letter(payload, str(payload["_error"]))
            await self._queue.ack(entry.entry_id)
            return

        msg_id = str(payload.get("msg_id") or entry.entry_id)
        if await self._dedup.is_done(msg_id):
            logger.info("消息已完成，跳过 msg_id=%s", msg_id)
            await self._queue.ack(entry.entry_id)
            return

        processing_lock = self._dedup.processing_lock(msg_id)
        token = await processing_lock.acquire()
        if token is None:
            logger.info("消息正在被其他 worker 处理，跳过 msg_id=%s", msg_id)
            await self._queue.ack(entry.entry_id)
            return

        try:
            await self._process(payload, entry, msg_id)
        finally:
            await processing_lock.release(token)

    async def _process(
        self,
        payload: dict,
        entry: QueueEntry,
        msg_id: str,
    ) -> None:
        inbound = InboundMessage.from_dict(payload)
        if not inbound.msg_id:
            inbound.msg_id = msg_id

        if inbound.message_type != "text":
            await self._reply_safely(inbound, "目前只支持文本消息，请直接发文字。")
            await self._dedup.mark_done(msg_id)
            await self._queue.ack(entry.entry_id)
            return

        if not inbound.text:
            await self._reply_safely(inbound, "请直接输入你的问题。")
            await self._dedup.mark_done(msg_id)
            await self._queue.ack(entry.entry_id)
            return

        conversation_lock = self._dedup.conversation_lock(inbound.conversation_id)
        conversation_token = await self._acquire_conversation_lock(conversation_lock)
        try:
            answer = await self._dedup.get_answer(msg_id)
            if answer is None:
                history = await self._memory.load(inbound.conversation_id)
                try:
                    async with asyncio.timeout(self._settings.agent_timeout):
                        answer = await self._llm.run(inbound.text, history)
                except TimeoutError as exc:
                    raise RuntimeError("Agent 处理超时") from exc
                await self._dedup.set_answer(msg_id, answer)

            await self._dingtalk.reply(inbound, answer)
            await self._memory.append(
                inbound.conversation_id,
                inbound.text,
                answer,
            )
            await self._dedup.mark_done(msg_id)
            await self._queue.ack(entry.entry_id)
            logger.info(
                "处理完成 msg_id=%s conversation=%s",
                msg_id,
                inbound.conversation_id,
            )

        except ReplyInProgress:
            await self._queue.retry(payload, delay_seconds=5)
            await self._queue.ack(entry.entry_id)
            logger.info("回复进行中，5 秒后重试 msg_id=%s", msg_id)

        except Exception as exc:  # noqa: BLE001 - 单条消息失败不能拖垮 worker
            try:
                await self._handle_failure(payload, inbound, msg_id, exc)
            finally:
                await self._queue.ack(entry.entry_id)

        finally:
            if conversation_token:
                await conversation_lock.release(conversation_token)

    async def _handle_failure(
        self,
        payload: dict,
        inbound: InboundMessage,
        msg_id: str,
        exc: Exception,
    ) -> None:
        attempts = await self._dedup.incr_attempts(msg_id)
        logger.warning(
            "处理失败 msg_id=%s attempts=%s error=%s",
            msg_id,
            attempts,
            exc,
        )

        max_attempts = max(1, self._settings.max_attempts)
        if attempts < max_attempts:
            delay = self._settings.retry_base_seconds * (2 ** (attempts - 1))
            await self._queue.retry(payload, delay_seconds=delay)
            return

        await self._queue.dead_letter(payload, str(exc))
        await self._dedup.mark_done(msg_id)
        await self._reply_safely(
            inbound,
            "抱歉，这个问题我暂时没处理成功，请稍后再试。",
        )

    async def _reply_safely(self, inbound: InboundMessage, text: str) -> None:
        try:
            await self._dingtalk.reply(inbound, text)
        except Exception:
            logger.exception("兜底回复失败 msg_id=%s", inbound.msg_id)

    async def _acquire_conversation_lock(self, lock: RedisLock) -> str | None:
        token = await lock.acquire()
        if token:
            return token

        deadline = time.monotonic() + max(0.0, self._settings.conversation_lock_wait)
        while time.monotonic() < deadline and not self._stop.is_set():
            await asyncio.sleep(0.5)
            token = await lock.acquire()
            if token:
                return token

        logger.warning("等待会话锁超时，继续处理（可能与其他消息交错）")
        return None

    # ------------------------------------------------------------------
    # 后台巡检
    # ------------------------------------------------------------------
    async def _retry_loop(self) -> None:
        while not self._stop.is_set():
            try:
                payloads = await self._queue.pop_due_retries(limit=20)
                for payload in payloads:
                    await self._queue.requeue(payload)
                if payloads:
                    logger.info("延迟重试重新入队 %d 条", len(payloads))
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("重试巡检失败")
            await self._sleep(1)

    async def _claim_loop(self) -> None:
        consumer = f"{self._base_consumer}-claimer"
        while not self._stop.is_set():
            try:
                entries = await self._queue.claim_stale(
                    consumer,
                    min_idle_ms=self._settings.claim_min_idle_ms,
                    count=10,
                )
                for entry in entries:
                    await self._handle_entry(entry)
                if entries:
                    logger.info("巡检认领 %d 条消息", len(entries))
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("消息巡检失败")
            await self._sleep(30)

    async def _sleep(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=seconds)
        except TimeoutError:
            pass
