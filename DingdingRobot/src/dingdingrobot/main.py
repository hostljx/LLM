"""命令行入口。

用法：

    python -m dingdingrobot.main gateway     # 只接收钉钉消息并入队
    python -m dingdingrobot.main worker      # 只消费队列、调 LLM、回复
    python -m dingdingrobot.main all         # 本地开发：接收 + 消费一起跑

生产环境建议 gateway 和 worker 分开进程，worker 可以多开几个。
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from .config import Settings, settings
from .gateway import run_gateway
from .queue import RedisQueue, create_redis
from .worker import Worker

logger = logging.getLogger(__name__)


def setup_logging(level: str) -> None:
    # Windows 控制台默认 GBK，中文日志容易乱码，尽量切成 UTF-8。
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    logging.getLogger("dingtalk_stream").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dingdingrobot",
        description="钉钉 Stream 机器人（Redis 队列 + LLM 工具调用）",
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="all",
        choices=("gateway", "worker", "all"),
        help="运行模式，默认 all",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="日志级别，默认 INFO",
    )
    parser.add_argument(
        "--consumer-name",
        default="worker",
        help="消费组内 consumer 名前缀，多进程部署时用来区分",
    )
    return parser


async def _run(app_settings: Settings, mode: str, consumer_name: str) -> None:
    gateway_queue: RedisQueue | None = None
    worker: Worker | None = None
    tasks: list[asyncio.Task] = []

    try:
        if mode in ("gateway", "all"):
            app_settings.require_dingtalk()
            gateway_redis = create_redis(app_settings.redis_url)
            gateway_queue = RedisQueue(
                gateway_redis,
                stream=app_settings.stream_key,
                group=app_settings.group_name,
                dlq=app_settings.dlq_key,
                retry_key=app_settings.retry_key,
                dedup_ttl=app_settings.dedup_ttl,
                maxlen=app_settings.queue_maxlen,
                key_prefix=app_settings.key_prefix,
            )
            await gateway_queue.ensure_group()
            tasks.append(
                asyncio.create_task(
                    run_gateway(app_settings, gateway_queue),
                    name="gateway",
                )
            )

        if mode in ("worker", "all"):
            worker = Worker(app_settings, consumer_name)
            tasks.append(
                asyncio.create_task(worker.run(), name="worker")
            )

        await asyncio.gather(*tasks)
    finally:
        if worker is not None:
            await worker.stop()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if gateway_queue is not None:
            await gateway_queue.close()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)

    try:
        asyncio.run(_run(settings, args.mode, args.consumer_name))
    except KeyboardInterrupt:
        logger.info("收到 Ctrl+C，已退出")
    except Exception:
        logger.exception("启动或运行失败")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
