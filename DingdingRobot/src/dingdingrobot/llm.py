"""LLM 调用 + 工具循环。

使用 OpenAI 兼容接口，默认指向本机 Ollama。
一次 run() 的流程：

    用户问题
      -> 模型决定是否调用工具
      -> 工具结果回填给模型
      -> 直到模型给出最终文本（或达到最大轮数）

工具执行放在线程里，避免同步函数阻塞事件循环。
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any

import httpx
from openai import AsyncOpenAI

from .config import Settings
from .tools import registry

logger = logging.getLogger(__name__)

TOOL_TIMEOUT_SECONDS = 20.0

_SYSTEM_PROMPT = """你是钉钉里的智能助手，回答要简洁、直接，使用中文。

当前时间：{now}（Asia/Shanghai）。

规则：
1. 时间、天气、计算、单位换算、订单等问题必须先调用工具，禁止凭记忆或常识编造；最终回答必须与工具结果一致。
2. 工具返回 ok=false 时，向用户说明原因，不要假装成功。
3. 钉钉文本消息不支持复杂排版，不要使用 Markdown 表格或代码块。
4. 不要向用户暴露工具名、JSON、内部错误信息，用自然语言回答。
5. 普通闲聊直接回答，不必调用工具。
6. 如果工具结果里 source=mock，或 note 说明是演示/占位数据，必须在回答中明确标注“演示数据”，不得当作真实信息。"""


class LLMError(RuntimeError):
    """LLM 调用失败，worker 会按策略重试。"""


def _now_text() -> str:
    try:
        from zoneinfo import ZoneInfo

        return datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:  # noqa: BLE001 - tzdata 缺失时退回 UTC
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._http = httpx.AsyncClient(
            trust_env=settings.llm_trust_env,
            timeout=httpx.Timeout(settings.llm_timeout, connect=10.0),
        )
        self._client = AsyncOpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            http_client=self._http,
            max_retries=0,
            timeout=settings.llm_timeout,
        )
        self._tools = registry.schemas()

    async def close(self) -> None:
        try:
            await self._client.close()
        finally:
            if not self._http.is_closed:
                await self._http.aclose()

    async def run(
        self,
        user_text: str,
        history: list[dict[str, Any]] | None = None,
    ) -> str:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _SYSTEM_PROMPT.format(now=_now_text())}
        ]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_text})

        try:
            for _ in range(max(1, self._settings.llm_max_tool_rounds)):
                response = await self._client.chat.completions.create(
                    model=self._settings.llm_model,
                    messages=messages,
                    tools=self._tools,
                    temperature=self._settings.llm_temperature,
                )
                message = response.choices[0].message

                if not message.tool_calls:
                    text = (message.content or "").strip()
                    if text:
                        return text
                    raise LLMError("模型返回了空回答")

                messages.append(self._assistant_message(message))
                for tool_call in message.tool_calls:
                    result = await self._execute_tool(tool_call)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(result, ensure_ascii=False),
                        }
                    )

            # 工具轮数用尽，强制模型给出最终文本
            response = await self._client.chat.completions.create(
                model=self._settings.llm_model,
                messages=messages,
                temperature=self._settings.llm_temperature,
            )
            text = (response.choices[0].message.content or "").strip()
            if text:
                return text
            raise LLMError("工具调用轮数用尽，模型仍未给出回答")

        except LLMError:
            raise
        except Exception as exc:  # noqa: BLE001 - 统一转成可重试错误
            logger.exception("LLM 调用失败")
            raise LLMError(f"LLM 调用失败: {type(exc).__name__}: {exc}") from exc

    @staticmethod
    def _assistant_message(message: Any) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
                for tool_call in message.tool_calls
            ],
        }

    async def _execute_tool(self, tool_call: Any) -> dict[str, Any]:
        name = tool_call.function.name
        arguments = tool_call.function.arguments
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(registry.execute, name, arguments),
                timeout=TOOL_TIMEOUT_SECONDS,
            )
            logger.info("工具调用 %s -> ok=%s", name, result.get("ok"))
            return result
        except TimeoutError:
            logger.warning("工具执行超时: %s", name)
            return {"ok": False, "error": f"工具 {name} 执行超时"}
        except Exception as exc:  # noqa: BLE001
            logger.exception("工具执行异常: %s", name)
            return {"ok": False, "error": f"工具 {name} 执行异常: {exc}"}
