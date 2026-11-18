"""工具定义与注册表。

一个工具 = 一个普通 Python 函数 + 一段 JSON Schema 描述。
LLM 通过 function calling 选择工具并给出参数，注册表负责：

- 按名字查找工具
- 把 LLM 返回的 JSON 字符串参数解析成 dict
- 校验必填参数
- 把任何异常转成结构化错误，避免一次工具失败把整个对话打挂
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# 工具函数约定：接收关键字参数，返回可 JSON 序列化的 dict。
# 成功时建议包含 "ok": True，业务失败时返回 {"ok": False, "error": "..."}。
ToolHandler = Callable[..., dict[str, Any]]


@dataclass(frozen=True, slots=True)
class Tool:
    """一个可被 LLM 调用的工具。"""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler

    def schema(self) -> dict[str, Any]:
        """转成 OpenAI Chat Completions 的 tools 条目格式。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """工具注册表，进程内全局唯一。"""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise ValueError(f"工具重复注册: {tool.name}")
        self._tools[tool.name] = tool
        logger.debug("注册工具: %s", tool.name)
        return tool

    def tool(
        self,
        *,
        name: str,
        description: str,
        parameters: dict[str, Any],
    ) -> Callable[[ToolHandler], ToolHandler]:
        """装饰器形式注册，函数本身仍可直接调用，方便单测。"""

        def decorator(func: ToolHandler) -> ToolHandler:
            self.register(
                Tool(
                    name=name,
                    description=description,
                    parameters=parameters,
                    handler=func,
                )
            )
            return func

        return decorator

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        """给 LLM 的工具清单，顺序与注册顺序一致。"""
        return [tool.schema() for tool in self._tools.values()]

    def execute(
        self,
        name: str,
        arguments: str | Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """执行工具。

        arguments 支持 LLM 返回的 JSON 字符串，也支持直接传 dict。
        任何异常都会被吞掉并转成 {"ok": False, "error": ...}，交给 LLM
        去解释或换个方式重试。
        """
        tool = self._tools.get(name)
        if tool is None:
            return {
                "ok": False,
                "error": f"未知工具: {name}",
                "available_tools": self.names(),
            }

        args, parse_error = self._parse_arguments(arguments)
        if parse_error:
            return {"ok": False, "error": parse_error}

        missing = [
            key
            for key in tool.parameters.get("required", [])
            if args.get(key) is None or args.get(key) == ""
        ]
        if missing:
            return {"ok": False, "error": f"缺少必填参数: {', '.join(missing)}"}

        try:
            result = tool.handler(**args)
        except TypeError as exc:
            # 大概率是 LLM 多传/少传/拼错了参数名
            return {"ok": False, "error": f"参数不匹配: {exc}"}
        except Exception as exc:  # noqa: BLE001 - 工具失败不能影响主流程
            logger.exception("工具执行失败: %s", name)
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

        if not isinstance(result, dict):
            result = {"ok": True, "result": result}

        try:
            json.dumps(result, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            return {"ok": False, "error": f"工具返回值无法序列化: {exc}"}

        return result

    @staticmethod
    def _parse_arguments(
        arguments: str | Mapping[str, Any] | None,
    ) -> tuple[dict[str, Any], str | None]:
        """返回 (参数字典, 错误信息)。错误信息为 None 表示解析成功。"""
        if arguments is None:
            return {}, None
        if isinstance(arguments, Mapping):
            return dict(arguments), None
        if not isinstance(arguments, str):
            return {}, f"参数类型不支持: {type(arguments).__name__}"

        text = arguments.strip()
        if not text:
            return {}, None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            return {}, f"参数不是合法 JSON: {exc}"
        if not isinstance(parsed, dict):
            return {}, "参数必须是 JSON 对象"
        return parsed, None

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools
