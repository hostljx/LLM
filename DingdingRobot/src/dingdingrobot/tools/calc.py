"""工具 2：计算器（AST 白名单安全求值）。"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any

from .registry import registry

_BIN_OPS: dict[type[ast.operator], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_MAX_EXPRESSION_LENGTH = 200
_MAX_POWER_EXPONENT = 100
_MAX_POWER_BASE = 1_000_000


def _eval_node(node: ast.AST) -> int | float:
    """只允许数字、括号和四则运算，杜绝 eval/import/属性访问。"""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("只支持数字常量")
        return node.value

    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"不支持的运算符: {type(node.op).__name__}")

        left = _eval_node(node.left)
        right = _eval_node(node.right)

        if isinstance(node.op, ast.Pow):
            if abs(right) > _MAX_POWER_EXPONENT:
                raise ValueError("指数过大")
            if abs(left) > _MAX_POWER_BASE:
                raise ValueError("底数过大")

        return op(left, right)

    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"不支持的一元运算符: {type(node.op).__name__}")
        return op(_eval_node(node.operand))

    raise ValueError(f"不支持的语法: {type(node).__name__}")


@registry.tool(
    name="calculator",
    description=(
        "计算数学表达式，支持 + - * / // % ** 和括号。"
        "需要做算术时调用，不要自己心算。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "数学表达式，例如 (12 + 8) * 3 / 5",
            }
        },
        "required": ["expression"],
        "additionalProperties": False,
    },
)
def calculator(expression: str) -> dict[str, Any]:
    expr = expression.strip()
    if not expr:
        return {"ok": False, "error": "表达式不能为空"}
    if len(expr) > _MAX_EXPRESSION_LENGTH:
        return {"ok": False, "error": f"表达式过长（最多 {_MAX_EXPRESSION_LENGTH} 字符）"}

    try:
        tree = ast.parse(expr, mode="eval")
        result = _eval_node(tree.body)
    except ZeroDivisionError:
        return {"ok": False, "error": "除数不能为 0"}
    except (ValueError, SyntaxError, OverflowError) as exc:
        return {"ok": False, "error": f"无法计算: {exc}"}

    if isinstance(result, complex):
        return {"ok": False, "error": "结果为复数，暂不支持"}
    if isinstance(result, float) and not math.isfinite(result):
        return {"ok": False, "error": "结果超出可表示范围"}

    return {"ok": True, "expression": expr, "result": result}
