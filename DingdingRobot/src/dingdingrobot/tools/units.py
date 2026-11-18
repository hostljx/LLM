"""工具 5：单位换算。"""

from __future__ import annotations

import math
from typing import Any

from .registry import registry

# 基准单位：长度 -> 米，重量 -> 千克
_LENGTH_TO_M = {
    "mm": 0.001,
    "cm": 0.01,
    "m": 1.0,
    "km": 1000.0,
    "in": 0.0254,
    "ft": 0.3048,
    "yd": 0.9144,
    "mi": 1609.344,
}
_WEIGHT_TO_KG = {
    "g": 0.001,
    "kg": 1.0,
    "t": 1000.0,
    "lb": 0.45359237,
    "oz": 0.028349523125,
}
_TEMPERATURE_UNITS = {"c", "f", "k"}

_ALIASES = {
    "毫米": "mm",
    "厘米": "cm",
    "米": "m",
    "千米": "km",
    "公里": "km",
    "英寸": "in",
    "英尺": "ft",
    "码": "yd",
    "英里": "mi",
    "克": "g",
    "千克": "kg",
    "公斤": "kg",
    "吨": "t",
    "磅": "lb",
    "盎司": "oz",
    "摄氏度": "c",
    "华氏度": "f",
    "开尔文": "k",
    "°c": "c",
    "℃": "c",
    "°f": "f",
    "℉": "f",
}

_ALL_UNITS = set(_LENGTH_TO_M) | set(_WEIGHT_TO_KG) | _TEMPERATURE_UNITS


def _normalize_unit(unit: str) -> str:
    key = unit.strip().lower()
    key = _ALIASES.get(key, key)
    if key not in _ALL_UNITS and key.endswith("s"):
        key = key[:-1]
    return key


def _to_celsius(value: float, unit: str) -> float:
    if unit == "c":
        return value
    if unit == "f":
        return (value - 32.0) * 5.0 / 9.0
    return value - 273.15


def _from_celsius(value: float, unit: str) -> float:
    if unit == "c":
        return value
    if unit == "f":
        return value * 9.0 / 5.0 + 32.0
    return value + 273.15


@registry.tool(
    name="convert_units",
    description=(
        "进行单位换算，支持长度（mm/cm/m/km/in/ft/yd/mi）、"
        "重量（g/kg/t/lb/oz）和温度（c/f/k）。"
        "用户问“5 公里等于多少英里”“100 华氏度是多少摄氏度”时调用。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "value": {
                "type": "number",
                "description": "要换算的数值",
            },
            "from_unit": {
                "type": "string",
                "description": "原单位，例如 km、kg、c",
            },
            "to_unit": {
                "type": "string",
                "description": "目标单位，例如 mi、lb、f",
            },
        },
        "required": ["value", "from_unit", "to_unit"],
        "additionalProperties": False,
    },
)
def convert_units(value: float, from_unit: str, to_unit: str) -> dict[str, Any]:
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return {"ok": False, "error": "value 必须是数字"}
    if not math.isfinite(amount):
        return {"ok": False, "error": "value 必须是有限数字"}

    src = _normalize_unit(from_unit)
    dst = _normalize_unit(to_unit)
    unknown = [unit for unit in (src, dst) if unit not in _ALL_UNITS]
    if unknown:
        return {
            "ok": False,
            "error": f"不支持的单位: {', '.join(unknown)}",
            "supported": sorted(_ALL_UNITS),
        }

    if src in _LENGTH_TO_M and dst in _LENGTH_TO_M:
        category = "length"
        result = amount * _LENGTH_TO_M[src] / _LENGTH_TO_M[dst]
    elif src in _WEIGHT_TO_KG and dst in _WEIGHT_TO_KG:
        category = "weight"
        result = amount * _WEIGHT_TO_KG[src] / _WEIGHT_TO_KG[dst]
    elif src in _TEMPERATURE_UNITS and dst in _TEMPERATURE_UNITS:
        category = "temperature"
        result = _from_celsius(_to_celsius(amount, src), dst)
    else:
        return {
            "ok": False,
            "error": f"单位类型不一致: {src} 和 {dst} 不能互相换算",
        }

    return {
        "ok": True,
        "category": category,
        "value": amount,
        "from_unit": src,
        "to_unit": dst,
        "result": round(result, 6),
    }
