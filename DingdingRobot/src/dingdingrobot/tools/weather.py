"""工具 3：天气查询（和风天气 QWeather 实时数据）。"""

from __future__ import annotations

import logging
import re
from typing import Any

import httpx

from ..config import settings
from .registry import registry

logger = logging.getLogger(__name__)

_QWEATHER_ERRORS = {
    "204": "没有查询到该城市的天气数据",
    "400": "天气查询参数有误",
    "401": "天气服务认证失败，请检查 API Key",
    "402": "天气服务额度已用完",
    "403": "天气服务没有权限",
    "404": "没有找到该城市",
    "429": "天气查询太频繁，请稍后再试",
    "500": "天气服务暂时不可用",
}


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _error_text(code: str) -> str:
    return _QWEATHER_ERRORS.get(str(code), f"天气服务返回错误（{code}）")


def _http_error_result(
    exc: httpx.HTTPStatusError, city: str
) -> dict[str, Any]:
    status = exc.response.status_code
    if status in (400, 404):
        return {"ok": False, "error": f"没有找到城市：{city}"}
    return {"ok": False, "error": _error_text(str(status))}


def _get_json(path: str, params: dict[str, Any]) -> dict[str, Any]:
    base = settings.weather_api_base.rstrip("/")
    url = f"{base}{path}"
    with httpx.Client(
        timeout=settings.weather_timeout,
        trust_env=settings.weather_trust_env,
    ) as client:
        # Key 放在 header 里，避免出现在 httpx 的 URL 日志中
        response = client.get(
            url,
            params=params,
            headers={"X-QW-Api-Key": settings.weather_api_key},
        )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError("天气服务返回格式异常")
    return data


_CJK_RE = re.compile(r"[一-鿿]")
_PLACE_SUFFIXES = ("城市", "市", "区", "县", "镇", "州", "盟", "旗")


def _normalize_place(value: str) -> str:
    for suffix in _PLACE_SUFFIXES:
        value = value.replace(suffix, "")
    return value.strip()


def _location_matches(city: str, location: dict[str, Any]) -> bool:
    """和风天气的模糊搜索会把乱码匹配到无关城市，做个保守校验。"""
    if not _CJK_RE.search(city):
        return True

    query = _normalize_place(city)
    names = "".join(
        str(location.get(key) or "") for key in ("name", "adm1", "adm2")
    )
    target = _normalize_place(names)
    if not query or not target:
        return True
    if len(query) == 1:
        return query in target

    query_grams = {query[index : index + 2] for index in range(len(query) - 1)}
    target_grams = {target[index : index + 2] for index in range(len(target) - 1)}
    return bool(query_grams & target_grams)


def _advice(condition: str, temperature: int) -> str:
    if "雪" in condition:
        return "可能有雪，注意保暖和防滑"
    if any(keyword in condition for keyword in ("雨", "雷")):
        return "出门记得带伞，注意安全"
    if "霾" in condition or "雾" in condition:
        return "能见度较低，出行注意安全"
    if temperature >= 30:
        return "天气较热，注意补水和防晒"
    if temperature <= 10:
        return "天气较冷，注意保暖"
    return "天气适宜，适合外出"


@registry.tool(
    name="get_weather",
    description=(
        "查询指定城市的实时天气，数据来自和风天气。"
        "用户问天气、气温、下雨、穿衣建议时调用。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "城市名称，例如 北京、上海、成都、佛山",
            }
        },
        "required": ["city"],
        "additionalProperties": False,
    },
)
def get_weather(city: str) -> dict[str, Any]:
    city = (city or "").strip()
    if not city:
        return {"ok": False, "error": "城市名称不能为空"}

    if not settings.weather_api_key:
        return {
            "ok": False,
            "error": "未配置天气 API Key，请在 .env 中填写 WEATHER_API_KEY",
        }
    if not settings.weather_api_base:
        return {
            "ok": False,
            "error": "未配置天气 API 地址，请在 .env 中填写 WEATHER_API_BASE",
        }

    # 1. 城市名 -> 和风天气 Location ID
    try:
        geo = _get_json(
            "/geo/v2/city/lookup",
            {
                "location": city,
                "number": 5,
                "lang": "zh",
            },
        )
    except httpx.HTTPStatusError as exc:
        logger.warning(
            "天气城市查询 HTTP %s city=%s",
            exc.response.status_code,
            city,
        )
        return _http_error_result(exc, city)
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("天气城市查询失败 city=%s error=%s", city, type(exc).__name__)
        return {"ok": False, "error": "天气服务暂时不可用，请稍后再试"}

    code = str(geo.get("code", ""))
    if code != "200":
        logger.warning("天气城市查询返回错误 city=%s code=%s", city, code)
        return {"ok": False, "error": _error_text(code)}

    locations = [
        item
        for item in (geo.get("location") or [])
        if isinstance(item, dict) and _location_matches(city, item)
    ]
    if not locations:
        return {"ok": False, "error": f"没有找到城市：{city}"}

    location = locations[0]
    location_id = location.get("id")
    if not location_id:
        return {"ok": False, "error": f"没有找到城市：{city}"}

    # 2. Location ID -> 实时天气
    try:
        data = _get_json(
            "/v7/weather/now",
            {
                "location": location_id,
                "lang": "zh",
                "unit": "m",
            },
        )
    except httpx.HTTPStatusError as exc:
        logger.warning(
            "天气查询 HTTP %s city=%s",
            exc.response.status_code,
            city,
        )
        return _http_error_result(exc, city)
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("天气查询失败 city=%s error=%s", city, type(exc).__name__)
        return {"ok": False, "error": "天气服务暂时不可用，请稍后再试"}

    code = str(data.get("code", ""))
    if code != "200":
        logger.warning("天气查询返回错误 city=%s code=%s", city, code)
        return {"ok": False, "error": _error_text(code)}

    now = data.get("now") or {}
    if not now:
        return {"ok": False, "error": "天气服务没有返回实时数据"}

    temperature = _to_int(now.get("temp"))
    feels_like = _to_int(now.get("feelsLike"), temperature)
    humidity = _to_int(now.get("humidity"))
    condition = str(now.get("text") or "未知")
    wind_level = str(now.get("windScale") or "")
    wind_direction = str(now.get("windDir") or "")
    observed_at = str(now.get("obsTime") or "")

    display_city = str(location.get("name") or city)
    logger.info(
        "天气查询成功 city=%s condition=%s temperature=%s",
        display_city,
        condition,
        temperature,
    )

    return {
        "ok": True,
        "source": "qweather",
        "city": display_city,
        "province": str(location.get("adm1") or ""),
        "district": str(location.get("adm2") or ""),
        "condition": condition,
        "temperature_c": temperature,
        "feels_like_c": feels_like,
        "humidity_percent": humidity,
        "wind_level": wind_level,
        "wind_direction": wind_direction,
        "observed_at": observed_at,
        "advice": _advice(condition, temperature),
    }
