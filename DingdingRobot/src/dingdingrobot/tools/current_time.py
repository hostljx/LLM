"""工具 1：获取当前时间。"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .registry import registry

_DEFAULT_TIMEZONE = "Asia/Shanghai"


@registry.tool(
    name="get_current_time",
    description=(
        "获取指定时区的当前日期和时间。"
        "当用户问“现在几点”“今天几号”“今天星期几”时调用。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "timezone": {
                "type": "string",
                "description": (
                    "IANA 时区名，例如 Asia/Shanghai、America/New_York，"
                    "不填默认 Asia/Shanghai"
                ),
            }
        },
        "required": [],
        "additionalProperties": False,
    },
)
def get_current_time(timezone: str = _DEFAULT_TIMEZONE) -> dict[str, Any]:
    try:
        tz = ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError):
        return {"ok": False, "error": f"未知时区: {timezone}"}

    now = datetime.now(tz)
    weekdays = ("星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日")
    return {
        "ok": True,
        "timezone": timezone,
        "datetime": now.isoformat(timespec="seconds"),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "weekday": weekdays[now.weekday()],
        "utc_offset": now.strftime("%z"),
        "timestamp": int(now.timestamp()),
    }
