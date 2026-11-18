"""工具 4：订单查询（当前为演示数据）。"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from typing import Any

from .registry import registry


@registry.tool(
    name="query_order",
    description=(
        "按订单号查询订单状态、金额和商品。"
        "当前返回演示数据，用于验证工具调用链路。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "order_id": {
                "type": "string",
                "description": "订单号，例如 DD20261116001",
            }
        },
        "required": ["order_id"],
        "additionalProperties": False,
    },
)
def query_order(order_id: str) -> dict[str, Any]:
    order_id = order_id.strip()
    if not order_id:
        return {"ok": False, "error": "订单号不能为空"}

    seed = int(hashlib.sha256(order_id.encode("utf-8")).hexdigest()[:8], 16)
    statuses = ("待付款", "已付款", "已发货", "已签收")
    products = ("机械键盘", "无线鼠标", "27 寸显示器", "USB-C 扩展坞")
    quantity = 1 + seed % 3
    unit_price = 99 + seed % 900
    amount = round(quantity * unit_price + (seed % 100) / 100, 2)
    created = datetime.now(timezone.utc) - timedelta(days=seed % 15, hours=seed % 24)

    return {
        "ok": True,
        "source": "mock",
        "order_id": order_id,
        "status": statuses[seed % len(statuses)],
        "amount": amount,
        "currency": "CNY",
        "items": [{"name": products[seed % len(products)], "quantity": quantity}],
        "created_at": created.isoformat(timespec="seconds"),
        "note": "占位数据，接入真实订单系统后替换",
    }
