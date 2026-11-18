"""工具包。

导入本包即完成内置工具注册：

    from dingdingrobot.tools import registry

    registry.names()    # 查看已注册工具
    registry.schemas()  # 传给 LLM 的 tools 参数
    registry.execute("calculator", '{"expression": "1 + 2"}')

每个工具一个模块，新增工具时：
1. 新建 tools/xxx.py，用 @registry.tool 装饰一个函数；
2. 在下面按顺序 import 一次（顺序决定 schema 顺序）。
"""

from .base import Tool, ToolRegistry
from .registry import registry

# 导入即触发 @registry.tool 注册。
from . import current_time  # noqa: F401
from . import calc  # noqa: F401
from . import weather  # noqa: F401
from . import order  # noqa: F401
from . import units  # noqa: F401

__all__ = ["Tool", "ToolRegistry", "registry"]
