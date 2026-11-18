"""全局工具注册表实例。

单独放一个模块，避免 builtin 与包入口互相 import 造成循环依赖。
"""

from .base import Tool, ToolRegistry

registry = ToolRegistry()

__all__ = ["Tool", "ToolRegistry", "registry"]
