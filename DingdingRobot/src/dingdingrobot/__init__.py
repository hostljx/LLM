"""钉钉 Stream 机器人。

延迟导入 main，避免 `import dingdingrobot` 时就读配置、装依赖。
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__", "main"]


def main(argv: list[str] | None = None) -> int:
    from .main import main as _main

    return _main(argv)
