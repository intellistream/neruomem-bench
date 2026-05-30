"""MapFunction 兼容层。

单元测试或轻量脚本环境下，`sage.foundation` 可能不可用。
这里提供一个最小回退实现，使算子模块可以被导入并通过 monkeypatch
 注入 `call_service` 进行局部测试。
"""

from __future__ import annotations

import logging

try:
    from sage.foundation import MapFunction  # type: ignore
except ImportError:

    class MapFunction:  # type: ignore[override]
        """用于测试环境的最小兼容实现。"""

        def __init__(self, *args, **kwargs):
            del args, kwargs
            self.logger = logging.getLogger(self.__class__.__name__)

        def call_service(self, *args, **kwargs):
            del args, kwargs
            raise RuntimeError(
                "sage.foundation.MapFunction is unavailable in this environment; "
                "inject call_service manually in tests or run under Sage runtime."
            )