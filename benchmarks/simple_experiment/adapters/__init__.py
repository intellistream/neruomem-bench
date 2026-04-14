"""adapters 包 — 第三方记忆体适配器层

提供 BaseSimpleMemoryAdapter 抽象基类、AdapterRegistry 注册表，
以及 mem0 的参考实现 Mem0Adapter。

新增适配器步骤：
    1. 创建 my_adapter.py，继承 BaseSimpleMemoryAdapter
    2. 用 @AdapterRegistry.register("my_name") 装饰类
    3. 在本文件末尾 import 使注册生效
"""

from .adapter_registry import AdapterRegistry
from .base_adapter import BaseSimpleMemoryAdapter
from .simple_adapter_factory import SimpleAdapterFactory

# 自动触发所有适配器注册
from . import mem0_adapter  # noqa: F401

__all__ = [
    "BaseSimpleMemoryAdapter",
    "AdapterRegistry",
    "SimpleAdapterFactory",
]
