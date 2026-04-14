"""AdapterRegistry — 第三方记忆体适配器注册表

用法（在适配器实现文件中）：
    from .adapter_registry import AdapterRegistry

    @AdapterRegistry.register("mem0")
    class Mem0Adapter(BaseSimpleMemoryAdapter):
        ...

用法（在工厂/调用方中）：
    adapter = AdapterRegistry.create("mem0", config_dict)
"""

from __future__ import annotations

from typing import Any, Type

from .base_adapter import BaseSimpleMemoryAdapter


class AdapterRegistry:
    """简单记忆体适配器注册表（装饰器模式）。"""

    _registry: dict[str, Type[BaseSimpleMemoryAdapter]] = {}

    @classmethod
    def register(cls, name: str):
        """注册适配器类。

        Args:
            name: 适配器标识名，与 config ``services_type`` 中最后一段对应，
                  例如 ``services_type: "simple.mem0"`` → name="mem0"
        """

        def decorator(adapter_cls: Type[BaseSimpleMemoryAdapter]):
            cls._registry[name] = adapter_cls
            return adapter_cls

        return decorator

    @classmethod
    def create(cls, name: str, config: dict[str, Any]) -> BaseSimpleMemoryAdapter:
        """根据名称和配置字典创建适配器实例。

        Args:
            name:   适配器标识名（已注册）
            config: 该适配器的配置字典（来自 YAML ``services.<name>`` 块）

        Raises:
            ValueError: 适配器未注册时抛出
        """
        if name not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Adapter '{name}' not registered. Available: {available}"
            )
        return cls._registry[name](config)

    @classmethod
    def available(cls) -> list[str]:
        """返回所有已注册的适配器名列表。"""
        return list(cls._registry.keys())
