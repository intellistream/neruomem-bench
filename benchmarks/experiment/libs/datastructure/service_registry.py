"""MemoryServiceRegistry - 记忆服务注册表

装饰器模式，统一注册和创建记忆服务。

使用示例：
    # 注册
    @MemoryServiceRegistry.register("lsh_hash")
    class LSHHashService(BaseMemoryService):
        ...

    # 创建
    service = MemoryServiceRegistry.create("lsh_hash", collection, config)

    # 查看已注册
    MemoryServiceRegistry.list_registered()  # ["lsh_hash"]
"""

from __future__ import annotations

import logging
from typing import Any

from .base_service import BaseMemoryService

logger = logging.getLogger(__name__)


class MemoryServiceRegistry:
    """记忆服务注册表

    Attributes:
        _registry: 服务类注册表 {service_type: ServiceClass}
    """

    _registry: dict[str, type[BaseMemoryService]] = {}

    @classmethod
    def register(cls, service_type: str):
        """装饰器：注册记忆服务类

        Args:
            service_type: 服务类型名称

        Returns:
            装饰后的类（原封返回）

        示例：
            @MemoryServiceRegistry.register("my_service")
            class MyService(BaseMemoryService):
                ...
        """

        def decorator(service_class: type[BaseMemoryService]):
            if not issubclass(service_class, BaseMemoryService):
                msg = f"{service_class} must be a subclass of BaseMemoryService"
                raise TypeError(msg)

            if service_type in cls._registry:
                logger.warning(f"Service '{service_type}' already registered, overwriting")

            cls._registry[service_type] = service_class
            logger.info(f"Registered service '{service_type}' -> {service_class.__name__}")
            return service_class

        return decorator

    @classmethod
    def create(cls, service_type: str, collection, config: dict[str, Any] | None = None) -> BaseMemoryService:
        """创建记忆服务实例

        Args:
            service_type: 服务类型名称
            collection: SimpleCollection 实例
            config: 服务配置

        Returns:
            记忆服务实例

        Raises:
            ValueError: 服务类型未注册
        """
        if service_type not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys()))
            msg = f"Unknown service type: '{service_type}'. Available: [{available}]"
            raise ValueError(msg)

        return cls._registry[service_type](collection, config or {})

    @classmethod
    def list_registered(cls) -> list[str]:
        """列出所有已注册的服务类型"""
        return sorted(cls._registry.keys())

    @classmethod
    def get_service_class(cls, service_type: str) -> type[BaseMemoryService] | None:
        """获取服务类（不创建实例）"""
        return cls._registry.get(service_type)
