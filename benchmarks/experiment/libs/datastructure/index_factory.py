"""IndexFactory - 索引工厂

注册表模式，统一创建各种索引类型。

使用示例：
    # 创建索引
    index = IndexFactory.create("lsh", {"num_perm": 128, "threshold": 0.5})

    # 注册自定义索引
    IndexFactory.register("my_index", MyIndexClass)

    # 查看已注册类型
    IndexFactory.list_types()  # ["lsh", "my_index"]
"""

from __future__ import annotations

import logging
from typing import Any

from .base_index import BaseIndex

logger = logging.getLogger(__name__)


class IndexFactory:
    """索引工厂 - 注册表模式创建索引

    Attributes:
        _registry: 索引类型注册表 {type_name: IndexClass}
    """

    _registry: dict[str, type[BaseIndex]] = {}

    @classmethod
    def create(cls, index_type: str, config: dict[str, Any] | None = None) -> BaseIndex:
        """创建索引实例

        Args:
            index_type: 索引类型名称（如 "lsh"）
            config: 索引配置字典

        Returns:
            索引实例

        Raises:
            ValueError: 索引类型未注册
        """
        if index_type not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys()))
            msg = f"Unknown index type: '{index_type}'. Available: [{available}]"
            raise ValueError(msg)

        return cls._registry[index_type](config or {})

    @classmethod
    def register(cls, index_type: str, index_class: type[BaseIndex]) -> None:
        """注册索引类型

        Args:
            index_type: 索引类型名称
            index_class: 索引类（必须继承 BaseIndex）

        Raises:
            TypeError: index_class 不是 BaseIndex 子类
        """
        if not issubclass(index_class, BaseIndex):
            msg = f"{index_class} must be a subclass of BaseIndex"
            raise TypeError(msg)

        if index_type in cls._registry:
            logger.warning(f"Index type '{index_type}' already registered, overwriting")

        cls._registry[index_type] = index_class
        logger.info(f"Registered index type '{index_type}' -> {index_class.__name__}")

    @classmethod
    def list_types(cls) -> list[str]:
        """列出所有已注册的索引类型"""
        return sorted(cls._registry.keys())

    @classmethod
    def is_registered(cls, index_type: str) -> bool:
        """检查索引类型是否已注册"""
        return index_type in cls._registry
