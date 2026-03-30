"""BaseMemoryService - 记忆服务抽象基类

定义记忆服务的标准接口：insert / retrieve / delete / get。
具体实现（如 LSH、向量检索等）继承此类并实现抽象方法。

扩展方式：
    1. 继承 BaseMemoryService
    2. 实现 _setup_indexes(), insert(), retrieve()
    3. 使用 @MemoryServiceRegistry.register("my_service") 装饰器注册

示例：
    @MemoryServiceRegistry.register("my_service")
    class MyService(BaseMemoryService):
        def _setup_indexes(self):
            self.collection.add_index("main", "my_index", self.config)
        def insert(self, text, metadata=None, **kwargs):
            return self.collection.insert(text, metadata)
        def retrieve(self, query, **kwargs):
            return self.collection.retrieve("main", query)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .collection import SimpleCollection

logger = logging.getLogger(__name__)


class BaseMemoryService(ABC):
    """记忆服务抽象基类

    Attributes:
        collection: 数据容器（SimpleCollection 实例）
        config: 服务配置字典
    """

    def __init__(self, collection: SimpleCollection, config: dict[str, Any] | None = None):
        self.collection = collection
        self.config = config or {}
        self._setup_indexes()

    @abstractmethod
    def _setup_indexes(self) -> None:
        """初始化所需的索引

        在此方法中调用 self.collection.add_index() 创建索引。
        """

    @abstractmethod
    def insert(self, text: str, metadata: dict[str, Any] | None = None, **kwargs: Any) -> str:
        """插入记忆

        Args:
            text: 记忆文本内容
            metadata: 元数据
            **kwargs: 扩展参数

        Returns:
            数据 ID
        """

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        threshold: float | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """检索记忆

        Args:
            query: 查询文本
            top_k: 返回最多 top_k 条
            threshold: 相似度阈值
            **kwargs: 扩展参数

        Returns:
            检索结果列表 [{text, metadata, ...}, ...]
        """

    def delete(self, data_id: str) -> bool:
        """删除记忆"""
        return self.collection.delete(data_id)

    def get(self, data_id: str) -> dict[str, Any] | None:
        """获取单条记忆"""
        return self.collection.get(data_id)

    def list_indexes(self) -> list[dict[str, Any]]:
        """列出该服务管理的所有索引"""
        return self.collection.list_indexes()
