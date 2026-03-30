"""BaseIndex - 索引抽象基类

所有记忆索引的统一接口定义。

索引负责管理数据的快速检索结构（如 LSH、FAISS、BM25、图等），
但不存储原始数据本身——原始数据由 Collection 管理。

扩展方式：
    1. 继承 BaseIndex
    2. 实现所有抽象方法
    3. 通过 IndexFactory.register() 注册

示例：
    class MyIndex(BaseIndex):
        def add(self, data_id, text, metadata):
            ...
        def query(self, query, **params):
            ...
        # ... 实现其他抽象方法

    IndexFactory.register("my_index", MyIndex)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class BaseIndex(ABC):
    """索引抽象基类

    所有索引实现必须继承此类并实现全部抽象方法。

    Attributes:
        config: 索引配置字典
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    @abstractmethod
    def add(self, data_id: str, text: str = "", metadata: dict[str, Any] | None = None, **kwargs: Any) -> None:
        """添加数据到索引

        Args:
            data_id: 数据唯一标识
            text: 原始文本
            metadata: 元数据字典
            **kwargs: 扩展参数（如预计算的 embedding 向量）
        """

    @abstractmethod
    def remove(self, data_id: str) -> None:
        """从索引移除数据

        Args:
            data_id: 数据唯一标识
        """

    @abstractmethod
    def query(self, query: Any, **params: Any) -> list[str]:
        """查询索引，返回匹配的 data_id 列表

        Args:
            query: 查询内容（类型取决于具体索引：文本/向量/节点 ID 等）
            **params: 查询参数（top_k, threshold 等）

        Returns:
            匹配的 data_id 列表
        """

    @abstractmethod
    def contains(self, data_id: str) -> bool:
        """检查 data_id 是否在索引中"""

    @abstractmethod
    def size(self) -> int:
        """返回索引中的数据条数"""

    @abstractmethod
    def save(self, save_dir: Path) -> None:
        """持久化索引到磁盘

        Args:
            save_dir: 保存目录路径
        """

    @abstractmethod
    def load(self, load_dir: Path) -> None:
        """从磁盘加载索引

        Args:
            load_dir: 加载目录路径
        """

    @abstractmethod
    def clear(self) -> None:
        """清空索引"""
