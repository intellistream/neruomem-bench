"""SimpleCollection - 简化版统一数据容器

纯内存实现，管理原始数据 + 多索引。

职责：
- 存储原始数据 (text + metadata)
- 管理多个索引 (通过 IndexFactory 创建)
- 提供 insert / get / delete / query 统一接口

与 neuromem 完整版的区别：
- 仅支持内存存储（无 Redis/SageDB 后端）
- 无 StorageFactory 依赖
- 专为 benchmark 场景设计

使用示例：
    collection = SimpleCollection("my_memory")
    collection.add_index("lsh_idx", "lsh", {"num_perm": 128, "threshold": 0.5})
    data_id = collection.insert("今天天气很好", {"speaker": "user"})
    results = collection.retrieve("lsh_idx", "天气")
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class SimpleCollection:
    """简化版统一数据容器

    Attributes:
        name: Collection 名称
        raw_data: 原始数据 {data_id: {text, metadata, created_at}}
        indexes: 索引容器 {index_name: IndexObject}
        index_metadata: 索引配置信息
    """

    def __init__(self, name: str, config: dict[str, Any] | None = None):
        self.name = name
        self.config = config or {}
        self.raw_data: dict[str, dict[str, Any]] = {}
        self.indexes: dict[str, Any] = {}
        self.index_metadata: dict[str, dict[str, Any]] = {}

    def insert(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
        index_names: list[str] | None = None,
    ) -> str:
        """插入数据

        Args:
            text: 原始文本
            metadata: 元数据
            index_names: 加入哪些索引（None = 所有）

        Returns:
            data_id (SHA256 哈希)
        """
        data_id = self._generate_id(text, metadata)

        self.raw_data[data_id] = {
            "text": text,
            "metadata": metadata or {},
            "created_at": time.time(),
        }

        targets = list(self.indexes.keys()) if index_names is None else index_names
        for idx_name in targets:
            if idx_name in self.indexes:
                self.indexes[idx_name].add(data_id, text, metadata or {})

        return data_id

    def get(self, data_id: str) -> dict[str, Any] | None:
        """获取原始数据"""
        return self.raw_data.get(data_id)

    def delete(self, data_id: str) -> bool:
        """删除数据（同时从所有索引移除）"""
        if data_id not in self.raw_data:
            return False

        for index in self.indexes.values():
            index.remove(data_id)

        del self.raw_data[data_id]
        return True

    def size(self) -> int:
        """数据条数"""
        return len(self.raw_data)

    # ==================== 索引管理 ====================

    def add_index(self, name: str, index_type: str, config: dict[str, Any] | None = None) -> bool:
        """添加索引

        Args:
            name: 索引名称
            index_type: 索引类型（需已在 IndexFactory 注册）
            config: 索引配置

        Returns:
            是否成功（False = 已存在）
        """
        if name in self.indexes:
            logger.warning(f"Index '{name}' already exists in {self.name}")
            return False

        from .index_factory import IndexFactory

        index = IndexFactory.create(index_type, config or {})
        self.indexes[name] = index
        self.index_metadata[name] = {
            "type": index_type,
            "config": config or {},
            "created_at": time.time(),
        }
        return True

    def remove_index(self, name: str) -> bool:
        """删除索引（不影响原始数据）"""
        if name not in self.indexes:
            return False
        del self.indexes[name]
        del self.index_metadata[name]
        return True

    def list_indexes(self) -> list[dict[str, Any]]:
        """列出所有索引信息"""
        return [
            {"name": name, "type": meta["type"], "config": meta["config"]}
            for name, meta in self.index_metadata.items()
        ]

    def query_by_index(self, index_name: str, query: Any, **params: Any) -> list[str]:
        """通过索引查询 data_id 列表

        Raises:
            ValueError: 索引不存在
        """
        if index_name not in self.indexes:
            msg = f"Index '{index_name}' not found in {self.name}"
            raise ValueError(msg)
        return self.indexes[index_name].query(query, **params)

    def retrieve(self, index_name: str, query: Any, **params: Any) -> list[dict[str, Any]]:
        """检索完整数据（query + get 快捷方式）

        Returns:
            [{id, text, metadata, created_at}, ...]
        """
        data_ids = self.query_by_index(index_name, query, **params)
        results = []
        for id_ in data_ids:
            data = self.raw_data.get(id_)
            if data is not None:
                entry = data.copy()
                entry["id"] = id_
                results.append(entry)
        return results

    # ==================== 内部方法 ====================

    def _generate_id(self, text: str, metadata: dict[str, Any] | None = None) -> str:
        """基于内容生成稳定的 SHA256 ID"""
        key = text
        if metadata:
            key += json.dumps(metadata, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def __len__(self) -> int:
        return len(self.raw_data)

    def __contains__(self, data_id: str) -> bool:
        return data_id in self.raw_data

    def __repr__(self) -> str:
        return (
            f"SimpleCollection(name='{self.name}', "
            f"data_count={len(self.raw_data)}, "
            f"index_count={len(self.indexes)})"
        )
