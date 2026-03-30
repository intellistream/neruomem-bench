"""LSHHashService - LSH 记忆服务

完整的 LSH 记忆服务示例，展示如何基于 BaseMemoryService 和 LSHIndex
构建一个可用的记忆存储/检索服务。

功能：
    - insert: 文本记忆写入（自动生成 MinHash 签名）
    - retrieve: 近似文本检索（基于 Jaccard 相似度）
    - find_duplicates: 重复/近似重复检测

使用示例：
    from benchmarks.experiment.libs.datastructure import SimpleCollection, MemoryServiceRegistry

    # 导入 LSH 模块以触发 @register 装饰器
    import benchmarks.experiment.libs.datastructure.lsh  # noqa: F401

    collection = SimpleCollection("test")
    service = MemoryServiceRegistry.create("lsh_hash", collection, {
        "n_gram": 3,
        "num_perm": 128,
        "threshold": 0.5,
    })

    service.insert("今天天气很好，阳光明媚")
    service.insert("今天天气不错，万里无云")
    results = service.retrieve("天气怎么样")
"""

from __future__ import annotations

import logging
from typing import Any

from ..base_service import BaseMemoryService
from ..service_registry import MemoryServiceRegistry

logger = logging.getLogger(__name__)


@MemoryServiceRegistry.register("lsh_hash")
class LSHHashService(BaseMemoryService):
    """基于 LSH 的文本记忆服务

    使用 MinHash + LSH 实现文本级近似匹配。
    适用于文本去重、相似对话检测等场景。

    配置参数：
        n_gram (int): shingle 字符 n-gram 大小，默认 3
        num_perm (int): MinHash 排列数，默认 128
        threshold (float): Jaccard 相似度阈值，默认 0.5
    """

    DEFAULT_CONFIG = {
        "n_gram": 3,
        "num_perm": 128,
        "threshold": 0.5,
    }

    def _setup_indexes(self) -> None:
        """创建 LSH 索引"""
        lsh_config = {
            "n_gram": self.config.get("n_gram", self.DEFAULT_CONFIG["n_gram"]),
            "num_perm": self.config.get("num_perm", self.DEFAULT_CONFIG["num_perm"]),
            "threshold": self.config.get("threshold", self.DEFAULT_CONFIG["threshold"]),
        }
        self.collection.add_index("lsh_index", "lsh", lsh_config)

    def insert(self, text: str, metadata: dict[str, Any] | None = None, **kwargs: Any) -> str:
        """插入文本记忆

        Args:
            text: 记忆文本
            metadata: 元数据
            **kwargs: 扩展参数

        Returns:
            data_id
        """
        return self.collection.insert(text, metadata, index_names=["lsh_index"])

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        threshold: float | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """检索相似文本记忆

        Args:
            query: 查询文本
            top_k: 返回最多条数
            threshold: Jaccard 相似度阈值（覆盖默认）
            **kwargs: 扩展参数

        Returns:
            [{id, text, metadata, created_at, score}, ...]
        """
        params: dict[str, Any] = {"top_k": top_k}
        if threshold is not None:
            params["threshold"] = threshold

        # 查询匹配的 data_id
        data_ids = self.collection.query_by_index("lsh_index", query, **params)

        # 构建结果，附带排序分数
        results = []
        for rank, data_id in enumerate(data_ids):
            data = self.collection.get(data_id)
            if data is not None:
                entry = data.copy()
                entry["id"] = data_id
                entry["score"] = 1.0 / (rank + 1)  # 基于排名的分数
                results.append(entry)

        return results

    def find_duplicates(self, text: str, threshold: float | None = None) -> list[dict[str, Any]]:
        """查找近似重复记忆

        用于记忆去重场景：插入前检查是否已有相似记忆。

        Args:
            text: 待检查文本
            threshold: Jaccard 相似度阈值

        Returns:
            近似重复的记忆列表
        """
        return self.retrieve(query=text, top_k=10, threshold=threshold)
