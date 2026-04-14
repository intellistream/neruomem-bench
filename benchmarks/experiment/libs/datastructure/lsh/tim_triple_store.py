"""TiMTripleStoreService - TiM 专用三元组记忆服务

完全自包含实现，不依赖 sage.neuromem 的数据结构层。
使用本地 SimpleCollection + LSHIndex，适合作为 neuromem-bench 示例。

TiM 论文要求的数据结构特性：
    1. 存储三元组文本（subject predicate object 的自然语言重构）
    2. 可选地存储 embedding 向量（由 PreInsert/PreRetrieval 算子负责计算）
    3. 检索支持两种模式：
       - 向量模式（vector 不为 None）：余弦相似度排序
       - 文本模式（vector 为 None）：MinHash Jaccard 近似检索
    4. 支持 delete：用于 PostInsert 语义巩固（合并后删除旧条目）
    5. 支持 get_recent：按插入时间倒序取最近 N 条

与 sage LSHHashService 的区别：
    - 使用 SimpleCollection（纯内存，无后端依赖）
    - 使用本地 LSHIndex（datasketch MinHash）
    - 向量检索用 numpy 余弦相似度，无需 FAISS
    - 接口签名保持兼容 sage 服务接口，便于 pipeline 算子透明调用

注册名称：
    "tim_triple_store" → MemoryServiceRegistry
"""

from __future__ import annotations

import logging
import math
from typing import Any

from ..base_service import BaseMemoryService
from ..service_registry import MemoryServiceRegistry

logger = logging.getLogger(__name__)


@MemoryServiceRegistry.register("tim_triple_store")
class TiMTripleStoreService(BaseMemoryService):
    """TiM 三元组记忆服务 — 完全自包含，无 sage 数据结构依赖。

    检索策略：
        - 优先向量（余弦相似度）：当 retrieve(vector=...) 提供向量时使用
        - 回退文本（MinHash LSH）：无向量时使用 datasketch 近似检索

    配置参数 (services.tim_triple_store):
        n_gram    (int)   : MinHash shingle 大小，默认 3
        num_perm  (int)   : MinHash 排列数，默认 128
        threshold (float) : MinHash 相似度阈值，默认 0.3
                            （三元组文本短，需调低阈值保证召回）
    """

    DEFAULT_CONFIG: dict[str, Any] = {
        "n_gram": 3,
        "num_perm": 128,
        "threshold": 0.3,
    }

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _setup_indexes(self) -> None:
        """创建 MinHash LSH 索引（文本模式）"""
        lsh_cfg = {
            "n_gram":    self.config.get("n_gram",    self.DEFAULT_CONFIG["n_gram"]),
            "num_perm":  self.config.get("num_perm",  self.DEFAULT_CONFIG["num_perm"]),
            "threshold": self.config.get("threshold", self.DEFAULT_CONFIG["threshold"]),
        }
        self.collection.add_index("lsh_index", "lsh", lsh_cfg)
        logger.info(
            f"[TiMTripleStore] LSH index created: n_gram={lsh_cfg['n_gram']} "
            f"num_perm={lsh_cfg['num_perm']} threshold={lsh_cfg['threshold']}"
        )

    # ── Insert ────────────────────────────────────────────────────────────────

    def insert(
        self,
        entry: str,
        vector: Any = None,
        metadata: dict[str, Any] | None = None,
        *,
        insert_mode: str = "passive",
        insert_params: dict[str, Any] | None = None,
    ) -> str:
        """插入三元组文本到记忆。

        Args:
            entry:        三元组重构文本（如 "Alice works at TechCorp"）
            vector:       由 PreInsert 算子计算的 embedding 向量（list[float]）
            metadata:     额外元数据（subject/predicate/object 等）
            insert_mode:  "passive"（常规）| "active"（主动写入）
            insert_params: 保留扩展参数（当前未使用）

        Returns:
            data_id (str) — 本条记忆的唯一 ID
        """
        ext_meta: dict[str, Any] = dict(metadata or {})
        if vector is not None:
            ext_meta["embedding"] = vector
        ext_meta["insert_mode"] = insert_mode

        data_id = self.collection.insert(
            text=entry,
            metadata=ext_meta,
            index_names=["lsh_index"],
        )
        logger.debug(f"[TiMTripleStore] Inserted {data_id[:8]}... text={entry[:60]}")
        return data_id

    # ── Retrieve ──────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str | None = None,
        vector: Any = None,
        metadata: dict[str, Any] | None = None,
        top_k: int = 5,
        hints: dict[str, Any] | None = None,
        threshold: float | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """检索相似三元组。

        向量模式（vector 不为 None）：
            遍历全部记忆，按余弦相似度倒序返回 top_k 条。
        文本模式（vector 为 None，query 必须提供）：
            通过 MinHash LSH 近似检索。

        Returns:
            [{"id": str, "text": str, "metadata": dict, "score": float}, ...]
        """
        if vector is not None:
            return self._retrieve_by_vector(vector, top_k)
        if query is not None:
            return self._retrieve_by_text(query, top_k, threshold)
        # 既无 vector 也无 query → 返回最近 top_k 条
        return self.get_recent(top_k)

    def _retrieve_by_vector(
        self, vector: list[float], top_k: int
    ) -> list[dict[str, Any]]:
        """余弦相似度向量检索。"""
        candidates: list[tuple[str, float]] = []

        for data_id, item in self.collection.raw_data.items():
            stored_vec = item.get("metadata", {}).get("embedding")
            if stored_vec is None:
                continue
            score = _cosine_similarity(vector, stored_vec)
            candidates.append((data_id, score))

        # 如果没有任何条目有 embedding，回退到返回全部
        if not candidates:
            return self.get_recent(top_k)

        candidates.sort(key=lambda x: x[1], reverse=True)
        results: list[dict[str, Any]] = []
        for data_id, score in candidates[:top_k]:
            item = self.collection.get(data_id)
            if item is not None:
                entry: dict[str, Any] = item.copy()
                entry["id"] = data_id
                entry["score"] = score
                results.append(entry)
        return results

    def _retrieve_by_text(
        self, query: str, top_k: int, threshold: float | None
    ) -> list[dict[str, Any]]:
        """MinHash 文本检索。"""
        params: dict[str, Any] = {"top_k": top_k}
        if threshold is not None:
            params["threshold"] = threshold

        data_ids = self.collection.query_by_index("lsh_index", query, **params)

        results: list[dict[str, Any]] = []
        for rank, data_id in enumerate(data_ids):
            item = self.collection.get(data_id)
            if item is not None:
                entry = item.copy()
                entry["id"] = data_id
                entry["score"] = 1.0 / (rank + 1)
                results.append(entry)
        return results

    # ── get_recent ────────────────────────────────────────────────────────────

    def get_recent(self, limit: int = 10) -> list[dict[str, Any]]:
        """按插入时间倒序返回最近 limit 条记忆。"""
        all_items = [
            {"id": did, **item}
            for did, item in self.collection.raw_data.items()
        ]
        all_items.sort(key=lambda x: x.get("created_at", 0.0), reverse=True)
        return all_items[:limit]

    def get_stats(self) -> dict[str, Any]:
        """返回服务当前统计快照（供 pipeline_caller 记录）。"""
        return {
            "service_type": "tim_triple_store",
            "total_entries": self.collection.size(),
            "indexes": self.collection.list_indexes(),
        }


# ── Cosine similarity (no numpy dependency) ───────────────────────────────────

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Pure-Python cosine similarity — avoids numpy dependency."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
