"""LSHIndex - 基于 MinHash 的 LSH 索引

使用 datasketch 库实现文本级 LSH（Locality-Sensitive Hashing）。

原理：
    1. 文本 → character n-gram shingles（如 "hello" → {"hel", "ell", "llo"}）
    2. shingles → MinHash 签名（固定长度的哈希摘要）
    3. MinHash → LSH 桶（相似文本大概率落入同一桶）
    4. 查询时仅在同桶内比较，实现亚线性时间近似最近邻检索

配置参数：
    - n_gram: shingle 大小（默认 3）
    - num_perm: MinHash 排列数，越大越精确（默认 128）
    - threshold: Jaccard 相似度阈值（默认 0.5）

依赖：
    pip install datasketch
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from datasketch import MinHash, MinHashLSH

from ..base_index import BaseIndex

logger = logging.getLogger(__name__)


class LSHIndex(BaseIndex):
    """MinHash + LSH 文本近似匹配索引

    内部维护三个数据结构：
        - lsh: MinHashLSH 对象（LSH 桶结构）
        - minhashes: {data_id: MinHash}（签名缓存）
        - texts: {data_id: str}（原始文本缓存，用于持久化）

    Attributes:
        n_gram: shingle 的字符 n-gram 大小
        num_perm: MinHash 排列数
        threshold: 默认 Jaccard 相似度阈值
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)

        self.n_gram = self.config.get("n_gram", 3)
        self.num_perm = self.config.get("num_perm", 128)
        self.threshold = self.config.get("threshold", 0.5)

        self.lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        self.minhashes: dict[str, MinHash] = {}
        self.texts: dict[str, str] = {}

    def _get_shingles(self, text: str) -> set[str]:
        """将文本转换为 n-gram shingles

        Examples:
            >>> self._get_shingles("hello")  # n_gram=3
            {'hel', 'ell', 'llo'}
        """
        if len(text) < self.n_gram:
            return {text}
        return {text[i : i + self.n_gram] for i in range(len(text) - self.n_gram + 1)}

    def _create_minhash(self, text: str) -> MinHash:
        """创建文本的 MinHash 签名"""
        minhash = MinHash(num_perm=self.num_perm)
        for shingle in self._get_shingles(text):
            minhash.update(shingle.encode("utf-8"))
        return minhash

    def add(
        self, data_id: str, text: str = "", metadata: dict[str, Any] | None = None, **kwargs: Any
    ) -> None:
        """添加文本到 LSH 索引

        如果 data_id 已存在，先移除旧的再添加新的。

        Note:
            当前仅支持文本模式。如需向量 LSH，请使用 FAISS 等向量索引。
        """
        if data_id in self.minhashes:
            self.lsh.remove(data_id)

        minhash = self._create_minhash(text)
        self.lsh.insert(data_id, minhash)
        self.minhashes[data_id] = minhash
        self.texts[data_id] = text

    def remove(self, data_id: str) -> None:
        """从索引移除数据"""
        if data_id not in self.minhashes:
            return
        self.lsh.remove(data_id)
        del self.minhashes[data_id]
        del self.texts[data_id]

    def query(self, query: str | None = None, **params: Any) -> list[str]:
        """查询相似文本

        Args:
            query: 查询文本
            **params:
                threshold: 相似度阈值（覆盖默认值）
                top_k: 返回结果数量上限

        Returns:
            匹配的 data_id 列表

        Raises:
            ValueError: query 为 None
        """
        if query is None:
            raise ValueError("LSHIndex requires a query text (str)")

        threshold = params.get("threshold", self.threshold)
        top_k = params.get("top_k")

        query_minhash = self._create_minhash(query)

        if threshold != self.threshold:
            temp_lsh = MinHashLSH(threshold=threshold, num_perm=self.num_perm)
            for data_id, mh in self.minhashes.items():
                temp_lsh.insert(data_id, mh)
            results = temp_lsh.query(query_minhash)
        else:
            results = self.lsh.query(query_minhash)

        if top_k is not None and top_k > 0:
            results = results[:top_k]

        return results

    def contains(self, data_id: str) -> bool:
        return data_id in self.minhashes

    def size(self) -> int:
        return len(self.minhashes)

    def clear(self) -> None:
        self.lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        self.minhashes.clear()
        self.texts.clear()

    def save(self, save_dir: Path) -> None:
        """持久化索引（保存配置 + 文本，LSH 对象从文本重建）"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        with (save_dir / "config.json").open("w", encoding="utf-8") as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)

        with (save_dir / "texts.json").open("w", encoding="utf-8") as f:
            json.dump(self.texts, f, ensure_ascii=False, indent=2)

        logger.info(f"LSHIndex saved to {save_dir}")

    def load(self, load_dir: Path) -> None:
        """从磁盘加载索引（重建 LSH 结构）"""
        load_dir = Path(load_dir)

        with (load_dir / "config.json").open("r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.n_gram = self.config.get("n_gram", 3)
        self.num_perm = self.config.get("num_perm", 128)
        self.threshold = self.config.get("threshold", 0.5)

        with (load_dir / "texts.json").open("r", encoding="utf-8") as f:
            self.texts = json.load(f)

        self.lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        self.minhashes.clear()

        for data_id, text in self.texts.items():
            minhash = self._create_minhash(text)
            self.lsh.insert(data_id, minhash)
            self.minhashes[data_id] = minhash

        logger.info(f"LSHIndex loaded from {load_dir}, {len(self.texts)} items")
