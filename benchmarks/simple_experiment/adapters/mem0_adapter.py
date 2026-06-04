"""Mem0Adapter — mem0 记忆体适配器

依赖：pip install mem0ai
文档：https://docs.mem0.ai

本地模式需要配置 llm、embedder、vector_store（例如 Qdrant）。
托管 API 模式只需 api_key。

配置示例（mem0_locomo.yaml）：
    services:
      services_type: "simple.mem0"
      mem0:
        top_k: 5
        user_id_scope: "auto"
        mem0_config:
          llm:
            provider: "openai"
            config:
              model: "meta-llama/Llama-3.1-8B-Instruct"
              openai_base_url: "http://localhost:18000/v1"
              api_key: "EMPTY"
          embedder:
            provider: "openai"
            config:
              model: "BAAI/bge-m3"
              openai_base_url: "http://localhost:18001/v1"
              api_key: "EMPTY"
          vector_store:
            provider: "qdrant"
            config:
              collection_name: "mem0_bench"
              host: "localhost"
              port: 6333
"""

from __future__ import annotations

from typing import Any

from .adapter_registry import AdapterRegistry
from .base_adapter import BaseSimpleMemoryAdapter


@AdapterRegistry.register("mem0")
class Mem0Adapter(BaseSimpleMemoryAdapter):
    """mem0 黑盒适配器。

    Args:
        config: ``services.mem0`` 配置块，支持以下字段：
            top_k (int):           搜索返回条数，默认 5
            user_id (str):         用户 ID（由 SimpleAdapterFactory 注入，无需手动设置）
            mem0_config (dict):    透传给 mem0.Memory.from_config() 的完整配置
                                   若为空则使用 mem0 默认配置
    """

    def __init__(self, config: dict[str, Any]) -> None:
        try:
            from mem0 import Memory  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "mem0 is not installed. Run: pip install mem0ai"
            ) from exc

        self._top_k: int = config.get("top_k", 5)
        self._user_id: str = config.get("user_id", "neuromem_bench")

        mem0_config: dict[str, Any] = config.get("mem0_config", {})
        if mem0_config:
            self._memory = Memory.from_config(mem0_config)
        else:
            self._memory = Memory()

    @property
    def name(self) -> str:
        return "mem0"

    def add(self, text: str, metadata: dict[str, Any] | None = None) -> str:
        """插入一段文本，返回第一条生成记忆的 ID（或空字符串）。"""
        result = self._memory.add(
            text,
            user_id=self._user_id,
            metadata=metadata or {},
        )
        # mem0 v1+ 返回 {"results": [{"id": ..., "memory": ..., ...}]}
        # mem0 旧版可能直接返回列表
        if isinstance(result, dict):
            records = result.get("results", [])
        elif isinstance(result, list):
            records = result
        else:
            records = []

        if records:
            return str(records[0].get("id", ""))
        return ""

    def search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """检索与 query 相关的记忆，返回标准化结果列表。"""
        limit = top_k if top_k is not None else self._top_k
        result = self._memory.search(
            query,
            limit=limit,
            filters={"user_id": self._user_id},
        )

        # 同上，兼容新旧 mem0 返回格式
        if isinstance(result, dict):
            records = result.get("results", [])
        elif isinstance(result, list):
            records = result
        else:
            records = []

        normalized: list[dict[str, Any]] = []
        for r in records:
            normalized.append(
                {
                    "id": str(r.get("id", "")),
                    # mem0 v1+ 用 "memory" 字段存文本，旧版用 "text"
                    "text": r.get("memory", r.get("text", "")),
                    "score": float(r.get("score", 0.0)),
                    "metadata": r.get("metadata", {}),
                }
            )
        return normalized

    def clear(self) -> None:
        """清空当前 user_id 下的所有记忆。"""
        self._memory.delete_all(filters={"user_id": self._user_id})

    def get_stats(self) -> dict[str, Any]:
        """返回当前记忆条数统计。"""
        try:
            all_memories = self._memory.get_all(filters={"user_id": self._user_id})
            if isinstance(all_memories, dict):
                all_memories = all_memories.get("results", [])
            count = len(all_memories) if all_memories else 0
        except Exception:
            count = -1
        return {
            "adapter": self.name,
            "memory_count": count,
            "user_id": self._user_id,
        }
