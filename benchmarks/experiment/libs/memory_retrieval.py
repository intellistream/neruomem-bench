"""MemoryRetrieval - 记忆检索算子

纯透传模式：调用记忆服务的 retrieve 方法，返回原始检索结果。
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

from sage.foundation import MapFunction

from benchmarks.experiment.utils import process_logger


@dataclass
class RetrievalStats:
    """检索统计"""

    retrieved: int
    time_ms: float
    service_name: str


class MemoryRetrieval(MapFunction):
    """记忆检索算子 - 纯透传模式"""

    def __init__(self, config=None):
        super().__init__()
        self.config = config
        services_type = config.get("services.services_type")
        if not services_type:
            raise ValueError("Missing required config: services.services_type")
        self.service_name = services_type.split(".")[-1]
        self.verbose = config.get("runtime.memory_test_verbose", True)

        service_cfg = f"services.{self.service_name}"
        self.retrieval_top_k = config.get(f"{service_cfg}.retrieval_top_k", 10)

    def execute(self, data: dict[str, Any]) -> dict[str, Any]:
        start_time = time.perf_counter()
        start = time.time()

        query = data.get("question")
        vector = data.get("query_embedding")
        metadata = data.get("metadata", {})
        retrieve_params = data.get("retrieve_params", {})

        sub_queries = retrieve_params.get("sub_queries", [])
        multi_query = retrieve_params.get("multi_query", [])
        queries = sub_queries or multi_query

        if queries and len(queries) >= 1:
            all_results = []
            seen_texts: set[str] = set()
            query_embeddings = retrieve_params.get(
                "sub_query_embeddings", []
            ) or retrieve_params.get("expanded_embeddings", [])

            for idx, single_query in enumerate(queries, 1):
                query_vector = query_embeddings[idx - 1] if idx <= len(query_embeddings) else None
                sub_results = self.call_service(
                    self.service_name,
                    method="retrieve",
                    query=single_query,
                    vector=query_vector,
                    metadata=metadata,
                    top_k=self.retrieval_top_k,
                    timeout=60.0,
                )
                for result in sub_results or []:
                    text = result.get("text", "")
                    if text and text not in seen_texts:
                        seen_texts.add(text)
                        all_results.append(result)
            results = all_results
        else:
            results = self.call_service(
                self.service_name,
                method="retrieve",
                query=query,
                vector=vector,
                metadata=metadata,
                top_k=self.retrieval_top_k,
                timeout=60.0,
            )

        elapsed = (time.time() - start) * 1000
        stats = RetrievalStats(
            retrieved=len(results) if results else 0,
            time_ms=elapsed,
            service_name=self.service_name,
        )

        data["memory_data"] = results
        data["retrieval_stats"] = asdict(stats)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        data.setdefault("stage_timings", {})["memory_retrieval_ms"] = elapsed_ms

        result_texts = [r.get("text", "")[:100] for r in (results or [])[:5]]
        process_logger.log_service(
            "RETRIEVE",
            f"Query: {query}\nResults: {stats.retrieved} items\nTop results: {result_texts}",
        )

        print(
            f"  [MemoryRetrieval] 检索: {stats.retrieved}条 | 耗时: {elapsed_ms:.2f}ms", flush=True
        )

        return data
