"""MemoryRetrieval - 记忆检索算子

纯透传模式：调用记忆服务的 retrieve 方法，返回原始检索结果。
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

from benchmarks.experiment.libs._map_function_compat import MapFunction
from benchmarks.experiment.libs.runtime_adapter import (
    build_query_request,
    build_sub_query_requests,
    extract_service_telemetry,
    merge_retrieval_results,
    normalize_retrieval_results,
    query_request_to_dict,
)
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

        query_request = build_query_request(data, self.retrieval_top_k)
        data["retrieval_request"] = query_request_to_dict(query_request)

        retrieve_params = data.get("retrieve_params", {})
        sub_requests = build_sub_query_requests(query_request, retrieve_params)

        if sub_requests:
            raw_results = []
            for request in sub_requests:
                sub_results = self.call_service(
                    self.service_name,
                    method="retrieve",
                    timeout=60.0,
                    **request.to_retrieve_kwargs(),
                )
                raw_results.extend(sub_results or [])
            results = merge_retrieval_results(raw_results)
        else:
            raw_results = self.call_service(
                self.service_name,
                method="retrieve",
                timeout=60.0,
                **query_request.to_retrieve_kwargs(),
            )
            results = normalize_retrieval_results(raw_results or [])

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

        telemetry = extract_service_telemetry(
            self.call_service,
            self.service_name,
            event_types={"retrieve"},
        )
        if telemetry is not None:
            data.setdefault("service_telemetry", {})["retrieve"] = telemetry

        result_texts = [r.get("text", "")[:100] for r in (results or [])[:5]]
        process_logger.log_service(
            "RETRIEVE",
            "Query: "
            f"{query_request.query}\nResults: {stats.retrieved} items\nTop results: {result_texts}",
        )

        print(
            f"  [MemoryRetrieval] 检索: {stats.retrieved}条 | 耗时: {elapsed_ms:.2f}ms", flush=True
        )

        return data
