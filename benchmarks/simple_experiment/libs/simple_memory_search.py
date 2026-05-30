"""SimpleMemorySearch — 黑盒记忆检索算子

替代原 pipeline 中的「PreRetrieval + MemoryRetrieval + PostRetrieval」三段式设计。
直接用问题文本调用适配器的 search() 方法，并将结果格式化为 history_text，
保持与 MemoryEvaluation 的接口兼容（MemoryEvaluation 读取 data["history_text"]）。

Pipeline 位置：测试子 pipeline 的第一个 Map 节点
    PipelineServiceSource → SimpleMemorySearch → MemoryEvaluation → PipelineServiceSink
"""

from __future__ import annotations

import time
from typing import Any

from benchmarks.experiment.libs._map_function_compat import MapFunction
from benchmarks.experiment.libs.runtime_adapter import (
    build_query_request,
    normalize_retrieval_results,
    query_request_to_dict,
)
from benchmarks.experiment.utils import process_logger


class SimpleMemorySearch(MapFunction):
    """黑盒记忆检索算子 — 直接调用适配器 search()"""

    def __init__(self, config) -> None:
        super().__init__()
        services_type = config.get("services.services_type")
        if not services_type:
            raise ValueError("Missing required config: services.services_type")
        # e.g. "simple.mem0" → "mem0"
        self.adapter_name: str = services_type.split(".")[-1]
        self.top_k: int = config.get(
            f"services.{self.adapter_name}.top_k", 5
        )
        # 检索结果格式化前缀（与 PostRetrieval 默认值保持一致）
        self.conversation_format: str = config.get(
            "operators.simple_retrieval.conversation_format_prompt",
            "The following is some history information.\n",
        )
        self.verbose: bool = config.get("runtime.memory_test_verbose", True)

    def execute(self, data: dict[str, Any]) -> dict[str, Any]:
        start_time = time.perf_counter()

        query_request = build_query_request(data, self.top_k)
        data["retrieval_request"] = query_request_to_dict(query_request)

        question: str = data.get("question", "")
        question_idx = data.get("question_idx", 0)

        raw_results: list[dict[str, Any]] = self.call_service(
            self.adapter_name,
            method="search",
            query=query_request.query,
            top_k=query_request.top_k,
        ) or []
        results = normalize_retrieval_results(raw_results)

        # 存入 memory_data 供下游使用（与 MemoryRetrieval 输出 key 一致）
        data["memory_data"] = results

        # 构建 history_text 供 MemoryEvaluation 直接拼接
        data["history_text"] = self._format_history(results)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        data.setdefault("stage_timings", {})["search_ms"] = elapsed_ms

        process_logger.log_service(
            "SEARCH",
            f"Q{question_idx}: {question}\n"
            f"Retrieved {len(results)} memories in {elapsed_ms:.1f}ms",
        )

        if self.verbose:
            print(
                f"  [SimpleMemorySearch] 检索: {len(results)} 条 | 耗时: {elapsed_ms:.2f}ms",
                flush=True,
            )

        return data

    def _format_history(self, results: list[dict[str, Any]]) -> str:
        """将检索结果列表转换为 MemoryEvaluation 可直接使用的文本块。"""
        if not results:
            return ""
        lines: list[str] = [self.conversation_format]
        for r in results:
            text = r.get("text", "").strip()
            if text:
                lines.append(text)
        return "\n".join(lines)
