"""Runtime schema 适配层。

优先复用 neuromem 提供的统一 runtime schema；若当前环境中的
 isage-neuromem 版本尚未包含这些类型，则退回到本地兼容实现，
 保持 benchmark 流水线可继续工作。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    from sage.neuromem.runtime import (  # type: ignore
        MemoryEntry as RuntimeMemoryEntry,
        QueryRequest as RuntimeQueryRequest,
        RetrievalResult as RuntimeRetrievalResult,
    )
except ImportError:

    @dataclass(slots=True)
    class RuntimeMemoryEntry:
        entry_id: str | None = None
        text: str = ""
        metadata: dict[str, Any] = field(default_factory=dict)
        vector: Any = None
        created_at: float | None = None
        updated_at: float | None = None

        def to_dict(self) -> dict[str, Any]:
            result = {
                "text": self.text,
                "metadata": dict(self.metadata),
            }
            if self.entry_id is not None:
                result["id"] = self.entry_id
            if self.vector is not None:
                result["vector"] = self.vector
            if self.created_at is not None:
                result["created_at"] = self.created_at
            if self.updated_at is not None:
                result["updated_at"] = self.updated_at
            return result

    @dataclass(slots=True)
    class RuntimeQueryRequest:
        query: str | None = None
        vector: Any = None
        metadata: dict[str, Any] | None = None
        top_k: int = 5
        hints: dict[str, Any] | None = None
        threshold: float | None = None
        filters: dict[str, Any] | None = None
        extra_params: dict[str, Any] = field(default_factory=dict)

        def to_retrieve_kwargs(self) -> dict[str, Any]:
            kwargs: dict[str, Any] = {"top_k": self.top_k}
            if self.query is not None:
                kwargs["query"] = self.query
            if self.vector is not None:
                kwargs["vector"] = self.vector
            if self.metadata is not None:
                kwargs["metadata"] = dict(self.metadata)
            if self.hints is not None:
                kwargs["hints"] = dict(self.hints)
            if self.threshold is not None:
                kwargs["threshold"] = self.threshold
            if self.filters is not None:
                kwargs["filters"] = dict(self.filters)
            kwargs.update(self.extra_params)
            return kwargs

    @dataclass(slots=True)
    class RuntimeRetrievalResult:
        entry_id: str
        text: str
        metadata: dict[str, Any] = field(default_factory=dict)
        score: float | None = None
        vector: Any = None
        source_index: str | None = None
        rank: int | None = None
        raw: dict[str, Any] = field(default_factory=dict)

        @classmethod
        def from_service_record(
            cls,
            record: dict[str, Any],
            *,
            rank: int | None = None,
        ) -> RuntimeRetrievalResult:
            raw = dict(record)
            entry_id = str(raw.pop("id", raw.pop("entry_id", "")))
            return cls(
                entry_id=entry_id,
                text=str(raw.pop("text", raw.pop("content", ""))),
                metadata=dict(raw.pop("metadata", {})),
                score=raw.pop("score", None),
                vector=raw.pop("vector", None),
                source_index=raw.pop("source_index", None),
                rank=raw.pop("rank", rank),
                raw=raw,
            )

        def to_dict(self) -> dict[str, Any]:
            result = {
                "id": self.entry_id,
                "text": self.text,
                "metadata": dict(self.metadata),
            }
            if self.score is not None:
                result["score"] = self.score
            if self.vector is not None:
                result["vector"] = self.vector
            if self.source_index is not None:
                result["source_index"] = self.source_index
            if self.rank is not None:
                result["rank"] = self.rank
            result.update(self.raw)
            return result


_RESERVED_RETRIEVE_PARAMS = {
    "sub_queries",
    "multi_query",
    "sub_query_embeddings",
    "expanded_embeddings",
    "hints",
    "threshold",
    "filters",
}


def normalize_memory_entry(raw_entry: dict[str, Any]) -> RuntimeMemoryEntry:
    """将 bench 的松散 entry 字典转换为统一 MemoryEntry。"""
    vector = raw_entry.get("embedding")
    if vector is None:
        vector = raw_entry.get("vector")

    return RuntimeMemoryEntry(
        entry_id=raw_entry.get("id") or raw_entry.get("entry_id"),
        text=raw_entry.get("text", ""),
        metadata=dict(raw_entry.get("metadata", {})),
        vector=vector,
        created_at=raw_entry.get("created_at"),
        updated_at=raw_entry.get("updated_at"),
    )


def memory_entry_to_bench_dict(
    entry: RuntimeMemoryEntry,
    raw_entry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """将统一 MemoryEntry 转回 bench 兼容字典。"""
    source = raw_entry or {}
    payload = entry.to_dict() if hasattr(entry, "to_dict") else {
        "id": getattr(entry, "entry_id", None),
        "text": entry.text,
        "metadata": dict(entry.metadata),
    }

    result = {
        "text": payload.get("text", ""),
        "metadata": dict(payload.get("metadata", {})),
        "insert_mode": source.get("insert_mode", "passive"),
        "insert_method": source.get("insert_method", "default"),
        "insert_params": source.get("insert_params"),
    }

    entry_id = payload.get("id") or payload.get("entry_id")
    if entry_id is not None:
        result["id"] = entry_id

    vector = payload.get("vector", source.get("embedding", source.get("vector")))
    if vector is not None:
        result["embedding"] = vector

    for key in ("created_at", "updated_at"):
        if key in payload and payload[key] is not None:
            result[key] = payload[key]

    return result


def build_query_request(data: dict[str, Any], top_k: int) -> RuntimeQueryRequest:
    """从 benchmark 流水线数据构建统一 QueryRequest。"""
    retrieve_params = dict(data.get("retrieve_params", {}))
    extra_params = {
        key: value
        for key, value in retrieve_params.items()
        if key not in _RESERVED_RETRIEVE_PARAMS
    }

    return RuntimeQueryRequest(
        query=data.get("question"),
        vector=data.get("query_embedding"),
        metadata=dict(data.get("metadata", {})),
        top_k=top_k,
        hints=retrieve_params.get("hints"),
        threshold=retrieve_params.get("threshold"),
        filters=retrieve_params.get("filters"),
        extra_params=extra_params,
    )


def query_request_to_dict(request: RuntimeQueryRequest) -> dict[str, Any]:
    """转换为便于调试和记录的请求字典。"""
    if hasattr(request, "to_retrieve_kwargs"):
        return request.to_retrieve_kwargs()
    return {
        "query": getattr(request, "query", None),
        "vector": getattr(request, "vector", None),
        "metadata": getattr(request, "metadata", None),
        "top_k": getattr(request, "top_k", 5),
    }


def build_sub_query_requests(
    base_request: RuntimeQueryRequest,
    retrieve_params: dict[str, Any],
) -> list[RuntimeQueryRequest]:
    """根据多查询参数生成子请求列表。"""
    sub_queries = retrieve_params.get("sub_queries", [])
    multi_query = retrieve_params.get("multi_query", [])
    queries = sub_queries or multi_query
    if not queries:
        return []

    query_embeddings = retrieve_params.get("sub_query_embeddings", []) or retrieve_params.get(
        "expanded_embeddings", []
    )

    requests: list[RuntimeQueryRequest] = []
    for idx, single_query in enumerate(queries):
        vector = query_embeddings[idx] if idx < len(query_embeddings) else None
        requests.append(
            RuntimeQueryRequest(
                query=single_query,
                vector=vector,
                metadata=(
                    dict(base_request.metadata) if getattr(base_request, "metadata", None) else None
                ),
                top_k=base_request.top_k,
                hints=(
                    dict(base_request.hints) if getattr(base_request, "hints", None) else None
                ),
                threshold=base_request.threshold,
                filters=(
                    dict(base_request.filters) if getattr(base_request, "filters", None) else None
                ),
                extra_params=dict(getattr(base_request, "extra_params", {})),
            )
        )
    return requests


def normalize_retrieval_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """标准化 service 返回结果。"""
    normalized = []
    for idx, record in enumerate(results, 1):
        item = RuntimeRetrievalResult.from_service_record(record, rank=idx)
        normalized.append(item.to_dict())
    return normalized


def merge_retrieval_results(result_batches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """合并多查询结果并按文本去重。"""
    normalized_items = []
    seen_keys: set[str] = set()

    for record in result_batches:
        item = RuntimeRetrievalResult.from_service_record(record)
        dedupe_key = item.text or item.entry_id
        if dedupe_key and dedupe_key in seen_keys:
            continue
        if dedupe_key:
            seen_keys.add(dedupe_key)
        normalized_items.append(item)

    merged = []
    for idx, item in enumerate(normalized_items, 1):
        item.rank = idx
        merged.append(item.to_dict())
    return merged


def extract_service_telemetry(
    call_service,
    service_name: str,
    *,
    recent_limit: int = 10,
    event_types: set[str] | None = None,
) -> dict[str, Any] | None:
    """从记忆服务中提取遥测摘要与最近事件。

    优先通过 ``get_stats()`` 获取兼容的摘要信息；若服务支持
    ``get_telemetry_events()``，则补充最近事件列表。
    """

    try:
        stats = call_service(service_name, method="get_stats", timeout=10.0) or {}
    except Exception:
        return None

    if not isinstance(stats, dict):
        return None

    telemetry = stats.get("telemetry")
    if not isinstance(telemetry, dict):
        return None

    recent_events: list[dict[str, Any]] = []
    try:
        raw_recent = call_service(
            service_name,
            method="get_telemetry_events",
            limit=recent_limit,
            timeout=10.0,
        )
    except Exception:
        raw_recent = []

    if isinstance(raw_recent, list):
        recent_events = [event for event in raw_recent if isinstance(event, dict)]

    if event_types:
        recent_events = [
            event for event in recent_events if event.get("event_type") in event_types
        ]

    payload: dict[str, Any] = {
        "service_type": stats.get("service_type"),
        "collection_name": stats.get("collection_name"),
        "summary": telemetry,
    }
    if isinstance(stats.get("neural_memory"), dict):
        payload["learning"] = dict(stats["neural_memory"])
    if recent_events:
        payload["recent_events"] = recent_events
        payload["last_event"] = recent_events[-1]
    elif isinstance(telemetry.get("last_event"), dict):
        payload["last_event"] = telemetry["last_event"]
    return payload