"""MemoryInsert / MemoryRetrieval 的 runtime schema 适配测试。"""

from __future__ import annotations

import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from benchmarks.experiment.libs.memory_insert import MemoryInsert
from benchmarks.experiment.libs.memory_retrieval import MemoryRetrieval
from benchmarks.simple_experiment.libs.simple_memory_add import SimpleMemoryAdd
from benchmarks.simple_experiment.libs.simple_memory_search import SimpleMemorySearch


def test_memory_insert_normalizes_entries(monkeypatch):
    config = {
        "services.services_type": "partitional.fifo_queue",
        "runtime.memory_insert_verbose": False,
    }
    operator = MemoryInsert(config)

    captured: dict[str, object] = {}

    def fake_call_service(service_name, **kwargs):
        if kwargs.get("method") in {"get_stats", "get_telemetry_events"}:
            return {}
        captured["service_name"] = service_name
        captured.update(kwargs)
        return "entry-1"

    monkeypatch.setattr(operator, "call_service", fake_call_service)

    result = operator.execute(
        {
            "memory_entries": [
                {
                    "text": "hello world",
                    "embedding": [0.1, 0.2],
                    "metadata": {"topic": "demo"},
                    "insert_mode": "active",
                    "insert_params": {"priority": 1},
                }
            ]
        }
    )

    assert captured["service_name"] == "fifo_queue"
    assert captured["method"] == "insert"
    assert captured["entry"] == "hello world"
    assert captured["vector"] == [0.1, 0.2]
    assert captured["metadata"] == {"topic": "demo"}
    assert result["memory_entries"][0]["embedding"] == [0.1, 0.2]
    assert result["insert_stats"]["entry_ids"] == ["entry-1"]


def test_memory_retrieval_builds_query_request_and_normalizes_results(monkeypatch):
    config = {
        "services.services_type": "partitional.fifo_queue",
        "services.fifo_queue.retrieval_top_k": 3,
        "runtime.memory_test_verbose": False,
    }
    operator = MemoryRetrieval(config)

    calls: list[dict[str, object]] = []

    def fake_call_service(service_name, **kwargs):
        if kwargs.get("method") in {"get_stats", "get_telemetry_events"}:
            return {}
        calls.append({"service_name": service_name, **kwargs})
        if kwargs.get("query") == "sub query 1":
            return [
                {
                    "id": "doc-1",
                    "text": "alpha",
                    "metadata": {"source": "s1"},
                    "score": 0.9,
                    "extra": "keep",
                }
            ]
        return [
            {
                "id": "doc-2",
                "text": "alpha",
                "metadata": {"source": "s2"},
                "score": 0.8,
            },
            {
                "entry_id": "doc-3",
                "text": "beta",
                "metadata": {"source": "s3"},
                "score": 0.7,
            },
        ]

    monkeypatch.setattr(operator, "call_service", fake_call_service)

    result = operator.execute(
        {
            "question": "main question",
            "query_embedding": [0.9],
            "metadata": {"session": "demo"},
            "retrieve_params": {
                "sub_queries": ["sub query 1", "sub query 2"],
                "sub_query_embeddings": [[0.1], [0.2]],
                "filters": {"topic": "memory"},
                "hints": {"rewrite": True},
                "threshold": 0.6,
            },
        }
    )

    assert len(calls) == 2
    assert calls[0]["service_name"] == "fifo_queue"
    assert calls[0]["query"] == "sub query 1"
    assert calls[0]["vector"] == [0.1]
    assert calls[0]["filters"] == {"topic": "memory"}
    assert calls[0]["hints"] == {"rewrite": True}
    assert calls[0]["threshold"] == 0.6
    assert result["retrieval_request"]["query"] == "main question"
    assert result["retrieval_request"]["top_k"] == 3
    assert len(result["memory_data"]) == 2
    assert result["memory_data"][0]["text"] == "alpha"
    assert result["memory_data"][0]["rank"] == 1
    assert result["memory_data"][0]["extra"] == "keep"
    assert result["memory_data"][1]["text"] == "beta"
    assert result["retrieval_stats"]["retrieved"] == 2


def test_simple_memory_add_prepares_normalized_entry(monkeypatch):
    config = {
        "services.services_type": "simple.mock",
        "runtime.memory_insert_verbose": False,
    }
    operator = SimpleMemoryAdd(config)

    captured: dict[str, object] = {}

    def fake_call_service(service_name, **kwargs):
        captured["service_name"] = service_name
        captured.update(kwargs)
        return "simple-entry-1"

    monkeypatch.setattr(operator, "call_service", fake_call_service)

    result = operator.execute(
        {
            "session_id": 1,
            "dialog_id": 2,
            "dialogs": [
                {"speaker": "Alice", "text": "hello"},
                {"speaker": "Bob", "text": "world"},
            ],
        }
    )

    assert captured["service_name"] == "mock"
    assert captured["method"] == "add"
    assert captured["text"] == "Alice: hello\nBob: world"
    assert captured["metadata"] == {"session_id": 1, "dialog_id": 2, "source": "dialog"}
    assert result["memory_entries"][0]["text"] == "Alice: hello\nBob: world"
    assert result["add_stats"]["entry_id"] == "simple-entry-1"


def test_simple_memory_search_normalizes_results(monkeypatch):
    config = {
        "services.services_type": "simple.mock",
        "services.mock.top_k": 2,
        "runtime.memory_test_verbose": False,
        "operators.simple_retrieval.conversation_format_prompt": "History:\n",
    }
    operator = SimpleMemorySearch(config)

    calls: list[dict[str, object]] = []

    def fake_call_service(service_name, **kwargs):
        calls.append({"service_name": service_name, **kwargs})
        return [
            {"entry_id": "m1", "text": "first", "metadata": {"role": "user"}, "score": 0.9},
            {"id": "m2", "text": "second", "metadata": {"role": "assistant"}, "score": 0.8},
        ]

    monkeypatch.setattr(operator, "call_service", fake_call_service)

    result = operator.execute({"question": "what happened?", "question_idx": 1})

    assert calls[0]["service_name"] == "mock"
    assert calls[0]["query"] == "what happened?"
    assert result["retrieval_request"]["query"] == "what happened?"
    assert result["memory_data"][0]["id"] == "m1"
    assert result["memory_data"][0]["rank"] == 1
    assert result["history_text"] == "History:\n\nfirst\nsecond"