from __future__ import annotations

import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import patch

import yaml

_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

_workspace_root = Path(__file__).resolve().parents[3]
_neuromem_repo = _workspace_root / "neuromem"
if _neuromem_repo.exists():
    neuromem_repo_str = str(_neuromem_repo)
    if neuromem_repo_str not in sys.path:
        sys.path.append(neuromem_repo_str)

try:
    import sage.foundation  # type: ignore  # noqa: F401
except ModuleNotFoundError:
    sage_module = types.ModuleType("sage")
    foundation_module = types.ModuleType("sage.foundation")

    class _CompatMapFunction:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def call_service(self, *args, **kwargs):
            del args, kwargs
            raise RuntimeError("inject call_service manually in tests")

    foundation_module.MapFunction = _CompatMapFunction
    sage_module.foundation = foundation_module
    sys.modules.setdefault("sage", sage_module)
    sys.modules["sage.foundation"] = foundation_module

from benchmarks.experiment.libs.memory_insert import MemoryInsert
from benchmarks.experiment.libs.memory_retrieval import MemoryRetrieval
from benchmarks.experiment.libs.post_insert import PostInsert
from benchmarks.experiment.libs.post_retrieval import PostRetrieval
from benchmarks.experiment.libs.pre_insert import PreInsert
from benchmarks.experiment.libs.pre_retrieval import PreRetrieval
from benchmarks.experiment.utils.config.config_loader import RuntimeConfig
from test.benchmark.test_fifo_locomo_mock import MockLLMGenerator, _make_call_service


class MockEmbeddingGenerator:
    def __init__(self, **kwargs):
        self.model_name = "mock-embedding"
        self.base_url = "http://localhost:0/v1"

    @classmethod
    def from_config(cls, config):
        del config
        return cls()

    def embed(self, text: str):
        if "rain" in text.lower():
            return [0.0, 1.0]
        return [1.0, 0.0]

    def embed_batch(self, texts: list[str]):
        return [self.embed(text) for text in texts]

    def is_available(self) -> bool:
        return True


def _write_config(tmp_path: Path) -> RuntimeConfig:
    raw_config = {
        "runtime": {
            "dataset": "mock_locomo",
            "memory_name": "online_continual_memory_test",
            "test_segments": 1,
            "api_key": "mock-key",
            "base_url": "http://localhost:0/v1",
            "model_name": "mock-llm",
            "max_tokens": 64,
            "temperature": 0.0,
            "seed": 42,
            "memory_insert_verbose": False,
            "memory_test_verbose": False,
        },
        "services": {
            "services_type": "partitional.online_continual_memory",
            "online_continual_memory": {
                "collection_type": "neural_continual",
                "feature_dim": 32,
                "learning_rate": 0.2,
                "replay_buffer_size": 32,
                "replay_batch_size": 4,
                "retrieval_top_k": 5,
                "telemetry_limit": 20,
            },
        },
        "operators": {
            "pre_insert": {"action": "none"},
            "post_insert": {"action": "none"},
            "pre_retrieval": {"action": "none"},
            "post_retrieval": {
                "action": "none",
                "conversation_format_prompt": "The following is some history information.\n",
            },
        },
    }

    config_path = tmp_path / "online_continual_memory.yaml"
    config_path.write_text(yaml.safe_dump(raw_config, sort_keys=False), encoding="utf-8")
    return RuntimeConfig.load(str(config_path), task_id="mock-01")


def _create_online_continual_service(config: RuntimeConfig):
    from sage.neuromem.memory_collection import NeuralMemoryCollection
    from sage.neuromem.services import MemoryServiceRegistry
    from sage.neuromem.services.partitional import OnlineContinualMemoryService  # noqa: F401

    collection = NeuralMemoryCollection(
        name="online_continual_memory_test",
        config=config.get("services.online_continual_memory", {}),
    )
    return MemoryServiceRegistry.create(
        "online_continual_memory",
        collection,
        config.get("services.online_continual_memory", {}),
    )


def test_operator_chain_uses_online_continual_memory(tmp_path):
    config = _write_config(tmp_path)
    service = _create_online_continual_service(config)
    call_service = _make_call_service(service)

    with (
        patch("benchmarks.experiment.libs.pre_insert.operator.LLMGenerator", MockLLMGenerator),
        patch(
            "benchmarks.experiment.libs.pre_insert.operator.EmbeddingGenerator",
            MockEmbeddingGenerator,
        ),
        patch("benchmarks.experiment.libs.pre_retrieval.operator.LLMGenerator", MockLLMGenerator),
        patch(
            "benchmarks.experiment.libs.pre_retrieval.operator.EmbeddingGenerator",
            MockEmbeddingGenerator,
        ),
        patch("benchmarks.experiment.libs.post_insert.operator.LLMGenerator", MockLLMGenerator),
        patch(
            "benchmarks.experiment.libs.post_insert.operator.EmbeddingGenerator",
            MockEmbeddingGenerator,
        ),
        patch("benchmarks.experiment.libs.post_retrieval.operator.LLMGenerator", MockLLMGenerator),
        patch(
            "benchmarks.experiment.libs.post_retrieval.operator.EmbeddingGenerator",
            MockEmbeddingGenerator,
        ),
    ):
        pre_insert = PreInsert(config)
        mem_insert = MemoryInsert(config)
        post_insert = PostInsert(config)
        pre_retrieval = PreRetrieval(config)
        mem_retrieval = MemoryRetrieval(config)
        post_retrieval = PostRetrieval(config)

        for op in [mem_insert, mem_retrieval, post_insert, post_retrieval]:
            op.call_service = call_service

        insert_packets = [
            {
                "task_id": "mock-01",
                "session_id": 0,
                "dialog_id": 0,
                "dialogs": [{"speaker": "Alice", "text": "Today is sunny and warm."}],
                "packet_idx": 0,
                "total_packets": 2,
                "is_session_end": False,
            },
            {
                "task_id": "mock-01",
                "session_id": 0,
                "dialog_id": 1,
                "dialogs": [{"speaker": "Alice", "text": "I really like rainy days."}],
                "packet_idx": 1,
                "total_packets": 2,
                "is_session_end": True,
            },
        ]

        last_insert_output = None
        for packet in insert_packets:
            data = pre_insert.execute(dict(packet))
            data = mem_insert.execute(data)
            data = post_insert.execute(data)
            last_insert_output = data

        assert last_insert_output is not None
        insert_telemetry = last_insert_output["service_telemetry"]["insert"]
        assert insert_telemetry["summary"]["by_type"]["insert"]["count"] == 2
        assert insert_telemetry["learning"]["training_steps"] == 2
        assert insert_telemetry["learning"]["samples_seen"] == 2
        assert insert_telemetry["last_event"]["attributes"]["avg_loss"] >= 0.0

        retrieval_data = {
            "task_id": "mock-01",
            "session_id": 0,
            "dialog_id": 1,
            "dialogs": insert_packets[-1]["dialogs"],
            "question": "Does Alice like rainy days?",
            "question_idx": 1,
            "question_metadata": {"category": "factual"},
        }

        result = pre_retrieval.execute(dict(retrieval_data))
        result = mem_retrieval.execute(result)
        result = post_retrieval.execute(result)

        assert len(result["memory_data"]) >= 1
        assert any("rainy days" in item["text"].lower() for item in result["memory_data"])
        retrieve_telemetry = result["service_telemetry"]["retrieve"]
        assert retrieve_telemetry["summary"]["by_type"]["retrieve"]["count"] == 1
        assert retrieve_telemetry["learning"]["last_query"]["candidate_count"] == 2
        assert retrieve_telemetry["last_event"]["attributes"]["result_count"] >= 1
        assert "rainy days" in result["history_text"].lower()