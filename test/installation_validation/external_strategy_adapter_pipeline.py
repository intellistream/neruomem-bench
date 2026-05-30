from __future__ import annotations

import argparse
import importlib
import logging
import os
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
NEUROMEM_REPO = WORKSPACE_ROOT / "neuromem"
if NEUROMEM_REPO.exists():
    neuromem_repo_str = str(NEUROMEM_REPO)
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
            self.logger = logging.getLogger(self.__class__.__name__)

        def call_service(self, *args, **kwargs):
            del args, kwargs
            raise RuntimeError("inject call_service manually in validation script")

    foundation_module.MapFunction = _CompatMapFunction
    sage_module.foundation = foundation_module
    sys.modules.setdefault("sage", sage_module)
    sys.modules["sage.foundation"] = foundation_module

from benchmarks.experiment.libs.memory_evaluation import MemoryEvaluation
from benchmarks.experiment.libs.memory_insert import MemoryInsert
from benchmarks.experiment.libs.memory_retrieval import MemoryRetrieval
from benchmarks.experiment.libs.post_insert import PostInsert
from benchmarks.experiment.libs.post_retrieval import PostRetrieval
from benchmarks.experiment.libs.pre_insert import PreInsert
from benchmarks.experiment.libs.pre_retrieval import PreRetrieval
from benchmarks.experiment.utils import RuntimeConfig, calculate_test_thresholds, process_logger
from benchmarks.experiment.utils.dataloader import DataLoaderFactory
from test.benchmark.test_fifo_locomo_mock import MockLLMGenerator, MockLocomoLoader
from test.benchmark.test_fifo_locomo_mock import _make_call_service


FLOWRAG_REPO = WORKSPACE_ROOT / "FlowRAG"
STREAMFP_REPO = WORKSPACE_ROOT / "streamfp"
DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "experiment"
    / "config"
    / "fifo_external_adapters_mock.yaml"
)


class MockEmbeddingGenerator:
    def __init__(self, **kwargs):
        self.model_name = "mock-embedding"
        self.base_url = "http://localhost:0/v1"

    @classmethod
    def from_config(cls, config):
        del config
        return cls()

    def embed(self, text: str):
        text_lower = text.lower()
        if "rain" in text_lower:
            return [0.0, 1.0]
        return [1.0, 0.0]

    def embed_batch(self, texts: list[str]):
        return [self.embed(text) for text in texts]

    def is_available(self) -> bool:
        return True


class _IdentityFeat:
    def patch_embed(self, examples):
        return examples


class _DummyPrompt(nn.Module):
    def __init__(self):
        super().__init__()
        self.e_pool_size = 2
        self.n_tasks = 1
        self.task_count = 0
        self.e_p_0 = nn.Parameter(torch.ones(2, 2, 768))


class _DummyStreamModel:
    def __init__(self):
        self.prompt = _DummyPrompt()
        self.feat = _IdentityFeat()


def _add_repo_to_path(repo_path: Path) -> None:
    repo_str = str(repo_path)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def _build_flowrag_index(index_dir: Path) -> None:
    _add_repo_to_path(FLOWRAG_REPO)
    index_module = importlib.import_module("src.retrieval.index")

    faiss_index = index_module.FaissIndex(str(index_dir))
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    documents = ["external-sunny-memory", "external-rain-memory"]
    faiss_index.build(embeddings, documents, index_type="FLAT")
    faiss_index.save("tiny")


def _prepare_config(config_path: Path, task_id: str, flowrag_index_dir: Path) -> RuntimeConfig:
    config = RuntimeConfig.load(str(config_path), task_id=task_id)
    adapter_specs = config.get("services.fifo_queue.strategy_adapters") or []

    _add_repo_to_path(STREAMFP_REPO)
    streamprompt_module = importlib.import_module("dataselections.streamprompt")
    selector = streamprompt_module.Streamprompt(measure="cosine")
    model = _DummyStreamModel()

    def build_examples(entries, context):
        del context
        text_lower = entries[0].text.lower()
        fill_value = 1.0 if "rain" in text_lower else -1.0
        return torch.full((1, 2, 768), fill_value=fill_value, dtype=torch.float32)

    for spec in adapter_specs:
        adapter_name = spec.get("name") or spec.get("type")
        if adapter_name == "streamfp_selector":
            spec["selector"] = selector
            spec["model"] = model
            spec["examples_builder"] = build_examples
        elif adapter_name == "flowrag_retriever":
            spec["index_dir"] = str(flowrag_index_dir)

    return config


def _create_fifo_service(config: RuntimeConfig):
    from sage.neuromem.memory_collection import UnifiedCollection
    from sage.neuromem.services.partitional import FIFOQueueService  # noqa: F401
    from sage.neuromem.services import MemoryServiceRegistry

    collection = UnifiedCollection(name="fifo_external_adapters_mock")
    return MemoryServiceRegistry.create(
        "fifo_queue",
        collection,
        config.get("services.fifo_queue", {}),
    )


def run_pipeline(config: RuntimeConfig) -> None:
    task_id = config.get("task_id", "unknown")
    loader = MockLocomoLoader()

    pre_insert = PreInsert(config)
    mem_insert = MemoryInsert(config)
    post_insert = PostInsert(config)
    pre_retrieval = PreRetrieval(config)
    mem_retrieval = MemoryRetrieval(config)
    post_retrieval = PostRetrieval(config)
    mem_evaluation = MemoryEvaluation(config)

    fifo_service = _create_fifo_service(config)
    mock_call_service = _make_call_service(fifo_service)
    for op in [mem_insert, mem_retrieval, post_insert, post_retrieval]:
        op.call_service = mock_call_service

    total_questions = loader.question_count(task_id)
    test_thresholds = calculate_test_thresholds(
        total_questions, segments=config.get("runtime.test_segments", 2)
    )
    next_threshold_idx = 0
    total_inserted = 0
    saw_flowrag_result = False

    for session_id, max_dialog_idx in loader.sessions(task_id):
        dialog_ptr = 0
        while dialog_ptr <= max_dialog_idx:
            dialogs = loader.get_dialog(task_id, session_id, dialog_ptr)
            dialog_len = len(dialogs) if dialogs else 2

            insert_data = {
                "task_id": task_id,
                "session_id": session_id,
                "dialog_id": dialog_ptr,
                "dialogs": dialogs,
                "packet_idx": dialog_ptr,
                "total_packets": loader.dialog_count(task_id),
                "is_session_end": (dialog_ptr + dialog_len) > max_dialog_idx,
            }

            data = pre_insert.execute(dict(insert_data))
            data = mem_insert.execute(data)
            data = post_insert.execute(data)

            total_inserted += data.get("insert_stats", {}).get("inserted", 0)

            visible_dialog = dialog_ptr + dialog_len - 1
            current_questions = loader.get_evaluation(task_id, session_id, visible_dialog)
            current_count = len(current_questions)
            should_test = (
                next_threshold_idx < len(test_thresholds)
                and current_count >= test_thresholds[next_threshold_idx]
            )

            if should_test:
                for q_idx, qa in enumerate(current_questions):
                    test_data = {
                        "task_id": task_id,
                        "session_id": session_id,
                        "dialog_id": dialog_ptr,
                        "dialogs": dialogs,
                        "question": qa["question"],
                        "question_idx": q_idx + 1,
                        "question_metadata": qa,
                    }
                    td = pre_retrieval.execute(dict(test_data))
                    td = mem_retrieval.execute(td)
                    td = post_retrieval.execute(td)
                    td = mem_evaluation.execute(td)
                    saw_flowrag_result = saw_flowrag_result or any(
                        item.get("metadata", {}).get("source") == "flowrag"
                        for item in td.get("memory_data", [])
                    )
                next_threshold_idx += 1

            dialog_ptr += dialog_len

    stored_items = fifo_service.get_recent(limit=10)
    saw_streamfp_metadata = any(
        item.get("metadata", {}).get("strategy_adapter") == "streamfp_selector"
        and item.get("metadata", {}).get("streamfp_score") is not None
        for item in stored_items
    )

    assert total_inserted == 2, f"expected 2 kept entries after streamfp gating, got {total_inserted}"
    assert saw_flowrag_result, "expected FlowRAG retrieval augmentation in benchmark loop"
    assert saw_streamfp_metadata, "expected stored entries to preserve streamfp metadata"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate external strategy adapters in bench")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--task_id", type=str, default="mock-01")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not FLOWRAG_REPO.exists():
        raise FileNotFoundError(f"FlowRAG repo not found: {FLOWRAG_REPO}")
    if not STREAMFP_REPO.exists():
        raise FileNotFoundError(f"streamfp repo not found: {STREAMFP_REPO}")

    DataLoaderFactory.register("mock_locomo", MockLocomoLoader)

    with tempfile.TemporaryDirectory(prefix="bench-ext-adapter-") as tmpdir:
        tmp_path = Path(tmpdir)
        os.environ["PROCESS_LOG_DIR"] = str(tmp_path / "logs")

        flowrag_index_dir = tmp_path / "flowrag_index"
        _build_flowrag_index(flowrag_index_dir)
        config = _prepare_config(Path(args.config), args.task_id, flowrag_index_dir)

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
            patch("benchmarks.experiment.libs.memory_evaluation.LLMGenerator", MockLLMGenerator),
        ):
            try:
                run_pipeline(config)
            finally:
                process_logger.close()
                os.environ.pop("PROCESS_LOG_DIR", None)

    print("\n✅ external strategy adapter benchmark validation passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())