from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import patch

_project_root = str(Path(__file__).resolve().parents[2])
if _project_root in sys.path:
    sys.path.remove(_project_root)
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
from test.benchmark.test_fifo_locomo_mock import MockLLMGenerator, _make_call_service


DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "experiment"
    / "config"
    / "online_continual_memory_locomo_pipeline.yaml"
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
        score = 0.0
        if "innovatetech" in text_lower:
            score += 2.0
        if "oakland" in text_lower:
            score += 2.0
        if "tokyo" in text_lower:
            score += 2.0
        if "duolingo" in text_lower:
            score += 2.0
        if "emma" in text_lower:
            score += 1.0
        if score == 0.0:
            score = 0.5
        return [score, 1.0]

    def embed_batch(self, texts: list[str]):
        return [self.embed(text) for text in texts]

    def is_available(self) -> bool:
        return True


def _create_online_continual_service(config: RuntimeConfig):
    from sage.neuromem.memory_collection import NeuralMemoryCollection
    from sage.neuromem.services import MemoryServiceRegistry
    from sage.neuromem.services.partitional import OnlineContinualMemoryService  # noqa: F401

    collection = NeuralMemoryCollection(
        name=config.get("runtime.memory_name", "online_continual_memory"),
        config=config.get("services.online_continual_memory", {}),
    )
    return MemoryServiceRegistry.create(
        "online_continual_memory",
        collection,
        config.get("services.online_continual_memory", {}),
    )


def run_pipeline(config: RuntimeConfig) -> None:
    task_id = config.get("task_id", "unknown")
    loader = DataLoaderFactory.create(config.get("runtime.dataset", "locomo"))

    pre_insert = PreInsert(config)
    mem_insert = MemoryInsert(config)
    post_insert = PostInsert(config)
    pre_retrieval = PreRetrieval(config)
    mem_retrieval = MemoryRetrieval(config)
    post_retrieval = PostRetrieval(config)
    mem_evaluation = MemoryEvaluation(config)

    service = _create_online_continual_service(config)
    mock_call_service = _make_call_service(service)
    for op in [mem_insert, mem_retrieval, post_insert, post_retrieval]:
        op.call_service = mock_call_service

    total_questions = loader.question_count(task_id)
    test_thresholds = calculate_test_thresholds(
        total_questions, segments=config.get("runtime.test_segments", 4)
    )
    next_threshold_idx = 0
    total_inserted = 0
    last_insert_output = None
    last_retrieval_output = None

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
            last_insert_output = data

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
                    last_retrieval_output = td
                next_threshold_idx += 1

            dialog_ptr += dialog_len

    expected_insertions = loader.dialog_count(task_id)
    assert total_inserted == expected_insertions, (
        f"expected {expected_insertions} inserted entries, got {total_inserted}"
    )
    assert last_insert_output is not None, "expected insert pipeline output"
    assert last_retrieval_output is not None, "expected retrieval pipeline output"

    insert_telemetry = last_insert_output["service_telemetry"]["insert"]
    retrieve_telemetry = last_retrieval_output["service_telemetry"]["retrieve"]

    telemetry_limit = int(config.get("services.online_continual_memory.telemetry_limit", 100))
    insert_event_count = insert_telemetry["summary"]["by_type"]["insert"]["count"]
    assert 1 <= insert_event_count <= min(total_inserted, telemetry_limit)
    assert insert_telemetry["learning"]["training_steps"] == total_inserted
    assert insert_telemetry["learning"]["samples_seen"] == total_inserted
    assert insert_telemetry["last_event"]["event_type"] == "insert"
    assert insert_telemetry["last_event"]["attributes"]["avg_loss"] >= 0.0

    assert len(last_retrieval_output.get("memory_data", [])) >= 1
    assert retrieve_telemetry["summary"]["by_type"]["retrieve"]["count"] >= 1
    assert retrieve_telemetry["learning"]["last_query"]["candidate_count"] >= 1
    assert retrieve_telemetry["last_event"]["attributes"]["result_count"] >= 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate online continual memory benchmark pipeline on local LoCoMo"
    )
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--task_id", type=str, default="conv-mini-01")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not NEUROMEM_REPO.exists():
        raise FileNotFoundError(f"neuromem repo not found: {NEUROMEM_REPO}")

    with tempfile.TemporaryDirectory(prefix="bench-online-continual-locomo-") as tmpdir:
        os.environ["PROCESS_LOG_DIR"] = str(Path(tmpdir) / "logs")
        config = RuntimeConfig.load(str(args.config), task_id=args.task_id)

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

    print("\n✅ online continual memory LoCoMo validation passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())