from __future__ import annotations

import argparse
import copy
import importlib
import json
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

BENCH_REPO = Path(__file__).resolve().parents[2]
if BENCH_REPO.parent.name == "third_party":
    NEUROMEM_REPO = BENCH_REPO.parent.parent
    WORKSPACE_ROOT = BENCH_REPO.parent
else:
    WORKSPACE_ROOT = BENCH_REPO.parent
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


FLOWRAG_REPO = (
    WORKSPACE_ROOT / "flowrag"
    if (WORKSPACE_ROOT / "flowrag").exists()
    else WORKSPACE_ROOT / "FlowRAG"
)
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


def _prepare_config(
    config_path: Path,
    task_id: str,
    flowrag_index_dir: Path,
    *,
    enable_write_gate: bool,
    enable_retrieval_expansion: bool,
) -> RuntimeConfig:
    config = RuntimeConfig.load(str(config_path), task_id=task_id)
    adapter_specs = copy.deepcopy(config.get("services.fifo_queue.strategy_adapters") or [])
    filtered_specs = []
    for spec in adapter_specs:
        adapter_name = spec.get("name") or spec.get("type")
        if adapter_name == "streamfp_selector" and not enable_write_gate:
            continue
        if adapter_name == "flowrag_retriever" and not enable_retrieval_expansion:
            continue
        filtered_specs.append(spec)

    config._config.setdefault("services", {}).setdefault("fifo_queue", {})[
        "strategy_adapters"
    ] = filtered_specs

    _add_repo_to_path(STREAMFP_REPO)
    streamprompt_module = importlib.import_module("dataselections.streamprompt")
    selector = streamprompt_module.Streamprompt(measure="cosine")
    model = _DummyStreamModel()

    def build_examples(entries, context):
        del context
        text_lower = entries[0].text.lower()
        fill_value = (
            1.0
            if "really like rainy days" in text_lower or "feel peaceful" in text_lower
            else -1.0
        )
        return torch.full((1, 2, 768), fill_value=fill_value, dtype=torch.float32)

    for spec in filtered_specs:
        adapter_name = spec.get("name") or spec.get("type")
        if adapter_name == "streamfp_selector":
            spec["repo_path"] = str(STREAMFP_REPO)
            spec["selector"] = selector
            spec["model"] = model
            spec["examples_builder"] = build_examples
        elif adapter_name == "flowrag_retriever":
            spec["repo_path"] = str(FLOWRAG_REPO)
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


def run_pipeline(
    config: RuntimeConfig,
    *,
    scenario_name: str,
    enable_write_gate: bool,
    enable_retrieval_expansion: bool,
) -> dict[str, object]:
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
    insert_candidates = 0
    total_inserted = 0
    insert_counts: list[int] = []
    expanded_queries = 0
    retrieval_result_counts: list[int] = []

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

            inserted = int(data.get("insert_stats", {}).get("inserted", 0))
            insert_candidates += 1
            total_inserted += inserted
            insert_counts.append(inserted)

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
                    retrieval_result_count = len(td.get("memory_data", []))
                    retrieval_result_counts.append(retrieval_result_count)
                    expansion_present = any(
                        item.get("metadata", {}).get("source") == "flowrag"
                        for item in td.get("memory_data", [])
                    )
                    if expansion_present:
                        expanded_queries += 1
                next_threshold_idx += 1

            dialog_ptr += dialog_len

    stored_items = fifo_service.get_recent(limit=10)
    streamfp_items = [
        item
        for item in stored_items
        if item.get("metadata", {}).get("strategy_adapter") == "streamfp_selector"
        and item.get("metadata", {}).get("streamfp_score") is not None
    ]
    saw_streamfp_metadata = any(
        item.get("metadata", {}).get("strategy_adapter") == "streamfp_selector"
        and item.get("metadata", {}).get("streamfp_score") is not None
        for item in stored_items
    )
    skipped_entries = insert_candidates - total_inserted
    summary: dict[str, object] = {
        "scenario": scenario_name,
        "task_id": task_id,
        "write_gate_enabled": enable_write_gate,
        "retrieval_expansion_enabled": enable_retrieval_expansion,
        "insert_candidates": insert_candidates,
        "inserted_entries": total_inserted,
        "skipped_entries": skipped_entries,
        "insert_counts": insert_counts,
        "retrieval_queries": len(retrieval_result_counts),
        "expanded_queries": expanded_queries,
        "avg_retrieval_result_count": (
            sum(retrieval_result_counts) / len(retrieval_result_counts)
            if retrieval_result_counts
            else 0.0
        ),
        "max_retrieval_result_count": max(retrieval_result_counts) if retrieval_result_counts else 0,
        "last_retrieval_result_count": retrieval_result_counts[-1] if retrieval_result_counts else 0,
        "stored_entries": len(stored_items),
        "stored_entries_with_gate_metadata": len(streamfp_items),
        "gate_scores": [
            float(item.get("metadata", {}).get("streamfp_score")) for item in streamfp_items
        ],
    }

    if enable_write_gate:
        assert total_inserted == 1, f"expected 1 kept entry after write gating, got {total_inserted}"
        assert skipped_entries == 1, f"expected 1 skipped entry after write gating, got {skipped_entries}"
        assert saw_streamfp_metadata, "expected stored entries to preserve gate metadata"
    else:
        assert total_inserted == 2, f"expected 2 kept entries without write gating, got {total_inserted}"
        assert skipped_entries == 0, f"expected 0 skipped entries without write gating, got {skipped_entries}"

    if enable_retrieval_expansion:
        assert expanded_queries >= 1, "expected at least one retrieval query with expansion"
    else:
        assert expanded_queries == 0, "did not expect retrieval expansion without the expansion module"

    return summary


def run_ablation_scenarios(config_path: Path, task_id: str, flowrag_index_dir: Path) -> dict[str, object]:
    scenarios = [
        ("fifo_base", False, False),
        ("fifo_write_gate", True, False),
        ("fifo_retrieval_expansion", False, True),
        ("fifo_gate_plus_expansion", True, True),
    ]
    scenario_summaries: list[dict[str, object]] = []

    for scenario_name, enable_write_gate, enable_retrieval_expansion in scenarios:
        config = _prepare_config(
            config_path,
            task_id,
            flowrag_index_dir,
            enable_write_gate=enable_write_gate,
            enable_retrieval_expansion=enable_retrieval_expansion,
        )
        scenario_summaries.append(
            run_pipeline(
                config,
                scenario_name=scenario_name,
                enable_write_gate=enable_write_gate,
                enable_retrieval_expansion=enable_retrieval_expansion,
            )
        )

    return {
        "task_id": task_id,
        "summary_type": "mechanism_ablation",
        "scenarios": scenario_summaries,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate external strategy adapters in bench")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--task_id", type=str, default="mock-01")
    parser.add_argument("--summary-json", type=str, default="")
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
                summary = run_ablation_scenarios(Path(args.config), args.task_id, flowrag_index_dir)
            finally:
                process_logger.close()
                os.environ.pop("PROCESS_LOG_DIR", None)

    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\n📦 adapter summary written to {summary_path}")

    print("\n✅ external strategy adapter benchmark validation passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
