from __future__ import annotations

import json
from pathlib import Path

from benchmarks.evaluation.analysis.export_locomo_summary import (
    export_strategy_summary,
    summarize_task_file,
)


def test_summarize_task_file_computes_locomo_metrics(tmp_path: Path) -> None:
    artifact = {
        "experiment_info": {"dataset": "locomo", "task_id": "conv-test"},
        "dataset_statistics": {
            "total_sessions": 2,
            "total_messages": 10,
            "total_questions": 4,
        },
        "test_results": [
            {
                "test_index": 1,
                "questions": [
                    {
                        "predicted_answer": "Paris",
                        "reference_answer": "Paris",
                        "category": 4,
                    },
                    {
                        "predicted_answer": "Not mentioned in the conversation.",
                        "reference_answer": "unknown",
                        "category": 5,
                    },
                ],
            }
        ],
        "timing_summary": {
            "insert_timings": {
                "summary": {
                    "pre_insert_ms": {"avg_ms": 1.0},
                    "memory_insert_ms": {"avg_ms": 2.0},
                    "post_insert_ms": {"avg_ms": 0.5},
                }
            },
            "retrieval_timings": {
                "summary": {
                    "pre_retrieval_ms": {"avg_ms": 3.0},
                    "memory_retrieval_ms": {"avg_ms": 4.0},
                    "post_retrieval_ms": {"avg_ms": 0.25},
                }
            },
        },
        "memory_snapshots": [
            {
                "neural_memory": {
                    "training_steps": 0,
                    "samples_seen": 0,
                    "active_buffer_items": 0,
                },
                "storage": {"total_entries": 0, "parameter_count": 1024},
            }
        ],
    }
    json_path = tmp_path / "conv-test_0001.json"
    json_path.write_text(json.dumps(artifact), encoding="utf-8")

    summary = summarize_task_file(json_path)

    assert summary["task_id"] == "conv-test"
    assert summary["dataset_questions"] == 4
    assert summary["evaluated_questions"] == 2
    assert summary["evaluation_mode"] == "partial_trace"
    assert summary["response_multiplicity"] == 0.5
    assert summary["abstain_count"] == 1
    assert summary["avg_locomo_f1"] == 1.0
    assert summary["avg_exact_match"] == 0.5
    assert summary["category_scores"]["4"] == 1.0
    assert summary["category_scores"]["5"] == 1.0
    assert summary["timings_ms"]["memory_retrieval"] == 4.0
    assert summary["final_snapshot"]["parameter_count"] == 1024


def test_summarize_task_file_supports_simple_dataset_stats_keys(tmp_path: Path) -> None:
    artifact = {
        "experiment_info": {"dataset": "locomo", "task_id": "conv-simple"},
        "dataset_statistics": {
            "sessions": 3,
            "messages": 12,
            "questions": 5,
        },
        "test_results": [
            {
                "test_index": 1,
                "questions": [
                    {
                        "predicted_answer": "Not mentioned in the conversation.",
                        "reference_answer": "unknown",
                        "category": 5,
                    }
                ],
            }
        ],
        "timing_summary": {},
        "memory_snapshots": [],
    }
    json_path = tmp_path / "conv-simple_0001.json"
    json_path.write_text(json.dumps(artifact), encoding="utf-8")

    summary = summarize_task_file(json_path)

    assert summary["sessions"] == 3
    assert summary["messages"] == 12
    assert summary["dataset_questions"] == 5


def test_export_strategy_summary_writes_protocol_aware_markdown(tmp_path: Path) -> None:
    strategy_dir = tmp_path / "mem0"
    strategy_dir.mkdir()
    artifact = {
        "experiment_info": {"dataset": "locomo", "task_id": "conv-protocol"},
        "dataset_statistics": {
            "sessions": 1,
            "messages": 2,
            "questions": 3,
        },
        "test_results": [
            {
                "test_index": 1,
                "questions": [
                    {
                        "question_index": 1,
                        "question_text": "Q1",
                        "predicted_answer": "A1",
                        "reference_answer": "A1",
                        "category": 4,
                    },
                    {
                        "question_index": 1,
                        "question_text": "Q1",
                        "predicted_answer": "A1",
                        "reference_answer": "A1",
                        "category": 4,
                    },
                    {
                        "question_index": 2,
                        "question_text": "Q2",
                        "predicted_answer": "A2",
                        "reference_answer": "A2",
                        "category": 4,
                    },
                ],
            }
        ],
        "timing_summary": {},
        "memory_snapshots": [],
    }
    json_path = strategy_dir / "conv-protocol_0001.json"
    json_path.write_text(json.dumps(artifact), encoding="utf-8")

    output_dir = tmp_path / "output"
    summary = export_strategy_summary(strategy_dir, output_dir)
    markdown = (output_dir / "summary.md").read_text(encoding="utf-8")

    assert summary["aggregate"]["evaluation_mode"] == "single_pass"
    assert summary["aggregate"]["total_dataset_questions"] == 3
    assert summary["aggregate"]["total_evaluated_questions"] == 3
    assert "- Total unique dataset questions: 3" in markdown
    assert "- Total scored responses: 3" in markdown
    assert "| Task | Unique Q | Scored Resp. | Protocol |" in markdown


def test_export_strategy_summary_prefers_newer_mtime_over_filename(tmp_path: Path) -> None:
    strategy_dir = tmp_path / "online_continual_memory_full"
    strategy_dir.mkdir()

    older_name_newer_time = strategy_dir / "conv-30_0834.json"
    newer_name_older_time = strategy_dir / "conv-30_1211.json"

    base_artifact = {
        "experiment_info": {"dataset": "locomo", "task_id": "conv-30"},
        "dataset_statistics": {"sessions": 1, "messages": 2, "questions": 1},
        "test_results": [
            {
                "test_index": 1,
                "questions": [
                    {
                        "question_index": 1,
                        "question_text": "Q1",
                        "predicted_answer": "A1",
                        "reference_answer": "A1",
                        "category": 4,
                    }
                ],
            }
        ],
        "timing_summary": {},
        "memory_snapshots": [],
    }

    older_payload = dict(base_artifact)
    older_payload["experiment_info"] = {"dataset": "locomo", "task_id": "conv-30"}
    newer_payload = dict(base_artifact)
    newer_payload["experiment_info"] = {"dataset": "locomo", "task_id": "conv-30"}
    older_name_newer_time.write_text(json.dumps(older_payload), encoding="utf-8")
    newer_name_older_time.write_text(json.dumps(newer_payload), encoding="utf-8")

    older_mtime = 1_700_000_000
    newer_mtime = older_mtime + 10
    newer_name_older_time.touch()
    older_name_newer_time.touch()
    import os
    os.utime(newer_name_older_time, (older_mtime, older_mtime))
    os.utime(older_name_newer_time, (newer_mtime, newer_mtime))

    output_dir = tmp_path / "output"
    summary = export_strategy_summary(strategy_dir, output_dir, tasks=["conv-30"])

    assert summary["tasks"][0]["source_file"].endswith("conv-30_0834.json")