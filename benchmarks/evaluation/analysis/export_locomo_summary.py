"""Export submission-oriented LoCoMo summaries from benchmark artifacts.

This script converts raw benchmark JSON files into compact CSV/JSON/Markdown
summaries suitable for paper writing. It intentionally avoids plotting
dependencies so it can run in lightweight environments.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from .utils.indicators import LoCoMoEvaluator, exact_match
except ImportError:
    from utils.indicators import LoCoMoEvaluator, exact_match


def _safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0

def _category_sort_key(category: str) -> tuple[int, str | int]:
    try:
        return (0, int(category))
    except (TypeError, ValueError):
        return (1, category)


def _infer_dataset_question_count(data: dict[str, Any], dataset_stats: dict[str, Any]) -> int:
    dataset_questions = dataset_stats.get(
        "total_questions", dataset_stats.get("questions", 0)
    )
    if dataset_questions:
        return int(dataset_questions)

    unique_question_keys: set[tuple[Any, ...]] = set()
    for test in data.get("test_results", []):
        for question in test.get("questions", []):
            question_index = question.get("question_index")
            if question_index not in (None, ""):
                unique_question_keys.add(("question_index", int(question_index)))
                continue

            question_id = question.get("question_id")
            if question_id not in (None, ""):
                unique_question_keys.add(("question_id", str(question_id)))
                continue

            question_text = str(question.get("question_text", "")).strip()
            reference_answer = str(question.get("reference_answer", "")).strip()
            if question_text or reference_answer:
                unique_question_keys.add(("question_text", question_text, reference_answer))

    return len(unique_question_keys)


def _infer_evaluation_mode(dataset_questions: int, evaluated_questions: int) -> str:
    if dataset_questions <= 0:
        return "scored_trace_only"
    if evaluated_questions == dataset_questions:
        return "single_pass"
    if evaluated_questions > dataset_questions:
        return "multi_pass_trace"
    return "partial_trace"


def _format_ratio(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}x"


def _question_key(question: dict[str, Any]) -> tuple[Any, ...] | None:
    question_index = question.get("question_index")
    if question_index not in (None, ""):
        return ("question_index", int(question_index))

    question_id = question.get("question_id")
    if question_id not in (None, ""):
        return ("question_id", str(question_id))

    question_text = str(question.get("question_text", "")).strip()
    reference_answer = str(question.get("reference_answer", "")).strip()
    if question_text or reference_answer:
        return ("question_text", question_text, reference_answer)

    return None


def _summarize_question_set(
    questions: list[dict[str, Any]],
    dataset_questions: int,
) -> dict[str, Any]:
    evaluator = LoCoMoEvaluator()
    question_scores: list[float] = []
    em_scores: list[float] = []
    abstain_count = 0
    category_scores: dict[str, list[float]] = defaultdict(list)
    category_counts: dict[str, int] = defaultdict(int)

    for question in questions:
        prediction = str(question.get("predicted_answer", ""))
        reference = str(question.get("reference_answer", ""))
        raw_category = question.get("category", 0)
        if isinstance(raw_category, (int, float)):
            category = str(int(raw_category))
        else:
            category = str(raw_category or 0)
        score = evaluator.evaluate_single(prediction, reference, **question)
        question_scores.append(score)
        em_scores.append(exact_match(prediction, reference))
        category_scores[category].append(score)
        category_counts[category] += 1

        if prediction.strip().lower() == "not mentioned in the conversation.":
            abstain_count += 1

    evaluated_questions = len(questions)
    evaluation_mode = _infer_evaluation_mode(dataset_questions, evaluated_questions)
    response_multiplicity = (
        evaluated_questions / dataset_questions if dataset_questions > 0 else None
    )

    return {
        "evaluated_questions": evaluated_questions,
        "evaluation_mode": evaluation_mode,
        "response_multiplicity": response_multiplicity,
        "avg_locomo_f1": _safe_mean(question_scores),
        "avg_exact_match": _safe_mean(em_scores),
        "abstain_count": abstain_count,
        "abstain_rate": (abstain_count / evaluated_questions) if evaluated_questions else 0.0,
        "category_scores": {category: _safe_mean(scores) for category, scores in sorted(category_scores.items())},
        "category_counts": {category: count for category, count in sorted(category_counts.items())},
    }


def _single_pass_latest_view(data: dict[str, Any], dataset_questions: int) -> dict[str, Any]:
    latest_by_question: dict[tuple[Any, ...], dict[str, Any]] = {}
    for test in data.get("test_results", []):
        for question in test.get("questions", []):
            key = _question_key(question)
            if key is None:
                continue
            latest_by_question[key] = question

    summary = _summarize_question_set(list(latest_by_question.values()), dataset_questions)
    summary["evaluation_mode"] = "single_pass_latest"
    return summary


def _latest_task_files(strategy_dir: Path, tasks: list[str] | None = None) -> list[Path]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for json_file in sorted(strategy_dir.glob("*.json")):
        task_id = json_file.stem.rsplit("_", 1)[0]
        if tasks is not None and task_id not in tasks:
            continue
        grouped[task_id].append(json_file)
    return [
        max(paths, key=lambda path: (path.stat().st_mtime, path.name))
        for _, paths in sorted(grouped.items())
    ]


def summarize_task_file(json_path: Path) -> dict[str, Any]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    dataset_stats = data.get("dataset_statistics", {})
    dataset_questions = _infer_dataset_question_count(data, dataset_stats)
    all_questions = [
        question
        for test in data.get("test_results", [])
        for question in test.get("questions", [])
    ]
    trace_summary = _summarize_question_set(all_questions, dataset_questions)
    single_pass_latest = _single_pass_latest_view(data, dataset_questions)

    timing_summary = data.get("timing_summary", {})
    insert_summary = timing_summary.get("insert_timings", {}).get("summary", {})
    retrieval_summary = timing_summary.get("retrieval_timings", {}).get("summary", {})
    final_snapshot = (data.get("memory_snapshots") or [{}])[-1]
    neural_memory = final_snapshot.get("neural_memory", {})
    storage = final_snapshot.get("storage", {})

    return {
        "strategy": json_path.parent.name,
        "task_id": data.get("experiment_info", {}).get("task_id", json_path.stem),
        "dataset": data.get("experiment_info", {}).get("dataset", "locomo"),
        "source_file": str(json_path),
        "sessions": dataset_stats.get("total_sessions", dataset_stats.get("sessions", 0)),
        "messages": dataset_stats.get("total_messages", dataset_stats.get("messages", 0)),
        "dataset_questions": dataset_questions,
        "evaluated_questions": trace_summary["evaluated_questions"],
        "evaluation_mode": trace_summary["evaluation_mode"],
        "response_multiplicity": trace_summary["response_multiplicity"],
        "avg_locomo_f1": trace_summary["avg_locomo_f1"],
        "avg_exact_match": trace_summary["avg_exact_match"],
        "abstain_count": trace_summary["abstain_count"],
        "abstain_rate": trace_summary["abstain_rate"],
        "category_scores": trace_summary["category_scores"],
        "category_counts": trace_summary["category_counts"],
        "single_pass_latest": single_pass_latest,
        "timings_ms": {
            "pre_insert": insert_summary.get("pre_insert_ms", {}).get("avg_ms", 0.0),
            "memory_insert": insert_summary.get("memory_insert_ms", {}).get("avg_ms", 0.0),
            "post_insert": insert_summary.get("post_insert_ms", {}).get("avg_ms", 0.0),
            "pre_retrieval": retrieval_summary.get("pre_retrieval_ms", {}).get("avg_ms", 0.0),
            "memory_retrieval": retrieval_summary.get("memory_retrieval_ms", {}).get("avg_ms", 0.0),
            "post_retrieval": retrieval_summary.get("post_retrieval_ms", {}).get("avg_ms", 0.0),
        },
        "final_snapshot": {
            "training_steps": neural_memory.get("training_steps", 0),
            "samples_seen": neural_memory.get("samples_seen", 0),
            "active_buffer_items": neural_memory.get("active_buffer_items", 0),
            "stored_entries": storage.get("total_entries", 0),
            "parameter_count": storage.get("parameter_count", 0),
        },
    }


def summarize_strategy(strategy_dir: Path, tasks: list[str] | None = None) -> dict[str, Any]:
    task_files = _latest_task_files(strategy_dir, tasks)
    task_summaries = [summarize_task_file(path) for path in task_files]

    all_categories = sorted(
        {category for summary in task_summaries for category in summary["category_scores"].keys()},
        key=_category_sort_key,
    )
    weighted_category_scores: dict[str, float] = {}
    for category in all_categories:
        total_weight = sum(summary["category_counts"].get(category, 0) for summary in task_summaries)
        if total_weight == 0:
            weighted_category_scores[category] = 0.0
            continue
        weighted_sum = sum(
            summary["category_scores"].get(category, 0.0)
            * summary["category_counts"].get(category, 0)
            for summary in task_summaries
        )
        weighted_category_scores[category] = weighted_sum / total_weight

    total_eval_questions = sum(summary["evaluated_questions"] for summary in task_summaries)
    total_dataset_questions = sum(summary["dataset_questions"] for summary in task_summaries)
    total_abstains = sum(summary["abstain_count"] for summary in task_summaries)
    evaluation_modes = {summary["evaluation_mode"] for summary in task_summaries}
    total_single_pass_questions = sum(
        summary["single_pass_latest"]["evaluated_questions"] for summary in task_summaries
    )
    total_single_pass_abstains = sum(
        summary["single_pass_latest"]["abstain_count"] for summary in task_summaries
    )

    single_pass_categories = sorted(
        {
            category
            for summary in task_summaries
            for category in summary["single_pass_latest"]["category_scores"].keys()
        },
        key=_category_sort_key,
    )
    weighted_single_pass_scores: dict[str, float] = {}
    for category in single_pass_categories:
        total_weight = sum(
            summary["single_pass_latest"]["category_counts"].get(category, 0)
            for summary in task_summaries
        )
        if total_weight == 0:
            weighted_single_pass_scores[category] = 0.0
            continue
        weighted_sum = sum(
            summary["single_pass_latest"]["category_scores"].get(category, 0.0)
            * summary["single_pass_latest"]["category_counts"].get(category, 0)
            for summary in task_summaries
        )
        weighted_single_pass_scores[category] = weighted_sum / total_weight

    aggregate = {
        "strategy": strategy_dir.name,
        "task_count": len(task_summaries),
        "tasks": task_summaries,
        "aggregate": {
            "avg_locomo_f1": _safe_mean([summary["avg_locomo_f1"] for summary in task_summaries]),
            "avg_exact_match": _safe_mean([summary["avg_exact_match"] for summary in task_summaries]),
            "avg_pre_insert_ms": _safe_mean(
                [summary["timings_ms"]["pre_insert"] for summary in task_summaries]
            ),
            "avg_memory_insert_ms": _safe_mean(
                [summary["timings_ms"]["memory_insert"] for summary in task_summaries]
            ),
            "avg_pre_retrieval_ms": _safe_mean(
                [summary["timings_ms"]["pre_retrieval"] for summary in task_summaries]
            ),
            "avg_memory_retrieval_ms": _safe_mean(
                [summary["timings_ms"]["memory_retrieval"] for summary in task_summaries]
            ),
            "total_dataset_questions": total_dataset_questions,
            "total_evaluated_questions": total_eval_questions,
            "evaluation_mode": (
                next(iter(evaluation_modes)) if len(evaluation_modes) == 1 else "mixed"
            ),
            "response_multiplicity": (
                total_eval_questions / total_dataset_questions
                if total_dataset_questions
                else None
            ),
            "total_abstain_count": total_abstains,
            "overall_abstain_rate": (total_abstains / total_eval_questions)
            if total_eval_questions
            else 0.0,
            "weighted_category_scores": weighted_category_scores,
            "single_pass_latest": {
                "avg_locomo_f1": _safe_mean(
                    [summary["single_pass_latest"]["avg_locomo_f1"] for summary in task_summaries]
                ),
                "avg_exact_match": _safe_mean(
                    [summary["single_pass_latest"]["avg_exact_match"] for summary in task_summaries]
                ),
                "total_dataset_questions": total_dataset_questions,
                "total_evaluated_questions": total_single_pass_questions,
                "evaluation_mode": "single_pass_latest",
                "response_multiplicity": (
                    total_single_pass_questions / total_dataset_questions
                    if total_dataset_questions
                    else None
                ),
                "total_abstain_count": total_single_pass_abstains,
                "overall_abstain_rate": (
                    total_single_pass_abstains / total_single_pass_questions
                    if total_single_pass_questions
                    else 0.0
                ),
                "weighted_category_scores": weighted_single_pass_scores,
            },
            "zeroed_final_snapshots": sum(
                1
                for summary in task_summaries
                if summary["final_snapshot"]["training_steps"] == 0
                and summary["final_snapshot"]["samples_seen"] == 0
                and summary["final_snapshot"]["active_buffer_items"] == 0
                and summary["final_snapshot"]["stored_entries"] == 0
            ),
        },
    }
    return aggregate


def _write_task_csv(summary: dict[str, Any], output_dir: Path) -> None:
    task_csv = output_dir / "task_summary.csv"
    with task_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "task_id",
                "sessions",
                "messages",
                "dataset_questions",
                "evaluated_questions",
                "evaluation_mode",
                "response_multiplicity",
                "avg_locomo_f1",
                "avg_exact_match",
                "abstain_count",
                "abstain_rate",
                    "single_pass_latest_questions",
                    "single_pass_latest_f1",
                    "single_pass_latest_exact_match",
                    "single_pass_latest_abstain_count",
                    "single_pass_latest_abstain_rate",
                "pre_insert_ms",
                "memory_insert_ms",
                "pre_retrieval_ms",
                "memory_retrieval_ms",
            ]
        )
        for task in summary["tasks"]:
            writer.writerow(
                [
                    task["task_id"],
                    task["sessions"],
                    task["messages"],
                    task["dataset_questions"],
                    task["evaluated_questions"],
                    task["evaluation_mode"],
                    f"{task['response_multiplicity']:.6f}"
                    if task["response_multiplicity"] is not None
                    else "",
                    f"{task['avg_locomo_f1']:.6f}",
                    f"{task['avg_exact_match']:.6f}",
                    task["abstain_count"],
                    f"{task['abstain_rate']:.6f}",
                    task["single_pass_latest"]["evaluated_questions"],
                    f"{task['single_pass_latest']['avg_locomo_f1']:.6f}",
                    f"{task['single_pass_latest']['avg_exact_match']:.6f}",
                    task["single_pass_latest"]["abstain_count"],
                    f"{task['single_pass_latest']['abstain_rate']:.6f}",
                    f"{task['timings_ms']['pre_insert']:.6f}",
                    f"{task['timings_ms']['memory_insert']:.6f}",
                    f"{task['timings_ms']['pre_retrieval']:.6f}",
                    f"{task['timings_ms']['memory_retrieval']:.6f}",
                ]
            )


def _write_markdown(summary: dict[str, Any], output_dir: Path) -> None:
    aggregate = summary["aggregate"]
    lines = [
        f"# LoCoMo Submission Summary: {summary['strategy']}",
        "",
        f"Generated: {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        "## Aggregate",
        "",
        f"- Tasks: {summary['task_count']}",
        f"- Evaluation protocol: {aggregate['evaluation_mode']}",
        f"- Total unique dataset questions: {aggregate['total_dataset_questions']}",
        f"- Total scored responses: {aggregate['total_evaluated_questions']}",
        f"- Response multiplicity: {_format_ratio(aggregate['response_multiplicity'])}",
        f"- Avg LoCoMo F1: {aggregate['avg_locomo_f1']:.4f}",
        f"- Avg Exact Match: {aggregate['avg_exact_match']:.4f}",
        f"- Abstain rate: {aggregate['overall_abstain_rate']:.2%}",
        f"- Single-pass-latest F1: {aggregate['single_pass_latest']['avg_locomo_f1']:.4f}",
        f"- Single-pass-latest EM: {aggregate['single_pass_latest']['avg_exact_match']:.4f}",
        f"- Single-pass-latest abstain rate: {aggregate['single_pass_latest']['overall_abstain_rate']:.2%}",
        f"- Avg pre-insert latency: {aggregate['avg_pre_insert_ms']:.2f} ms",
        f"- Avg memory-insert latency: {aggregate['avg_memory_insert_ms']:.2f} ms",
        f"- Avg pre-retrieval latency: {aggregate['avg_pre_retrieval_ms']:.2f} ms",
        f"- Avg memory-retrieval latency: {aggregate['avg_memory_retrieval_ms']:.2f} ms",
        f"- Zeroed final snapshots: {aggregate['zeroed_final_snapshots']}/{summary['task_count']}",
        "",
        "## Weighted Category F1",
        "",
        "| Category | F1 |",
        "|---|---:|",
    ]
    for category, score in aggregate["weighted_category_scores"].items():
        lines.append(f"| {category} | {score:.4f} |")

    lines.extend(
        [
            "",
            "## Task Summary",
            "",
            "| Task | Unique Q | Scored Resp. | Protocol | Trace F1 | SP-latest F1 | Trace abstain | SP-latest abstain | PreIns | Insert | PreRet | Ret |",
            "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for task in summary["tasks"]:
        lines.append(
            "| {task_id} | {dataset_questions} | {evaluated_questions} | {evaluation_mode} | {avg_locomo_f1:.4f} | {single_pass_f1:.4f} | "
            "{abstain_rate:.2%} | {single_pass_abstain:.2%} | {pre_insert:.2f} | {memory_insert:.2f} | {pre_retrieval:.2f} | {memory_retrieval:.2f} |".format(
                task_id=task["task_id"],
                dataset_questions=task["dataset_questions"],
                evaluated_questions=task["evaluated_questions"],
                evaluation_mode=task["evaluation_mode"],
                avg_locomo_f1=task["avg_locomo_f1"],
                single_pass_f1=task["single_pass_latest"]["avg_locomo_f1"],
                abstain_rate=task["abstain_rate"],
                single_pass_abstain=task["single_pass_latest"]["abstain_rate"],
                pre_insert=task["timings_ms"]["pre_insert"],
                memory_insert=task["timings_ms"]["memory_insert"],
                pre_retrieval=task["timings_ms"]["pre_retrieval"],
                memory_retrieval=task["timings_ms"]["memory_retrieval"],
            )
        )

    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_strategy_summary(strategy_dir: Path, output_dir: Path, tasks: list[str] | None = None) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_strategy(strategy_dir, tasks)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_task_csv(summary, output_dir)
    _write_markdown(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Export compact LoCoMo paper summaries")
    parser.add_argument("--strategy-dir", required=True, help="Directory containing LoCoMo result JSONs")
    parser.add_argument("--output-dir", required=True, help="Directory to write JSON/CSV/Markdown summaries")
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Optional task IDs to include, e.g. conv-26 conv-30 conv-41",
    )
    args = parser.parse_args()

    summary = export_strategy_summary(
        Path(args.strategy_dir),
        Path(args.output_dir),
        tasks=args.tasks,
    )
    print(
        "Exported LoCoMo summary for "
        f"{summary['strategy']} with {summary['task_count']} task(s) to {args.output_dir}"
    )


if __name__ == "__main__":
    main()