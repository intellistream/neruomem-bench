#!/usr/bin/env python3
"""
Simple Experiment 轮次分析器

基于配置文件驱动的分析脚本，用于黑盒第三方记忆体的评估结果分析。
与 benchmarks.evaluation.analysis.round_analyzer 结构一致，
但使用简化的视觉编码（无 Dimension×System×Strategy 编码）。

使用方法:
    # 分析所有适配器
    python -m benchmarks.simple_experiment.evaluation.analysis.round_analyzer \
        --config locomo --all

    # 分析指定适配器
    python -m benchmarks.simple_experiment.evaluation.analysis.round_analyzer \
        --config locomo --path mem0

    # 仅验证数据
    python -m benchmarks.simple_experiment.evaluation.analysis.round_analyzer \
        --config locomo --all --validate-only
"""

from __future__ import annotations

import argparse
import csv
import datetime
import sys
from pathlib import Path

import numpy as np
import yaml

from .utils.data_loader import (
    CategoryAnalyzer,
    DataLoader,
    RoundAnalyzer,
    TimeBreakdownAnalyzer,
)
from .utils.indicators import SimpleLoCoMoEvaluator  # noqa: F401 — 触发注册
from .utils.plotting import (
    plot_category_comparison,
    plot_comparison,
    plot_cost_effectiveness_comparison,
    plot_single_strategy,
)
from .utils.validators import (
    discover_adapter_dirs,
    print_validation_report,
    validate_experiment_dir,
)


# ── 路径工具 ─────────────────────────────────────────────────────────────────


def load_config(config_name: str) -> dict:
    """加载 config/ 目录下的 YAML 配置"""
    config_dir = Path(__file__).parent / "config"
    config_file = config_dir / f"{config_name}.yaml"
    if not config_file.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_file}")
    with open(config_file, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_project_root() -> Path:
    """项目根目录（从 analysis → evaluation → simple_experiment → benchmarks → root）"""
    return Path(__file__).parent.parent.parent.parent.parent


def resolve_path(path: str) -> Path:
    if Path(path).is_absolute():
        return Path(path)
    return get_project_root() / path


# ── 核心分析流程 ─────────────────────────────────────────────────────────────


def run_analysis(
    config: dict,
    strategies: list[str],
    base_dir: str | None = None,
    output_dir: str | None = None,
    validate_only: bool = False,
):
    """执行完整分析流程"""
    base_path = resolve_path(base_dir or config["paths"]["base_dir"])
    out_path = resolve_path(output_dir or config["paths"]["output_dir"])

    if not base_path.exists():
        print(f"错误: 数据目录不存在: {base_path}")
        sys.exit(1)

    out_path.mkdir(parents=True, exist_ok=True)

    # ── 验证 ──────────────────────────────────────────────────────────────
    print("=" * 70)
    print("数据验证")
    print("=" * 70)

    validation_results = []
    valid_strategies = []

    for strategy in strategies:
        result = validate_experiment_dir(base_path / strategy)
        validation_results.append(result)
        if result["valid"]:
            valid_strategies.append(strategy)

    print_validation_report(validation_results)

    if validate_only:
        return

    if not valid_strategies:
        print("错误: 没有有效的适配器目录")
        sys.exit(1)

    # ── 初始化分析器 ──────────────────────────────────────────────────────
    evaluator_name = config.get("evaluator", {}).get("name", "simple_locomo")
    loader = DataLoader(base_path)
    analyzer = RoundAnalyzer(evaluator_name)
    category_analyzer = CategoryAnalyzer(evaluator_name)
    time_analyzer = TimeBreakdownAnalyzer()

    print("=" * 70)
    print(f"开始分析 ({len(valid_strategies)} 个适配器)")
    print("=" * 70)

    tasks_filter = config.get("tasks")
    if tasks_filter:
        print(f"\n任务过滤: 仅评估 {len(tasks_filter)} 个任务: {tasks_filter}")
    else:
        print("\n任务过滤: 未指定，评估所有任务")

    # ── 逐适配器分析 ──────────────────────────────────────────────────────
    all_f1: dict[str, dict[int, float]] = {}
    all_insert: dict[str, dict[int, float]] = {}
    all_retrieval: dict[str, dict[int, float]] = {}
    all_category_f1: dict[str, dict] = {}
    all_insert_breakdown: dict[str, dict[str, float]] = {}
    all_retrieval_breakdown: dict[str, dict[str, float]] = {}

    for strategy in valid_strategies:
        print(f"\n[{strategy}]")

        f1_metrics = analyzer.aggregate_across_tasks(loader, strategy, "f1", tasks_filter)
        insert_metrics = analyzer.aggregate_across_tasks(
            loader, strategy, "insert_time", tasks_filter
        )
        retrieval_metrics = analyzer.aggregate_across_tasks(
            loader, strategy, "retrieval_time", tasks_filter
        )

        print(f"  F1 Scores: {f1_metrics}")
        print(f"  Insert Times (ms): {insert_metrics}")
        print(f"  Retrieval Times (ms): {retrieval_metrics}")

        all_f1[strategy] = f1_metrics
        all_insert[strategy] = insert_metrics
        all_retrieval[strategy] = retrieval_metrics

        category_f1 = category_analyzer.aggregate_across_tasks(loader, strategy, tasks_filter)
        all_category_f1[strategy] = category_f1
        print(f"  Category F1: {category_f1}")

        insert_breakdown = time_analyzer.aggregate_across_tasks(
            loader, strategy, "insert", tasks_filter
        )
        retrieval_breakdown = time_analyzer.aggregate_across_tasks(
            loader, strategy, "retrieval", tasks_filter
        )
        all_insert_breakdown[strategy] = insert_breakdown
        all_retrieval_breakdown[strategy] = retrieval_breakdown
        print(
            f"  Insert Breakdown: pre={insert_breakdown['pre']:.2f}, "
            f"memory={insert_breakdown['memory']:.2f}, post={insert_breakdown['post']:.2f}"
        )
        print(
            f"  Retrieval Breakdown: pre={retrieval_breakdown['pre']:.2f}, "
            f"memory={retrieval_breakdown['memory']:.2f}, post={retrieval_breakdown['post']:.2f}"
        )

        # 单适配器图表
        if config.get("output", {}).get("charts", {}).get("single_strategy", True):
            plot_single_strategy(
                strategy,
                f1_metrics,
                insert_metrics,
                retrieval_metrics,
                out_path / f"{strategy}_metrics.png",
            )
            print(f"  ✓ Saved: {strategy}_metrics.png")

    # ── 对比图表 ──────────────────────────────────────────────────────────
    print("\n[Comparison Charts]")

    if config.get("output", {}).get("charts", {}).get("comparison", True):
        plot_comparison(all_f1, "F1 Score", "F1 Score", out_path / "comparison_f1.png")
        print("  ✓ Saved: comparison_f1.png")

        plot_comparison(
            all_insert, "Insert Time", "Time (ms)", out_path / "comparison_insert_time.png"
        )
        print("  ✓ Saved: comparison_insert_time.png")

        plot_comparison(
            all_retrieval,
            "Retrieval Time",
            "Time (ms)",
            out_path / "comparison_retrieval_time.png",
        )
        print("  ✓ Saved: comparison_retrieval_time.png")

        if all_category_f1:
            category_labels = config.get("categories", {}).get("labels", None)
            plot_category_comparison(
                all_category_f1,
                out_path / "comparison_category_f1.png",
                title="F1 Score by Question Category",
                category_labels=category_labels,
            )
            print("  ✓ Saved: comparison_category_f1.png")

        # Cost-Effectiveness
        if config.get("output", {}).get("charts", {}).get("cost_effectiveness", True):
            strategies_data = {}
            for strat in valid_strategies:
                if strat in all_f1 and strat in all_insert:
                    strategies_data[strat] = (all_f1[strat], all_insert[strat])
            if strategies_data:
                plot_cost_effectiveness_comparison(
                    strategies_data,
                    "Simple Experiment",
                    out_path / "comparison_cost_effectiveness.png",
                    title="Cost-Effectiveness Comparison",
                )
                print("  ✓ Saved: comparison_cost_effectiveness.png")

    # ── CSV ────────────────────────────────────────────────────────────────
    if "csv" in config.get("output", {}).get("formats", []):
        generate_csv(all_f1, out_path / "f1_scores.csv", "F1")
        generate_csv(all_insert, out_path / "insert_times.csv", "Insert Time (ms)")
        generate_csv(all_retrieval, out_path / "retrieval_times.csv", "Retrieval Time (ms)")
        generate_category_csv(all_category_f1, out_path / "category_f1_scores.csv")
        generate_breakdown_csv(all_insert_breakdown, out_path / "insert_breakdown.csv")
        generate_breakdown_csv(all_retrieval_breakdown, out_path / "retrieval_breakdown.csv")
        print("\n✓ CSV files saved")

    # ── Markdown ──────────────────────────────────────────────────────────
    if "md" in config.get("output", {}).get("formats", []):
        generate_markdown(
            config,
            valid_strategies,
            all_f1,
            all_insert,
            all_retrieval,
            all_category_f1,
            all_insert_breakdown,
            all_retrieval_breakdown,
            out_path / "analysis_report.md",
        )
        print("✓ Markdown report saved")

    print(f"\n{'=' * 70}")
    print(f"分析完成! 结果保存至: {out_path}")
    print(f"{'=' * 70}")


# ── CSV 生成 ─────────────────────────────────────────────────────────────────


def generate_csv(
    metrics: dict[str, dict[int, float]],
    output_path: Path,
    metric_name: str,
):
    if not metrics:
        return
    all_rounds = sorted(set().union(*[set(m.keys()) for m in metrics.values()]))
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["Adapter"] + [f"R{r}" for r in all_rounds] + ["Mean"]
        writer.writerow(header)
        for adapter, round_metrics in metrics.items():
            values = [round_metrics.get(r, 0) for r in all_rounds]
            mean_val = float(np.mean(values)) if values else 0
            row = (
                [adapter]
                + [f"{v:.6f}" if v < 1 else f"{v:.2f}" for v in values]
                + [f"{mean_val:.6f}" if mean_val < 1 else f"{mean_val:.2f}"]
            )
            writer.writerow(row)


def generate_category_csv(
    category_metrics: dict[str, dict],
    output_path: Path,
):
    if not category_metrics:
        return
    all_categories = sorted(
        set().union(*[set(m.keys()) for m in category_metrics.values()]), key=str
    )
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["Adapter"] + [str(c) for c in all_categories] + ["Mean"]
        writer.writerow(header)
        for adapter, cat_metrics in category_metrics.items():
            values = [cat_metrics.get(c, 0) for c in all_categories]
            mean_val = float(np.mean(values)) if values else 0
            row = [adapter] + [f"{v:.6f}" for v in values] + [f"{mean_val:.6f}"]
            writer.writerow(row)


def generate_breakdown_csv(
    breakdown_metrics: dict[str, dict[str, float]],
    output_path: Path,
):
    if not breakdown_metrics:
        return
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["Adapter", "Pre (ms)", "Memory (ms)", "Post (ms)", "Total (ms)"]
        writer.writerow(header)
        for adapter, breakdown in breakdown_metrics.items():
            row = [
                adapter,
                f"{breakdown.get('pre', 0):.2f}",
                f"{breakdown.get('memory', 0):.2f}",
                f"{breakdown.get('post', 0):.2f}",
                f"{breakdown.get('total', 0):.2f}",
            ]
            writer.writerow(row)


# ── Markdown 报告 ────────────────────────────────────────────────────────────


def generate_markdown(
    config: dict,
    adapters: list[str],
    f1_metrics: dict,
    insert_metrics: dict,
    retrieval_metrics: dict,
    category_f1_metrics: dict,
    insert_breakdown_metrics: dict,
    retrieval_breakdown_metrics: dict,
    output_path: Path,
):
    dataset_name = config.get("dataset", {}).get("name", "Unknown")
    all_rounds = sorted(set().union(*[set(m.keys()) for m in f1_metrics.values()]))
    all_categories = (
        sorted(set().union(*[set(m.keys()) for m in category_f1_metrics.values()]), key=str)
        if category_f1_metrics
        else []
    )

    md = f"# {dataset_name.upper()} Simple Experiment — Round Analysis Report\n\n"
    md += (
        f"**Generated:** "
        f"{datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )

    md += "## Summary\n\n"
    md += f"- **Dataset:** {dataset_name}\n"
    md += f"- **Adapters:** {len(adapters)}\n"
    md += f"- **Rounds:** {len(all_rounds)}\n"
    md += f"- **Categories:** {len(all_categories)}\n\n"

    # F1 表
    md += "## F1 Scores by Round\n\n"
    md += "| Adapter | " + " | ".join([f"R{r}" for r in all_rounds]) + " | Mean |\n"
    md += "|---------|" + "|".join(["------" for _ in all_rounds]) + "|------|\n"
    for adapter in adapters:
        vals = [f1_metrics.get(adapter, {}).get(r, 0) for r in all_rounds]
        mean_val = float(np.mean(vals)) if vals else 0
        md += (
            f"| {adapter} | "
            + " | ".join([f"{v:.4f}" for v in vals])
            + f" | {mean_val:.4f} |\n"
        )
    md += "\n"

    # Category F1 表
    if category_f1_metrics and all_categories:
        category_labels = config.get("categories", {}).get("labels", {})
        cat_headers = [category_labels.get(c, str(c)) for c in all_categories]
        md += "## F1 Scores by Category\n\n"
        md += "| Adapter | " + " | ".join(cat_headers) + " | Mean |\n"
        md += "|---------|" + "|".join(["------" for _ in all_categories]) + "|------|\n"
        for adapter in adapters:
            vals = [category_f1_metrics.get(adapter, {}).get(c, 0) for c in all_categories]
            mean_val = float(np.mean(vals)) if vals else 0
            md += (
                f"| {adapter} | "
                + " | ".join([f"{v:.4f}" for v in vals])
                + f" | {mean_val:.4f} |\n"
            )
        md += "\n"

    # Insert 时间表
    md += "## Insert Time (ms) by Round\n\n"
    md += "| Adapter | " + " | ".join([f"R{r}" for r in all_rounds]) + " | Mean |\n"
    md += "|---------|" + "|".join(["------" for _ in all_rounds]) + "|------|\n"
    for adapter in adapters:
        vals = [insert_metrics.get(adapter, {}).get(r, 0) for r in all_rounds]
        mean_val = float(np.mean(vals)) if vals else 0
        md += (
            f"| {adapter} | "
            + " | ".join([f"{v:.2f}" for v in vals])
            + f" | {mean_val:.2f} |\n"
        )
    md += "\n"

    # Insert 分解表
    if insert_breakdown_metrics:
        md += "## Insert Time Breakdown (ms)\n\n"
        md += "| Adapter | Pre (ms) | Memory (ms) | Post (ms) | Total (ms) |\n"
        md += "|---------|----------|-------------|-----------|------------|\n"
        for adapter in adapters:
            b = insert_breakdown_metrics.get(adapter, {})
            md += (
                f"| {adapter} | {b.get('pre', 0):.2f} | {b.get('memory', 0):.2f} "
                f"| {b.get('post', 0):.2f} | {b.get('total', 0):.2f} |\n"
            )
        md += "\n"

    # Retrieval 时间表
    md += "## Retrieval Time (ms) by Round\n\n"
    md += "| Adapter | " + " | ".join([f"R{r}" for r in all_rounds]) + " | Mean |\n"
    md += "|---------|" + "|".join(["------" for _ in all_rounds]) + "|------|\n"
    for adapter in adapters:
        vals = [retrieval_metrics.get(adapter, {}).get(r, 0) for r in all_rounds]
        mean_val = float(np.mean(vals)) if vals else 0
        md += (
            f"| {adapter} | "
            + " | ".join([f"{v:.2f}" for v in vals])
            + f" | {mean_val:.2f} |\n"
        )
    md += "\n"

    # Retrieval 分解表
    if retrieval_breakdown_metrics:
        md += "## Retrieval Time Breakdown (ms)\n\n"
        md += "| Adapter | Pre (ms) | Memory (ms) | Post (ms) | Total (ms) |\n"
        md += "|---------|----------|-------------|-----------|------------|\n"
        for adapter in adapters:
            b = retrieval_breakdown_metrics.get(adapter, {})
            md += (
                f"| {adapter} | {b.get('pre', 0):.2f} | {b.get('memory', 0):.2f} "
                f"| {b.get('post', 0):.2f} | {b.get('total', 0):.2f} |\n"
            )
        md += "\n"

    # Performance Insights
    if f1_metrics:
        md += "## Performance Insights\n\n"
        best_f1_adapter = max(
            adapters,
            key=lambda a: float(
                np.mean(list(f1_metrics.get(a, {}).values())) if f1_metrics.get(a) else 0
            ),
        )
        best_f1 = float(np.mean(list(f1_metrics.get(best_f1_adapter, {}).values())))
        md += f"- **Best F1 Score**: {best_f1_adapter} ({best_f1:.4f})\n"

        if insert_metrics:
            fastest = min(
                adapters,
                key=lambda a: float(
                    np.mean(list(insert_metrics.get(a, {}).values()))
                    if insert_metrics.get(a)
                    else float("inf")
                ),
            )
            t = float(np.mean(list(insert_metrics.get(fastest, {}).values())))
            md += f"- **Fastest Insert**: {fastest} ({t:.2f} ms)\n"

        if retrieval_metrics:
            fastest = min(
                adapters,
                key=lambda a: float(
                    np.mean(list(retrieval_metrics.get(a, {}).values()))
                    if retrieval_metrics.get(a)
                    else float("inf")
                ),
            )
            t = float(np.mean(list(retrieval_metrics.get(fastest, {}).values())))
            md += f"- **Fastest Retrieval**: {fastest} ({t:.2f} ms)\n"

        md += "\n"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)


# ── CLI 入口 ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Simple Experiment 轮次分析器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 分析所有适配器
  python -m benchmarks.simple_experiment.evaluation.analysis.round_analyzer \\
      --config locomo --all

  # 分析指定适配器
  python -m benchmarks.simple_experiment.evaluation.analysis.round_analyzer \\
      --config locomo --path mem0

  # 仅验证数据
  python -m benchmarks.simple_experiment.evaluation.analysis.round_analyzer \\
      --config locomo --all --validate-only
        """,
    )

    parser.add_argument("--config", required=True, help="配置文件名（不含 .yaml 后缀）")
    parser.add_argument("--path", nargs="+", help="要分析的适配器目录名称")
    parser.add_argument("--all", action="store_true", help="分析所有适配器目录")
    parser.add_argument("--base-dir", help="数据基础目录（覆盖配置）")
    parser.add_argument("--output-dir", help="输出目录（覆盖配置）")
    parser.add_argument("--validate-only", action="store_true", help="仅验证数据")

    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        sys.exit(1)

    base_dir = resolve_path(args.base_dir or config["paths"]["base_dir"])

    if args.all:
        strategies = discover_adapter_dirs(base_dir)
        if not strategies:
            print(f"错误: {base_dir} 下未找到适配器目录")
            sys.exit(1)
    elif args.path:
        strategies = args.path
    else:
        print("错误: 请指定 --path 或 --all")
        parser.print_help()
        sys.exit(1)

    run_analysis(
        config,
        strategies,
        base_dir=str(base_dir),
        output_dir=args.output_dir,
        validate_only=args.validate_only,
    )


if __name__ == "__main__":
    main()
