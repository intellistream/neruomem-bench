#!/usr/bin/env python3
"""
通用轮次分析器

基于配置文件驱动的分析脚本，支持多种数据集

使用方法:
    # 分析指定目录
    python round_analyzer.py --config locomo --path PostInsert_MemoryOS_forgetting_curve

    # 分析所有目录
    python round_analyzer.py --config locomo --all

    # 指定前缀
    python round_analyzer.py --config locomo --prefix PostInsert_
"""

from __future__ import annotations

import argparse
import csv
import datetime
import sys
from pathlib import Path

import numpy as np
import yaml

# 支持直接运行和作为模块导入
try:
    from .utils.data_loader import (
        CategoryAnalyzer,
        DataLoader,
        RoundAnalyzer,
        TimeBreakdownAnalyzer,
    )
    from .utils.plotting import (
        plot_category_comparison,
        plot_comparison,
        plot_cost_effectiveness_comparison,
        plot_single_strategy,
    )
    from .utils.validators import (
        discover_experiment_dirs,
        print_validation_report,
        validate_experiment_dir,
    )
except ImportError:
    from utils.data_loader import (
        CategoryAnalyzer,
        DataLoader,
        RoundAnalyzer,
        TimeBreakdownAnalyzer,
    )
    from utils.plotting import (
        plot_category_comparison,
        plot_comparison,
        plot_cost_effectiveness_comparison,
        plot_single_strategy,
    )
    from utils.validators import (
        discover_experiment_dirs,
        print_validation_report,
        validate_experiment_dir,
    )


def load_config(config_name: str) -> dict:
    """加载配置文件"""
    config_dir = Path(__file__).parent / "config"
    config_file = config_dir / f"{config_name}.yaml"

    if not config_file.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_file}")

    with open(config_file, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_project_root() -> Path:
    """获取项目根目录（从analysis目录向上3级）"""
    # analysis -> evaluation -> benchmarks -> project_root
    return Path(__file__).parent.parent.parent.parent


def resolve_path(path: str) -> Path:
    """解析相对路径为绝对路径"""
    if Path(path).is_absolute():
        return Path(path)
    return get_project_root() / path


def run_analysis(
    config: dict,
    strategies: list[str],
    base_dir: str | None = None,
    output_dir: str | None = None,
    validate_only: bool = False,
):
    """
    执行分析

    Args:
        config: 配置字典
        strategies: 要分析的策略列表
        base_dir: 数据目录（覆盖配置）
        output_dir: 输出目录（覆盖配置）
        validate_only: 仅验证不分析
    """
    # 解析路径（支持相对于项目根目录的路径）
    base_path = resolve_path(base_dir or config["paths"]["base_dir"])
    out_path = resolve_path(output_dir or config["paths"]["output_dir"])

    if not base_path.exists():
        print(f"错误: 数据目录不存在: {base_path}")
        sys.exit(1)

    out_path.mkdir(parents=True, exist_ok=True)

    # 验证数据
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
        print("错误: 没有有效的策略目录")
        sys.exit(1)

    # 初始化分析器
    evaluator_name = config.get("evaluator", {}).get("name", "generic_f1")
    loader = DataLoader(base_path)
    analyzer = RoundAnalyzer(evaluator_name)
    category_analyzer = CategoryAnalyzer(evaluator_name)
    time_analyzer = TimeBreakdownAnalyzer()

    print("=" * 70)
    print(f"开始分析 ({len(valid_strategies)} 个策略)")
    print("=" * 70)

    # 获取任务过滤列表（如果配置中指定）
    tasks_filter = config.get("tasks")
    if tasks_filter:
        print(f"\n任务过滤: 仅评估 {len(tasks_filter)} 个任务: {tasks_filter}")
    else:
        print("\n任务过滤: 未指定，评估所有任务")

    # 存储所有指标
    all_f1 = {}
    all_insert = {}
    all_retrieval = {}
    all_category_f1 = {}  # Category F1
    all_insert_breakdown = {}  # Insert时间分解
    all_retrieval_breakdown = {}  # Retrieval时间分解

    for strategy in valid_strategies:
        print(f"\n[{strategy}]")

        # 聚合指标（传递任务过滤器）
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

        # Category F1分析
        category_f1 = category_analyzer.aggregate_across_tasks(loader, strategy, tasks_filter)
        all_category_f1[strategy] = category_f1
        print(f"  Category F1: {category_f1}")

        # 时间分解分析
        insert_breakdown = time_analyzer.aggregate_across_tasks(
            loader, strategy, "insert", tasks_filter
        )
        retrieval_breakdown = time_analyzer.aggregate_across_tasks(
            loader, strategy, "retrieval", tasks_filter
        )
        all_insert_breakdown[strategy] = insert_breakdown
        all_retrieval_breakdown[strategy] = retrieval_breakdown
        print(
            f"  Insert Breakdown: pre={insert_breakdown['pre']:.2f}, memory={insert_breakdown['memory']:.2f}, post={insert_breakdown['post']:.2f}"
        )
        print(
            f"  Retrieval Breakdown: pre={retrieval_breakdown['pre']:.2f}, memory={retrieval_breakdown['memory']:.2f}, post={retrieval_breakdown['post']:.2f}"
        )

        # 绘制单策略图
        if config.get("output", {}).get("charts", {}).get("single_strategy", True):
            plot_single_strategy(
                strategy,
                f1_metrics,
                insert_metrics,
                retrieval_metrics,
                out_path / f"{strategy}_metrics.png",
            )
            print(f"  ✓ Saved: {strategy}_metrics.png")

    # 绘制对比图
    print("\n[Comparison Charts]")

    if config.get("output", {}).get("charts", {}).get("comparison", True):
        plot_comparison(all_f1, "F1 Score", "F1 Score", out_path / "comparison_f1.png")
        print("  ✓ Saved: comparison_f1.png")

        plot_comparison(
            all_insert, "Insert Time", "Time (ms)", out_path / "comparison_insert_time.png"
        )
        print("  ✓ Saved: comparison_insert_time.png")

        plot_comparison(
            all_retrieval, "Retrieval Time", "Time (ms)", out_path / "comparison_retrieval_time.png"
        )
        print("  ✓ Saved: comparison_retrieval_time.png")

        # Category F1 对比图
        if all_category_f1:
            category_labels = config.get("categories", {}).get("labels", None)
            plot_category_comparison(
                all_category_f1,
                out_path / "comparison_category_f1.png",
                title="F1 Score by Question Category",
                category_labels=category_labels,
            )
            print("  ✓ Saved: comparison_category_f1.png")

        # Cost-Effectiveness 对比图
        if config.get("output", {}).get("charts", {}).get("cost_effectiveness", True):
            # 准备数据：{config_name: (f1_by_round, time_by_round)}
            # 根据维度使用对应阶段的时间
            strategies_data = {}

            # 从第一个策略中提取维度信息（PreInsert/PostInsert/PreRetrieval/PostRetrieval）
            dimension_name = None
            if valid_strategies:
                first_strategy = valid_strategies[0]
                for dim in ["PreInsert", "PostInsert", "PreRetrieval", "PostRetrieval"]:
                    if dim in first_strategy:
                        dimension_name = dim
                        break

            for strategy in strategies:
                if strategy not in all_f1:
                    continue

                # 根据维度选择对应阶段的时间
                time_by_round = {}
                if dimension_name == "PostInsert" and strategy in all_insert_breakdown:
                    # PostInsert: 使用post_insert时间
                    breakdown = all_insert_breakdown[strategy]
                    # breakdown是平均值，需要为每个round复制（假设时间相对稳定）
                    for round_num in all_f1[strategy]:
                        time_by_round[round_num] = breakdown.get("post", 0)
                elif dimension_name == "PreInsert" and strategy in all_insert_breakdown:
                    # PreInsert: 使用pre_insert时间
                    breakdown = all_insert_breakdown[strategy]
                    for round_num in all_f1[strategy]:
                        time_by_round[round_num] = breakdown.get("pre", 0)
                elif dimension_name == "PostRetrieval" and strategy in all_retrieval_breakdown:
                    # PostRetrieval: 使用post_retrieval时间
                    breakdown = all_retrieval_breakdown[strategy]
                    for round_num in all_f1[strategy]:
                        time_by_round[round_num] = breakdown.get("post", 0)
                elif dimension_name == "PreRetrieval" and strategy in all_retrieval_breakdown:
                    # PreRetrieval: 使用pre_retrieval时间
                    breakdown = all_retrieval_breakdown[strategy]
                    for round_num in all_f1[strategy]:
                        time_by_round[round_num] = breakdown.get("pre", 0)
                else:
                    # 降级方案：使用总时间
                    if "Retrieval" in strategy and strategy in all_retrieval:
                        time_by_round = all_retrieval[strategy]
                    elif strategy in all_insert:
                        time_by_round = all_insert[strategy]

                if time_by_round:
                    strategies_data[strategy] = (all_f1[strategy], time_by_round)

            if strategies_data:
                # 使用检测到的维度名（PreInsert/PostInsert等），而不是config名
                dimension = dimension_name if dimension_name else config.get("name", "Experiment")
                plot_cost_effectiveness_comparison(
                    strategies_data,
                    dimension,
                    out_path / "comparison_cost_effectiveness.png",
                    title="Cost-Effectiveness Comparison",
                )
                print("  ✓ Saved: comparison_cost_effectiveness.png")

    # 生成CSV
    if "csv" in config.get("output", {}).get("formats", []):
        generate_csv(all_f1, out_path / "f1_scores.csv", "F1")
        generate_csv(all_insert, out_path / "insert_times.csv", "Insert Time (ms)")
        generate_csv(all_retrieval, out_path / "retrieval_times.csv", "Retrieval Time (ms)")
        generate_category_csv(all_category_f1, out_path / "category_f1_scores.csv")
        generate_breakdown_csv(all_insert_breakdown, out_path / "insert_breakdown.csv", "Insert")
        generate_breakdown_csv(
            all_retrieval_breakdown, out_path / "retrieval_breakdown.csv", "Retrieval"
        )
        print("\n✓ CSV files saved")

    # 生成Markdown报告
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


def generate_csv(
    metrics: dict[str, dict[int, float]],
    output_path: Path,
    metric_name: str,
):
    """生成CSV文件"""
    if not metrics:
        return

    # 获取所有轮次
    all_rounds = sorted(set().union(*[set(m.keys()) for m in metrics.values()]))

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # 表头
        header = ["Strategy"] + [f"R{r}" for r in all_rounds] + ["Mean"]
        writer.writerow(header)

        # 数据行
        for strategy, round_metrics in metrics.items():
            values = [round_metrics.get(r, 0) for r in all_rounds]
            mean_val = float(np.mean(values)) if values else 0
            row = (
                [strategy]
                + [f"{v:.6f}" if v < 1 else f"{v:.2f}" for v in values]
                + [f"{mean_val:.6f}" if mean_val < 1 else f"{mean_val:.2f}"]
            )
            writer.writerow(row)


def generate_category_csv(
    category_metrics: dict[str, dict[int, float]],
    output_path: Path,
):
    """生成Category F1 CSV文件"""
    if not category_metrics:
        return

    # 获取所有Category
    all_categories = sorted(set().union(*[set(m.keys()) for m in category_metrics.values()]))

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # 表头
        header = ["Strategy"] + [f"Cat{c}" for c in all_categories] + ["Mean"]
        writer.writerow(header)

        # 数据行
        for strategy, cat_metrics in category_metrics.items():
            values = [cat_metrics.get(c, 0) for c in all_categories]
            mean_val = float(np.mean(values)) if values else 0
            row = [strategy] + [f"{v:.6f}" for v in values] + [f"{mean_val:.6f}"]
            writer.writerow(row)


def generate_breakdown_csv(
    breakdown_metrics: dict[str, dict[str, float]],
    output_path: Path,
    timing_type: str,
):
    """生成时间分解CSV文件"""
    if not breakdown_metrics:
        return

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # 表头
        header = ["Strategy", "Pre (ms)", "Memory (ms)", "Post (ms)", "Total (ms)"]
        writer.writerow(header)

        # 数据行
        for strategy, breakdown in breakdown_metrics.items():
            row = [
                strategy,
                f"{breakdown.get('pre', 0):.2f}",
                f"{breakdown.get('memory', 0):.2f}",
                f"{breakdown.get('post', 0):.2f}",
                f"{breakdown.get('total', 0):.2f}",
            ]
            writer.writerow(row)


def generate_markdown(
    config: dict,
    strategies: list[str],
    f1_metrics: dict,
    insert_metrics: dict,
    retrieval_metrics: dict,
    category_f1_metrics: dict,
    insert_breakdown_metrics: dict,
    retrieval_breakdown_metrics: dict,
    output_path: Path,
):
    """生成Markdown报告"""
    dataset_name = config.get("dataset", {}).get("name", "Unknown")

    # 获取所有轮次
    all_rounds = sorted(set().union(*[set(m.keys()) for m in f1_metrics.values()]))
    # 获取所有Category
    all_categories = (
        sorted(set().union(*[set(m.keys()) for m in category_f1_metrics.values()]))
        if category_f1_metrics
        else []
    )

    md = f"# {dataset_name.upper()} Round Analysis Report\n\n"
    md += f"**Generated:** {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    md += "## Summary\n\n"
    md += f"- **Dataset:** {dataset_name}\n"
    md += f"- **Strategies:** {len(strategies)}\n"
    md += f"- **Rounds:** {len(all_rounds)}\n"
    md += f"- **Categories:** {len(all_categories)}\n\n"

    # F1对比表
    md += "## F1 Scores by Round\n\n"
    md += "| Strategy | " + " | ".join([f"R{r}" for r in all_rounds]) + " | Mean |\n"
    md += "|----------|" + "|".join(["------" for _ in all_rounds]) + "|------|\n"

    for strategy in strategies:
        vals = [f1_metrics.get(strategy, {}).get(r, 0) for r in all_rounds]
        mean_val = float(np.mean(vals)) if vals else 0
        md += f"| {strategy} | " + " | ".join([f"{v:.4f}" for v in vals]) + f" | {mean_val:.4f} |\n"

    md += "\n"

    # Category F1对比表
    if category_f1_metrics and all_categories:
        category_labels = config.get("categories", {}).get("labels", {})
        md += "## F1 Scores by Category\n\n"
        cat_headers = [category_labels.get(c, f"Cat{c}") for c in all_categories]
        md += "| Strategy | " + " | ".join(cat_headers) + " | Mean |\n"
        md += "|----------|" + "|".join(["------" for _ in all_categories]) + "|------|\n"

        for strategy in strategies:
            vals = [category_f1_metrics.get(strategy, {}).get(c, 0) for c in all_categories]
            mean_val = float(np.mean(vals)) if vals else 0
            md += (
                f"| {strategy} | "
                + " | ".join([f"{v:.4f}" for v in vals])
                + f" | {mean_val:.4f} |\n"
            )

        md += "\n"

    # Insert时间表
    md += "## Insert Time (ms) by Round\n\n"
    md += "| Strategy | " + " | ".join([f"R{r}" for r in all_rounds]) + " | Mean |\n"
    md += "|----------|" + "|".join(["------" for _ in all_rounds]) + "|------|\n"

    for strategy in strategies:
        vals = [insert_metrics.get(strategy, {}).get(r, 0) for r in all_rounds]
        mean_val = float(np.mean(vals)) if vals else 0
        md += f"| {strategy} | " + " | ".join([f"{v:.2f}" for v in vals]) + f" | {mean_val:.2f} |\n"

    md += "\n"

    # Insert时间分解表
    if insert_breakdown_metrics:
        md += "## Insert Time Breakdown (ms)\n\n"
        md += "**Pre-Insert Stage**: Data preprocessing, embedding generation, and consolidation operations.\n\n"
        md += "**Memory-Insert Stage**: Core memory insertion operations (vector indexing, storage writes).\n\n"
        md += "**Post-Insert Stage**: Post-processing, secondary index updates, and cleanup.\n\n"

        md += "| Strategy | Pre (ms) | Memory (ms) | Post (ms) | Total (ms) |\n"
        md += "|----------|----------|-------------|-----------|------------|\n"

        for strategy in strategies:
            b = insert_breakdown_metrics.get(strategy, {})
            md += f"| {strategy} | {b.get('pre', 0):.2f} | {b.get('memory', 0):.2f} | {b.get('post', 0):.2f} | {b.get('total', 0):.2f} |\n"

        md += "\n"

    # Retrieval时间表
    md += "## Retrieval Time (ms) by Round\n\n"
    md += "| Strategy | " + " | ".join([f"R{r}" for r in all_rounds]) + " | Mean |\n"
    md += "|----------|" + "|".join(["------" for _ in all_rounds]) + "|------|\n"

    for strategy in strategies:
        vals = [retrieval_metrics.get(strategy, {}).get(r, 0) for r in all_rounds]
        mean_val = float(np.mean(vals)) if vals else 0
        md += f"| {strategy} | " + " | ".join([f"{v:.2f}" for v in vals]) + f" | {mean_val:.2f} |\n"

    md += "\n"

    # Retrieval时间分解表
    if retrieval_breakdown_metrics:
        md += "## Retrieval Time Breakdown (ms)\n\n"
        md += "**Pre-Retrieval Stage**: Query preprocessing, embedding generation, and query expansion.\n\n"
        md += "**Memory-Retrieval Stage**: Core memory retrieval operations (vector search, ranking).\n\n"
        md += "**Post-Retrieval Stage**: Result post-processing, re-ranking, and formatting.\n\n"

        md += "| Strategy | Pre (ms) | Memory (ms) | Post (ms) | Total (ms) |\n"
        md += "|----------|----------|-------------|-----------|------------|\n"

        for strategy in strategies:
            b = retrieval_breakdown_metrics.get(strategy, {})
            md += f"| {strategy} | {b.get('pre', 0):.2f} | {b.get('memory', 0):.2f} | {b.get('post', 0):.2f} | {b.get('total', 0):.2f} |\n"

        md += "\n"

    # 添加综合性能洞察部分
    if insert_breakdown_metrics and retrieval_breakdown_metrics:
        md += "## Performance Insights\n\n"

        # 找出最优策略
        best_f1_strategy = max(
            strategies,
            key=lambda s: float(
                np.mean(list(f1_metrics.get(s, {}).values())) if f1_metrics.get(s) else 0
            ),
        )
        best_f1_score = float(np.mean(list(f1_metrics.get(best_f1_strategy, {}).values())))

        # 找出最快的插入策略
        if insert_metrics:
            fastest_insert = min(
                strategies,
                key=lambda s: float(
                    np.mean(list(insert_metrics.get(s, {}).values()))
                    if insert_metrics.get(s)
                    else float("inf")
                ),
            )
            fastest_insert_time = float(
                np.mean(list(insert_metrics.get(fastest_insert, {}).values()))
            )

        # 找出最快的检索策略
        if retrieval_metrics:
            fastest_retrieval = min(
                strategies,
                key=lambda s: float(
                    np.mean(list(retrieval_metrics.get(s, {}).values()))
                    if retrieval_metrics.get(s)
                    else float("inf")
                ),
            )
            fastest_retrieval_time = float(
                np.mean(list(retrieval_metrics.get(fastest_retrieval, {}).values()))
            )

        md += "### Top Performers\n\n"
        md += f"- **Best F1 Score**: {best_f1_strategy} ({best_f1_score:.4f})\n"
        if insert_metrics:
            md += f"- **Fastest Insert**: {fastest_insert} ({fastest_insert_time:.2f} ms)\n"
        if retrieval_metrics:
            md += (
                f"- **Fastest Retrieval**: {fastest_retrieval} ({fastest_retrieval_time:.2f} ms)\n"
            )
        md += "\n"

        md += "### Stage-wise Performance Analysis\n\n"

        # Insert阶段分析
        md += "#### Insert Stage Analysis\n\n"

        # 分析哪些策略在pre-insert阶段最慢
        md += "**Pre-Insert Stage Impact**: Strategies with longest pre-processing times:\n\n"
        pre_insert_sorted = sorted(
            [(s, insert_breakdown_metrics.get(s, {}).get("pre", 0)) for s in strategies],
            key=lambda x: x[1],
            reverse=True,
        )[:3]
        for i, (strat, time) in enumerate(pre_insert_sorted, 1):
            md += f"{i}. {strat}: {time:.2f} ms\n"
        md += "\n"

        # 分析哪些策略在post-insert阶段最慢
        md += "**Post-Insert Stage Impact**: Strategies with longest post-processing times:\n\n"
        post_insert_sorted = sorted(
            [(s, insert_breakdown_metrics.get(s, {}).get("post", 0)) for s in strategies],
            key=lambda x: x[1],
            reverse=True,
        )[:3]
        for i, (strat, time) in enumerate(post_insert_sorted, 1):
            md += f"{i}. {strat}: {time:.2f} ms\n"
        md += "\n"

        # Retrieval阶段分析
        md += "#### Retrieval Stage Analysis\n\n"

        # 分析哪些策略在pre-retrieval阶段最慢
        md += "**Pre-Retrieval Stage Impact**: Strategies with longest pre-processing times:\n\n"
        pre_retrieval_sorted = sorted(
            [(s, retrieval_breakdown_metrics.get(s, {}).get("pre", 0)) for s in strategies],
            key=lambda x: x[1],
            reverse=True,
        )[:3]
        for i, (strat, time) in enumerate(pre_retrieval_sorted, 1):
            md += f"{i}. {strat}: {time:.2f} ms\n"
        md += "\n"

        # 分析哪些策略在post-retrieval阶段最慢
        md += "**Post-Retrieval Stage Impact**: Strategies with longest post-processing times:\n\n"
        post_retrieval_sorted = sorted(
            [(s, retrieval_breakdown_metrics.get(s, {}).get("post", 0)) for s in strategies],
            key=lambda x: x[1],
            reverse=True,
        )[:3]
        for i, (strat, time) in enumerate(post_retrieval_sorted, 1):
            md += f"{i}. {strat}: {time:.2f} ms\n"
        md += "\n"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)


def main():
    parser = argparse.ArgumentParser(
        description="通用轮次分析器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 分析LoCoMo的指定目录
  python round_analyzer.py --config locomo --path PostInsert_MemoryOS_forgetting_curve

  # 分析所有PostInsert目录
  python round_analyzer.py --config locomo --all --prefix PostInsert_

  # 仅验证数据
  python round_analyzer.py --config locomo --all --validate-only
        """,
    )

    parser.add_argument(
        "--config",
        required=True,
        help="配置文件名（不含.yaml后缀）",
    )
    parser.add_argument(
        "--path",
        nargs="+",
        help="要分析的目录名称",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="分析所有匹配前缀的目录",
    )
    parser.add_argument(
        "--prefix",
        default="PostInsert_",
        help="目录前缀（默认: PostInsert_）",
    )
    parser.add_argument(
        "--base-dir",
        help="数据基础目录（覆盖配置）",
    )
    parser.add_argument(
        "--output-dir",
        help="输出目录（覆盖配置）",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="仅验证数据，不执行分析",
    )

    args = parser.parse_args()

    # 加载配置
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        sys.exit(1)

    # 解析基础目录路径
    base_dir = resolve_path(args.base_dir or config["paths"]["base_dir"])

    # 确定要分析的策略
    if args.all:
        strategies = discover_experiment_dirs(base_dir, args.prefix)
        if not strategies:
            print(f"错误: 未找到以 '{args.prefix}' 开头的目录")
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
