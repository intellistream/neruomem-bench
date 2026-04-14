"""plotting.py — simple_experiment 专用绘图工具

与 benchmarks.evaluation.analysis.utils.plotting 相比，简化了视觉编码系统：
- 不使用 Dimension×System×Strategy 标记/颜色体系
- 适配器名称直接作为图例标签
- 重用通用的图表样式
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


# 默认颜色方案（最多支持 12 个适配器）
ADAPTER_COLORS = [
    "#3498DB",  # 蓝
    "#E74C3C",  # 红
    "#2ECC71",  # 绿
    "#9B59B6",  # 紫
    "#F39C12",  # 橙
    "#1ABC9C",  # 青
    "#E91E63",  # 粉
    "#795548",  # 棕
    "#607D8B",  # 蓝灰
    "#FF5722",  # 深橙
    "#8BC34A",  # 浅绿
    "#00BCD4",  # 天蓝
]

ADAPTER_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "p", "<", ">"]


def _get_color(idx: int) -> str:
    return ADAPTER_COLORS[idx % len(ADAPTER_COLORS)]


def _get_marker(idx: int) -> str:
    return ADAPTER_MARKERS[idx % len(ADAPTER_MARKERS)]


def plot_comparison(
    strategies_metrics: dict[str, dict[int, float]],
    metric_name: str,
    ylabel: str,
    output_path: Path | str,
    title: str | None = None,
    figsize: tuple = (9, 6),
):
    """
    绘制多适配器的对比折线图

    Args:
        strategies_metrics: {adapter_name: {round_idx: value}}
        metric_name: 指标名称（用于标题）
        ylabel: Y轴标签
        output_path: 输出路径
        title: 自定义标题
        figsize: 图片尺寸
    """
    fig, ax = plt.subplots(figsize=figsize)

    for idx, (adapter, metrics) in enumerate(strategies_metrics.items()):
        if not metrics:
            continue
        rounds = sorted(metrics.keys())
        values = [metrics[r] for r in rounds]
        ax.plot(
            rounds,
            values,
            marker=_get_marker(idx),
            color=_get_color(idx),
            linestyle="-",
            label=adapter,
            linewidth=1.5,
            markersize=6,
            alpha=0.85,
        )

    ax.set_xlabel("Round", fontsize=14, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=14, fontweight="bold")
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(fontsize=10, loc="best")

    all_rounds: set[int] = set()
    for m in strategies_metrics.values():
        all_rounds.update(m.keys())
    if all_rounds:
        ax.set_xticks(sorted(all_rounds))

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_single_strategy(
    strategy_name: str,
    f1_metrics: dict[int, float],
    insert_metrics: dict[int, float],
    retrieval_metrics: dict[int, float],
    output_path: Path | str,
):
    """
    单适配器：三合一子图（F1 / 插入时间 / 检索时间）
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    all_rounds = sorted(
        set(f1_metrics.keys()) | set(insert_metrics.keys()) | set(retrieval_metrics.keys())
    )

    def _draw(ax, data, ylabel, color, marker):
        values = [data.get(r, 0) for r in all_rounds]
        ax.plot(all_rounds, values, marker=marker, color=color, linewidth=1.5, markersize=6)
        ax.set_xlabel("Round", fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
        ax.set_xticks(all_rounds)

    _draw(axes[0], f1_metrics, "F1 Score", "#3498DB", "o")
    _draw(axes[1], insert_metrics, "Insert Time (ms)", "#E74C3C", "s")
    _draw(axes[2], retrieval_metrics, "Retrieval Time (ms)", "#2ECC71", "^")

    fig.suptitle(f"Adapter: {strategy_name}", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_category_comparison(
    strategies_metrics: dict[str, dict],
    output_path: Path | str,
    title: str = "F1 Score by Question Category",
    category_labels: dict | None = None,
    figsize: tuple = (9, 6),
):
    """
    多适配器 Category F1 分组柱状图

    Args:
        strategies_metrics: {adapter_name: {category_key: f1_score}}
        output_path: 输出路径
        title: 标题
        category_labels: 可选类别标签映射
        figsize: 图片尺寸
    """
    if not strategies_metrics:
        return

    fig, ax = plt.subplots(figsize=figsize)

    all_categories: set = set()
    for m in strategies_metrics.values():
        all_categories.update(m.keys())
    categories = sorted(all_categories, key=str)

    if category_labels is None:
        category_labels = {}

    adapters = list(strategies_metrics.keys())
    n = len(adapters)
    bar_width = 0.8 / max(n, 1)
    x = np.arange(len(categories))

    for i, (adapter, metrics) in enumerate(strategies_metrics.items()):
        values = [metrics.get(cat, 0) for cat in categories]
        offset = (i - n / 2 + 0.5) * bar_width
        color = _get_color(i)
        bars = ax.bar(x + offset, values, bar_width, label=adapter, color=color, alpha=0.85)
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=45,
                )

    xlabels = [category_labels.get(cat, str(cat)) for cat in categories]
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=30, ha="right", fontsize=10)
    ax.set_xlabel("Question Category", fontsize=12, fontweight="bold")
    ax.set_ylabel("F1 Score", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")

    all_vals = [v for m in strategies_metrics.values() for v in m.values() if v > 0]
    if all_vals:
        ax.set_ylim(0, min(1.15, max(all_vals) * 1.25))

    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_cost_effectiveness_comparison(
    strategies_data: dict[str, tuple[dict[int, float], dict[int, float]]],
    dimension: str,
    output_path: Path,
    title: str = "Cost-Effectiveness Comparison",
):
    """
    多适配器 Cost-Effectiveness 对比图

    CE(R_i) = F1(R_i) / Time_ms(R_i) × 1000

    Args:
        strategies_data: {adapter_name: (f1_by_round, time_by_round)}
        dimension: 维度标签（如 "mem0"）
        output_path: 输出路径
        title: 标题
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    for idx, (adapter, (f1_by_round, time_by_round)) in enumerate(strategies_data.items()):
        rounds = sorted(f1_by_round.keys())
        ce_values = [
            (f1_by_round[r] / time_by_round[r] * 1000) if time_by_round.get(r, 0) > 0 else 0
            for r in rounds
        ]
        ax.plot(
            rounds,
            ce_values,
            marker=_get_marker(idx),
            color=_get_color(idx),
            linestyle="-",
            label=adapter,
            linewidth=1.5,
            markersize=6,
            alpha=0.85,
        )

    ax.set_xlabel("Round", fontsize=14, fontweight="bold")
    ax.set_ylabel("CE (F1 / s)", fontsize=14, fontweight="bold")
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(fontsize=10, loc="best")
    ax.set_title(title, fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
