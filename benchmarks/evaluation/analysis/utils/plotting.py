"""
绘图工具

职责:
- 提供通用的绑图函数
- 支持单策略和多策略对比图
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


# ============================================================================
# 视觉编码系统 - 用于区分记忆体系统和操作策略
# ============================================================================

# 记忆体系统 → 标记形状
SYSTEM_MARKERS = {
    "TiM": "o",  # 圆形
    "MemoryOS": "s",  # 正方形
    "Mem0g": "^",  # 三角形
}

# 系统名称显示映射（用于图例）
SYSTEM_DISPLAY_NAMES = {
    "TiM": "Lsh Hash",
    "MemoryOS": "Queue-Segment",
    "Mem0g": "Property Graph",
}

# 操作策略 → 颜色（深色高对比度配色 - 适合打印和学术论文）
STRATEGY_COLORS = {
    # 基线
    "none": "#1f77b4",  # 深蓝（None基线）
    # PostRetrieval策略 - 深色高对比度
    "semantic": "#ff7f0e",  # 深橙（Rerank）
    "top_k": "#2ca02c",  # 深绿（Filter）
    "augment": "#9467bd",  # 深紫（Augment）
    "multi_query": "#e377c2",  # 深粉（Merge）
    # 其他增强操作
    "enrich": "#17becf",  # 深青（Enrich）
    # 复杂处理
    "summarize": "#bcbd22",  # 橄榄绿（Summarize）
    # 压缩/重写 - 单一操作
    "compress": "#d62728",  # 深红（Compress）
    "rewrite": "#e377c2",  # 深粉（Rewrite）
    "triplet_extract": "#9467bd",  # 深紫（Triplet Extract）
    "transform": "#8c564b",  # 棕色（Transform）
    # 组合操作 - 使用独立颜色
    "rewrite_compress": "#e377c2",  # 深粉（Rewrite-Compress）
    "rewrite_triplet_extract": "#9467bd",  # 深紫（Rewrite-Triplet-Extract）
    "enrich_summarize": "#17becf",  # 深青（Enrich-Summarize）
    # PostInsert 特有策略
    "forgetting_curve": "#ff7f0e",  # 深橙（Forgetting Curve）
    "link_evolution": "#2ca02c",  # 深绿（Link Evolution）
    "llm_crud": "#bcbd22",  # 橄榄绿（LLM CRUD）
    "heat_migration": "#d62728",  # 深红（Heat Migration）
    # PostRetrieval 特有策略
    "rerank": "#1f77b4",  # 深蓝（Rerank）
    "filter": "#2ca02c",  # 深绿（Filter）
    # PreRetrieve 特有策略
    "decompose": "#ff7f0e",  # 深橙（Decompose）
    "embedding": "#1f77b4",  # 深蓝（Embedding）
    "keyword_extract": "#bcbd22",  # 橄榄绿（Keyword Extract）
    "validate": "#9467bd",  # 深紫（Validate）
}

# 辅助: 线型 - 统一使用实线，避免视觉混乱
STRATEGY_LINESTYLES = {
    "none": "-",
    "semantic": "-",
    "top_k": "-",
    "augment": "-",
    "enrich": "-",
    "multi_query": "-",
    "summarize": "-",
    "compress": "-",
    "rewrite": "-",
}


def parse_config_name(config_name: str) -> tuple[str, str, str, str]:
    """
    解析配置名称，提取系统和策略

    Args:
        config_name: 如 "PreInsert_TiM_rewrite_compress"
                     (使用内部配置名称，如 TiM, MemoryOS, Mem0g)

    Returns:
        (dimension, system, strategy_key, strategy_label)
        例如: ("PreInsert", "TiM", "rewrite", "Rewrite-Compress")
        注意: system 返回的是内部名称，需要通过 SYSTEM_DISPLAY_NAMES 转换为显示名称
    """
    parts = config_name.split("_")
    dimension = parts[0] if parts else "Unknown"
    system = parts[1] if len(parts) > 1 else "Unknown"

    # 提取策略部分（第3个之后的所有parts）
    strategy_parts = parts[2:] if len(parts) > 2 else ["none"]

    # 如果只有一个策略词且是"none"
    if len(strategy_parts) == 1 and strategy_parts[0] == "none":
        return dimension, system, "none", "None"

    # 构建完整的策略字符串（用下划线连接）
    full_strategy = "_".join(strategy_parts)

    # 首先检查是否有完整匹配的组合策略
    strategy_key = full_strategy if full_strategy in STRATEGY_COLORS else None

    # 如果没有完整匹配，尝试匹配第一个已知策略前缀
    if strategy_key is None:
        matched = False
        for known_strategy in STRATEGY_COLORS:
            if known_strategy in strategy_parts[0].lower():
                strategy_key = known_strategy
                matched = True
                break
        # 如果仍然没有匹配，使用完整策略字符串作为key（避免冲突）
        if not matched:
            strategy_key = full_strategy

    # 构建显示标签
    # PreInsert策略标签简化映射
    preinsert_label_map = {
        "enrich_summarize": "Enrich",
        "rewrite_compress": "Rewrite",
        "rewrite_triplet_extract": "Rewrite",
    }

    # PreRetrieval策略标签映射（专用优化）
    preretrieval_label_map = {
        "embedding": "None",
        "keyword_extract": "Optimize",
        "decompose": "Enhancement",
        "validate": "Validate",
    }

    # PostRetrieval策略标签映射
    postretrieval_label_map = {
        "multi_query": "Merge",
        "semantic": "Rerank",
        "top_k": "Filter",
        "augment": "Augment",
        "none": "None",
    }

    # 根据维度应用不同的映射
    if dimension == "PreRetrieval" or dimension == "PreRetrieve":
        # PreRetrieval使用专用映射
        full_strategy_lower = "_".join(strategy_parts)
        if full_strategy_lower in preretrieval_label_map:
            strategy_label = preretrieval_label_map[full_strategy_lower]
        else:
            strategy_label = strategy_parts[0].replace("_", " ").title()
    elif dimension == "PostRetrieval":
        # PostRetrieval使用专用映射
        full_strategy_lower = "_".join(strategy_parts)
        if full_strategy_lower in postretrieval_label_map:
            strategy_label = postretrieval_label_map[full_strategy_lower]
        else:
            strategy_label = strategy_parts[0].replace("_", " ").title()
    elif len(strategy_parts) == 1:
        # 单个策略词
        strategy_label = strategy_parts[0].replace("_", " ").title()
    else:
        # 多个策略词，先检查是否有简化映射
        full_strategy_lower = "_".join(strategy_parts)
        if full_strategy_lower in preinsert_label_map:
            strategy_label = preinsert_label_map[full_strategy_lower]
        else:
            # 连接成 "Prefix-Action" 格式
            # 例如: rewrite_compress → Rewrite-Compress
            strategy_label = "-".join(p.capitalize() for p in strategy_parts)

    return dimension, system, strategy_key, strategy_label


def get_visual_encoding(config_name: str) -> dict:
    """
    获取配置的视觉编码

    Returns:
        {marker, color, linestyle, system, strategy, strategy_label}
    """
    dimension, system, strategy_key, strategy_label = parse_config_name(config_name)

    return {
        "marker": SYSTEM_MARKERS.get(system, "o"),
        "color": STRATEGY_COLORS.get(strategy_key, "#808080"),
        "linestyle": STRATEGY_LINESTYLES.get(strategy_key, "-"),
        "system": system,
        "strategy_key": strategy_key,
        "strategy_label": strategy_label,
        "dimension": dimension,
    }


# 默认颜色方案（向后兼容）
DEFAULT_COLORS = [
    "#3498DB",  # 蓝色
    "#E74C3C",  # 红色
    "#2ECC71",  # 绿色
    "#9B59B6",  # 紫色
    "#F39C12",  # 橙色
    "#1ABC9C",  # 青色
    "#E91E63",  # 粉色
    "#795548",  # 棕色
]


def plot_comparison(
    strategies_metrics: dict[str, dict[int, float]],
    metric_name: str,
    ylabel: str,
    output_path: Path | str,
    title: str | None = None,
    figsize: tuple = (9, 6),
    use_visual_encoding: bool = True,
):
    """
    绘制多个策略的对比图

    Args:
        strategies_metrics: {strategy_name: {round_idx: value}}
        metric_name: 指标名称（用于标题）
        ylabel: Y轴标签
        output_path: 输出文件路径
        title: 自定义标题
        figsize: 图片尺寸
        use_visual_encoding: 是否使用视觉编码系统（标记=系统，颜色=策略）
    """
    fig, ax = plt.subplots(figsize=figsize)

    # 检测是否是PreInsert/PostInsert/PreRetrieval/PostRetrieval类型的配置
    is_dimension_config = any(
        any(dim in name for dim in ["PreInsert", "PostInsert", "PreRetrieval", "PostRetrieval"])
        for name in strategies_metrics
    )

    use_visual_encoding = use_visual_encoding and is_dimension_config

    if use_visual_encoding:
        # 使用视觉编码: 标记=系统, 颜色=策略
        # 去重: 使用集合记录已绘制的配置
        plotted_configs = {}

        for strategy, metrics in strategies_metrics.items():
            if not metrics:
                continue

            rounds = sorted(metrics.keys())
            values = [metrics[r] for r in rounds]

            encoding = get_visual_encoding(strategy)

            # 生成唯一键用于去重 - 使用strategy_label而不是strategy_key以区分不同的组合操作
            unique_key = f"{encoding['system']}_{encoding['strategy_label']}"

            # 如果已经绘制过相同的system+strategy组合，跳过
            if unique_key in plotted_configs:
                continue

            plotted_configs[unique_key] = encoding

            display_system = SYSTEM_DISPLAY_NAMES.get(encoding["system"], encoding["system"])
            ax.plot(
                rounds,
                values,
                marker=encoding["marker"],
                color=encoding["color"],
                linestyle=encoding["linestyle"],
                label=f"{display_system}-{encoding['strategy_label']}",  # 使用显示名称
                linewidth=1.5,
                markersize=6,
                alpha=0.85,
            )
    else:
        # 使用原始配色方案
        colors = plt.cm.Set3(np.linspace(0, 1, max(len(strategies_metrics), 8)))

        for (strategy, metrics), color in zip(strategies_metrics.items(), colors):
            if not metrics:
                continue

            rounds = sorted(metrics.keys())
            values = [metrics[r] for r in rounds]

            ax.plot(
                rounds,
                values,
                marker="o",
                label=strategy,
                color=color,
                linewidth=1.5,
                markersize=6,
                alpha=0.8,
            )

    ax.set_xlabel("Round", fontsize=16, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=16, fontweight="bold")

    # 设置刻度标签字体大小
    ax.tick_params(axis="both", labelsize=14)

    # 使用双图例或标准图例
    if use_visual_encoding and plotted_configs:
        create_dual_legend(ax, list(plotted_configs.values()))
    else:
        ax.legend(fontsize=9, loc="best", ncol=1)

    # 设置x轴刻度
    all_rounds = set()
    for m in strategies_metrics.values():
        all_rounds.update(m.keys())
    if all_rounds:
        ax.set_xticks(sorted(all_rounds))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_dual_legend(ax, plotted_configs):
    """
    创建双图例: 分别显示系统(标记)和策略(颜色)
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    # 提取唯一的系统和策略
    systems = sorted({c["system"] for c in plotted_configs})
    strategies = sorted(
        {c["strategy_label"] for c in plotted_configs}, key=lambda x: (x != "None", x)
    )  # None排第一

    # 系统标记的颜色映射（深色，与淡色策略形成对比）
    system_colors = {
        "TiM": "#2E86AB",  # 深蓝
        "MemoryOS": "#A23B72",  # 深紫红
        "Mem0g": "#F18F01",  # 深橙
    }

    # 创建系统图例 (标记形状)
    system_handles = [
        Line2D(
            [0],
            [0],
            marker=SYSTEM_MARKERS.get(sys, "o"),
            color=system_colors.get(sys, "#666666"),
            linestyle="",
            markersize=14,
            label=SYSTEM_DISPLAY_NAMES.get(sys, sys),  # 使用显示名称
            markeredgewidth=1.5,
            markerfacecolor=system_colors.get(sys, "#666666"),
        )
        for sys in systems
        if sys in SYSTEM_MARKERS
    ]

    # 创建策略图例 (颜色) - 反向查找颜色
    strategy_handles = []
    for strat_label in strategies:
        # 找到对应的strategy_key
        color = "#808080"  # 默认灰色
        for config in plotted_configs:
            if config["strategy_label"] == strat_label:
                color = config["color"]
                break

        strategy_handles.append(
            Patch(facecolor=color, label=strat_label, edgecolor="white", linewidth=0.5)
        )

    # 添加双图例 - 放在底部x轴标签下方，一行一个
    if strategy_handles:
        legend1 = ax.legend(
            handles=strategy_handles,
            title="Operation",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.2),
            fontsize=10,
            title_fontsize=12,
            framealpha=0.95,
            edgecolor="gray",
            ncol=100,  # 设置足够大的值确保一行显示
            columnspacing=0.5,
            handlelength=1.0,
            handletextpad=0.4,
        )
        ax.add_artist(legend1)  # 保留第一个图例

    if system_handles:
        ax.legend(
            handles=system_handles,
            title="Memory Base",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.30),
            fontsize=10,
            title_fontsize=12,
            framealpha=0.95,
            edgecolor="gray",
            ncol=100,  # 设置足够大的值确保一行显示
            columnspacing=0.5,
            handlelength=1.0,
            handletextpad=0.4,
        )


def plot_single_strategy(
    strategy_name: str,
    f1_metrics: dict[int, float],
    insert_metrics: dict[int, float],
    retrieval_metrics: dict[int, float],
    output_path: Path | str,
):
    """
    为单个策略绘制3个指标的子图

    Args:
        strategy_name: 策略名称
        f1_metrics: F1分数 {round: score}
        insert_metrics: 插入时间 {round: ms}
        retrieval_metrics: 检索时间 {round: ms}
        output_path: 输出文件路径
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # 获取所有轮次
    all_rounds = sorted(
        set(f1_metrics.keys()) | set(insert_metrics.keys()) | set(retrieval_metrics.keys())
    )

    # F1分数
    f1_values = [f1_metrics.get(r, 0) for r in all_rounds]
    axes[0].plot(
        all_rounds,
        f1_values,
        marker="o",
        color="#3498DB",
        linewidth=1.5,
        markersize=6,
    )
    axes[0].set_xlabel("Round", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("F1 Score", fontsize=12, fontweight="bold")
    axes[0].set_xticks(all_rounds)

    # Insert时间
    insert_values = [insert_metrics.get(r, 0) for r in all_rounds]
    axes[1].plot(
        all_rounds,
        insert_values,
        marker="s",
        color="#E74C3C",
        linewidth=1.5,
        markersize=6,
    )
    axes[1].set_xlabel("Round", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Time (ms)", fontsize=12, fontweight="bold")
    axes[1].set_xticks(all_rounds)

    # Retrieval时间
    retrieval_values = [retrieval_metrics.get(r, 0) for r in all_rounds]
    axes[2].plot(
        all_rounds,
        retrieval_values,
        marker="^",
        color="#2ECC71",
        linewidth=1.5,
        markersize=6,
    )
    axes[2].set_xlabel("Round", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("Time (ms)", fontsize=12, fontweight="bold")
    axes[2].set_xticks(all_rounds)

    fig.suptitle(f"Strategy: {strategy_name}", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_bar_comparison(
    data: dict[str, float],
    ylabel: str,
    output_path: Path | str,
    title: str = "",
    figsize: tuple = (8, 5),
):
    """
    绘制柱状对比图

    Args:
        data: {label: value}
        ylabel: Y轴标签
        output_path: 输出路径
        title: 标题
    """
    fig, ax = plt.subplots(figsize=figsize)

    labels = list(data.keys())
    values = list(data.values())
    colors = DEFAULT_COLORS[: len(labels)]

    bars = ax.bar(range(len(labels)), values, color=colors, alpha=0.8)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")

    # 添加数值标签
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{val:.3f}" if val < 10 else f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_category_comparison(
    strategies_metrics: dict[str, dict[int, float]],
    output_path: Path | str,
    title: str = "F1 Score by Category",
    category_labels: dict[int, str] | None = None,
    figsize: tuple = (9, 6),
):
    """
    绘制多策略的Category F1对比图（分组柱状图）

    Args:
        strategies_metrics: {strategy_name: {category: f1_score}}
        output_path: 输出路径
        title: 标题
        category_labels: 可选的Category标签映射 {1: "Multi-Answer", ...}
        figsize: 图片尺寸
    """
    if not strategies_metrics:
        return

    fig, ax = plt.subplots(figsize=figsize)

    # 获取所有Category
    all_categories = set()
    for metrics in strategies_metrics.values():
        all_categories.update(metrics.keys())
    categories = sorted(all_categories)

    # 设置默认标签
    if category_labels is None:
        category_labels = {
            1: "Cat1: Multi-Answer",
            2: "Cat2: Time-Related",
            3: "Cat3: Clean-Comments",
            4: "Cat4: Standard",
            5: "Cat5: Not-Mentioned",
        }

    strategies = list(strategies_metrics.keys())
    n_strategies = len(strategies)
    n_categories = len(categories)

    bar_width = 0.8 / n_strategies
    x = np.arange(n_categories)

    colors = plt.cm.Set2(np.linspace(0, 1, n_strategies))

    for i, (strategy, metrics) in enumerate(strategies_metrics.items()):
        values = [metrics.get(cat, 0) for cat in categories]
        offset = (i - n_strategies / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, values, bar_width, label=strategy, color=colors[i], alpha=0.85)

        # 添加数值标签
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

    # 设置x轴标签
    x_labels = [category_labels.get(cat, f"Cat{cat}") for cat in categories]
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=10)

    ax.set_xlabel("Question Category", fontsize=12, fontweight="bold")
    ax.set_ylabel("F1 Score", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.set_ylim(0, min(1.1, max(v for m in strategies_metrics.values() for v in m.values()) * 1.2))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_time_breakdown(
    strategies_breakdowns: dict[str, dict[str, float]],
    output_path: Path | str,
    title: str = "Time Breakdown",
    figsize: tuple = (9, 6),
):
    """
    绘制时间分解堆叠柱状图

    Args:
        strategies_breakdowns: {strategy: {"pre": ms, "memory": ms, "post": ms}}
        output_path: 输出路径
        title: 标题
        figsize: 图片尺寸
    """
    if not strategies_breakdowns:
        return

    fig, ax = plt.subplots(figsize=figsize)

    strategies = list(strategies_breakdowns.keys())
    n_strategies = len(strategies)
    x = np.arange(n_strategies)
    bar_width = 0.6

    # 颜色
    colors = {
        "pre": "#3498DB",  # 蓝色 - 预处理
        "memory": "#E74C3C",  # 红色 - 核心Memory
        "post": "#2ECC71",  # 绿色 - 后处理
    }

    # 堆叠绘制
    pre_values = [strategies_breakdowns[s].get("pre", 0) for s in strategies]
    memory_values = [strategies_breakdowns[s].get("memory", 0) for s in strategies]
    post_values = [strategies_breakdowns[s].get("post", 0) for s in strategies]

    ax.bar(x, pre_values, bar_width, label="Pre-processing", color=colors["pre"], alpha=0.85)
    ax.bar(
        x,
        memory_values,
        bar_width,
        bottom=pre_values,
        label="Memory Operation",
        color=colors["memory"],
        alpha=0.85,
    )
    ax.bar(
        x,
        post_values,
        bar_width,
        bottom=[p + m for p, m in zip(pre_values, memory_values)],
        label="Post-processing",
        color=colors["post"],
        alpha=0.85,
    )

    # 添加总时间标签
    for i, s in enumerate(strategies):
        total = strategies_breakdowns[s].get(
            "total", sum([pre_values[i], memory_values[i], post_values[i]])
        )
        ax.text(i, total, f"{total:.1f}ms", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=45, ha="right", fontsize=10)
    ax.set_xlabel("Strategy", fontsize=12, fontweight="bold")
    ax.set_ylabel("Time (ms)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_time_breakdown_by_round(
    breakdown_by_round: dict[int, dict[str, float]],
    output_path: Path | str,
    title: str = "Time Breakdown by Round",
    figsize: tuple = (10, 6),
):
    """
    绘制单策略按轮次的时间分解面积图

    Args:
        breakdown_by_round: {round_idx: {"pre": ms, "memory": ms, "post": ms}}
        output_path: 输出路径
        title: 标题
    """
    if not breakdown_by_round:
        return

    fig, ax = plt.subplots(figsize=figsize)

    rounds = sorted(breakdown_by_round.keys())
    pre_values = [breakdown_by_round[r].get("pre", 0) for r in rounds]
    memory_values = [breakdown_by_round[r].get("memory", 0) for r in rounds]
    post_values = [breakdown_by_round[r].get("post", 0) for r in rounds]

    # 堆叠面积图
    ax.stackplot(
        rounds,
        pre_values,
        memory_values,
        post_values,
        labels=["Pre-processing", "Memory Operation", "Post-processing"],
        colors=["#3498DB", "#E74C3C", "#2ECC71"],
        alpha=0.7,
    )

    ax.set_xlabel("Round", fontsize=12, fontweight="bold")
    ax.set_ylabel("Time (ms)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    ax.set_xticks(rounds)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_cost_effectiveness_evolution(
    f1_by_round: dict[int, float],
    time_by_round: dict[int, float],
    config_name: str,
    output_path: Path,
    title: str = None,
) -> None:
    """
    绘制 Cost-Effectiveness 演化曲线

    Cost-Effectiveness 定义：CE(R_i) = F1(R_i) / Time(R_i) × 1000
    物理意义：每花费 1000ms 能获得多少 F1 分数

    Args:
        f1_by_round: {round: f1_score}
        time_by_round: {round: time_ms}
        config_name: 配置名称（如 "PreInsert_Mem0g_none"）
        output_path: 输出路径
        title: 自定义标题
    """
    dimension, system, strategy_key, strategy_label = parse_config_name(config_name)

    if title is None:
        title = f"{config_name}\nCost-Effectiveness Evolution"

    # 计算 Cost-Effectiveness
    rounds = sorted(f1_by_round.keys())
    ce_values = [
        (f1_by_round[r] / time_by_round[r] * 1000) if time_by_round.get(r, 0) > 0 else 0
        for r in rounds
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    # 获取视觉编码
    color = STRATEGY_COLORS.get(strategy_key, "#000000")
    linestyle = STRATEGY_LINESTYLES.get(strategy_key, "-")
    marker = SYSTEM_MARKERS.get(system, "o")
    display_system = SYSTEM_DISPLAY_NAMES.get(system, system)

    # 绘制曲线
    ax.plot(
        rounds,
        ce_values,
        label=f"{display_system} - {strategy_label}",  # 使用显示名称
        color=color,
        linestyle=linestyle,
        marker=marker,
        linewidth=1.5,
        markersize=6,
        alpha=0.9,
    )

    # 添加退化率注释
    if len(ce_values) >= 2:
        degradation = (ce_values[-1] - ce_values[0]) / ce_values[0] * 100
        ax.annotate(
            f"Degradation: {degradation:.1f}%",
            xy=(rounds[-1], ce_values[-1]),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "yellow", "alpha": 0.5},
            arrowprops={"arrowstyle": "->", "connectionstyle": "arc3,rad=0", "lw": 1.5},
        )

    ax.set_xlabel("Round", fontsize=13, fontweight="bold")
    ax.set_ylabel("Cost-Effectiveness", fontsize=13, fontweight="bold")
    ax.set_xticks(rounds)
    ax.set_xticklabels([f"R{r}" for r in rounds])
    ax.legend(loc="best", frameon=True, framealpha=0.95, edgecolor="black")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_cost_effectiveness_comparison(
    strategies_data: dict[str, tuple[dict, dict]],
    dimension: str,
    output_path: Path,
    title: str = None,
) -> None:
    """
    绘制多策略 Cost-Effectiveness 对比曲线（使用对数坐标，符合 ICML 标准）

    Args:
        strategies_data: {config_name: (f1_by_round, time_by_round)}
        dimension: 维度名称（如 "PreInsert"）
        output_path: 输出路径
        title: 自定义标题
    """
    if title is None:
        title = f"{dimension}\nCost-Effectiveness Comparison"

    # 收集所有数据
    plotted_configs = {}
    data_to_plot = []  # 存储 (rounds, ce_values, color, linestyle, marker, unique_key, encoding)

    for config_name, (f1_by_round, time_by_round) in strategies_data.items():
        _, system, strategy_key, strategy_label = parse_config_name(config_name)

        unique_key = f"{system}_{strategy_label}"
        if unique_key in plotted_configs:
            continue

        rounds = sorted(f1_by_round.keys())
        ce_values = [
            (f1_by_round[r] / time_by_round[r] * 1000) if time_by_round.get(r, 0) > 0 else 0
            for r in rounds
        ]

        color = STRATEGY_COLORS.get(strategy_key, "#000000")
        linestyle = STRATEGY_LINESTYLES.get(strategy_key, "-")
        marker = SYSTEM_MARKERS.get(system, "o")

        encoding = {
            "system": system,
            "strategy_key": strategy_key,
            "strategy_label": strategy_label,
            "color": color,
            "marker": marker,
            "linestyle": linestyle,
        }
        plotted_configs[unique_key] = encoding
        data_to_plot.append((rounds, ce_values, color, linestyle, marker, unique_key, encoding))

    # 使用"高瘦"布局（适合4图并排，适度增加宽度）
    fig, ax = plt.subplots(figsize=(4.5, 5))

    # 跟踪实际使用的策略和系统（用于生成解耦图例）
    used_strategies = set()
    used_systems = set()

    for rounds, ce_values, color, linestyle, marker, _unique_key, encoding in data_to_plot:
        # 过滤掉零值（对数坐标无法显示）
        valid_data = [(r, ce) for r, ce in zip(rounds, ce_values) if ce > 0]
        if not valid_data:
            continue

        valid_rounds, valid_ce = zip(*valid_data)

        # 记录使用的策略和系统
        used_strategies.add(
            (encoding["strategy_key"], encoding["strategy_label"], encoding["color"])
        )
        used_systems.add((encoding["system"], encoding["marker"]))

        ax.plot(
            valid_rounds,
            valid_ce,
            color=color,
            linestyle=linestyle,
            marker=marker,
            linewidth=2.5,
            markersize=7,
            alpha=0.9,
        )

    # 设置对数坐标
    ax.set_yscale("log")

    # 简化标签（去除单位，留给 Figure Caption）
    # 只在 PreInsert 时显示 Y 轴标签，其他维度不显示（4图并排时节省空间）
    ax.set_xlabel("Round", fontsize=14)
    if dimension == "PreInsert":
        ax.set_ylabel("Cost-Effectiveness", fontsize=14)
    else:
        ax.set_ylabel("")  # 其他维度不显示 Y 轴标签

    # 设置刻度标签字体大小
    ax.tick_params(axis="both", labelsize=12)

    # 设置 x 轴刻度
    all_rounds = set()
    for f1_by_round, _ in strategies_data.values():
        all_rounds.update(f1_by_round.keys())
    if all_rounds:
        ax.set_xticks(sorted(all_rounds))

    # 添加网格线（对数坐标下更清晰）
    ax.grid(True, which="both", linestyle="--", alpha=0.3)

    # 创建解耦图例（颜色=操作，形状=系统）
    from matplotlib.lines import Line2D

    # 第一组：颜色代表操作（Operation）
    legend_operations = []
    for _strategy_key, strategy_label, color in sorted(used_strategies, key=lambda x: x[1]):
        legend_operations.append(Line2D([0], [0], color=color, lw=3, label=strategy_label))

    # 第二组：形状代表系统（Memory Base）
    legend_systems = []
    for system, marker in sorted(used_systems, key=lambda x: x[0]):
        # 使用显示名称，如果没有映射则使用原始名称
        display_name = SYSTEM_DISPLAY_NAMES.get(system, system)
        legend_systems.append(
            Line2D(
                [0],
                [0],
                color="gray",
                marker=marker,
                linestyle="None",
                markersize=8,
                markeredgewidth=1.5,
                label=display_name,
            )
        )

    # 合并图例（操作 + 系统）
    final_handles = legend_operations + legend_systems

    # 图例放在图表外部下方，紧凑横向排列
    ax.legend(
        handles=final_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        frameon=False,
        fontsize=10,
        handlelength=1.8,
        handletextpad=0.5,
        columnspacing=1.0,
    )

    # 使用固定边距确保所有维度图表高度一致
    # 即使某些维度不显示 Y 轴标签，也保留相同的左边距
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
