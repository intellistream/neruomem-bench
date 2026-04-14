"""
改进的视觉编码系统 - 用于四个记忆操作维度的图表

设计原则:
    - 使用标记形状(marker)表示记忆体系统
    - 使用颜色表示操作策略 (none/enrich/compress等)
    - 使用线型(linestyle)作为辅助区分

视觉编码:
    标记 (Marker) = 记忆体系统
        ○ (circle)     → Lsh Hash (内部名: TiM)
        □ (square)     → Queue-Segment (内部名: MemoryOS)
        △ (triangle_up) → Property Graph (内部名: Mem0g)

    颜色 (Color) = 操作策略
        灰色   → none (基线)
        蓝色   → semantic/top_k (语义相关)
        绿色   → augment/enrich (增强)
        橙色   → multi_query/summarize (复杂处理)
        红色   → compress/rewrite (压缩/重写)

注意:
    - SYSTEM_MARKERS 使用内部配置名称作为键 (TiM, MemoryOS, Mem0g)
    - SYSTEM_DISPLAY_NAMES 将内部名称映射到显示名称
    - 图例中显示的是 SYSTEM_DISPLAY_NAMES 中的友好名称
"""

from __future__ import annotations

import matplotlib.pyplot as plt

# ============================================================================
# 视觉编码配置
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

# 操作策略 → 颜色 (鲜艳版本)
STRATEGY_COLORS = {
    # 基线
    "none": "#808080",  # 灰色
    # 增强操作 (鲜艳绿色)
    "enrich": "#00D084",  # 鲜绿
    "enrich_summarize": "#00D084",  # 鲜绿
    "transform_summarize": "#4CAF50",  # 草绿
    "augment": "#66FF66",  # 亮绿
    # 压缩/重写 (鲜艳红色)
    "rewrite_compress": "#FF1744",  # 鲜红
    "rewrite_triplet_extract": "#D500F9",  # 紫红
    "compress": "#E74C3C",  # 橙红
    "rewrite": "#C0392B",  # 深红
    # 语义相关 (鲜艳蓝色)
    "semantic": "#00B0FF",  # 天蓝
    "top_k": "#2196F3",  # 亮蓝
    # 复杂处理 (鲜艳橙色)
    "multi_query": "#FF9100",  # 鲜橙
    "summarize": "#FFC107",  # 金黄
    # PreRetrieve 特有策略
    "decompose": "#FFB399",  # 珊瑚粉（Decompose）
    "embedding": "#99DDFF",  # 天蓝（Embedding）
    "keyword_extract": "#FFDD99",  # 浅金黄（Keyword Extract）
    "validate": "#DD99FF",  # 淡紫（Validate）
}

# 辅助: 线型 (可选，用于进一步区分)
STRATEGY_LINESTYLES = {
    "none": "-",  # 实线
    "semantic": "-",  # 实线
    "top_k": "-",  # 实线
    "augment": "--",  # 虚线
    "enrich": "--",  # 虚线
    "multi_query": "-.",  # 点划线
    "summarize": "-.",  # 点划线
    "compress": ":",  # 点线
    "rewrite": ":",  # 点线
}


# ============================================================================
# 解析配置名称
# ============================================================================


def parse_config_name(config_name: str) -> tuple[str, str, str]:
    """
    解析配置名称，提取系统、维度、策略

    Args:
        config_name: 如 "PreInsert_TiM_enrich_summarize"
                     (使用内部配置名称，如 TiM, MemoryOS, Mem0g)

    Returns:
        (dimension, system, strategy)
        如 ("PreInsert", "TiM", "enrich_summarize")
        注意: system 返回的是内部名称，需要通过 SYSTEM_DISPLAY_NAMES 转换为显示名称
    """
    parts = config_name.split("_")

    # 第一部分: 维度 (PreInsert/PostInsert/PreRetrieval/PostRetrieval)
    dimension = parts[0]

    # 第二部分: 系统 (TiM/MemoryOS/Mem0g - 内部配置名称)
    system = parts[1] if len(parts) > 1 else "Unknown"

    # 第三部分及之后: 策略 (保持完整组合名)
    strategy = "_".join(parts[2:]) if len(parts) > 2 else "none"

    return dimension, system, strategy


def format_strategy_label(strategy: str) -> str:
    """
    格式化策略标签

    规则:
    - none → "None-action"
    - enrich_summarize → "Enrich-Summarize"
    - rewrite_compress → "Rewrite-Compress"
    """
    if strategy == "none":
        return "None-action"

    # 将下划线替换为连字符，并首字母大写
    words = strategy.split("_")
    return "-".join(word.capitalize() for word in words)


def get_visual_encoding(config_name: str) -> dict:
    """
    获取配置的视觉编码

    Returns:
        {
            'marker': 标记形状,
            'color': 颜色,
            'linestyle': 线型,
            'label': 图例标签,
            'formatted_strategy': 格式化的策略名,
            'system': 系统名,
            'strategy': 策略名
        }
    """
    dimension, system, strategy = parse_config_name(config_name)

    # 获取颜色 (优先匹配完整策略名，否则匹配前缀)
    color = STRATEGY_COLORS.get(strategy)
    if color is None:
        # 尝试匹配前缀
        for known_strategy in STRATEGY_COLORS:
            if strategy.startswith(known_strategy):
                color = STRATEGY_COLORS[known_strategy]
                break
        else:
            color = "#808080"  # 默认灰色

    formatted_strategy = format_strategy_label(strategy)
    display_system = SYSTEM_DISPLAY_NAMES.get(system, system)

    return {
        "marker": SYSTEM_MARKERS.get(system, "o"),
        "color": color,
        "linestyle": STRATEGY_LINESTYLES.get(strategy, "-"),
        "label": f"{display_system} - {formatted_strategy}",  # 使用显示名称
        "formatted_strategy": formatted_strategy,
        "system": system,  # 保留内部名称供后续处理
        "display_system": display_system,  # 添加显示名称
        "strategy": strategy,
        "dimension": dimension,
    }


# ============================================================================
# 改进的绘图函数
# ============================================================================


def plot_comparison_with_encoding(
    strategies_metrics: dict[str, dict[int, float]],
    metric_name: str,
    ylabel: str,
    output_path,
    title: str | None = None,
    figsize: tuple = (12, 7),
    show_legend_separately: bool = True,
):
    """
    使用视觉编码系统的对比图

    新特性:
        - 标记形状 = 记忆体系统
        - 颜色 = 操作策略
        - 可选的分离图例
    """
    fig, ax = plt.subplots(figsize=figsize)

    # 按系统和策略分组
    plotted_configs = []

    for config_name, metrics in strategies_metrics.items():
        if not metrics:
            continue

        # 获取视觉编码
        encoding = get_visual_encoding(config_name)

        rounds = sorted(metrics.keys())
        values = [metrics[r] for r in rounds]

        ax.plot(
            rounds,
            values,
            marker=encoding["marker"],
            color=encoding["color"],
            linestyle=encoding["linestyle"],
            label=encoding["label"],
            linewidth=2.0,
            markersize=8,
            alpha=0.8,
        )

        plotted_configs.append(encoding)

    # 设置基本属性
    ax.set_xlabel("Test Round", fontweight="bold", fontsize=12)
    ax.set_ylabel(ylabel, fontweight="bold", fontsize=12)
    ax.set_title(title or f"{metric_name} by Round", fontweight="bold", fontsize=14)
    ax.grid(True, alpha=0.4, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)

    # 图例处理
    if show_legend_separately:
        # 创建分离的图例: 一个用于系统，一个用于策略
        create_dual_legend(ax, plotted_configs)
    else:
        # 标准图例
        ax.legend(fontsize=9, loc="best", framealpha=0.95, ncol=2)

    # 设置x轴刻度
    all_rounds = set()
    for m in strategies_metrics.values():
        all_rounds.update(m.keys())
    if all_rounds:
        ax.set_xticks(sorted(all_rounds))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")

    # PDF版本
    from pathlib import Path

    pdf_path = Path(output_path).with_suffix(".pdf")
    plt.savefig(pdf_path, format="pdf", dpi=300, bbox_inches="tight")

    plt.close()

    return fig


def create_dual_legend(ax, plotted_configs):
    """
    创建双图例: 分别显示系统(标记)和策略(颜色)
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    # 提取唯一的系统和策略 (去重)
    seen_systems = set()
    seen_strategies = set()
    unique_systems = []
    unique_strategies = []

    for c in plotted_configs:
        if c["system"] not in seen_systems:
            seen_systems.add(c["system"])
            unique_systems.append(c["system"])
        if c["strategy"] not in seen_strategies:
            seen_strategies.add(c["strategy"])
            unique_strategies.append((c["strategy"], c["formatted_strategy"], c["color"]))

    # 创建系统图例 (标记形状) - 使用黑色
    system_handles = [
        Line2D(
            [0],
            [0],
            marker=SYSTEM_MARKERS.get(sys, "o"),
            color="black",
            linestyle="",
            markersize=10,
            markeredgewidth=1.5,
            label=SYSTEM_DISPLAY_NAMES.get(sys, sys),  # 使用显示名称
        )
        for sys in unique_systems
        if sys in SYSTEM_MARKERS
    ]

    # 创建策略图例 (颜色) - 使用格式化标签
    strategy_handles = [
        Patch(facecolor=color, edgecolor="white", linewidth=0.5, label=formatted)
        for strategy, formatted, color in unique_strategies
    ]

    # 添加双图例
    legend1 = ax.legend(
        handles=system_handles,
        title="Memory System",
        loc="upper left",
        fontsize=10,
        title_fontsize=11,
        framealpha=0.95,
        edgecolor="gray",
    )
    ax.add_artist(legend1)  # 保留第一个图例

    ax.legend(
        handles=strategy_handles,
        title="Operation",
        loc="upper right",
        fontsize=9,
        title_fontsize=11,
        framealpha=0.95,
        edgecolor="gray",
    )


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")

    # 测试配置解析
    test_configs = [
        "PreInsert_TiM_enrich_summarize",
        "PostRetrieval_MemoryOS_multi_query",
        "PreRetrieval_Mem0g_semantic",
        "PostInsert_TiM_none",
    ]

    print("配置解析测试:")
    print("=" * 60)
    for config in test_configs:
        encoding = get_visual_encoding(config)
        print(
            f"{config:40s} → {encoding['system']:10s} {encoding['strategy']:12s} "
            f"{encoding['marker']} {encoding['color']}"
        )

    print("\n" + "=" * 60)
    print("视觉编码示例图生成...")

    # 生成示例图
    import random

    strategies_metrics = {}
    for config in test_configs:
        strategies_metrics[config] = {i: 0.3 + random.random() * 0.2 for i in range(1, 6)}

    plot_comparison_with_encoding(
        strategies_metrics,
        "F1 Score",
        "F1 Score",
        "test_visual_encoding.png",
        title="Visual Encoding Example (Marker=System, Color=Strategy)",
    )

    print("✓ 生成示例图: test_visual_encoding.png")
    print("\n图例说明:")
    print("  ○ (圆形)   = Lsh Hash")
    print("  □ (正方形) = Queue-Segment")
    print("  △ (三角形) = Property Graph")
    print("\n  灰色 = none (基线)")
    print("  蓝色 = semantic/top_k")
    print("  绿色 = augment/enrich")
    print("  橙色 = multi_query/summarize")
    print("  红色 = compress/rewrite")
