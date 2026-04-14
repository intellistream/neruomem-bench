"""
美化补丁: 增强现有plotting.py的学术级图表输出

应用方式:
    1. 备份原文件: cp plotting.py plotting.py.bak
    2. 将本文件内容添加到plotting.py的顶部

Features:
    - 色盲友好调色板
    - Times字体
    - PDF输出支持
    - 统一学术风格
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

try:
    import seaborn as sns

    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Warning: seaborn not installed. Using default styles.")

# ============================================================================
# 学术级配置
# ============================================================================

# 设置seaborn风格 (如果可用)
if HAS_SEABORN:
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)

# 全局matplotlib配置
ACADEMIC_RC_PARAMS = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "gray",
    "figure.titlesize": 14,
    "lines.linewidth": 2.0,
    "lines.markersize": 6,
    "grid.alpha": 0.4,
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.format": "pdf",  # 默认PDF
}

# 色盲友好调色板 (来自Wong 2011 + ColorBrewer)
COLORBLIND_PALETTE = [
    "#0173B2",  # 蓝色
    "#029E73",  # 绿色
    "#D55E00",  # 橙色
    "#CC78BC",  # 紫色
    "#ECE133",  # 黄色
    "#56B4E9",  # 浅蓝
    "#CA9161",  # 棕色
    "#949494",  # 灰色
    "#FBAFE4",  # 粉色
    "#E69F00",  # 金色
]

# 替换默认颜色
DEFAULT_COLORS = COLORBLIND_PALETTE


# ============================================================================
# 辅助函数
# ============================================================================


def enable_academic_style():
    """启用学术级图表风格"""
    plt.rcParams.update(ACADEMIC_RC_PARAMS)
    print("✓ Academic style enabled (Times font, colorblind-friendly palette)")


def save_figure_academic(fig_or_path, output_path=None, **kwargs):
    """
    保存图表为PDF + PNG双格式

    Args:
        fig_or_path: matplotlib Figure对象或输出路径
        output_path: 如果第一个参数是Figure，则需要此参数
        **kwargs: 传递给savefig的额外参数
    """
    from pathlib import Path

    if isinstance(fig_or_path, plt.Figure):
        fig = fig_or_path
        path = Path(output_path)
    else:
        fig = plt.gcf()
        path = Path(fig_or_path)

    # 默认参数
    save_kwargs = {
        "dpi": 300,
        "bbox_inches": "tight",
    }
    save_kwargs.update(kwargs)

    # 保存PDF (论文用)
    pdf_path = path.with_suffix(".pdf")
    fig.savefig(pdf_path, format="pdf", **save_kwargs)

    # 保存PNG (预览用)
    png_path = path.with_suffix(".png")
    fig.savefig(png_path, format="png", **save_kwargs)

    return pdf_path, png_path


def get_strategy_color(strategy_name: str) -> str:
    """
    为策略分配一致的颜色

    优先级系统:
        - inverted_vectorstore: 蓝色 (最佳)
        - feature_queue_*: 绿色系 (次优)
        - fifo/segment: 灰色/棕色 (基线)
        - graph/hash: 紫色/粉色系
    """
    color_map = {
        "inverted_vectorstore": "#0173B2",
        "feature_queue_vectorstore": "#029E73",
        "feature_summary_vectorstore": "#D55E00",
        "feature_queue_summary": "#ECE133",
        "semantic_inverted_kg": "#CC78BC",
        "fifo_queue": "#949494",
        "segment": "#CA9161",
        "lsh_hash": "#FBAFE4",
        "linknote_graph": "#56B4E9",
        "property_graph": "#E69F00",
    }

    # 精确匹配
    if strategy_name in color_map:
        return color_map[strategy_name]

    # 模糊匹配
    for key, color in color_map.items():
        if key in strategy_name.lower():
            return color

    # 默认从调色板选择
    hash_val = sum(ord(c) for c in strategy_name)
    return COLORBLIND_PALETTE[hash_val % len(COLORBLIND_PALETTE)]


# ============================================================================
# 增强版绘图函数
# ============================================================================


def plot_comparison_enhanced(
    strategies_metrics: dict[str, dict[int, float]],
    metric_name: str,
    ylabel: str,
    output_path,
    title: str | None = None,
    figsize: tuple = (10, 6),
    enable_pdf: bool = True,
    highlight_best: bool = True,
):
    """
    增强版对比图 - 使用学术风格

    新特性:
        - 色盲友好配色
        - Times字体
        - 自动高亮最佳配置
        - PDF双输出
    """
    # 应用学术风格
    with plt.rc_context(ACADEMIC_RC_PARAMS):
        fig, ax = plt.subplots(figsize=figsize)

        # 计算每个策略的平均值 (用于排序和高亮)
        avg_values = {}
        for strategy, metrics in strategies_metrics.items():
            if metrics:
                avg_values[strategy] = np.mean(list(metrics.values()))

        best_strategy = max(avg_values, key=avg_values.get) if avg_values else None

        for strategy, metrics in strategies_metrics.items():
            if not metrics:
                continue

            rounds = sorted(metrics.keys())
            values = [metrics[r] for r in rounds]

            # 使用策略专属颜色
            color = get_strategy_color(strategy)

            # 高亮最佳策略
            linewidth = 2.5 if (highlight_best and strategy == best_strategy) else 2.0
            alpha = 0.95 if (highlight_best and strategy == best_strategy) else 0.75

            ax.plot(
                rounds,
                values,
                marker="o",
                label=strategy,
                color=color,
                linewidth=linewidth,
                markersize=7 if strategy == best_strategy else 6,
                alpha=alpha,
            )

        ax.set_xlabel("Test Round", fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_title(title or f"{metric_name} by Round", fontweight="bold", pad=15)

        ax.grid(True, alpha=0.4, linestyle="--", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.legend(fontsize=9, loc="best", framealpha=0.95, edgecolor="gray")

        # 设置x轴刻度
        all_rounds = set()
        for m in strategies_metrics.values():
            all_rounds.update(m.keys())
        if all_rounds:
            ax.set_xticks(sorted(all_rounds))

        plt.tight_layout()

        # 保存
        if enable_pdf:
            save_figure_academic(fig, output_path)
        else:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    # 启用学术风格
    enable_academic_style()

    # 示例数据
    test_data = {
        "inverted_vectorstore": {1: 0.42, 2: 0.40, 3: 0.38, 4: 0.37, 5: 0.36},
        "feature_queue_vectorstore": {1: 0.40, 2: 0.38, 3: 0.36, 4: 0.35, 5: 0.34},
        "fifo_queue": {1: 0.20, 2: 0.18, 3: 0.16, 4: 0.15, 5: 0.14},
    }

    # 生成对比图
    plot_comparison_enhanced(
        test_data,
        "F1 Score",
        "F1 Score",
        "test_comparison_enhanced.pdf",
        title="DataStructure F1 Comparison (Enhanced)",
    )

    print("✓ Test figure generated: test_comparison_enhanced.pdf/png")
