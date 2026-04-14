"""data_loader.py — simple_experiment 专用数据加载器

直接复用 benchmarks.evaluation.analysis.utils.data_loader 的全部实现。
simple_experiment 输出 JSON 结构与 experiment 完全兼容，无需覆写。
"""

from benchmarks.evaluation.analysis.utils.data_loader import (  # noqa: F401
    CategoryAnalyzer,
    DataLoader,
    RoundAnalyzer,
    TaskData,
    TimeBreakdownAnalyzer,
)
