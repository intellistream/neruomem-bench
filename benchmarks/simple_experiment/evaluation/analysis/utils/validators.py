"""validators.py — simple_experiment 数据验证

直接复用 benchmarks.evaluation.analysis.utils.validators 的全部实现。
添加 discover_adapter_dirs 便利函数（无前缀约束，扫描所有子目录）。
"""

from __future__ import annotations

from pathlib import Path

from benchmarks.evaluation.analysis.utils.validators import (  # noqa: F401
    discover_experiment_dirs,
    discover_rounds,
    discover_tasks,
    print_validation_report,
    validate_experiment_dir,
    validate_task_data,
)


def discover_adapter_dirs(base_dir: Path | str) -> list[str]:
    """
    自动发现 base_dir 下所有子目录（即适配器目录）。

    simple_experiment 的策略目录不带任何固定前缀，
    目录名就是适配器名称（如 mem0、memgpt）。

    排除已知的非数据目录 output/。
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        return []

    exclude = {"output"}
    return sorted(
        item.name
        for item in base_path.iterdir()
        if item.is_dir() and item.name not in exclude
    )
