"""
数据验证工具

职责:
- 验证数据目录结构
- 检查数据完整性
- 自动发现可用的实验目录
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def discover_experiment_dirs(
    base_dir: Path | str,
    prefix: str = "PostInsert_",
) -> list[str]:
    """
    自动发现符合前缀的实验目录

    Args:
        base_dir: 数据基础目录
        prefix: 目录前缀（如 PostInsert_, PreInsert_）

    Returns:
        匹配的目录名列表
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        return []

    dirs = []
    for item in base_path.iterdir():
        if item.is_dir() and item.name.startswith(prefix):
            dirs.append(item.name)

    return sorted(dirs)


def discover_tasks(strategy_dir: Path | str) -> list[str]:
    """
    自动发现目录下的task（从JSON文件名解析）

    Args:
        strategy_dir: 策略目录路径

    Returns:
        task名称列表（如 ['conv-26', 'conv-30', ...]）
    """
    strategy_path = Path(strategy_dir)
    if not strategy_path.exists():
        return []

    tasks = set()
    for json_file in strategy_path.glob("*.json"):
        # 文件名格式: conv-26_1014.json -> task = conv-26
        name = json_file.stem
        if "_" in name:
            task = name.rsplit("_", 1)[0]
            tasks.add(task)

    return sorted(tasks)


def discover_rounds(data: dict) -> list[int]:
    """
    从数据中自动发现轮次

    Args:
        data: 加载的JSON数据

    Returns:
        轮次索引列表
    """
    test_results = data.get("test_results", [])
    rounds = []
    for test in test_results:
        round_idx = test.get("test_index")
        if round_idx is not None:
            rounds.append(round_idx)
    return sorted(rounds)


def validate_task_data(data: dict) -> dict[str, Any]:
    """
    验证单个task数据的完整性

    Args:
        data: 加载的JSON数据

    Returns:
        验证结果字典
    """
    result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "stats": {},
    }

    # 检查必要字段
    required_fields = ["test_results"]
    for field in required_fields:
        if field not in data:
            result["valid"] = False
            result["errors"].append(f"缺少必要字段: {field}")

    if not result["valid"]:
        return result

    # 检查test_results
    test_results = data.get("test_results", [])
    result["stats"]["round_count"] = len(test_results)

    if len(test_results) == 0:
        result["valid"] = False
        result["errors"].append("test_results为空")
        return result

    # 检查每个轮次
    total_questions = 0
    for test in test_results:
        questions = test.get("questions", [])
        total_questions += len(questions)

        if len(questions) == 0:
            result["warnings"].append(f"轮次 {test.get('test_index')} 没有问题")

    result["stats"]["total_questions"] = total_questions
    result["stats"]["rounds"] = discover_rounds(data)

    # 检查timing数据
    timing = data.get("timing_summary", {})
    result["stats"]["has_insert_timing"] = "insert_timings" in timing
    result["stats"]["has_retrieval_timing"] = "retrieval_timings" in timing

    return result


def validate_experiment_dir(
    strategy_dir: Path | str,
    min_tasks: int = 1,
) -> dict[str, Any]:
    """
    验证实验目录的完整性

    Args:
        strategy_dir: 策略目录路径
        min_tasks: 最少task数量

    Returns:
        验证结果
    """
    strategy_path = Path(strategy_dir)
    result = {
        "valid": True,
        "dir": str(strategy_path),
        "errors": [],
        "warnings": [],
        "tasks": [],
        "stats": {},
    }

    if not strategy_path.exists():
        result["valid"] = False
        result["errors"].append(f"目录不存在: {strategy_path}")
        return result

    # 发现tasks
    tasks = discover_tasks(strategy_path)
    result["tasks"] = tasks
    result["stats"]["task_count"] = len(tasks)

    if len(tasks) < min_tasks:
        result["valid"] = False
        result["errors"].append(f"task数量不足: {len(tasks)} < {min_tasks}")

    # 验证每个task
    valid_tasks = 0
    for task in tasks:
        json_files = list(strategy_path.glob(f"{task}_*.json"))
        if json_files:
            try:
                with open(json_files[0], encoding="utf-8") as f:
                    data = json.load(f)
                task_result = validate_task_data(data)
                if task_result["valid"]:
                    valid_tasks += 1
                else:
                    result["warnings"].extend([f"Task {task}: {e}" for e in task_result["errors"]])
            except Exception as e:
                result["warnings"].append(f"Task {task} 加载失败: {e}")

    result["stats"]["valid_task_count"] = valid_tasks

    return result


def print_validation_report(results: list[dict], verbose: bool = False):
    """打印验证报告"""
    print("=" * 60)
    print("数据验证报告")
    print("=" * 60)

    valid_count = sum(1 for r in results if r["valid"])
    print(f"\n总计: {len(results)} 个目录, {valid_count} 个有效")

    for r in results:
        status = "✓" if r["valid"] else "✗"
        dir_name = Path(r["dir"]).name
        task_count = r["stats"].get("task_count", 0)
        print(f"\n{status} {dir_name}")
        print(f"    Tasks: {task_count}")

        if r["errors"]:
            for e in r["errors"]:
                print(f"    ❌ {e}")

        if verbose and r["warnings"]:
            for w in r["warnings"]:
                print(f"    ⚠ {w}")

    print()
