"""
数据加载工具

职责:
- 加载不同格式的实验数据
- 自动检测数据格式
- 提供统一的数据访问接口
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from .indicators import BaseEvaluator, calc_insert_time, calc_retrieval_time, get_evaluator
from .validators import discover_rounds, discover_tasks


class TaskData:
    """单个Task的数据封装"""

    def __init__(self, data: dict, task_name: str = ""):
        self._data = data
        self.task_name = task_name

    @property
    def raw(self) -> dict:
        """原始数据"""
        return self._data

    @property
    def test_results(self) -> list[dict]:
        """测试结果列表"""
        return self._data.get("test_results", [])

    @property
    def rounds(self) -> list[int]:
        """自动检测的轮次列表"""
        return discover_rounds(self._data)

    @property
    def timing_summary(self) -> dict:
        """时间统计"""
        return self._data.get("timing_summary", {})

    def get_questions_by_round(self, round_idx: int) -> list[dict]:
        """获取指定轮次的问题"""
        for test in self.test_results:
            if test.get("test_index") == round_idx:
                return test.get("questions", [])
        return []

    def get_question_range(self, round_idx: int) -> tuple[int, int]:
        """获取指定轮次的问题范围（用于timing对应）"""
        for test in self.test_results:
            if test.get("test_index") == round_idx:
                q_range = test.get("question_range", {})
                start = q_range.get("start", 1) - 1  # 转为0-based
                end = q_range.get("end", 1)
                return start, end
        return 0, 0


class DataLoader:
    """通用数据加载器"""

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)

    def load_task(self, strategy: str, task: str) -> TaskData | None:
        """
        加载单个task的数据

        Args:
            strategy: 策略目录名
            task: task名称

        Returns:
            TaskData对象，加载失败返回None
        """
        strategy_dir = self.base_dir / strategy
        json_files = list(strategy_dir.glob(f"{task}_*.json"))

        if not json_files:
            return None

        try:
            with open(json_files[0], encoding="utf-8") as f:
                data = json.load(f)
            return TaskData(data, task)
        except Exception as e:
            print(f"Error loading {json_files[0]}: {e}")
            return None

    def iter_tasks(
        self, strategy: str, tasks_filter: list[str] | None = None
    ) -> Iterator[TaskData]:
        """
        迭代策略下的所有task

        Args:
            strategy: 策略目录名
            tasks_filter: 可选的任务过滤列表（如 ['conv-26', 'conv-30']）
                         如果提供，只加载这些任务；否则加载所有任务

        Yields:
            TaskData对象
        """
        strategy_dir = self.base_dir / strategy
        tasks = discover_tasks(strategy_dir)

        # 应用任务过滤
        if tasks_filter is not None:
            tasks = [t for t in tasks if t in tasks_filter]

        for task in tasks:
            task_data = self.load_task(strategy, task)
            if task_data is not None:
                yield task_data

    def get_tasks(self, strategy: str, tasks_filter: list[str] | None = None) -> list[str]:
        """获取策略下的所有task名称

        Args:
            strategy: 策略目录名
            tasks_filter: 可选的任务过滤列表

        Returns:
            任务名称列表
        """
        tasks = discover_tasks(self.base_dir / strategy)
        if tasks_filter is not None:
            tasks = [t for t in tasks if t in tasks_filter]
        return tasks


class RoundAnalyzer:
    """
    轮次分析器 - 按轮次聚合指标
    """

    def __init__(self, evaluator: BaseEvaluator | str = "generic_f1"):
        if isinstance(evaluator, str):
            self.evaluator = get_evaluator(evaluator)
        else:
            self.evaluator = evaluator

    def analyze_f1_by_round(self, task_data: TaskData) -> dict[int, float]:
        """
        按轮次分析F1分数

        Returns:
            {round_idx: avg_f1}
        """
        round_scores = {}

        for round_idx in task_data.rounds:
            questions = task_data.get_questions_by_round(round_idx)
            if not questions:
                continue

            scores = self.evaluator.evaluate_batch(questions)
            round_scores[round_idx] = float(np.mean(scores)) if scores else 0.0

        return round_scores

    def analyze_insert_time_by_round(self, task_data: TaskData) -> dict[int, float]:
        """
        按轮次分析插入时间

        Returns:
            {round_idx: avg_insert_time_ms}
        """
        timing = task_data.timing_summary.get("insert_timings", {})
        details = timing.get("details", [])

        round_times = {}

        if not details:
            return round_times

        # ------------------------------------------------------------------
        # LongMemEval 兼容：insert_timings.details 往往是“每次插入(对话/记忆条目)”的明细，
        # 但 test_results 里的 question_range 是“题目范围”，两者不是同一索引空间。
        # 此时改用 dialogs_inserted_count（通常为累计插入进度）对 details 进行分段。
        # 为适配 dialogs_inserted_count 可能按 message 计数的情况，按 total_messages 进行缩放。
        # ------------------------------------------------------------------
        try:
            test_results = task_data.test_results
            has_dialog_counts = bool(test_results) and all(
                isinstance(t.get("dialogs_inserted_count"), int) for t in test_results
            )
            if has_dialog_counts:
                # dialogs_inserted_count 通常为累计值（随 test_index 增长）
                ordered = sorted(
                    (
                        (int(t.get("test_index")), int(t.get("dialogs_inserted_count")))
                        for t in test_results
                        if t.get("test_index") is not None
                    ),
                    key=lambda x: x[0],
                )

                if ordered:
                    # 使用 dataset_statistics.total_messages 作为分母（更接近 dialogs_inserted_count 的计数口径）
                    ds_stats = task_data.raw.get("dataset_statistics", {})
                    total_messages = ds_stats.get("total_messages")
                    denom = None
                    if isinstance(total_messages, int) and total_messages > 0:
                        denom = total_messages
                    else:
                        # 回退：使用最大累计值
                        denom = max(c for _, c in ordered) or None

                    if denom:
                        n = len(details)
                        # 将累计计数映射到 details 的累计索引
                        cum_indices: list[tuple[int, int]] = []
                        last_idx = 0
                        for round_idx, cum_count in ordered:
                            # clamp to [0, n]
                            idx = int(round((cum_count / denom) * n))
                            if idx < last_idx:
                                idx = last_idx
                            if idx > n:
                                idx = n
                            cum_indices.append((round_idx, idx))
                            last_idx = idx

                        prev = 0
                        for round_idx, end in cum_indices:
                            seg = details[prev:end]
                            if seg:
                                times = [calc_insert_time(t) for t in seg]
                                round_times[round_idx] = float(np.mean(times)) if times else 0.0
                            else:
                                round_times[round_idx] = 0.0
                            prev = end

                        # 如果最后一段没被覆盖（极少数情况），不强行追加额外轮次
                        return round_times
        except Exception:
            # 任意异常时回退到旧逻辑，避免影响其它数据集
            pass

        for round_idx in task_data.rounds:
            start_idx, end_idx = task_data.get_question_range(round_idx)

            if start_idx < len(details) and end_idx <= len(details):
                round_timings = details[start_idx:end_idx]
                times = [calc_insert_time(t) for t in round_timings]
                round_times[round_idx] = float(np.mean(times)) if times else 0.0

        return round_times

    def analyze_retrieval_time_by_round(self, task_data: TaskData) -> dict[int, float]:
        """
        按轮次分析检索时间

        Returns:
            {round_idx: retrieval_time_ms}
        """
        timing = task_data.timing_summary.get("retrieval_timings", {})
        details = timing.get("details", [])

        round_times = {}
        for detail in details:
            round_idx = detail.get("test_index")
            if round_idx is not None:
                round_times[round_idx] = calc_retrieval_time(detail)

        return round_times

    def aggregate_across_tasks(
        self,
        loader: DataLoader,
        strategy: str,
        metric_func: str = "f1",
        tasks_filter: list[str] | None = None,
    ) -> dict[int, float]:
        """
        跨所有task聚合某个指标

        Args:
            loader: 数据加载器
            strategy: 策略目录名
            metric_func: 指标类型 ("f1", "insert_time", "retrieval_time")
            tasks_filter: 可选的任务过滤列表

        Returns:
            {round_idx: avg_metric}
        """
        from collections import defaultdict

        all_round_metrics = defaultdict(list)

        # 选择分析函数
        if metric_func == "f1":
            analyze_fn = self.analyze_f1_by_round
        elif metric_func == "insert_time":
            analyze_fn = self.analyze_insert_time_by_round
        elif metric_func == "retrieval_time":
            analyze_fn = self.analyze_retrieval_time_by_round
        else:
            raise ValueError(f"未知的指标类型: {metric_func}")

        for task_data in loader.iter_tasks(strategy, tasks_filter):
            round_metrics = analyze_fn(task_data)
            for round_idx, value in round_metrics.items():
                all_round_metrics[round_idx].append(value)

        # 计算每个轮次的平均值
        avg_metrics = {}
        for round_idx in sorted(all_round_metrics.keys()):
            values = all_round_metrics[round_idx]
            avg_metrics[round_idx] = float(np.mean(values)) if values else 0.0

        return avg_metrics


class CategoryAnalyzer:
    """
    Category分析器 - 按问题类别分析F1

    Category说明:
    - Category 1: 多答案问题
    - Category 2: 时间相关问题
    - Category 3: 需清理注释的问题
    - Category 4: 标准问题
    """

    def __init__(self, evaluator: BaseEvaluator | str = "generic_f1"):
        if isinstance(evaluator, str):
            self.evaluator = get_evaluator(evaluator)
        else:
            self.evaluator = evaluator

    def analyze_f1_by_category(self, task_data: TaskData) -> dict[int, float]:
        """
        按Category分析F1分数

        Returns:
            {category: avg_f1}
        """
        from collections import defaultdict

        category_scores = defaultdict(list)

        for test in task_data.test_results:
            questions = test.get("questions", [])
            for q in questions:
                category = q.get("category", 0)
                score = self.evaluator.evaluate_single(
                    q.get("predicted_answer", ""), q.get("reference_answer", ""), **q
                )
                category_scores[category].append(score)

        result = {}
        for cat in sorted(category_scores.keys()):
            scores = category_scores[cat]
            result[cat] = float(np.mean(scores)) if scores else 0.0

        return result

    def analyze_f1_by_category_and_round(self, task_data: TaskData) -> dict[int, dict[int, float]]:
        """
        按Category和轮次分析F1分数

        Returns:
            {round_idx: {category: avg_f1}}
        """
        from collections import defaultdict

        round_category_scores = defaultdict(lambda: defaultdict(list))

        for test in task_data.test_results:
            round_idx = test.get("test_index", 0)
            questions = test.get("questions", [])
            for q in questions:
                category = q.get("category", 0)
                score = self.evaluator.evaluate_single(
                    q.get("predicted_answer", ""), q.get("reference_answer", ""), **q
                )
                round_category_scores[round_idx][category].append(score)

        result = {}
        for round_idx in sorted(round_category_scores.keys()):
            result[round_idx] = {}
            for cat in sorted(round_category_scores[round_idx].keys()):
                scores = round_category_scores[round_idx][cat]
                result[round_idx][cat] = float(np.mean(scores)) if scores else 0.0

        return result

    def get_category_distribution(self, task_data: TaskData) -> dict[int, int]:
        """
        获取Category分布

        Returns:
            {category: count}
        """
        from collections import Counter

        categories = []
        for test in task_data.test_results:
            for q in test.get("questions", []):
                categories.append(q.get("category", 0))

        return dict(Counter(categories))

    def aggregate_across_tasks(
        self,
        loader: DataLoader,
        strategy: str,
        tasks_filter: list[str] | None = None,
    ) -> dict[int, float]:
        """
        跨所有task聚合Category F1

        Args:
            loader: 数据加载器
            strategy: 策略目录名
            tasks_filter: 可选的任务过滤列表

        Returns:
            {category: avg_f1}
        """
        from collections import defaultdict

        all_category_metrics = defaultdict(list)

        for task_data in loader.iter_tasks(strategy, tasks_filter):
            category_metrics = self.analyze_f1_by_category(task_data)
            for cat, value in category_metrics.items():
                all_category_metrics[cat].append(value)

        avg_metrics = {}
        for cat in sorted(all_category_metrics.keys()):
            values = all_category_metrics[cat]
            avg_metrics[cat] = float(np.mean(values)) if values else 0.0

        return avg_metrics


class TimeBreakdownAnalyzer:
    """
    时间分解分析器 - 分析pre/memory/post三阶段时间
    """

    def analyze_insert_breakdown(self, task_data: TaskData) -> dict[str, float]:
        """
        分析插入时间的三阶段分解

        Returns:
            {"pre": avg_ms, "memory": avg_ms, "post": avg_ms, "total": avg_ms}
        """
        timing = task_data.timing_summary.get("insert_timings", {})
        details = timing.get("details", [])

        if not details:
            return {"pre": 0.0, "memory": 0.0, "post": 0.0, "total": 0.0}

        pre_times = [d.get("pre_insert_ms", 0) for d in details]
        memory_times = [d.get("memory_insert_ms", 0) for d in details]
        post_times = [d.get("post_insert_ms", 0) for d in details]

        pre_avg = float(np.mean(pre_times))
        memory_avg = float(np.mean(memory_times))
        post_avg = float(np.mean(post_times))

        return {
            "pre": pre_avg,
            "memory": memory_avg,
            "post": post_avg,
            "total": pre_avg + memory_avg + post_avg,
        }

    def analyze_retrieval_breakdown(self, task_data: TaskData) -> dict[str, float]:
        """
        分析检索时间的三阶段分解

        Returns:
            {"pre": avg_ms, "memory": avg_ms, "post": avg_ms, "total": avg_ms}
        """
        timing = task_data.timing_summary.get("retrieval_timings", {})
        details = timing.get("details", [])

        if not details:
            return {"pre": 0.0, "memory": 0.0, "post": 0.0, "total": 0.0}

        pre_times = [d.get("pre_retrieval_ms", 0) for d in details]
        memory_times = [d.get("memory_retrieval_ms", 0) for d in details]
        post_times = [d.get("post_retrieval_ms", 0) for d in details]

        pre_avg = float(np.mean(pre_times))
        memory_avg = float(np.mean(memory_times))
        post_avg = float(np.mean(post_times))

        return {
            "pre": pre_avg,
            "memory": memory_avg,
            "post": post_avg,
            "total": pre_avg + memory_avg + post_avg,
        }

    def analyze_breakdown_by_round(
        self, task_data: TaskData, timing_type: str = "insert"
    ) -> dict[int, dict[str, float]]:
        """
        按轮次分析时间分解

        Args:
            timing_type: "insert" 或 "retrieval"

        Returns:
            {round_idx: {"pre": ms, "memory": ms, "post": ms, "total": ms}}
        """
        if timing_type == "insert":
            timing = task_data.timing_summary.get("insert_timings", {})
            pre_key, mem_key, post_key = "pre_insert_ms", "memory_insert_ms", "post_insert_ms"
        else:
            timing = task_data.timing_summary.get("retrieval_timings", {})
            pre_key, mem_key, post_key = (
                "pre_retrieval_ms",
                "memory_retrieval_ms",
                "post_retrieval_ms",
            )

        details = timing.get("details", [])
        result = {}

        if timing_type == "insert":
            # insert按question_range对应
            for round_idx in task_data.rounds:
                start_idx, end_idx = task_data.get_question_range(round_idx)
                if start_idx < len(details) and end_idx <= len(details):
                    round_details = details[start_idx:end_idx]
                    if round_details:
                        pre_avg = float(np.mean([d.get(pre_key, 0) for d in round_details]))
                        mem_avg = float(np.mean([d.get(mem_key, 0) for d in round_details]))
                        post_avg = float(np.mean([d.get(post_key, 0) for d in round_details]))
                        result[round_idx] = {
                            "pre": pre_avg,
                            "memory": mem_avg,
                            "post": post_avg,
                            "total": pre_avg + mem_avg + post_avg,
                        }
        else:
            # retrieval按test_index对应
            for detail in details:
                round_idx = detail.get("test_index")
                if round_idx is not None:
                    result[round_idx] = {
                        "pre": detail.get(pre_key, 0),
                        "memory": detail.get(mem_key, 0),
                        "post": detail.get(post_key, 0),
                        "total": detail.get(pre_key, 0)
                        + detail.get(mem_key, 0)
                        + detail.get(post_key, 0),
                    }

        return result

    def aggregate_across_tasks(
        self,
        loader: DataLoader,
        strategy: str,
        timing_type: str = "insert",
        tasks_filter: list[str] | None = None,
    ) -> dict[str, float]:
        """
        跨所有task聚合时间分解

        Args:
            loader: 数据加载器
            strategy: 策略目录名
            timing_type: 时间类型 ("insert" 或 "retrieval")
            tasks_filter: 可选的任务过滤列表

        Returns:
            {"pre": avg_ms, "memory": avg_ms, "post": avg_ms, "total": avg_ms}
        """
        from collections import defaultdict

        all_breakdowns = defaultdict(list)

        analyze_fn = (
            self.analyze_insert_breakdown
            if timing_type == "insert"
            else self.analyze_retrieval_breakdown
        )

        for task_data in loader.iter_tasks(strategy, tasks_filter):
            breakdown = analyze_fn(task_data)
            for key, value in breakdown.items():
                all_breakdowns[key].append(value)

        result = {}
        for key in ["pre", "memory", "post", "total"]:
            values = all_breakdowns.get(key, [])
            result[key] = float(np.mean(values)) if values else 0.0

        return result
