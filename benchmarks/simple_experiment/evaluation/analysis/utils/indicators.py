"""indicators.py — simple_experiment 专用评估指标

在 benchmarks.evaluation.analysis.utils.indicators 基础上扩展：
- SimpleLoCoMoEvaluator: 处理字符串类别（"factual"、"temporal" 等）
"""

from __future__ import annotations

from benchmarks.evaluation.analysis.utils.indicators import (  # noqa: F401 (re-export)
    BaseEvaluator,
    ExactMatchEvaluator,
    GenericF1Evaluator,
    LoCoMoEvaluator,
    calc_insert_time,
    calc_retrieval_time,
    exact_match,
    f1_score,
    f1_multi,
    get_evaluator,
    list_evaluators,
    normalize_answer,
    register_evaluator,
)


class SimpleLoCoMoEvaluator(BaseEvaluator):
    """
    Simple Experiment 用评估器。

    simple_experiment 输出的 category 字段为字符串（"factual"、"temporal" 等），
    不是 LoCoMo 原始的整数（1-5）。
    本评估器对字符串类别直接使用 F1 评分（不做 Category 1/3/5 的特殊处理）。

    如需更精细的 category-aware 处理，继承本类并覆写 evaluate_single。
    """

    def evaluate_single(
        self, prediction: str, reference: str, category=None, **kwargs
    ) -> float:
        return f1_score(prediction, reference)


# 注册到全局注册表
register_evaluator("simple_locomo", SimpleLoCoMoEvaluator)
