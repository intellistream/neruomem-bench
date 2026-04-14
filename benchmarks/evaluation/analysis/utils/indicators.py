"""
评估指标计算模块

职责:
- 提供各种评估指标的计算函数
- 支持不同数据集的特异性评估方式
- 指标注册机制便于扩展
"""

from __future__ import annotations

import string
from abc import ABC, abstractmethod
from collections import Counter

import numpy as np
import regex

try:
    from nltk.stem import PorterStemmer

    _ps = PorterStemmer()
except ImportError:
    _ps = None


# ============================================================================
# 基础文本处理
# ============================================================================


def normalize_answer(s: str) -> str:
    """标准化答案文本（去标点、小写、去停用词）"""
    if not isinstance(s, str):
        s = str(s)
    s = s.replace(",", "")
    exclude = set(string.punctuation)
    s = "".join(ch for ch in s if ch not in exclude)
    s = regex.sub(r"\b(a|an|the|and)\b", " ", s.lower())
    return " ".join(s.split())


def tokenize_with_stem(text: str) -> list[str]:
    """分词并进行词干提取"""
    if _ps is None:
        return normalize_answer(text).split()
    return [_ps.stem(w) for w in normalize_answer(text).split()]


# ============================================================================
# 基础指标函数
# ============================================================================


def f1_score(prediction: str, ground_truth: str) -> float:
    """
    计算token级别的F1分数

    Args:
        prediction: 预测答案
        ground_truth: 标准答案

    Returns:
        F1分数 (0.0 - 1.0)
    """
    pred_tokens = tokenize_with_stem(prediction)
    gt_tokens = tokenize_with_stem(ground_truth)

    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0.0

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)

    return (2 * precision * recall) / (precision + recall)


def exact_match(prediction: str, ground_truth: str) -> float:
    """
    精确匹配分数

    Returns:
        1.0 如果完全匹配，否则 0.0
    """
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0


def f1_multi(prediction: str, ground_truth: str, delimiter: str = ",") -> float:
    """
    处理多答案的F1计算（用于有多个正确答案的情况）

    Args:
        prediction: 预测答案（可能包含多个）
        ground_truth: 标准答案（逗号分隔的多个答案）
        delimiter: 分隔符

    Returns:
        平均F1分数
    """
    preds = [p.strip() for p in prediction.split(delimiter)]
    gts = [g.strip() for g in ground_truth.split(delimiter)]

    # 对每个ground truth，找与它最匹配的prediction
    scores = [max(f1_score(p, gt) for p in preds) for gt in gts]

    return float(np.mean(scores)) if scores else 0.0


# ============================================================================
# 数据集特异性评估器
# ============================================================================


class BaseEvaluator(ABC):
    """评估器基类"""

    @abstractmethod
    def evaluate_single(self, prediction: str, reference: str, **kwargs) -> float:
        """评估单个问答对"""

    def evaluate_batch(self, qa_pairs: list[dict]) -> list[float]:
        """批量评估"""
        return [
            self.evaluate_single(
                qa.get("predicted_answer", ""), qa.get("reference_answer", ""), **qa
            )
            for qa in qa_pairs
        ]


class LoCoMoEvaluator(BaseEvaluator):
    """
    LoCoMo数据集专用评估器

    特殊处理:
    - Category 1: 多答案F1
    - Category 3: 清理分号后注释
    - Category 5: 判断是否选择(b)或识别"信息未提及"
    """

    def evaluate_single(
        self, prediction: str, reference: str, category: int = 1, **kwargs
    ) -> float:
        """
        评估单个LoCoMo问答对

        Args:
            prediction: 预测答案
            reference: 参考答案
            category: 问题类别 (1-5)
        """
        pred = prediction.strip()
        answer = str(reference)

        # Category 5: 特殊处理 - 判断"信息未提及"
        if category == 5:
            return self._evaluate_category5(pred)

        # Category 3: 清理分号后的注释部分
        if category == 3:
            answer = answer.split(";")[0].strip()
            pred = pred.split(";")[0].strip()

        # Category 1: 多答案处理
        if category == 1 and "," in answer:
            return f1_multi(pred, answer)

        # 默认: 标准F1
        return f1_score(pred, answer)

    def _evaluate_category5(self, prediction: str) -> float:
        """Category 5 特殊评估：判断是否正确识别"信息未提及"。"""
        pred_lower = prediction.lower()

        selected_b = any(
            pattern in pred_lower
            for pattern in ["(b)", "option b", "answer is b", "select b", "choice b"]
        )

        is_not_mentioned = any(
            keyword in pred_lower
            for keyword in [
                "not mentioned",
                "no information",
                "not in the conversation",
                "cannot be determined",
            ]
        )

        return 1.0 if (selected_b or is_not_mentioned) else 0.0


class GenericF1Evaluator(BaseEvaluator):
    """通用F1评估器"""

    def evaluate_single(self, prediction: str, reference: str, **kwargs) -> float:
        return f1_score(prediction, reference)


class ExactMatchEvaluator(BaseEvaluator):
    """精确匹配评估器"""

    def evaluate_single(self, prediction: str, reference: str, **kwargs) -> float:
        return exact_match(prediction, reference)


# ============================================================================
# 评估器注册表
# ============================================================================


_EVALUATOR_REGISTRY: dict[str, type[BaseEvaluator]] = {
    "locomo": LoCoMoEvaluator,
    "generic_f1": GenericF1Evaluator,
    "exact_match": ExactMatchEvaluator,
}


def register_evaluator(name: str, evaluator_class: type[BaseEvaluator]):
    """注册新的评估器"""
    _EVALUATOR_REGISTRY[name] = evaluator_class


def get_evaluator(name: str) -> BaseEvaluator:
    """获取评估器实例"""
    if name not in _EVALUATOR_REGISTRY:
        raise ValueError(f"未知的评估器: {name}. 可用: {list(_EVALUATOR_REGISTRY.keys())}")
    return _EVALUATOR_REGISTRY[name]()


def list_evaluators() -> list[str]:
    """列出所有可用的评估器"""
    return list(_EVALUATOR_REGISTRY.keys())


# ============================================================================
# 时间指标计算
# ============================================================================


def calc_insert_time(timing_detail: dict) -> float:
    """计算单条插入的总时间（毫秒）"""
    return (
        timing_detail.get("pre_insert_ms", 0)
        + timing_detail.get("memory_insert_ms", 0)
        + timing_detail.get("post_insert_ms", 0)
    )


def calc_retrieval_time(timing_detail: dict) -> float:
    """计算单次检索的总时间（毫秒）"""
    return (
        timing_detail.get("pre_retrieval_ms", 0)
        + timing_detail.get("memory_retrieval_ms", 0)
        + timing_detail.get("post_retrieval_ms", 0)
    )
