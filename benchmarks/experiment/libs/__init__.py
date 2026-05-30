"""libs 包 - 记忆实验算子库

提供完整的 Pipeline 算子组件：

顶层算子（Pipeline 节点）：
- MemorySource: 数据源，从数据集读取对话轮次
- MemoryInsert: 记忆插入，调用记忆服务写入
- MemoryRetrieval: 记忆检索，调用记忆服务查询
- MemoryEvaluation: 记忆评估，使用 LLM 生成答案
- MemorySink: 结果收集，保存实验输出
- PipelineCaller: 主编排算子，协调子 Pipeline

四阶段可扩展 Operator：
- PreInsert: 插入前处理（数据清洗、转换、特征提取）
- PostInsert: 插入后处理（蒸馏、反思、遗忘等）
- PreRetrieval: 检索前处理（查询优化、Embedding）
- PostRetrieval: 检索后处理（重排、过滤、合并）

记忆数据结构（datastructure）：
- BaseIndex / IndexFactory: 索引抽象和工厂
- SimpleCollection: 统一数据容器（原始数据 + 多索引）
- BaseMemoryService / MemoryServiceRegistry: 记忆服务抽象和注册表
- lsh/: LSH 近似匹配示例（LSHIndex + LSHHashService）
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from benchmarks.experiment.libs.memory_evaluation import MemoryEvaluation as MemoryEvaluation
    from benchmarks.experiment.libs.memory_insert import MemoryInsert as MemoryInsert
    from benchmarks.experiment.libs.memory_retrieval import MemoryRetrieval as MemoryRetrieval
    from benchmarks.experiment.libs.memory_sink import MemorySink as MemorySink
    from benchmarks.experiment.libs.memory_source import MemorySource as MemorySource
    from benchmarks.experiment.libs.pipeline_caller import PipelineCaller as PipelineCaller
    from benchmarks.experiment.libs.post_insert import PostInsert as PostInsert
    from benchmarks.experiment.libs.post_retrieval import PostRetrieval as PostRetrieval
    from benchmarks.experiment.libs.pre_insert import PreInsert as PreInsert
    from benchmarks.experiment.libs.pre_retrieval import PreRetrieval as PreRetrieval


_EXPORT_MAP = {
    "MemorySource": ("benchmarks.experiment.libs.memory_source", "MemorySource"),
    "MemoryInsert": ("benchmarks.experiment.libs.memory_insert", "MemoryInsert"),
    "MemoryRetrieval": ("benchmarks.experiment.libs.memory_retrieval", "MemoryRetrieval"),
    "MemoryEvaluation": ("benchmarks.experiment.libs.memory_evaluation", "MemoryEvaluation"),
    "MemorySink": ("benchmarks.experiment.libs.memory_sink", "MemorySink"),
    "PipelineCaller": ("benchmarks.experiment.libs.pipeline_caller", "PipelineCaller"),
    "PreInsert": ("benchmarks.experiment.libs.pre_insert", "PreInsert"),
    "PostInsert": ("benchmarks.experiment.libs.post_insert", "PostInsert"),
    "PreRetrieval": ("benchmarks.experiment.libs.pre_retrieval", "PreRetrieval"),
    "PostRetrieval": ("benchmarks.experiment.libs.post_retrieval", "PostRetrieval"),
}


def __getattr__(name: str) -> Any:
    if name in _EXPORT_MAP:
        module_name, attr_name = _EXPORT_MAP[name]
        module = __import__(module_name, fromlist=[attr_name])
        return getattr(module, attr_name)
    raise AttributeError(f"module 'benchmarks.experiment.libs' has no attribute {name!r}")

__all__ = [
    # 顶层算子
    "MemorySource",
    "MemoryInsert",
    "MemoryRetrieval",
    "MemoryEvaluation",
    "MemorySink",
    "PipelineCaller",
    # 四阶段 Operator
    "PreInsert",
    "PostInsert",
    "PreRetrieval",
    "PostRetrieval",
]
