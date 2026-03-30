"""PreRetrieval Action 模块

提供查询预处理的策略框架。

开源版本包含：
- BasePreRetrievalAction: 抽象基类
- NoneAction: 透传策略
- EmbeddingAction: 查询向量化
- PreRetrieval: Pipeline 算子
- PreRetrievalActionRegistry: Action 注册表
"""

from .base import (
    BasePreRetrievalAction,
    PreRetrievalInput,
    PreRetrievalOutput,
)
from .operator import PreRetrieval
from .registry import PreRetrievalActionRegistry

__all__ = [
    "PreRetrieval",
    "BasePreRetrievalAction",
    "PreRetrievalInput",
    "PreRetrievalOutput",
    "PreRetrievalActionRegistry",
]
