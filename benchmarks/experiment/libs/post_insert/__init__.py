"""
PostInsert Action Module
========================

提供记忆插入后处理的策略框架。

开源版本包含：
- BasePostInsertAction: 抽象基类（开发接口）
- NoneAction: 透传策略（基本实现示例）
- PostInsert: Pipeline 算子
- PostInsertActionRegistry: Action 注册表

开发者可继承 BasePostInsertAction 实现自定义策略，例如：
- conflict_resolution: LLM CRUD / semantic consolidation
- decay_eviction: Forgetting curve / time decay
- structure_enrichment: Link evolution / graph construction
"""

from .base import BasePostInsertAction, PostInsertInput, PostInsertOutput
from .none_action import NoneAction
from .operator import PostInsert
from .registry import PostInsertActionRegistry, get_action

__all__ = [
    "PostInsert",
    "BasePostInsertAction",
    "PostInsertInput",
    "PostInsertOutput",
    "PostInsertActionRegistry",
    "get_action",
    "NoneAction",
]
