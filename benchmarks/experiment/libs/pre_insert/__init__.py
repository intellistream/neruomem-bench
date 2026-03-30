"""PreInsert Action 模块

提供记忆插入前预处理的策略框架。

开源版本包含：
- BasePreInsertAction: 抽象基类（开发接口）
- NoneAction: 透传策略（基本实现示例）
- PreInsert: Pipeline 算子
- PreInsertActionRegistry: Action 注册表

开发者可继承 BasePreInsertAction 实现自定义策略，例如：
- 关键词提取 (enrich.keyword)
- 摘要生成 (enrich.summarize)
- 事实抽取 (rewrite.fact_extract)
- 三元组提取 (rewrite.triplet_extract)
"""

from .base import BasePreInsertAction, PreInsertInput, PreInsertOutput
from .operator import PreInsert
from .registry import PreInsertActionRegistry

__all__ = [
    "PreInsert",
    "BasePreInsertAction",
    "PreInsertInput",
    "PreInsertOutput",
    "PreInsertActionRegistry",
]
