"""PostInsert Conflict Resolution Actions

包含冲突消解类的 PostInsert 算子:
- semantic_consolidation: 语义巩固 / 记忆合并 (TiM, MemGPT)
"""

from .semantic_consolidation import SemanticConsolidationAction

__all__ = ["SemanticConsolidationAction"]
