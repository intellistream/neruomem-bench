"""PreInsert Rewrite Actions

包含文本改写类的 PreInsert 算子:
- triplet_extract: 三元组提取 (TiM, HippoRAG)
"""

from .triplet_extract import TripleExtractAction

__all__ = ["TripleExtractAction"]
