"""统一数据集加载器模块

典型用法：
    >>> from benchmarks.experiment.utils.dataloader import DataLoaderFactory
    >>> loader = DataLoaderFactory.create("locomo")
    >>> sessions = loader.sessions("conv-26")
"""

from .base import BaseDataLoader
from .factory import DataLoaderFactory

__all__ = [
    "BaseDataLoader",
    "DataLoaderFactory",
]
