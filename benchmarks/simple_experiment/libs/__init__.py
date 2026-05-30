"""libs 包 — simple_experiment Pipeline 算子

黑盒测试三件套：
    SimpleMemoryAdd       替代 PreInsert + MemoryInsert + PostInsert
    SimpleMemorySearch    替代 PreRetrieval + MemoryRetrieval + PostRetrieval
    SimplePipelineCaller  简化版主编排算子（对应 PipelineCaller）
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .simple_memory_add import SimpleMemoryAdd as SimpleMemoryAdd
    from .simple_memory_search import SimpleMemorySearch as SimpleMemorySearch
    from .simple_pipeline_caller import SimplePipelineCaller as SimplePipelineCaller


_EXPORT_MAP = {
    "SimpleMemoryAdd": ("benchmarks.simple_experiment.libs.simple_memory_add", "SimpleMemoryAdd"),
    "SimpleMemorySearch": (
        "benchmarks.simple_experiment.libs.simple_memory_search",
        "SimpleMemorySearch",
    ),
    "SimplePipelineCaller": (
        "benchmarks.simple_experiment.libs.simple_pipeline_caller",
        "SimplePipelineCaller",
    ),
}


def __getattr__(name: str) -> Any:
    if name in _EXPORT_MAP:
        module_name, attr_name = _EXPORT_MAP[name]
        module = __import__(module_name, fromlist=[attr_name])
        return getattr(module, attr_name)
    raise AttributeError(
        f"module 'benchmarks.simple_experiment.libs' has no attribute {name!r}"
    )

__all__ = [
    "SimpleMemoryAdd",
    "SimpleMemorySearch",
    "SimplePipelineCaller",
]
