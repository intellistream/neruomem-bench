"""BaseSimpleMemoryAdapter — 黑盒记忆体适配器抽象基类

所有第三方记忆体适配器须继承此类并实现以下四个方法：
    add(text, metadata)  → str         插入一段文本，返回记忆 ID
    search(query, top_k) → list[dict]  检索，返回格式化结果列表
    clear()              → None        清空当前用户的所有记忆
    name                 → str         适配器标识名（只读属性）

search() 返回值格式与 sage.neuromem BaseMemoryService.retrieve() 一致：
    [{"id": str, "text": str, "score": float, "metadata": dict}, ...]
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseSimpleMemoryAdapter(ABC):
    """第三方记忆体黑盒适配器基类"""

    @abstractmethod
    def add(self, text: str, metadata: dict[str, Any] | None = None) -> str:
        """将文本插入记忆体，返回记忆 ID。"""

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """检索与 query 相关的记忆，返回最多 top_k 条结果。

        每条结果格式：
            {"id": str, "text": str, "score": float, "metadata": dict}
        """

    @abstractmethod
    def clear(self) -> None:
        """清空当前用户的所有记忆。"""

    @property
    @abstractmethod
    def name(self) -> str:
        """适配器标识名（小写，如 "mem0"）。"""

    def get_stats(self) -> dict[str, Any]:
        """返回统计信息（可选重写）。默认返回适配器名。"""
        return {"adapter": self.name}
