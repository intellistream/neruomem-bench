"""PostRetrieval Action 基类和数据模型

定义 PostRetrieval 所有 Action 的统一接口和数据流。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

UTC = UTC


@dataclass
class MemoryItem:
    """标准化记忆条目"""

    text: str
    score: float | None
    metadata: dict[str, Any]
    original_index: int

    def get_timestamp(self, field: str = "timestamp") -> datetime | None:
        value = self.metadata.get(field)
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            for fmt in (
                "%Y-%m-%dT%H:%M:%S.%f%z",
                "%Y-%m-%dT%H:%M:%S%z",
                "%Y-%m-%dT%H:%M:%S.%f",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S",
            ):
                try:
                    return datetime.strptime(value, fmt).replace(tzinfo=UTC)
                except Exception:  # noqa: BLE001
                    continue
        if isinstance(value, (int, float)):
            try:
                return datetime.fromtimestamp(float(value), tz=UTC)
            except Exception:  # noqa: BLE001
                return None
        return None


@dataclass
class PostRetrievalInput:
    """PostRetrieval 统一输入"""

    data: dict[str, Any]
    config: dict[str, Any]
    service_name: str


@dataclass
class PostRetrievalOutput:
    """PostRetrieval 统一输出"""

    memory_items: list[MemoryItem]
    metadata: dict[str, Any] = field(default_factory=dict)


class BasePostRetrievalAction(ABC):
    """PostRetrieval Action 基类

    所有 PostRetrieval Action 必须继承此类并实现 execute 方法。

    职责边界：
    - 对检索结果进行后处理（rerank, filter, merge, augment 等）
    - 可以多次调用记忆服务进行查询拼接
    - 不负责生成 embedding（由服务完成）
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self._init_action()

    @abstractmethod
    def _init_action(self) -> None:
        """初始化 Action 特定配置和工具"""

    @abstractmethod
    def execute(
        self,
        input_data: PostRetrievalInput,
        service: Any,
        llm: Any | None = None,
    ) -> PostRetrievalOutput:
        """执行 Action 逻辑"""

    def _convert_to_items(self, memory_data: list[dict[str, Any]]) -> list[MemoryItem]:
        items = []
        for idx, item in enumerate(memory_data):
            text = item.get("text") or item.get("content", "")
            items.append(
                MemoryItem(
                    text=text,
                    score=item.get("score"),
                    metadata=item.get("metadata", {}),
                    original_index=idx,
                )
            )
        return items

    def _items_to_dicts(self, items: list[MemoryItem]) -> list[dict[str, Any]]:
        return [
            {"text": item.text, "score": item.score, "metadata": item.metadata} for item in items
        ]
