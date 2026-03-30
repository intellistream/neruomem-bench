"""PostInsert Operator - 记忆插入后处理算子

Pipeline 位置: 第 2 层（插入后）
访问权限: 允许检索、删除、插入记忆服务（完整权限）

采用策略模式，通过 Action 注册表动态选择和执行后处理策略。
"""

from __future__ import annotations

import time
from typing import Any

from sage.foundation import MapFunction

from benchmarks.experiment.utils import (
    EmbeddingGenerator,
    LLMGenerator,
)

from .base import BasePostInsertAction, PostInsertInput, PostInsertOutput
from .registry import PostInsertActionRegistry


class _ServiceProxy:
    """Service proxy to wrap call_service calls into method-like interface"""

    def __init__(self, operator: MapFunction, service_name: str):
        self._operator = operator
        self._service_name = service_name

    def search(self, **kwargs) -> list[dict[str, Any]]:
        return self._operator.call_service(self._service_name, method="search", **kwargs)

    def retrieve(self, **kwargs) -> list[dict[str, Any]]:
        return self._operator.call_service(self._service_name, method="retrieve", **kwargs)

    def insert(self, **kwargs) -> str:
        return self._operator.call_service(self._service_name, method="insert", **kwargs)

    def update(self, **kwargs) -> bool:
        return self._operator.call_service(self._service_name, method="update", **kwargs)

    def delete(self, entry_id: str) -> bool:
        return self._operator.call_service(self._service_name, method="delete", entry_id=entry_id)

    def add_edge(
        self, from_node: str, to_node: str, weight: float = 1.0, edge_type: str = "relation"
    ) -> bool:
        return self._operator.call_service(
            self._service_name,
            method="add_edge",
            from_node=from_node,
            to_node=to_node,
            weight=weight,
            edge_type=edge_type,
        )


class PostInsert(MapFunction):
    """记忆插入后的后处理算子"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        services_type = config.get("services.services_type")
        if not services_type:
            raise ValueError("Missing required config: services.services_type")
        self.service_name = services_type.split(".")[-1]

        self._llm_generator = LLMGenerator.from_config(self.config)
        self._embedding_generator = EmbeddingGenerator.from_config(self.config)
        action_config = config.get("operators.post_insert", {})
        self.action_name = action_config.get("action", "none")

        action_type = None
        if self.action_name == "enhance":
            action_type = action_config.get("enhance_type")
        elif self.action_name == "migrate":
            action_type = action_config.get("migrate_type")

        action_key = f"{self.action_name}.{action_type}" if action_type else self.action_name

        try:
            action_class = PostInsertActionRegistry.get(action_key)
            self.action: BasePostInsertAction = action_class(action_config)
        except ValueError as e:
            print(f"[WARNING] {e}, using NoneAction as fallback")
            from .none_action import NoneAction

            self.action = NoneAction(action_config)
        if hasattr(self.action, "set_llm_generator"):
            self.action.set_llm_generator(self._llm_generator)
        if hasattr(self.action, "set_embedding_generator"):
            self.action.set_embedding_generator(self._embedding_generator)

    def execute(self, data: dict[str, Any]) -> dict[str, Any]:
        start_time = time.perf_counter()
        input_data = PostInsertInput(
            data=data,
            insert_stats=data.get("insert_stats", {}),
            service_name=self.service_name,
            is_session_end=data.get("is_session_end", False),
            config=self.config.get("operators.post_insert", {}),
        )
        service_proxy = _ServiceProxy(self, self.service_name)
        output: PostInsertOutput = self.action.execute(
            input_data,
            service=service_proxy,
            llm=self._llm_generator if self._llm_generator else None,
        )
        if output.details:
            data.setdefault("metadata", {}).update(output.details)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        dialog_count = len(data.dialogs) if hasattr(data, "dialogs") else 1

        print(
            f"  [PostInsert] 动作: {self.action_name} | 失败: 0条 | 耗时: {elapsed_ms:.2f}ms",
            flush=True,
        )

        if dialog_count > 0:
            per_entry_ms = elapsed_ms / dialog_count
            data.setdefault("stage_timings", {})["post_insert_ms"] = [per_entry_ms] * dialog_count
        else:
            data.setdefault("stage_timings", {})["post_insert_ms"] = []

        return data
