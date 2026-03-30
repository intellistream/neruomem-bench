"""PostRetrieval Operator - 记忆检索后处理算子

Pipeline 位置: 第 4 层（检索后）
访问权限: 允许多次检索记忆服务（不允许插入/删除）
"""

from __future__ import annotations

import time
from typing import Any

from sage.foundation import MapFunction

from benchmarks.experiment.utils import (
    EmbeddingGenerator,
    LLMGenerator,
)

from .base import (
    BasePostRetrievalAction,
    MemoryItem,
    PostRetrievalInput,
    PostRetrievalOutput,
)
from .registry import PostRetrievalActionRegistry


class _ServiceProxy:
    """Service proxy - PostRetrieval only allows search operations."""

    def __init__(self, operator: MapFunction, service_name: str):
        self._operator = operator
        self._service_name = service_name

    def search(self, **kwargs) -> list[dict[str, Any]]:
        return self._operator.call_service(self._service_name, method="search", **kwargs)

    def retrieve(self, **kwargs) -> list[dict[str, Any]]:
        return self._operator.call_service(self._service_name, method="retrieve", **kwargs)


class PostRetrieval(MapFunction):
    """记忆检索后的后处理算子"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        services_type = config.get("services.services_type")
        if not services_type:
            raise ValueError("Missing required config: services.services_type")
        self.service_name = services_type.split(".")[-1]

        self._llm_generator = LLMGenerator.from_config(self.config)
        self._embedding_generator = EmbeddingGenerator.from_config(self.config)
        action_config = config.get("operators.post_retrieval", {})
        self.action_name = action_config.get("action", "none")
        action_type = None
        if self.action_name in ["rerank", "filter", "merge"]:
            type_key = f"{self.action_name}_type"
            action_type = action_config.get(type_key)
        action_key = f"{self.action_name}.{action_type}" if action_type else self.action_name
        try:
            action_class = PostRetrievalActionRegistry.get(action_key)
            self.action: BasePostRetrievalAction = action_class(action_config)
        except ValueError as e:
            print(f"[WARNING] {e}, using NoneAction as fallback")
            from .none_action import NoneAction

            self.action = NoneAction(action_config)
        if hasattr(self.action, "set_llm_generator"):
            self.action.set_llm_generator(self._llm_generator)
        if hasattr(self.action, "set_embedding_generator"):
            self.action.set_embedding_generator(self._embedding_generator)
        self._conversation_format_prompt = action_config.get(
            "conversation_format_prompt", "The following is some history information.\n"
        )
        self._tier_retrieval_limits = action_config.get("tier_retrieval_limits", {})

    def execute(self, data: dict[str, Any]) -> dict[str, Any]:
        start_time = time.perf_counter()

        input_data = PostRetrievalInput(
            data=data,
            config=self.config.get("operators.post_retrieval", {}),
            service_name=self.service_name,
        )
        service_proxy = _ServiceProxy(self, self.service_name)
        output: PostRetrievalOutput = self.action.execute(
            input_data,
            service=service_proxy,
            llm=self._llm_generator if self._llm_generator else None,
        )
        output.memory_items = self._apply_tier_limits(output.memory_items)
        formatted_memory = self._format_conversation_history(output.memory_items)
        data["history_text"] = formatted_memory
        if output.memory_items:
            data["processed_memory_items"] = [
                {"text": item.text, "score": item.score, "metadata": item.metadata}
                for item in output.memory_items
            ]
        if output.metadata:
            data.setdefault("metadata", {}).update(output.metadata)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        data.setdefault("stage_timings", {})["post_retrieval_ms"] = elapsed_ms

        items_count = len(output.memory_items) if output.memory_items else 0
        print(
            f"  [PostRetrieval] 动作: {self.action_name} | 处理: {items_count}条 | 耗时: {elapsed_ms:.2f}ms",
            flush=True,
        )

        return data

    def _apply_tier_limits(self, items: list[MemoryItem]) -> list[MemoryItem]:
        if not self._tier_retrieval_limits:
            return items
        tier_items: dict[str, list[MemoryItem]] = {}
        for item in items:
            tier = item.metadata.get("tier", "default")
            if tier not in tier_items:
                tier_items[tier] = []
            tier_items[tier].append(item)
        limited_items: list[MemoryItem] = []
        for tier, tier_limit in self._tier_retrieval_limits.items():
            if tier in tier_items:
                limited_items.extend(tier_items[tier][:tier_limit])
        for tier, items_list in tier_items.items():
            if tier not in self._tier_retrieval_limits:
                limited_items.extend(items_list)
        return limited_items

    def _format_conversation_history(self, items: list[MemoryItem]) -> str:
        if not items:
            return ""
        has_tier_placeholders = (
            "{stm_memories}" in self._conversation_format_prompt
            or "{ltm_memories}" in self._conversation_format_prompt
        )
        if has_tier_placeholders:
            stm_items = [item for item in items if item.metadata.get("tier") == "stm"]
            ltm_items = [item for item in items if item.metadata.get("tier") == "ltm"]
            stm_text = "\n".join(item.text for item in stm_items) if stm_items else "None"
            ltm_text = "\n".join(item.text for item in ltm_items) if ltm_items else "None"
            result = self._conversation_format_prompt.replace("{stm_memories}", stm_text).replace(
                "{ltm_memories}", ltm_text
            )
        else:
            formatted = self._conversation_format_prompt
            for item in items:
                formatted += f"{item.text}\n"
            result = formatted.rstrip()
        return result
