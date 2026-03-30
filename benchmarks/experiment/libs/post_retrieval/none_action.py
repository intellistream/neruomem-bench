"""None Action - 透传策略

不做任何处理，直接透传检索结果。
"""

from typing import Any

from .base import BasePostRetrievalAction, PostRetrievalInput, PostRetrievalOutput


class NoneAction(BasePostRetrievalAction):
    """透传策略 - 不做任何处理"""

    def _init_action(self) -> None:
        pass

    def execute(
        self,
        input_data: PostRetrievalInput,
        service: Any,
        llm: Any | None = None,
    ) -> PostRetrievalOutput:
        memory_data = input_data.data.get("memory_data", [])
        items = self._convert_to_items(memory_data)
        return PostRetrievalOutput(memory_items=items, metadata={"action": "none"})
