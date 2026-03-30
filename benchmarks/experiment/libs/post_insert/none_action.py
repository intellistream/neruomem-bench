"""
None Action - Passthrough without post-processing

This is the basic PostInsert action that performs no post-insert optimization.
Use as a reference template for implementing custom PostInsert actions.
"""

from typing import Any

from .base import BasePostInsertAction, PostInsertInput, PostInsertOutput


class NoneAction(BasePostInsertAction):
    """Passthrough action - no post-insert processing."""

    def _init_action(self) -> None:
        """Initialize none action (no setup required)."""

    def execute(
        self,
        input_data: PostInsertInput,
        service: Any,
        llm: Any | None = None,
    ) -> PostInsertOutput:
        return PostInsertOutput(
            success=True,
            action="none",
            details={"message": "No post-insert processing performed"},
        )
