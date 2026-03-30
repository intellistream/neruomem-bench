"""
PostInsert Action Base Classes
===============================

Provides unified interface for all PostInsert actions.

Strategy Classification:
- conflict_resolution: LLM CRUD / semantic consolidation
- decay_eviction: Forgetting curve / time decay
- structure_enrichment: Link evolution / graph construction
- tier_migration: Heat-based layer migration
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PostInsertInput:
    """PostInsert unified input data structure.

    Attributes:
        data: Complete data from PreInsert + MemoryInsert stages
        insert_stats: Statistics from insertion (inserted count, entry_ids, etc.)
        service_name: Name of the memory service
        is_session_end: Whether this is the last packet in session
        config: Action-specific configuration
    """

    data: dict[str, Any]
    insert_stats: dict[str, Any]
    service_name: str
    is_session_end: bool = False
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class PostInsertOutput:
    """PostInsert unified output data structure.

    Attributes:
        success: Whether action executed successfully
        action: Action name
        details: Action-specific execution details
    """

    success: bool
    action: str
    details: dict[str, Any] = field(default_factory=dict)


class BasePostInsertAction(ABC):
    """Base class for all PostInsert actions.

    PostInsert actions handle memory optimization and maintenance after insertion,
    including conflict resolution, decay/eviction, structure enrichment, and tier migration.

    Strategy Type Classification:
    - STRATEGY_TYPE: conflict_resolution | decay_eviction | structure_enrichment | tier_migration
    - TRIGGER_MECHANISM: retrieval | temporal | threshold | semantic | hybrid
    - AVAILABLE_ACTIONS: List of actions this strategy can perform

    Attributes:
        config: Action-specific configuration dictionary
    """

    STRATEGY_TYPE: str = ""
    TRIGGER_MECHANISM: str = ""
    AVAILABLE_ACTIONS: list[str] = []

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self._init_action()

    @abstractmethod
    def _init_action(self) -> None:
        """Initialize action-specific configuration and tools."""

    @abstractmethod
    def execute(
        self,
        input_data: PostInsertInput,
        service: Any,
        llm: Any | None = None,
    ) -> PostInsertOutput:
        """Execute PostInsert action logic.

        Args:
            input_data: Input data structure
            service: Memory service instance (provides retrieve, delete, insert capabilities)
            llm: Optional LLM client
        """

    def _get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value with fallback."""
        keys = key.split(".")
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
