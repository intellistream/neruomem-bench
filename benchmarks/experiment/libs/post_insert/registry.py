"""
PostInsert Action Registry
===========================

Central registry for all PostInsert action strategies.

Open-source version includes only the base `none` action.
Developers can register custom actions via PostInsertActionRegistry.register().

Example:
    from benchmarks.experiment.libs.post_insert.registry import PostInsertActionRegistry
    from my_actions import MyConflictResolutionAction

    PostInsertActionRegistry.register("conflict_resolution.my_action", MyConflictResolutionAction)
"""

from .base import BasePostInsertAction
from .none_action import NoneAction


class PostInsertActionRegistry:
    """Registry for PostInsert action strategies."""

    _actions: dict[str, type[BasePostInsertAction]] = {}

    @classmethod
    def register(cls, name: str, action_class: type[BasePostInsertAction]) -> None:
        if not issubclass(action_class, BasePostInsertAction):
            raise ValueError(
                f"Action class must inherit from BasePostInsertAction, got {action_class}"
            )
        cls._actions[name] = action_class

    @classmethod
    def get(cls, name: str) -> type[BasePostInsertAction]:
        if name not in cls._actions:
            available = ", ".join(cls._actions.keys())
            raise ValueError(f"Unknown action: {name}. Available actions: {available}")
        return cls._actions[name]

    @classmethod
    def list_actions(cls) -> list[str]:
        return list(cls._actions.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        return name in cls._actions


# Register built-in actions
PostInsertActionRegistry.register("none", NoneAction)


def get_action(name: str, config: dict) -> BasePostInsertAction:
    """Convenience function to get and instantiate an action."""
    action_class = PostInsertActionRegistry.get(name)
    return action_class(config)
