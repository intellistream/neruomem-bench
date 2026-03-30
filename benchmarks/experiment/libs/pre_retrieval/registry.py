"""PreRetrieval Action 注册表

开源版本包含 none 和 embedding 两个基础 Action。
开发者可注册自定义 Action，例如：
- optimize.keyword_extract
- optimize.expand
- optimize.rewrite
- enhancement.decompose
- enhancement.multi_embed
- enhancement.route
"""

from .base import BasePreRetrievalAction
from .embedding_action import EmbeddingAction
from .none_action import NoneAction


class PreRetrievalActionRegistry:
    """PreRetrieval Action 注册表"""

    _actions: dict[str, type[BasePreRetrievalAction]] = {}

    @classmethod
    def register(cls, name: str, action_class: type[BasePreRetrievalAction]) -> None:
        cls._actions[name] = action_class

    @classmethod
    def get(cls, name: str) -> type[BasePreRetrievalAction]:
        if name not in cls._actions:
            raise ValueError(
                f"Unknown action: {name}. Available actions: {', '.join(cls._actions.keys())}"
            )
        return cls._actions[name]

    @classmethod
    def list_actions(cls) -> list[str]:
        return list(cls._actions.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        return name in cls._actions


# Register built-in actions
PreRetrievalActionRegistry.register("none", NoneAction)
PreRetrievalActionRegistry.register("embedding", EmbeddingAction)
