"""PostRetrieval Action 注册表

开源版本包含 none Action。
开发者可注册自定义 Action，例如：
- rerank.semantic / rerank.time_weighted / rerank.ppr / rerank.weighted
- filter.threshold / filter.token_budget / filter.top_k
- merge.link_expand / merge.multi_query / merge.multi_tier
- augment / augment.reinforce
"""

from .base import BasePostRetrievalAction
from .none_action import NoneAction


class PostRetrievalActionRegistry:
    """PostRetrieval Action 注册表"""

    _actions: dict[str, type[BasePostRetrievalAction]] = {}

    @classmethod
    def register(cls, name: str, action_class: type[BasePostRetrievalAction]) -> None:
        cls._actions[name] = action_class

    @classmethod
    def get(cls, name: str) -> type[BasePostRetrievalAction]:
        if name not in cls._actions:
            raise ValueError(
                f"Unknown PostRetrieval action: {name}. "
                f"Available actions: {list(cls._actions.keys())}"
            )
        return cls._actions[name]

    @classmethod
    def list_actions(cls) -> list[str]:
        return list(cls._actions.keys())

    @classmethod
    def has_action(cls, name: str) -> bool:
        return name in cls._actions


# Register built-in actions
PostRetrievalActionRegistry.register("none", NoneAction)
