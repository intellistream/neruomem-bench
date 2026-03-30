"""PreInsert Action 注册表

管理 PreInsert 阶段的算子(Action)注册与获取。

开源版本仅包含基础 none Action。
开发者可通过 PreInsertActionRegistry.register() 注册自定义 Action。

示例：
    # 注册自定义 Action
    from benchmarks.experiment.libs.pre_insert.registry import PreInsertActionRegistry
    from my_actions import MyCustomAction

    PreInsertActionRegistry.register("custom.my_action", MyCustomAction)
"""

from .base import BasePreInsertAction
from .none_action import NoneAction


class PreInsertActionRegistry:
    """PreInsert Action 注册表

    使用策略模式管理所有 Action，支持动态注册和获取。
    """

    _actions: dict[str, type[BasePreInsertAction]] = {}

    @classmethod
    def register(cls, name: str, action_class: type[BasePreInsertAction]) -> None:
        """注册一个 Action

        Args:
            name: Action 名称（支持点分隔的层级名，如 "enrich.keyword"）
            action_class: Action 类
        """
        cls._actions[name] = action_class

    @classmethod
    def get(cls, name: str) -> type[BasePreInsertAction]:
        """获取 Action 类

        Args:
            name: Action 名称

        Returns:
            Action 类

        Raises:
            ValueError: 如果 Action 未注册
        """
        if name not in cls._actions:
            raise ValueError(
                f"Unknown PreInsert action: '{name}'. "
                f"Available actions: {list(cls._actions.keys())}"
            )
        return cls._actions[name]

    @classmethod
    def list_actions(cls) -> list[str]:
        """列出所有已注册的 Action"""
        return list(cls._actions.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """检查 Action 是否已注册"""
        return name in cls._actions


# 注册内置 Action
PreInsertActionRegistry.register("none", NoneAction)
