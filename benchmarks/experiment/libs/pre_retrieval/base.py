"""PreRetrieval Action 基类和数据模型

定义了PreRetrieval阶段的统一接口和数据结构。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PreRetrievalInput:
    """PreRetrieval 统一输入数据模型

    Attributes:
        data: 包含用户查询的原始数据（必须包含 question 字段）
        config: Action 特定配置
    """

    data: dict[str, Any]
    config: dict[str, Any]

    @property
    def question(self) -> str:
        return self.data.get("question", "")


@dataclass
class PreRetrievalOutput:
    """PreRetrieval 统一输出数据模型

    Attributes:
        query: 处理后的查询文本
        query_embedding: 查询向量（如果生成）
        metadata: 额外元数据
        retrieve_mode: 检索模式（passive/active）
        retrieve_params: 检索参数
    """

    query: str
    query_embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    retrieve_mode: str = "passive"
    retrieve_params: dict[str, Any] | None = None


class BasePreRetrievalAction(ABC):
    """PreRetrieval Action 基类

    所有PreRetrieval Action必须继承此类并实现execute方法。
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self._init_action()

    @abstractmethod
    def _init_action(self) -> None:
        """初始化Action特定的配置和工具"""

    @abstractmethod
    def execute(self, input_data: PreRetrievalInput) -> PreRetrievalOutput:
        """执行Action逻辑"""

    def _get_config_value(
        self,
        key: str,
        default: Any = None,
        required: bool = False,
        context: str = "",
    ) -> Any:
        value = self.config.get(key, default)
        if required and value is None:
            ctx = f" ({context})" if context else ""
            raise ValueError(f"Missing required config: {key}{ctx}")
        return value
