"""数据集加载器抽象基类

定义所有数据集加载器必须实现的统一接口。

数据层级: sample → session → message
- session: 一次会话
- message: 单条消息（一方说的一句话）
- dialog: 一轮对话（通常包含2条消息，末尾可能是1条）

接口分类:
- 核心数据获取: get_dialog, get_evaluation
- 结构/统计查询: sessions, question_count, dialog_count, message_count, statistics
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseDataLoader(ABC):
    """统一的数据集加载器抽象基类

    所有数据集适配器必须继承此类并实现全部抽象方法。
    """

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """数据集名称标识"""

    @abstractmethod
    def get_dialog(self, task_id: str, session_x: int, dialog_y: int) -> list[dict[str, Any]]:
        """获取一轮对话内容

        Args:
            task_id: 任务标识，如 "conv-26"
            session_x: session 编号
            dialog_y: dialog 起始索引

        Returns:
            消息列表（1-2条），每项为 {"speaker": "...", "text": "...", ...}
        """

    @abstractmethod
    def get_evaluation(self, task_id: str, session_x: int, dialog_y: int) -> list[dict[str, Any]]:
        """获取当前可见的测试问题列表

        返回在指定 session/dialog 位置时可见的问题。

        Args:
            task_id: 任务标识
            session_x: session 编号
            dialog_y: dialog 索引

        Returns:
            问题列表，每项为问题字典
        """

    @abstractmethod
    def sessions(self, task_id: str) -> list[tuple[int, int]]:
        """获取任务的会话结构

        Args:
            task_id: 任务标识

        Returns:
            会话列表，每项为 (session_id, max_dialog_idx)
        """

    @abstractmethod
    def question_count(self, task_id: str) -> int:
        """获取有效问题总数"""

    @abstractmethod
    def dialog_count(self, task_id: str) -> int:
        """获取总对话轮次（数据包数量）"""

    @abstractmethod
    def message_count(self, task_id: str) -> int:
        """获取总消息数目"""

    @abstractmethod
    def statistics(self, task_id: str) -> dict[str, Any]:
        """获取数据集统计信息"""
