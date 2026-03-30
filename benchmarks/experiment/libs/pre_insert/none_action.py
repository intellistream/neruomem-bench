"""NoneAction - 透传策略

适用于不需要预处理的记忆体（如 MemoryBank, SCM）。
直接将原始对话转换为记忆条目，不做任何额外处理。

这是最基本的 PreInsert Action 实现，可作为自定义 Action 的参考模板。
"""

from .base import BasePreInsertAction, PreInsertInput, PreInsertOutput


class NoneAction(BasePreInsertAction):
    """透传 Action - 不做任何预处理

    使用场景：
    - MemoryBank: 直接存储原始对话
    - SCM: 保持原始对话格式
    - HippoRAG2: 仅需要 embedding，不需要额外处理
    """

    def _init_action(self) -> None:
        """无需初始化任何工具"""

    def execute(self, input_data: PreInsertInput) -> PreInsertOutput:
        """直接透传，将对话转换为单条记忆条目

        Args:
            input_data: 包含 dialogs 的输入数据

        Returns:
            包含单条记忆条目的输出
        """
        dialogs = input_data.data.get("dialogs", [])
        text = self._format_dialogue(dialogs)

        entry = {
            "text": text,
            "metadata": {
                "action": "none",
                "dialog_count": len(dialogs),
            },
        }

        entry = self._set_default_fields(entry)
        entry["insert_method"] = "none"

        return PreInsertOutput(memory_entries=[entry])
