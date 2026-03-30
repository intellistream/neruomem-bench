"""记忆实验数据源 - 支持多数据集的统一接口"""

from sage.foundation import BatchFunction

from benchmarks.experiment.utils.dataloader import DataLoaderFactory


class MemorySource(BatchFunction):
    """从多种数据集中逐个读取对话轮次的Source"""

    def __init__(self, config):
        super().__init__()
        self.dataset = config.get("dataset")
        self.task_id = config.get("task_id")

        self.loader = DataLoaderFactory.create(self.dataset)
        self.turns = self.loader.sessions(self.task_id)
        self.total_messages = self.loader.message_count(self.task_id)
        self.total_dialogs = self.loader.dialog_count(self.task_id)

        print(f"📊 样本 {self.task_id} 统计信息:")
        print(f"   - 总会话数: {len(self.turns)}")
        print(f"   - 总消息数: {self.total_messages}")
        print(f"   - 总对话轮次: {self.total_dialogs}")
        for idx, (session_id, max_dialog_idx) in enumerate(self.turns):
            msg_count = max_dialog_idx + 1
            print(f"   - 会话 {idx + 1} (session_id={session_id}): {msg_count} 条消息")

        self.session_idx = 0
        self.dialog_ptr = 0
        self.packet_idx = 0

    def execute(self):
        import time

        time.sleep(0.01)

        if self.session_idx >= len(self.turns):
            print(f"🏁 MemorySource 已完成：所有 {len(self.turns)} 个会话已处理完毕")
            return None

        session_id, max_dialog_idx = self.turns[self.session_idx]

        if self.dialog_ptr > max_dialog_idx:
            self.session_idx += 1
            self.dialog_ptr = 0
            if self.session_idx >= len(self.turns):
                return None
            session_id, max_dialog_idx = self.turns[self.session_idx]

        dialogs = self.loader.get_dialog(
            self.task_id, session_x=session_id, dialog_y=self.dialog_ptr
        )

        dialog_increment = len(dialogs) if dialogs else 2
        next_dialog_ptr = self.dialog_ptr + dialog_increment
        is_session_end = next_dialog_ptr > max_dialog_idx

        result = {
            "task_id": self.task_id,
            "session_id": session_id,
            "dialog_id": self.dialog_ptr,
            "dialogs": dialogs,
            "dialog_len": len(dialogs),
            "packet_idx": self.packet_idx,
            "total_packets": self.total_dialogs,
            "is_session_end": is_session_end,
        }

        self.dialog_ptr += dialog_increment
        self.packet_idx += 1

        return result
