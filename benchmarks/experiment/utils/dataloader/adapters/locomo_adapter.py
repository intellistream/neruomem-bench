"""本地 LoCoMo 数据集适配器

从 benchmarks/experiment/data/locomo/ 目录下的 JSON 文件加载数据。
无需 sage.data 依赖，可独立运行。

数据 JSON 格式：
    [
      {
        "task_id": "conv-mini-01",
        "sessions": [
          {
            "session_id": 0,
            "messages": [{"speaker": "...", "text": "..."}, ...]
          }
        ],
        "questions": [
          {
            "question": "...",
            "answer": "...",
            "category": "...",
            "evidence": "...",
            "visible_session": 0,
            "visible_dialog": 3
          }
        ]
      }
    ]

dialog_y 语义（与 LoCoMo 官方格式一致）：
    get_dialog(task_id, session_x, dialog_y) 返回 messages[dialog_y : dialog_y+2]
    每次推进 dialog_ptr += len(returned_messages)
    sessions() 返回 [(session_id, max_index_of_last_message)]
"""

from __future__ import annotations

import json
import os
from typing import Any

from benchmarks.experiment.utils.dataloader.base import BaseDataLoader


def _data_file() -> str:
    """默认数据文件路径（相对于 neuromem-bench root）。"""
    here = os.path.dirname(__file__)
    # adapters/ → dataloader/ → utils/ → experiment/
    experiment_root = os.path.join(here, "..", "..", "..")
    return os.path.join(experiment_root, "data", "locomo", "mini_locomo.json")


class LocalLocomoAdapter(BaseDataLoader):
    """本地 LoCoMo 格式数据集适配器。

    通过 data_file 参数可以指定任意 JSON 文件；
    默认加载 benchmarks/experiment/data/locomo/mini_locomo.json。

    数据路径可通过环境变量 LOCOMO_DATA_FILE 覆盖。
    """

    def __init__(self, data_file: str | None = None):
        path = data_file or os.environ.get("LOCOMO_DATA_FILE") or _data_file()
        path = os.path.realpath(path)
        with open(path, encoding="utf-8") as f:
            raw: list[dict[str, Any]] = json.load(f)

        # 构建索引: task_id → record
        self._records: dict[str, dict[str, Any]] = {}
        for rec in raw:
            record_id = rec.get("task_id") or rec.get("sample_id")
            if record_id:
                self._records[str(record_id)] = rec

    @property
    def dataset_name(self) -> str:
        return "locomo"

    # ── Core interface ────────────────────────────────────────────────────────

    def get_dialog(
        self, task_id: str, session_x: int, dialog_y: int
    ) -> list[dict[str, Any]]:
        """返回 session session_x 中从 dialog_y 开始的最多 2 条消息。"""
        session = self._get_session(task_id, session_x)
        msgs = session["messages"]
        return msgs[dialog_y : dialog_y + 2]

    def get_evaluation(
        self, task_id: str, session_x: int, dialog_y: int
    ) -> list[dict[str, Any]]:
        """返回在 (session_x, dialog_y) 位置及之前可见的所有问题。"""
        record = self._get_record(task_id)
        visible = []
        for q in record["questions"]:
            vs = q.get("visible_session", 0)
            vd = q.get("visible_dialog", 0)
            if (vs < session_x) or (vs == session_x and vd <= dialog_y):
                visible.append(q)
        return visible

    def sessions(self, task_id: str) -> list[tuple[int, int]]:
        """返回 [(session_id, last_message_index), ...] 列表。"""
        record = self._get_record(task_id)
        result = []
        for sess in record["sessions"]:
            sid = sess["session_id"]
            last_idx = len(sess["messages"]) - 1
            result.append((sid, last_idx))
        return result

    def question_count(self, task_id: str) -> int:
        return len(self._get_record(task_id)["questions"])

    def dialog_count(self, task_id: str) -> int:
        """总对话推进步数（每次推进最多 2 条消息）。"""
        record = self._get_record(task_id)
        count = 0
        for sess in record["sessions"]:
            n = len(sess["messages"])
            count += (n + 1) // 2  # 向上取整
        return count

    def message_count(self, task_id: str) -> int:
        record = self._get_record(task_id)
        return sum(len(s["messages"]) for s in record["sessions"])

    def statistics(self, task_id: str) -> dict[str, Any]:
        record = self._get_record(task_id)
        return {
            "task_id": task_id,
            "sessions": len(record["sessions"]),
            "messages": self.message_count(task_id),
            "questions": self.question_count(task_id),
        }

    def session_summary(self, task_id: str, session_x: int) -> str:
        """返回指定会话的摘要文本；若数据中不存在则返回空串。"""
        record = self._get_record(task_id)
        summary_block = record.get("session_summary", {})
        if not summary_block:
            return ""

        summary_keys = [
            f"session_{session_x + 1}_summary",
            f"session_{session_x}_summary",
        ]
        for key in summary_keys:
            summary_text = summary_block.get(key, "")
            if summary_text:
                return str(summary_text).strip()
        return ""

    def list_tasks(self) -> list[str]:
        """列出所有可用 task_id。"""
        return list(self._records.keys())

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_record(self, task_id: str) -> dict[str, Any]:
        if task_id not in self._records:
            available = list(self._records.keys())
            raise KeyError(
                f"task_id '{task_id}' not found. Available: {available}"
            )
        return self._records[task_id]

    def _get_session(self, task_id: str, session_x: int) -> dict[str, Any]:
        record = self._get_record(task_id)
        for sess in record["sessions"]:
            if sess["session_id"] == session_x:
                return sess
        raise KeyError(
            f"session_id {session_x} not found in task '{task_id}'."
        )
