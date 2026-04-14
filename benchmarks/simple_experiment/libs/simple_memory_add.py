"""SimpleMemoryAdd — 黑盒记忆插入算子

替代原 pipeline 中的「PreInsert + MemoryInsert + PostInsert」三段式设计。
直接将对话对拼接为纯文本后调用适配器的 add() 方法，不做任何预处理或后处理。

Pipeline 位置：插入子 pipeline 的唯一 Map 节点
    PipelineServiceSource → SimpleMemoryAdd → PipelineServiceSink
"""

from __future__ import annotations

import time
from typing import Any

from sage.foundation import MapFunction

from benchmarks.experiment.utils import process_logger


class SimpleMemoryAdd(MapFunction):
    """黑盒记忆插入算子 — 直接调用适配器 add()"""

    def __init__(self, config) -> None:
        super().__init__()
        services_type = config.get("services.services_type")
        if not services_type:
            raise ValueError("Missing required config: services.services_type")
        # e.g. "simple.mem0" → "mem0"
        self.adapter_name: str = services_type.split(".")[-1]
        self.verbose: bool = config.get("runtime.memory_insert_verbose", False)

    def execute(self, data: dict[str, Any]) -> dict[str, Any]:
        start_time = time.perf_counter()

        dialogs: list[dict[str, Any]] = data.get("dialogs", [])
        session_id = data.get("session_id")
        dialog_id = data.get("dialog_id")

        # 将多轮对话拼接为「Speaker: text」格式的单条文本
        lines: list[str] = []
        for turn in dialogs:
            speaker = turn.get("speaker", "Unknown")
            text = turn.get("text", "")
            if text.strip():
                lines.append(f"{speaker}: {text}")

        combined_text = "\n".join(lines)
        entry_id = ""
        inserted = 0

        if combined_text.strip():
            metadata: dict[str, Any] = {
                "session_id": session_id,
                "dialog_id": dialog_id,
                "source": "dialog",
            }
            entry_id = self.call_service(
                self.adapter_name,
                method="add",
                text=combined_text,
                metadata=metadata,
            )
            inserted = 1
            process_logger.log_service(
                "ADD", f"ID: {entry_id}\nText: {combined_text[:200]}"
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        if self.verbose:
            print(
                f"  [SimpleMemoryAdd] session={session_id} dialog={dialog_id} "
                f"| ID: {entry_id or 'N/A'} | 耗时: {elapsed_ms:.2f}ms",
                flush=True,
            )
        else:
            print(
                f"  [SimpleMemoryAdd] 插入: {inserted} 条 | 耗时: {elapsed_ms:.2f}ms",
                flush=True,
            )

        data["add_stats"] = {"inserted": inserted, "entry_id": entry_id}
        data.setdefault("stage_timings", {})["add_ms"] = elapsed_ms
        return data
