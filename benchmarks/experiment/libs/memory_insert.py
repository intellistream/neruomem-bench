"""MemoryInsert - 记忆插入算子

纯透传模式：遍历 PreInsert 输出的 memory_entries，调用记忆服务的 insert 方法。
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

from sage.foundation import MapFunction

from benchmarks.experiment.utils import process_logger


@dataclass
class InsertStats:
    """插入统计数据模型"""

    inserted: int
    failed: int
    entry_ids: list[str]
    entries: list[dict[str, Any]]
    errors: list[dict[str, Any]]


class MemoryInsert(MapFunction):
    """记忆插入算子 - 纯透传模式"""

    def __init__(self, config=None):
        super().__init__()
        self.config = config
        services_type = config.get("services.services_type")
        if not services_type:
            raise ValueError("Missing required config: services.services_type")
        self.service_name = services_type.split(".")[-1]
        self.verbose = config.get("runtime.memory_insert_verbose", False)

    def execute(self, data: dict[str, Any]) -> dict[str, Any]:
        memory_entries = data.get("memory_entries", [])
        stats = InsertStats(inserted=0, failed=0, entry_ids=[], entries=[], errors=[])

        batch_start = time.perf_counter()

        for entry in memory_entries:
            try:
                entry_id = self._insert_entry(entry)
                stats.inserted += 1
                stats.entry_ids.append(entry_id)
                stats.entries.append(
                    {
                        "id": entry_id,
                        "text": entry.get("text", ""),
                        "embedding": entry.get("embedding"),
                        "metadata": entry.get("metadata", {}),
                    }
                )
                process_logger.log_service(
                    "INSERT", f"ID: {entry_id}\nText: {entry.get('text', '')[:200]}"
                )
                if self.verbose:
                    self._log_insert(entry, entry_id)
            except Exception as e:
                stats.failed += 1
                stats.errors.append(
                    {"entry": entry.get("text", "")[:100], "error": str(e)}
                )
                self.logger.warning(f"Insert failed: {e}")

        batch_elapsed_ms = (time.perf_counter() - batch_start) * 1000
        print(
            f"  [MemoryInsert] 插入: {stats.inserted}条 | 失败: {stats.failed}条 | 耗时: {batch_elapsed_ms:.2f}ms",
            flush=True,
        )

        data["insert_stats"] = asdict(stats)
        dialog_count = len(data.dialogs) if hasattr(data, "dialogs") else 1
        if dialog_count > 0:
            per_dialog_ms = batch_elapsed_ms / dialog_count
            data.setdefault("stage_timings", {})["memory_insert_ms"] = [
                per_dialog_ms
            ] * dialog_count
        else:
            data.setdefault("stage_timings", {})["memory_insert_ms"] = []

        return data

    def _insert_entry(self, entry: dict[str, Any]) -> str:
        text = entry.get("text", "")
        if not text:
            raise ValueError("Entry text is empty")
        return self.call_service(
            self.service_name,
            method="insert",
            entry=text,
            vector=entry.get("embedding"),
            metadata=entry.get("metadata", {}),
            insert_mode=entry.get("insert_mode", "passive"),
            insert_params=entry.get("insert_params"),
            timeout=10.0,
        )

    def _log_insert(self, entry: dict[str, Any], entry_id: str) -> None:
        text = entry.get("text", "")[:50]
        mode = entry.get("insert_mode", "passive")
        self.logger.info(f"Inserted [{entry_id}] (mode={mode}): {text}...")
