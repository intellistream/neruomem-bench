"""SemanticConsolidationAction - LLM 驱动的记忆语义巩固

Strategy Type : conflict_resolution
Trigger       : retrieval (similarity-based) + threshold (min count)

TiM 核心机制:
    每次插入新记忆后，检索与其相似的历史记忆。
    若相似记忆数量超过阈值，调用 LLM 将其合并为一条更精炼的记忆，
    并删除原始条目，以此实现记忆去重和冲突消解。

配置参数 (operators.post_insert):
    retrieve_count  (int)  : 检索相似记忆的候选数量，默认 10
    min_merge_count (int)  : 触发合并的最低相似记忆数，默认 3
    merge_prompt    (str)  : LLM 合并提示词模板，含 {memories} 占位符
    merge_summary_only (bool): True 时仅合并摘要字段（SeCom 风格），默认 False
"""

from __future__ import annotations

from typing import Any

from ..base import BasePostInsertAction, PostInsertInput, PostInsertOutput


class SemanticConsolidationAction(BasePostInsertAction):
    """TiM 语义巩固：相似记忆 LLM 合并去重。"""

    STRATEGY_TYPE = "conflict_resolution"
    TRIGGER_MECHANISM = "retrieval"
    AVAILABLE_ACTIONS = ["MERGE", "NOOP"]

    def _init_action(self) -> None:
        self.retrieve_count: int = self._get_config("retrieve_count", 10)
        self.min_merge_count: int = self._get_config("min_merge_count", 3)
        self.merge_prompt: str = self._get_config(
            "merge_prompt",
            "Please consolidate the following similar memories into one concise "
            "memory:\n{memories}\nConsolidated memory:",
        )
        self.merge_summary_only: bool = self._get_config("merge_summary_only", False)

    def execute(
        self,
        input_data: PostInsertInput,
        service: Any,
        llm: Any | None = None,
    ) -> PostInsertOutput:
        if llm is None:
            return PostInsertOutput(
                success=False,
                action="semantic_consolidation",
                details={"error": "LLM client required for semantic consolidation"},
            )

        entries = input_data.insert_stats.get("entries", [])
        if not entries:
            return PostInsertOutput(
                success=True,
                action="semantic_consolidation",
                details={"message": "No entries to process"},
            )

        merged_count = 0
        deleted_count = 0

        for entry in entries:
            try:
                similar = self._retrieve_similar(service, entry, self.retrieve_count)

                if len(similar) >= self.min_merge_count:
                    merged_text = self._merge_memories(llm, similar)

                    for mem in similar:
                        service.delete(mem["id"])
                        deleted_count += 1

                    if self.merge_summary_only:
                        # SeCom mode: keep raw texts, store summary in metadata
                        texts = [m.get("text", "") for m in similar if m.get("text")]
                        insert_text = "\n\n---\n\n".join(texts)
                        insert_meta: dict[str, Any] = {
                            "merged_from": [m["id"] for m in similar],
                            "summary": merged_text,
                            "segment_count": len(similar),
                        }
                    else:
                        # TiM mode: merged summary *is* the new memory text
                        insert_text = merged_text
                        insert_meta = {"merged_from": [m["id"] for m in similar]}

                    # Re-use embedding from first similar memory if available
                    merged_vec = (
                        similar[0].get("embedding") if similar else None
                    )
                    service.insert(
                        entry=insert_text,
                        vector=merged_vec,
                        metadata=insert_meta,
                    )
                    merged_count += 1

            except Exception as e:
                input_data.data.setdefault("errors", []).append(
                    {
                        "entry_id": entry.get("id", "unknown"),
                        "action": "semantic_consolidation",
                        "error": str(e),
                    }
                )

        return PostInsertOutput(
            success=True,
            action="semantic_consolidation",
            details={
                "strategy_type": self.STRATEGY_TYPE,
                "trigger_mechanism": self.TRIGGER_MECHANISM,
                "merged_count": merged_count,
                "deleted_count": deleted_count,
                "processed_entries": len(entries),
            },
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _retrieve_similar(
        self, service: Any, entry: dict[str, Any], count: int
    ) -> list[dict[str, Any]]:
        """Retrieve memories similar to *entry* using its embedding vector."""
        if "embedding" not in entry or entry["embedding"] is None:
            return []

        results = service.retrieve(vector=entry["embedding"], top_k=count)

        entry_id = entry.get("id")
        normalized: list[dict[str, Any]] = []
        for r in results:
            rid = r.get("id") or r.get("entry_id") or r.get("node_id")
            if rid is not None and rid != entry_id:
                r["id"] = rid
                normalized.append(r)
        return normalized

    def _merge_memories(self, llm: Any, memories: list[dict[str, Any]]) -> str:
        """Call LLM to distill multiple memories into one."""
        memories_text = "\n".join(
            f"{i + 1}. {m.get('text', '')}" for i, m in enumerate(memories)
        )
        prompt = self.merge_prompt.replace("{memories}", memories_text)
        if hasattr(llm, "generate"):
            return llm.generate(prompt)
        return str(llm(prompt))
