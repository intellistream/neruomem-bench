"""PreInsert Operator - 记忆插入前预处理算子

Pipeline 位置: 第 1 层（插入前）
访问权限: 仅允许检索记忆服务（不允许插入/删除）

采用策略模式，通过 Action 注册表动态选择和执行预处理策略。
"""

from __future__ import annotations

import time
from typing import Any

from sage.foundation import MapFunction

from benchmarks.experiment.utils import (
    EmbeddingGenerator,
    LLMGenerator,
)

from .base import BasePreInsertAction, PreInsertInput, PreInsertOutput
from .registry import PreInsertActionRegistry


class PreInsert(MapFunction):
    """记忆插入前的预处理算子"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        services_type = config.get("services.services_type")
        if not services_type:
            raise ValueError("Missing required config: services.services_type")
        self.service_name = services_type.split(".")[-1]

        self._embedding_generator: EmbeddingGenerator = EmbeddingGenerator.from_config(self.config)
        self._llm_generator: LLMGenerator = LLMGenerator.from_config(self.config)

        print("\n" + "=" * 80)
        print("📋 [PreInsert Init] 模型配置信息")
        print("=" * 80)
        print("🤖 LLM 模型:")
        print(f"   - Model: {self._llm_generator.model_name}")
        print(f"   - Base URL: {config.get('runtime.base_url')}")
        print(f"   - Max Tokens: {self._llm_generator.max_tokens}")
        print(f"   - Temperature: {self._llm_generator.temperature}")
        if self._llm_generator.seed is not None:
            print(f"   - Seed: {self._llm_generator.seed}")

        print("\n🔢 Embedding 模型:")
        if self._embedding_generator.is_available():
            print(f"   - Model: {self._embedding_generator.model_name}")
            print(f"   - Base URL: {self._embedding_generator.base_url}")
        else:
            print("   - Status: Disabled (no embedding_base_url configured)")
        print("=" * 80 + "\n")

        action_config = config.get("operators.pre_insert", {})
        self.action_name = action_config.get("action", "none")

        action_type = None
        if self.action_name in ["transform", "extract", "score"]:
            type_key = f"{self.action_name}_type"
            action_type = action_config.get(type_key)

        action_key = f"{self.action_name}.{action_type}" if action_type else self.action_name

        try:
            action_class = PreInsertActionRegistry.get(action_key)
            self.action: BasePreInsertAction = action_class(action_config)
        except ValueError as e:
            print(f"[WARNING] {e}, using NoneAction as fallback")
            from .none_action import NoneAction

            self.action = NoneAction(action_config)

        if hasattr(self.action, "set_llm_generator"):
            self.action.set_llm_generator(self._llm_generator)
        if hasattr(self.action, "set_embedding_generator"):
            self.action.set_embedding_generator(self._embedding_generator)

    def execute(self, data: dict[str, Any]) -> dict[str, Any]:
        start_time = time.perf_counter()
        input_data = PreInsertInput(
            data=data,
            config=self.config.get("operators.pre_insert", {}),
            service_name=self.service_name,
        )
        output: PreInsertOutput = self.action.execute(input_data)
        self._generate_embeddings(output.memory_entries)
        data["memory_entries"] = output.memory_entries
        if output.metadata:
            data.setdefault("metadata", {}).update(output.metadata)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        dialog_count = len(data.dialogs) if hasattr(data, "dialogs") else 1

        entries_count = len(output.memory_entries)
        print(
            f"  [PreInsert] 动作: {self.action_name} | 生成: {entries_count}条 | 失败: 0条 | 耗时: {elapsed_ms:.2f}ms",
            flush=True,
        )

        if dialog_count > 0:
            per_entry_ms = elapsed_ms / dialog_count
            data.setdefault("stage_timings", {})["pre_insert_ms"] = [per_entry_ms] * dialog_count
        else:
            data.setdefault("stage_timings", {})["pre_insert_ms"] = []

        return data

    def _generate_embeddings(self, entries: list[dict[str, Any]]) -> None:
        """批量生成 embeddings"""
        if not self._embedding_generator or not self._embedding_generator.is_available():
            return

        texts_to_embed: list[str] = []
        indices_to_update: list[int] = []

        for i, entry in enumerate(entries):
            if "embedding" in entry and entry["embedding"] is not None:
                continue

            text_for_embed = (
                entry.get("summary", "")
                or entry.get("compressed_text", "")
                or entry.get("chunk_text", "")
                or entry.get("segment_text", "")
                or entry.get("fact", "")
                or entry.get("reconstructed_text", "")
                or entry.get("text", "")
            )

            if text_for_embed:
                texts_to_embed.append(text_for_embed)
                indices_to_update.append(i)

        if texts_to_embed:
            try:
                embeddings = self._embedding_generator.embed_batch(texts_to_embed)
                if embeddings:
                    for idx, embedding in zip(indices_to_update, embeddings):
                        if embedding:
                            entries[idx]["embedding"] = embedding
            except Exception as e:
                print(f"[WARNING] Batch embedding generation failed: {e}")
                print(f"  Failed to generate embeddings for {len(texts_to_embed)} entries")
