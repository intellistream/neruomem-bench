"""PreRetrieval Operator - 记忆检索前预处理算子

Pipeline 位置: 第 3 层（检索前）
访问权限: 仅允许检索记忆服务（不允许插入/删除）
"""

from __future__ import annotations

import time
from typing import Any

from sage.foundation import MapFunction

from benchmarks.experiment.utils import (
    EmbeddingGenerator,
    LLMGenerator,
    get_required_config,
)

from .base import BasePreRetrievalAction, PreRetrievalInput, PreRetrievalOutput
from .registry import PreRetrievalActionRegistry


class PreRetrieval(MapFunction):
    """记忆检索前的预处理算子"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        services_type = config.get("services.services_type")
        if not services_type:
            raise ValueError("Missing required config: services.services_type")
        self.service_name = services_type.split(".")[-1]

        self._llm_generator = LLMGenerator.from_config(self.config)
        self._embedding_generator = EmbeddingGenerator.from_config(self.config)

        print("\n" + "=" * 80)
        print("📋 [PreRetrieval Init] 模型配置信息")
        print("=" * 80)
        print("🤖 LLM 模型:")
        print(f"   - Model: {self._llm_generator.model_name}")
        print(f"   - Base URL: {self.config.get('runtime.base_url')}")
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

        action_config = self.config.get("operators.pre_retrieval", {})
        self.action_name = get_required_config(self.config, "operators.pre_retrieval.action")
        self.action_type = None

        if self.action_name == "optimize":
            self.action_type = get_required_config(
                self.config, "operators.pre_retrieval.optimize_type", "action=optimize"
            )
        elif self.action_name == "enhancement":
            self.action_type = get_required_config(
                self.config, "operators.pre_retrieval.enhancement_type", "action=enhancement"
            )

        action_key = (
            f"{self.action_name}.{self.action_type}" if self.action_type else self.action_name
        )
        action_class = PreRetrievalActionRegistry.get(action_key)
        self.action: BasePreRetrievalAction = action_class(action_config)

        if hasattr(self.action, "set_llm_generator"):
            self.action.set_llm_generator(self._llm_generator)
        if hasattr(self.action, "set_embedding_generator"):
            self.action.set_embedding_generator(self._embedding_generator)

    def execute(self, data: dict[str, Any]) -> dict[str, Any]:
        start_time = time.perf_counter()

        runtime_config = {
            "embedding_base_url": self.config.get("runtime.embedding_base_url"),
            "llm_base_url": self.config.get("runtime.llm_base_url"),
        }
        data["_runtime_config"] = runtime_config

        input_data = PreRetrievalInput(
            data=data, config=self.config.get("operators.pre_retrieval", {})
        )
        output: PreRetrievalOutput = self.action.execute(input_data)

        data.pop("_runtime_config", None)

        if output.metadata.get("needs_embedding") and self._embedding_generator:
            query_embedding = self._embedding_generator.embed(output.query)
            output.query_embedding = query_embedding
        data["question"] = output.query
        if output.query_embedding:
            data["query_embedding"] = output.query_embedding
        if output.retrieve_mode:
            data["retrieve_mode"] = output.retrieve_mode
        if output.retrieve_params:
            data["retrieve_params"] = output.retrieve_params
        else:
            data["retrieve_params"] = {}

        if output.metadata and "keywords" in output.metadata:
            data["retrieve_params"]["extracted_keywords"] = output.metadata["keywords"]

        if output.metadata:
            data.setdefault("metadata", {}).update(output.metadata)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        data.setdefault("stage_timings", {})["pre_retrieval_ms"] = elapsed_ms

        print(f"  [PreRetrieval] 动作: {self.action_name} | 耗时: {elapsed_ms:.2f}ms", flush=True)

        return data
