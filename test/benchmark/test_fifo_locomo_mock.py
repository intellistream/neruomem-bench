"""Mock LLM + FIFO Queue + LocoMo 基准运行测试

完全离线运行，不依赖外部 LLM / Embedding 服务和真实 LocoMo 数据集。
通过 Mock 替换三个外部依赖：
    1. LLM API → 固定返回 "mock answer"
    2. Embedding API → 不启用
    3. LocoMo 数据集 → MockLocomoLoader（3 条对话 + 2 个问题）

测试架构：
    组件级集成测试 —— 直接驱动算子链，绕过 Sage Runtime 多源限制。
    FIFO 记忆服务通过 MemoryServiceRegistry 直接创建，算子通过 mock call_service
    与真实 FIFO 服务交互。

    数据流：
        MockLocomoLoader → (遍历对话) → PreInsert → MemoryInsert → PostInsert
                         → (阈值触发) → PreRetrieval → MemoryRetrieval → PostRetrieval → MemoryEvaluation

测试覆盖：
    Test 1: FIFO 服务基础 insert/retrieve
    Test 2: 完整 LocoMo 基准流程（插入 + 检索 + 问答评估）

运行方式：
    python -m pytest test/benchmark/test_fifo_locomo_mock.py -v
    python -m test.benchmark.test_fifo_locomo_mock
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# Allow running as: python test/benchmark/test_fifo_locomo_mock.py
_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
from typing import Any
from unittest.mock import patch

import pytest

from benchmarks.experiment.utils.dataloader.base import BaseDataLoader
from benchmarks.experiment.utils.dataloader import DataLoaderFactory


# ============================================================================
# Mock 组件
# ============================================================================


class MockLocomoLoader(BaseDataLoader):
    """模拟 LocoMo 数据集

    结构：1 个 task ("mock-01")，1 个 session，3 轮对话，2 个测试问题。

    对话内容：
        session 0, dialog 0-1: Alice 和 Bob 讨论天气
        session 0, dialog 2:   Alice 说她喜欢下雨天

    测试问题：
        Q1: "What does Alice think about the weather?" (visible after dialog 1)
        Q2: "Does Alice like rainy days?" (visible after dialog 2)
    """

    DIALOGS = {
        0: [
            {"speaker": "Alice", "text": "Today the weather is really nice, sunny and warm."},
            {"speaker": "Bob", "text": "Yes, I heard it might rain tomorrow though."},
            {"speaker": "Alice", "text": "Actually, I really like rainy days. They feel peaceful."},
        ],
    }

    QUESTIONS = [
        {
            "question": "What does Alice think about the weather?",
            "answer": "Alice thinks the weather is really nice, sunny and warm.",
            "category": "factual",
            "evidence": "Alice: Today the weather is really nice, sunny and warm.",
            "visible_session": 0,
            "visible_dialog": 1,
        },
        {
            "question": "Does Alice like rainy days?",
            "answer": "Yes, Alice likes rainy days. They feel peaceful.",
            "category": "factual",
            "evidence": "Alice: Actually, I really like rainy days. They feel peaceful.",
            "visible_session": 0,
            "visible_dialog": 2,
        },
    ]

    @property
    def dataset_name(self) -> str:
        return "mock_locomo"

    def get_dialog(self, task_id: str, session_x: int, dialog_y: int) -> list[dict[str, Any]]:
        messages = self.DIALOGS.get(session_x, [])
        result = []
        for i in range(dialog_y, min(dialog_y + 2, len(messages))):
            result.append(messages[i])
        return result

    def get_evaluation(self, task_id: str, session_x: int, dialog_y: int) -> list[dict[str, Any]]:
        visible = []
        for q in self.QUESTIONS:
            if (q["visible_session"] < session_x) or (
                q["visible_session"] == session_x and q["visible_dialog"] <= dialog_y
            ):
                visible.append(q)
        return visible

    def sessions(self, task_id: str) -> list[tuple[int, int]]:
        return [(0, 2)]

    def question_count(self, task_id: str) -> int:
        return len(self.QUESTIONS)

    def dialog_count(self, task_id: str) -> int:
        return 2

    def message_count(self, task_id: str) -> int:
        return 3

    def statistics(self, task_id: str) -> dict[str, Any]:
        return {
            "dataset": self.dataset_name,
            "task_id": task_id,
            "sessions": 1,
            "messages": 3,
            "dialogs": 2,
            "questions": len(self.QUESTIONS),
        }


class MockLLMGenerator:
    """返回固定答案的 Mock LLM 生成器"""

    def __init__(self, **kwargs):
        self.model_name = "mock-llm"
        self.max_tokens = 64
        self.temperature = 0.0
        self.seed = 42

    @classmethod
    def from_config(cls, config, prefix="runtime"):
        return cls()

    def generate(self, prompt: str, **kwargs) -> str:
        return "mock answer based on context"

    def generate_json(self, prompt: str, default=None, **kwargs):
        return default or {}

    def generate_triples(self, prompt: str, **kwargs):
        return [], []


class MockEmbeddingGenerator:
    """不提供 embedding 的 Mock"""

    def __init__(self, **kwargs):
        self.model_name = "mock-emb"
        self.base_url = None

    @classmethod
    def from_config(cls, config):
        return cls()

    def embed(self, text: str):
        return None

    def embed_batch(self, texts: list[str]):
        return None

    def is_available(self) -> bool:
        return False


# ============================================================================
# 配置构造
# ============================================================================


def _make_config() -> dict[str, Any]:
    """构造最小测试配置"""
    return {
        "runtime": {
            "dataset": "mock_locomo",
            "memory_name": "fifo_test",
            "test_segments": 2,
            "api_key": "mock-key",
            "base_url": "http://localhost:0/v1",
            "model_name": "mock-llm",
            "max_tokens": 64,
            "temperature": 0.0,
            "seed": 42,
            "memory_insert_verbose": False,
            "memory_test_verbose": False,
        },
        "services": {
            "services_type": "partitional.fifo_queue",
            "fifo_queue": {
                "max_capacity": 100,
                "max_size": 100,
                "vector_dim": 0,
                "retrieval_top_k": 10,
            },
        },
        "operators": {
            "pre_insert": {"action": "none"},
            "post_insert": {"action": "none"},
            "pre_retrieval": {"action": "none"},
            "post_retrieval": {
                "action": "none",
                "conversation_format_prompt": "The following is some history information.\n",
            },
        },
    }


def _write_config_yaml(config: dict, tmpdir: str) -> str:
    import yaml

    config_path = os.path.join(tmpdir, "test_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    return config_path


def _create_fifo_service(config: dict[str, Any]):
    """直接创建 FIFO 记忆服务实例"""
    from sage.neuromem.memory_collection import UnifiedCollection
    from sage.neuromem.services import MemoryServiceRegistry

    collection = UnifiedCollection(name="fifo_test")
    fifo_config = config.get("services", {}).get("fifo_queue", {})
    return MemoryServiceRegistry.create("fifo_queue", collection, fifo_config)


def _make_call_service(fifo_service):
    """创建 mock call_service 函数，将算子的服务调用路由到真实 FIFO 服务"""

    def call_service(service_name, method=None, timeout=None, **kwargs):
        fn = getattr(fifo_service, method, None)
        if fn is None:
            raise AttributeError(f"FIFO service has no method '{method}'")
        return fn(**kwargs)

    return call_service


# ============================================================================
# Test 1: FIFO 服务基础功能
# ============================================================================


def test_fifo_service_basic():
    """验证 FIFO 服务的 insert / retrieve 基础操作"""
    print("\n--- Test 1: FIFO 服务基础功能 ---")

    config = _make_config()
    service = _create_fifo_service(config)

    # 插入 3 条记录
    ids = []
    texts = [
        "Alice: Today the weather is really nice.",
        "Bob: I heard it might rain tomorrow.",
        "Alice: I really like rainy days.",
    ]
    for text in texts:
        entry_id = service.insert(entry=text, metadata={"type": "dialog"})
        ids.append(entry_id)
        assert entry_id, f"insert 应返回非空 ID"
    print(f"  插入 {len(ids)} 条记录")

    # 检索
    results = service.retrieve(query="weather", top_k=10)
    assert len(results) == 3, f"FIFO 应返回全部 {len(texts)} 条, 实际 {len(results)}"
    for r in results:
        assert "text" in r
        assert "id" in r
    print(f"  检索返回 {len(results)} 条")

    # get_recent
    recent = service.get_recent(limit=2)
    assert len(recent) == 2, f"get_recent(2) 应返回 2 条, 实际 {len(recent)}"

    print("  ✅ FIFO 服务基础功能正常")


# ============================================================================
# Test 2: 完整 LocoMo 基准流程
# ============================================================================


def test_fifo_locomo_full_flow():
    """组件级集成测试：Mock LLM + FIFO + LocoMo 算子链

    模拟 memory_test_pipeline 的完整数据流：
    1. 读取 MockLocomoLoader 数据（等价 MemorySource）
    2. 对每批对话执行插入链：PreInsert → MemoryInsert → PostInsert
    3. 在阈值处触发测试链：PreRetrieval → MemoryRetrieval → PostRetrieval → MemoryEvaluation
    4. 验证插入条数和问答结果
    """
    print("\n--- Test 2: LocoMo 完整基准流程 ---")

    from benchmarks.experiment.utils import RuntimeConfig, process_logger

    DataLoaderFactory.register("mock_locomo", MockLocomoLoader)

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_config = _make_config()
        config_path = _write_config_yaml(raw_config, tmpdir)
        config = RuntimeConfig.load(config_path, task_id="mock-01")

        os.environ["PROCESS_LOG_DIR"] = os.path.join(tmpdir, "logs")
        process_logger.setup("mock_locomo", "fifo_test", "mock-01")

        # 创建 FIFO 服务
        fifo_service = _create_fifo_service(raw_config)
        mock_call_service = _make_call_service(fifo_service)

        # Patch LLM/Embedding 生成器
        with (
            patch(
                "benchmarks.experiment.libs.pre_insert.operator.LLMGenerator",
                MockLLMGenerator,
            ),
            patch(
                "benchmarks.experiment.libs.pre_insert.operator.EmbeddingGenerator",
                MockEmbeddingGenerator,
            ),
            patch(
                "benchmarks.experiment.libs.pre_retrieval.operator.LLMGenerator",
                MockLLMGenerator,
            ),
            patch(
                "benchmarks.experiment.libs.pre_retrieval.operator.EmbeddingGenerator",
                MockEmbeddingGenerator,
            ),
            patch(
                "benchmarks.experiment.libs.post_insert.operator.LLMGenerator",
                MockLLMGenerator,
            ),
            patch(
                "benchmarks.experiment.libs.post_insert.operator.EmbeddingGenerator",
                MockEmbeddingGenerator,
            ),
            patch(
                "benchmarks.experiment.libs.post_retrieval.operator.LLMGenerator",
                MockLLMGenerator,
            ),
            patch(
                "benchmarks.experiment.libs.post_retrieval.operator.EmbeddingGenerator",
                MockEmbeddingGenerator,
            ),
            patch(
                "benchmarks.experiment.libs.memory_evaluation.LLMGenerator",
                MockLLMGenerator,
            ),
        ):
            from benchmarks.experiment.libs.memory_evaluation import MemoryEvaluation
            from benchmarks.experiment.libs.memory_insert import MemoryInsert
            from benchmarks.experiment.libs.memory_retrieval import MemoryRetrieval
            from benchmarks.experiment.libs.post_insert import PostInsert
            from benchmarks.experiment.libs.post_retrieval import PostRetrieval
            from benchmarks.experiment.libs.pre_insert import PreInsert
            from benchmarks.experiment.libs.pre_retrieval import PreRetrieval
            from benchmarks.experiment.utils import calculate_test_thresholds

            # 实例化算子
            pre_insert = PreInsert(config)
            mem_insert = MemoryInsert(config)
            post_insert = PostInsert(config)
            pre_retrieval = PreRetrieval(config)
            mem_retrieval = MemoryRetrieval(config)
            post_retrieval = PostRetrieval(config)
            mem_evaluation = MemoryEvaluation(config)

            # 注入 call_service（路由到真实 FIFO 服务）
            for op in [mem_insert, mem_retrieval, post_insert, post_retrieval]:
                op.call_service = mock_call_service

            # ----- Phase 1: 模拟 MemorySource 遍历对话 -----
            loader = MockLocomoLoader()
            task_id = "mock-01"
            total_questions = loader.question_count(task_id)
            test_thresholds = calculate_test_thresholds(total_questions, segments=2)
            next_threshold_idx = 0
            total_inserted = 0
            all_answers = []

            print(f"  数据集: {loader.dataset_name}")
            print(f"  问题数: {total_questions}, 测试阈值: {test_thresholds}")

            for session_id, max_dialog_idx in loader.sessions(task_id):
                dialog_ptr = 0
                while dialog_ptr <= max_dialog_idx:
                    dialogs = loader.get_dialog(task_id, session_id, dialog_ptr)
                    dialog_len = len(dialogs) if dialogs else 2

                    insert_data = {
                        "task_id": task_id,
                        "session_id": session_id,
                        "dialog_id": dialog_ptr,
                        "dialogs": dialogs,
                        "packet_idx": dialog_ptr,
                        "total_packets": loader.dialog_count(task_id),
                        "is_session_end": (dialog_ptr + dialog_len) > max_dialog_idx,
                    }

                    # ---- 插入链 ----
                    data = pre_insert.execute(dict(insert_data))
                    data = mem_insert.execute(data)
                    data = post_insert.execute(data)

                    entries_inserted = data.get("insert_stats", {}).get("inserted", 0)
                    total_inserted += entries_inserted
                    print(
                        f"  Dialog {dialog_ptr}: 插入 {entries_inserted} 条"
                    )

                    # ---- 问题触发检查 ----
                    visible_dialog = dialog_ptr + dialog_len - 1
                    current_questions = loader.get_evaluation(
                        task_id, session_id, visible_dialog
                    )
                    current_count = len(current_questions)

                    should_test = (
                        next_threshold_idx < len(test_thresholds)
                        and current_count >= test_thresholds[next_threshold_idx]
                    )

                    if should_test:
                        print(
                            f"  阈值触发: {current_count} >= {test_thresholds[next_threshold_idx]}"
                            f" (阈值 {next_threshold_idx + 1}/{len(test_thresholds)})"
                        )
                        for q_idx, qa in enumerate(current_questions):
                            test_data = {
                                "task_id": task_id,
                                "session_id": session_id,
                                "dialog_id": dialog_ptr,
                                "dialogs": dialogs,
                                "question": qa["question"],
                                "question_idx": q_idx + 1,
                                "question_metadata": qa,
                            }

                            # ---- 测试链 ----
                            td = pre_retrieval.execute(dict(test_data))
                            td = mem_retrieval.execute(td)
                            td = post_retrieval.execute(td)
                            td = mem_evaluation.execute(td)

                            answer = td.get("answer", "")
                            all_answers.append(
                                {
                                    "question": qa["question"],
                                    "predicted": answer,
                                    "gold": qa["answer"],
                                }
                            )
                            print(f"    Q{q_idx + 1}: {qa['question'][:50]}")
                            print(f"    A: {answer[:50]}")

                        next_threshold_idx += 1

                    dialog_ptr += dialog_len

        # ----- 验证 -----
        print(f"\n  总插入: {total_inserted} 条")
        print(f"  总问答: {len(all_answers)} 个")

        assert total_inserted > 0, "应至少插入 1 条记忆"
        assert len(all_answers) > 0, "应至少回答 1 个问题"

        for ans in all_answers:
            assert ans["predicted"], f"问题 '{ans['question']}' 的回答不应为空"

        # 清理
        os.environ.pop("PROCESS_LOG_DIR", None)
        from benchmarks.experiment.utils.helpers.process_logger import ProcessLogger

        process_logger.close()
        ProcessLogger._initialized = False
        ProcessLogger._instance = None

        print("\n  ✅ LocoMo 完整基准流程通过")


# ============================================================================
# 主入口
# ============================================================================


def main() -> int:
    print("=" * 60)
    print("Mock FIFO + LocoMo 基准运行测试")
    print("=" * 60)

    test_fifo_service_basic()
    test_fifo_locomo_full_flow()

    print("\n" + "=" * 60)
    print("✅ 全部 FIFO + LocoMo 基准测试通过")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
