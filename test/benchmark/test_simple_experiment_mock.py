"""Mock 适配器 + LoCoMo 黑盒基准测试

完全离线运行，不依赖任何外部服务（LLM / Embedding / mem0 / Qdrant）。
通过 Mock 替换两个外部依赖：
    1. LLM API          → 固定返回 "mock answer"
    2. mem0 记忆体       → MockMemoryAdapter（内存字典实现）

测试架构：
    组件级集成测试 —— 直接驱动算子链，绕过 Sage Runtime 多源限制。
    MockMemoryAdapter 通过 call_service mock 直接注入算子。

    数据流：
        MockLocomoLoader → (遍历对话) → SimpleMemoryAdd
                         → (阈值触发) → SimpleMemorySearch → MemoryEvaluation

测试覆盖：
    Test 1: MockMemoryAdapter 基础 add/search 功能
    Test 2: 完整黑盒 LoCoMo 基准流程（插入 + 检索 + 问答评估）

运行方式：
    python -m pytest test/benchmark/test_simple_experiment_mock.py -v
    python -m test.benchmark.test_simple_experiment_mock
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from benchmarks.experiment.utils.dataloader.base import BaseDataLoader
from benchmarks.experiment.utils.dataloader import DataLoaderFactory


# ============================================================================
# Mock 组件
# ============================================================================


class MockLocomoLoader(BaseDataLoader):
    """与 test_fifo_locomo_mock.py 相同的 Mock LoCoMo 数据集。

    结构：1 个 task ("mock-simple-01")，1 个 session，3 轮对话，2 个问题。
    """

    DIALOGS = {
        0: [
            {"speaker": "Alice", "text": "Today the weather is really nice, sunny and warm."},
            {"speaker": "Bob",   "text": "Yes, I heard it might rain tomorrow though."},
            {"speaker": "Alice", "text": "Actually, I really like rainy days. They feel peaceful."},
        ],
    }

    QUESTIONS = [
        {
            "question": "What does Alice think about the weather?",
            "answer":   "Alice thinks the weather is really nice, sunny and warm.",
            "category": "factual",
            "evidence": "Alice: Today the weather is really nice, sunny and warm.",
            "visible_session": 0,
            "visible_dialog":  1,
        },
        {
            "question": "Does Alice like rainy days?",
            "answer":   "Yes, Alice likes rainy days. They feel peaceful.",
            "category": "factual",
            "evidence": "Alice: Actually, I really like rainy days. They feel peaceful.",
            "visible_session": 0,
            "visible_dialog":  2,
        },
    ]

    @property
    def dataset_name(self) -> str:
        return "mock_simple_locomo"

    def get_dialog(
        self, task_id: str, session_x: int, dialog_y: int
    ) -> list[dict[str, Any]]:
        messages = self.DIALOGS.get(session_x, [])
        return [messages[i] for i in range(dialog_y, min(dialog_y + 2, len(messages)))]

    def get_evaluation(
        self, task_id: str, session_x: int, dialog_y: int
    ) -> list[dict[str, Any]]:
        return [
            q for q in self.QUESTIONS
            if (q["visible_session"] < session_x)
            or (q["visible_session"] == session_x and q["visible_dialog"] <= dialog_y)
        ]

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
            "dialogs":  2,
            "questions": len(self.QUESTIONS),
        }


class MockMemoryAdapter:
    """纯内存 Mock 适配器，模拟 add / search / clear / get_stats 接口。"""

    def __init__(self) -> None:
        self._store: dict[str, str] = {}
        self._counter = 0

    def add(self, text: str, metadata: dict[str, Any] | None = None) -> str:
        self._counter += 1
        entry_id = f"mock-{self._counter}"
        self._store[entry_id] = text
        return entry_id

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """返回所有记忆（按插入顺序，截断到 top_k）。"""
        results = []
        for entry_id, text in list(self._store.items())[-top_k:]:
            results.append(
                {"id": entry_id, "text": text, "score": 1.0, "metadata": {}}
            )
        return results

    def clear(self) -> None:
        self._store.clear()
        self._counter = 0

    def get_stats(self) -> dict[str, Any]:
        return {"adapter": "mock", "memory_count": len(self._store)}


class MockLLMGenerator:
    """固定返回答案的 Mock LLM。"""

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


# ============================================================================
# 配置构造
# ============================================================================


def _make_config() -> dict[str, Any]:
    return {
        "runtime": {
            "dataset": "mock_simple_locomo",
            "memory_name": "mock_adapter",
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
            "services_type": "simple.mock",
            "mock": {
                "top_k": 5,
            },
        },
        "operators": {
            "simple_retrieval": {
                "conversation_format_prompt": "The following are relevant memories:\n",
            },
        },
    }


def _write_config_yaml(config: dict, tmpdir: str) -> str:
    import yaml

    config_path = os.path.join(tmpdir, "test_simple_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    return config_path


def _make_call_service(mock_adapter: MockMemoryAdapter):
    """创建 mock call_service，将算子服务调用路由到 MockMemoryAdapter。"""

    def call_service(service_name, method=None, timeout=None, **kwargs):
        fn = getattr(mock_adapter, method, None)
        if fn is None:
            raise AttributeError(
                f"MockMemoryAdapter has no method '{method}'"
            )
        return fn(**kwargs)

    return call_service


# ============================================================================
# Test 1: MockMemoryAdapter 基础功能
# ============================================================================


def test_mock_adapter_basic():
    """验证 MockMemoryAdapter 的 add / search / clear 基础操作。"""
    print("\n--- Test 1: MockMemoryAdapter 基础功能 ---")

    adapter = MockMemoryAdapter()

    texts = [
        "Alice: Today the weather is really nice.",
        "Bob: I heard it might rain tomorrow.",
        "Alice: I really like rainy days.",
    ]
    ids = []
    for text in texts:
        entry_id = adapter.add(text, metadata={"type": "dialog"})
        assert entry_id, "add 应返回非空 ID"
        ids.append(entry_id)
    print(f"  插入 {len(ids)} 条: {ids}")

    results = adapter.search("weather", top_k=10)
    assert len(results) == 3, f"search 应返回全部 3 条，实际 {len(results)}"
    for r in results:
        assert "text" in r and "id" in r
    print(f"  search 返回 {len(results)} 条")

    stats = adapter.get_stats()
    assert stats["memory_count"] == 3
    print(f"  get_stats: {stats}")

    adapter.clear()
    results_after = adapter.search("weather", top_k=10)
    assert len(results_after) == 0, "clear 后 search 应返回空"
    print("  clear 后 search 返回 0 条")

    print("  MockMemoryAdapter 基础功能正常")


# ============================================================================
# Test 2: 完整黑盒 LoCoMo 基准流程
# ============================================================================


def test_simple_experiment_full_flow():
    """组件级集成测试：Mock LLM + MockAdapter + LoCoMo 黑盒算子链。

    模拟 simple_pipeline 的完整数据流：
    1. 读取 MockLocomoLoader 数据（等价 MemorySource）
    2. 对每批对话执行 SimpleMemoryAdd
    3. 在阈值处触发 SimpleMemorySearch → MemoryEvaluation
    4. 验证插入条数和问答结果
    """
    print("\n--- Test 2: 黑盒 LoCoMo 完整基准流程 ---")

    from benchmarks.experiment.utils import RuntimeConfig, process_logger

    DataLoaderFactory.register("mock_simple_locomo", MockLocomoLoader)

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_config = _make_config()
        config_path = _write_config_yaml(raw_config, tmpdir)
        config = RuntimeConfig.load(config_path, task_id="mock-simple-01")

        os.environ["PROCESS_LOG_DIR"] = os.path.join(tmpdir, "logs")
        process_logger.setup("mock_simple_locomo", "mock_adapter", "mock-simple-01")

        mock_adapter = MockMemoryAdapter()
        mock_call_service = _make_call_service(mock_adapter)

        with patch(
            "benchmarks.experiment.libs.memory_evaluation.LLMGenerator",
            MockLLMGenerator,
        ):
            from benchmarks.experiment.libs.memory_evaluation import MemoryEvaluation
            from benchmarks.experiment.utils import calculate_test_thresholds
            from benchmarks.simple_experiment.libs.simple_memory_add import SimpleMemoryAdd
            from benchmarks.simple_experiment.libs.simple_memory_search import SimpleMemorySearch

            # 实例化算子并注入 mock call_service
            mem_add = SimpleMemoryAdd(config)
            mem_add.call_service = mock_call_service

            mem_search = SimpleMemorySearch(config)
            mem_search.call_service = mock_call_service

            mem_eval = MemoryEvaluation(config)

            # ----- Phase 1: 模拟 MemorySource 遍历对话 -----
            loader = MockLocomoLoader()
            task_id = "mock-simple-01"
            total_questions = loader.question_count(task_id)
            test_thresholds = calculate_test_thresholds(total_questions, segments=2)
            next_threshold_idx = 0
            total_added = 0
            all_answers = []

            print(f"  数据集: {loader.dataset_name}")
            print(
                f"  问题数: {total_questions}, 测试阈值: {test_thresholds}"
            )

            for session_id, max_dialog_idx in loader.sessions(task_id):
                dialog_ptr = 0
                while dialog_ptr <= max_dialog_idx:
                    dialogs = loader.get_dialog(task_id, session_id, dialog_ptr)
                    dialog_len = len(dialogs) if dialogs else 2

                    add_data = {
                        "task_id":       task_id,
                        "session_id":    session_id,
                        "dialog_id":     dialog_ptr,
                        "dialogs":       dialogs,
                        "packet_idx":    dialog_ptr,
                        "total_packets": loader.dialog_count(task_id),
                        "is_session_end": (dialog_ptr + dialog_len) > max_dialog_idx,
                    }

                    # ── 插入 ──
                    data = mem_add.execute(dict(add_data))
                    entries_added = data.get("add_stats", {}).get("inserted", 0)
                    total_added += entries_added
                    print(f"  Dialog {dialog_ptr}: 插入 {entries_added} 条")

                    # ── 阈值检查 ──
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
                            f"  阈值触发: {current_count} >= "
                            f"{test_thresholds[next_threshold_idx]}"
                        )
                        for q_idx, qa in enumerate(current_questions):
                            test_data = {
                                "task_id":           task_id,
                                "session_id":        session_id,
                                "dialog_id":         dialog_ptr,
                                "dialogs":           dialogs,
                                "question":          qa["question"],
                                "question_idx":      q_idx + 1,
                                "question_metadata": qa,
                            }
                            # 检索
                            search_out = mem_search.execute(dict(test_data))
                            assert "history_text" in search_out, \
                                "SimpleMemorySearch 应写入 history_text"
                            assert "memory_data" in search_out, \
                                "SimpleMemorySearch 应写入 memory_data"

                            # LLM 评估
                            eval_out = mem_eval.execute(dict(search_out))
                            assert "answer" in eval_out, \
                                "MemoryEvaluation 应写入 answer"
                            all_answers.append(eval_out["answer"])
                            print(
                                f"    Q{q_idx + 1}: {qa['question']}\n"
                                f"    A: {eval_out['answer']}"
                            )

                        next_threshold_idx += 1

                    dialog_ptr += dialog_len

        # ----- 验证 -----
        assert total_added > 0, "至少应插入 1 条记忆"
        assert len(all_answers) > 0, "至少应生成 1 条问答"
        for ans in all_answers:
            assert ans, "答案不应为空"

        print(
            f"\n  总插入: {total_added} 条 | 总回答: {len(all_answers)} 个"
        )
        print("  黑盒 LoCoMo 流程全部通过")

        process_logger.close()


# ============================================================================
# 直接运行入口
# ============================================================================

if __name__ == "__main__":
    test_mock_adapter_basic()
    test_simple_experiment_full_flow()
    print("\n✅ 所有测试通过")
