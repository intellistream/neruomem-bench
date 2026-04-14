"""mem0 集成冒烟测试（直接组件级驱动）

使用真实 LLM / Embedding 服务和 mini_locomo.json 数据集，
直接驱动 SimpleMemoryAdd + SimpleMemorySearch + MemoryEvaluation 算子链，
绕过 Sage Runtime 避免多 source 限制，等价于 simple_pipeline.py 的完整数据流。

依赖（运行前确认）：
    LLM   服务: http://localhost:18000/v1  (meta-llama/Llama-3.1-8B-Instruct)
    Embed 服务: http://localhost:18001/v1  (BAAI/bge-m3)
    pip install mem0ai faiss-cpu

运行方式：
    cd /home/zrc/mem/neuromem-bench
    python -m test.benchmark.test_mem0_integration

或仅在本机验证 mem0 add/search：
    python -m test.benchmark.test_mem0_integration --smoke-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import tempfile
from pathlib import Path

_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


# ============================================================================
# 配置
# ============================================================================

LLM_BASE_URL   = "http://localhost:18000/v1"
EMBED_BASE_URL = "http://localhost:18001/v1"
LLM_MODEL      = "meta-llama/Llama-3.1-8B-Instruct"
EMBED_MODEL    = "BAAI/bge-m3"
EMBED_DIM      = 1024   # BGE-M3 output dim

TASK_ID = "conv-mini-01"   # mini_locomo.json 里唯一的 task

MEM0_CONFIG = {
    "llm": {
        "provider": "openai",
        "config": {
            "model":            LLM_MODEL,
            "openai_base_url":  LLM_BASE_URL,
            "api_key":          "EMPTY",
            "max_tokens":       512,
            "temperature":      0.0,
        },
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model":            EMBED_MODEL,
            "openai_base_url":  EMBED_BASE_URL,
            "api_key":          "EMPTY",
            # embedding_dims 不传：vLLM 的 BGE-M3 不支持维度截断
        },
    },
    "vector_store": {
        "provider": "faiss",
        "config": {
            "collection_name":      "bench_smoke_test",
            "distance_strategy":    "cosine",
            "embedding_model_dims": EMBED_DIM,
        },
    },
}

RUNTIME_CONFIG_DICT = {
    "runtime": {
        "dataset":               "locomo",
        "memory_name":           "mem0",
        "test_segments":         2,
        "api_key":               "EMPTY",
        "base_url":              LLM_BASE_URL,
        "model_name":            LLM_MODEL,
        "max_tokens":            256,
        "temperature":           0.0,
        "seed":                  42,
        "memory_insert_verbose": False,
        "memory_test_verbose":   True,
        "prompt_template": (
            "Answer in a short phrase (3-8 words). "
            "Plain text only. No ** or formatting.\n"
            "- For 'what/who': just the thing/person\n"
            "- For 'when':    just the date/time\n"
            "- For 'why':     start with 'because'\n\n"
            "Q: {question}\nA:"
        ),
    },
    "services": {
        "services_type": "simple.mem0",
        "mem0": {
            "top_k":          5,
            "user_id_scope":  "auto",
            "mem0_config":    MEM0_CONFIG,
        },
    },
    "operators": {
        "simple_retrieval": {
            "conversation_format_prompt": "The following are relevant memories:\n",
        },
    },
}


# ============================================================================
# 工具
# ============================================================================

def _check_service(label: str, url: str) -> bool:
    import urllib.request
    try:
        urllib.request.urlopen(url, timeout=5)
        print(f"[OK]  {label} → {url}")
        return True
    except Exception as e:
        print(f"[ERR] {label} → {url}  ({e})")
        return False


def _write_config_yaml(config: dict, path: str) -> None:
    import yaml
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def _make_call_service(mem0_adapter):
    """将算子的 call_service 路由到真实 mem0 适配器。"""
    def call_service(service_name, method=None, timeout=None, **kwargs):
        fn = getattr(mem0_adapter, method, None)
        if fn is None:
            raise AttributeError(f"Adapter has no method '{method}'")
        return fn(**kwargs)
    return call_service


# ============================================================================
# Smoke Test: mem0 add / search 基础验证
# ============================================================================

def test_smoke():
    """独立验证 mem0 add / search，不依赖任何 benchmark 基础设施。"""
    print("\n" + "=" * 60)
    print("Smoke Test: mem0 add / search")
    print("=" * 60)

    from mem0 import Memory
    mem = Memory.from_config(MEM0_CONFIG)
    user_id = "smoke_test_user"

    # 确保干净状态
    try:
        mem.delete_all(user_id=user_id)
    except Exception:
        pass

    # 插入 3 段对话
    texts = [
        "Alice: Hey Bob, I just started a new job at InnovateTech last Monday.\n"
        "Bob: Congratulations! What role are you in?",
        "Alice: I'm a senior data engineer on the infrastructure team.\n"
        "Bob: That sounds exciting. Do you work remotely or in the office?",
        "Alice: Mostly remote. Our office is in San Francisco but I live in Austin.",
    ]
    for i, text in enumerate(texts):
        t0 = time.perf_counter()
        result = mem.add(text, user_id=user_id)
        ms = (time.perf_counter() - t0) * 1000
        if isinstance(result, dict):
            records = result.get("results", [])
        else:
            records = result or []
        print(f"  add [{i+1}]: {len(records)} memories stored ({ms:.0f}ms)")

    # 检索
    queries = ["Where does Alice work?", "Where does Alice live?"]
    for query in queries:
        t0 = time.perf_counter()
        result = mem.search(query, user_id=user_id, limit=3)
        ms = (time.perf_counter() - t0) * 1000
        if isinstance(result, dict):
            hits = result.get("results", [])
        else:
            hits = result or []
        print(f"  search '{query}': {len(hits)} results ({ms:.0f}ms)")
        for h in hits:
            text_field = h.get("memory", h.get("text", ""))
            print(f"    [{h.get('score', 0):.3f}] {text_field[:80]}")

    print("Smoke test passed.")


# ============================================================================
# Integration Test: 完整 LoCoMo 黑盒流程
# ============================================================================

def test_integration():
    """完整黑盒基准流程：mini_locomo.json + mem0 + LLM 评估。"""
    print("\n" + "=" * 60)
    print("Integration Test: mem0 × mini_locomo")
    print("=" * 60)

    from benchmarks.experiment.utils import RuntimeConfig, calculate_test_thresholds, process_logger
    from benchmarks.experiment.utils.dataloader import DataLoaderFactory
    from benchmarks.experiment.libs.memory_evaluation import MemoryEvaluation
    from benchmarks.simple_experiment.adapters.adapter_registry import AdapterRegistry
    from benchmarks.simple_experiment.libs.simple_memory_add import SimpleMemoryAdd
    from benchmarks.simple_experiment.libs.simple_memory_search import SimpleMemorySearch

    # 注册 mem0 适配器（触发 mem0_adapter.py 中的 @AdapterRegistry.register）
    import benchmarks.simple_experiment.adapters.mem0_adapter  # noqa: F401

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "config.yaml")
        _write_config_yaml(RUNTIME_CONFIG_DICT, config_path)
        config = RuntimeConfig.load(config_path, task_id=TASK_ID)

        os.environ["PROCESS_LOG_DIR"] = os.path.join(tmpdir, "logs")
        process_logger.setup("locomo", "mem0", TASK_ID)

        # 创建 mem0 适配器实例
        adapter_config = dict(RUNTIME_CONFIG_DICT["services"]["mem0"])
        adapter_config["user_id"] = f"integration_{int(time.time())}"
        adapter = AdapterRegistry.create("mem0", adapter_config)

        mock_call = _make_call_service(adapter)

        # 实例化算子
        mem_add = SimpleMemoryAdd(config)
        mem_add.call_service = mock_call

        mem_search = SimpleMemorySearch(config)
        mem_search.call_service = mock_call

        mem_eval = MemoryEvaluation(config)

        # 数据集
        loader = DataLoaderFactory.create("locomo")
        total_questions = loader.question_count(TASK_ID)
        test_thresholds = calculate_test_thresholds(total_questions, segments=2)
        print(f"  Task: {TASK_ID}  |  questions: {total_questions}  |  thresholds: {test_thresholds}")

        next_threshold_idx = 0
        total_added = 0
        all_qa: list[dict] = []
        t_start = time.perf_counter()

        for session_id, max_dialog_idx in loader.sessions(TASK_ID):
            dialog_ptr = 0
            while dialog_ptr <= max_dialog_idx:
                dialogs = loader.get_dialog(TASK_ID, session_id, dialog_ptr)
                dialog_len = len(dialogs) if dialogs else 2

                # ── Add ──
                add_data = {
                    "task_id":       TASK_ID,
                    "session_id":    session_id,
                    "dialog_id":     dialog_ptr,
                    "dialogs":       dialogs,
                    "packet_idx":    dialog_ptr,
                    "total_packets": loader.dialog_count(TASK_ID),
                    "is_session_end": (dialog_ptr + dialog_len) > max_dialog_idx,
                }
                data = mem_add.execute(dict(add_data))
                inserted = data.get("add_stats", {}).get("inserted", 0)
                total_added += inserted
                add_ms = data.get("stage_timings", {}).get("add_ms", 0)
                print(f"  session={session_id} dialog={dialog_ptr}: +{inserted} ({add_ms:.0f}ms)")

                # ── Threshold check ──
                visible_dialog = dialog_ptr + dialog_len - 1
                current_qs = loader.get_evaluation(TASK_ID, session_id, visible_dialog)
                current_count = len(current_qs)

                if (
                    next_threshold_idx < len(test_thresholds)
                    and current_count >= test_thresholds[next_threshold_idx]
                ):
                    print(f"\n  [Threshold {next_threshold_idx+1}] {current_count}/{total_questions} questions visible")
                    for q_idx, qa in enumerate(current_qs):
                        test_data = {
                            "task_id":           TASK_ID,
                            "session_id":        session_id,
                            "dialog_id":         dialog_ptr,
                            "dialogs":           dialogs,
                            "question":          qa["question"],
                            "question_idx":      q_idx + 1,
                            "question_metadata": qa,
                        }
                        search_out = mem_search.execute(dict(test_data))
                        eval_out   = mem_eval.execute(dict(search_out))
                        answer     = eval_out.get("answer", "")
                        ref        = qa.get("answer", "")
                        search_ms  = search_out.get("stage_timings", {}).get("search_ms", 0)
                        print(
                            f"  Q{q_idx+1}: {qa['question']}\n"
                            f"    Pred : {answer}\n"
                            f"    Ref  : {ref}\n"
                            f"    Search: {search_ms:.0f}ms"
                        )
                        all_qa.append({"question": qa["question"], "predicted": answer, "reference": ref})
                    next_threshold_idx += 1
                    print()

                dialog_ptr += dialog_len

        elapsed = time.perf_counter() - t_start
        stats = adapter.get_stats()
        print(f"\n{'=' * 60}")
        print(f"  Done in {elapsed:.1f}s")
        print(f"  Dialogs added  : {total_added}")
        print(f"  QA pairs answered: {len(all_qa)}")
        print(f"  Memory stats   : {stats}")
        print("=" * 60)

        process_logger.close()
    return all_qa


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-only", action="store_true", help="只运行 smoke test")
    args = parser.parse_args()

    print("Pre-flight: checking services...")
    ok_llm   = _check_service("LLM  (Llama-3.1-8B)", f"{LLM_BASE_URL}/models")
    ok_embed = _check_service("Embed (BGE-M3)",       f"{EMBED_BASE_URL}/models")
    if not (ok_llm and ok_embed):
        print("\n[ERROR] 请先启动模型服务再运行集成测试。")
        sys.exit(1)

    test_smoke()

    if not args.smoke_only:
        test_integration()

    print("\n✅ 所有测试完成")
