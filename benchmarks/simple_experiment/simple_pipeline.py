"""simple_pipeline.py — 黑盒记忆体基准测试入口

三层 Pipeline 架构（与 memory_test_pipeline.py 完全一致）：
    主 Pipeline:     MemorySource → SimplePipelineCaller → MemorySink
    记忆插入 Pipeline: PipelineServiceSource → SimpleMemoryAdd → PipelineServiceSink
    记忆测试 Pipeline: PipelineServiceSource → SimpleMemorySearch → MemoryEvaluation → PipelineServiceSink

与 memory_test_pipeline.py 的区别：
    - 记忆服务层：NeuromemServiceFactory / TiMServiceFactory  →  SimpleAdapterFactory
    - 插入子 pipeline：PreInsert + MemoryInsert + PostInsert  →  SimpleMemoryAdd
    - 测试子 pipeline 前三段：PreRetrieval + MemoryRetrieval + PostRetrieval  →  SimpleMemorySearch
    - 主调度算子：PipelineCaller  →  SimplePipelineCaller

用法：
    python -m benchmarks.simple_experiment.simple_pipeline \\
        --config benchmarks/simple_experiment/config/mem0_locomo.yaml \\
        --task_id conv-26
"""

from __future__ import annotations

from sage.foundation import CustomLogger
from sage.runtime import LocalEnvironment

from benchmarks.experiment.libs.memory_evaluation import MemoryEvaluation
from benchmarks.experiment.libs.memory_sink import MemorySink
from benchmarks.experiment.libs.memory_source import MemorySource
from benchmarks.experiment.pipeline_service import (
    PipelineBridge,
    PipelineService,
    PipelineServiceSink,
    PipelineServiceSource,
)
from benchmarks.experiment.utils import RuntimeConfig, parse_args, process_logger
from benchmarks.simple_experiment.adapters.simple_adapter_factory import SimpleAdapterFactory
from benchmarks.simple_experiment.libs.simple_memory_add import SimpleMemoryAdd
from benchmarks.simple_experiment.libs.simple_memory_search import SimpleMemorySearch
from benchmarks.simple_experiment.libs.simple_pipeline_caller import SimplePipelineCaller

# 复用 experiment 模块的多 pipeline 提交工具（不重复实现）
from benchmarks.experiment.memory_test_pipeline import (
    _LocalRuntimeContext,
    _inject_ctx,
    _partition_by_source,
    _submit_multi_pipeline,
)


def main() -> None:
    """黑盒记忆体基准测试主函数"""
    CustomLogger.disable_global_console_debug()

    args = parse_args()
    config = RuntimeConfig.load(args.config, args.task_id)

    dataset = config.get("runtime.dataset", "default")
    task_id = config.get("task_id", "unknown")
    memory_name = config.get("runtime.memory_name", "default")
    process_logger.setup(dataset, memory_name, task_id)

    env = LocalEnvironment("simple_memory_experiment")

    # ── 注册适配器服务 ────────────────────────────────────────────────────
    services_type = config.get("services.services_type")
    if not services_type:
        raise ValueError("Missing required config: services.services_type")

    factory = SimpleAdapterFactory.create(services_type, config)
    adapter_name = services_type.split(".")[-1]
    env.register_service_factory(adapter_name, factory)

    pipeline_service_timeout = config.get("runtime.pipeline_service_timeout", 300.0)

    # ── 注册两个子 pipeline 桥接服务 ────────────────────────────────────
    add_bridge = PipelineBridge()
    env.register_service(
        "memory_add_service",
        PipelineService,
        add_bridge,
        request_timeout=pipeline_service_timeout,
    )

    search_bridge = PipelineBridge()
    env.register_service(
        "memory_search_service",
        PipelineService,
        search_bridge,
        request_timeout=pipeline_service_timeout,
    )

    # ── 构建三条子 pipeline ──────────────────────────────────────────────

    # 记忆插入子 pipeline（黑盒单算子）
    (
        env.from_source(PipelineServiceSource, add_bridge)
        .map(SimpleMemoryAdd, config)
        .sink(PipelineServiceSink)
    )

    # 记忆测试子 pipeline（黑盒检索 + LLM 评估）
    (
        env.from_source(PipelineServiceSource, search_bridge)
        .map(SimpleMemorySearch, config)
        .map(MemoryEvaluation, config)
        .sink(PipelineServiceSink)
    )

    # 主 pipeline
    (
        env.from_batch(MemorySource, config)
        .map(SimplePipelineCaller, config)
        .sink(MemorySink, config)
    )

    # ── 启动（流式子 pipeline 在后台，批式主 pipeline 在前台）─────────────
    _submit_multi_pipeline(env, bridges=[add_bridge, search_bridge])

    process_logger.close()


if __name__ == "__main__":
    main()
