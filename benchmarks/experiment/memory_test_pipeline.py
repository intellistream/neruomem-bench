"""NeuroMem 记忆实验 - Pipeline 架构

三层 Pipeline 架构：
- 主 Pipeline: MemorySource → PipelineCaller → MemorySink
- 记忆插入 Pipeline: PreInsert → MemoryInsert → PostInsert
- 记忆测试 Pipeline: PreRetrieval → MemoryRetrieval → PostRetrieval → MemoryEvaluation
"""

from __future__ import annotations

from sage.foundation import CustomLogger
from sage.runtime import LocalEnvironment

from benchmarks.experiment.libs.memory_evaluation import MemoryEvaluation
from benchmarks.experiment.libs.memory_insert import MemoryInsert
from benchmarks.experiment.libs.memory_retrieval import MemoryRetrieval
from benchmarks.experiment.libs.memory_sink import MemorySink
from benchmarks.experiment.libs.memory_source import MemorySource
from benchmarks.experiment.libs.pipeline_caller import PipelineCaller
from benchmarks.experiment.libs.post_insert import PostInsert
from benchmarks.experiment.libs.post_retrieval import PostRetrieval
from benchmarks.experiment.libs.pre_insert import PreInsert
from benchmarks.experiment.libs.pre_retrieval import PreRetrieval
from benchmarks.experiment.pipeline_service import (
    PipelineBridge,
    PipelineService,
    PipelineServiceSink,
    PipelineServiceSource,
)
from benchmarks.experiment.utils import RuntimeConfig, parse_args, process_logger
from sage.neuromem.services import NeuromemServiceFactory


def main():
    """主函数"""
    CustomLogger.disable_global_console_debug()

    # 解析命令行参数并加载配置
    args = parse_args()
    config = RuntimeConfig.load(args.config, args.task_id)

    # 初始化过程日志
    dataset = config.get("runtime.dataset", "default")
    task_id = config.get("task_id", "unknown")
    memory_name = config.get("runtime.memory_name", "default")
    process_logger.setup(dataset, memory_name, task_id)

    # 创建环境
    env = LocalEnvironment("memory_test_experiment")

    # 注册服务 - 使用工厂模式动态创建服务
    services_type = config.get("services.services_type")
    if not services_type:
        raise ValueError("Missing required config: services.services_type")

    factory = NeuromemServiceFactory.create(services_type, config)

    registered_name = services_type.split(".")[-1]
    env.register_service_factory(registered_name, factory)

    pipeline_service_timeout = config.get("runtime.pipeline_service_timeout", 300.0)

    insert_bridge = PipelineBridge()
    env.register_service(
        "memory_insert_service",
        PipelineService,
        insert_bridge,
        request_timeout=pipeline_service_timeout,
    )

    test_bridge = PipelineBridge()
    env.register_service(
        "memory_test_service",
        PipelineService,
        test_bridge,
        request_timeout=pipeline_service_timeout,
    )

    # 创建 Pipeline
    # 记忆插入 Pipeline
    (
        env.from_source(PipelineServiceSource, insert_bridge)
        .map(PreInsert, config)
        .map(MemoryInsert, config)
        .map(PostInsert, config)
        .sink(PipelineServiceSink)
    )

    # 记忆测试（检索 + 评估）Pipeline
    (
        env.from_source(PipelineServiceSource, test_bridge)
        .map(PreRetrieval, config)
        .map(MemoryRetrieval, config)
        .map(PostRetrieval, config)
        .map(MemoryEvaluation, config)
        .sink(PipelineServiceSink)
    )

    # 主 Pipeline
    (env.from_batch(MemorySource, config).map(PipelineCaller, config).sink(MemorySink, config))

    # 启动并等待完成
    env.submit(autostop=True)

    # 关闭过程日志
    process_logger.close()


if __name__ == "__main__":
    main()
