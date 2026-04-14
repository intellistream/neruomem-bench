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
from benchmarks.experiment.libs.datastructure.tim_service_factory import TiMServiceFactory


# ── Multi-pipeline submission helper ─────────────────────────────────────────
# isage's lightweight PipelineCompiler only supports one SourceTransformation
# per compilation unit AND does not inject a RuntimeContext into operators.
# Both issues are resolved here without touching sage internals.


class _LocalRuntimeContext:
    """Minimal RuntimeContext that routes call_service to pre-instantiated services."""

    def __init__(self, services: dict, name: str = "local") -> None:
        self._services = services  # service_name → service instance
        self.name = name

    def call_service(
        self,
        service_name: str,
        *args,
        method: str | None = None,
        timeout: float | None = None,
        **kwargs,
    ):
        svc = self._services.get(service_name)
        if svc is None:
            raise RuntimeError(
                f"Service '{service_name}' not found. "
                f"Available: {list(self._services.keys())}"
            )
        if method is None:
            raise ValueError("call_service requires a 'method' argument")
        return getattr(svc, method)(*args, **kwargs)


def _partition_by_source(pipeline: list) -> list[list]:
    """Split env.pipeline (flat list) into sub-lists, one per source."""
    segments: list[list] = []
    current: list = []
    for t in pipeline:
        if type(t).__name__ in ("SourceTransformation", "BatchTransformation"):
            if current:
                segments.append(current)
            current = [t]
        else:
            current.append(t)
    if current:
        segments.append(current)
    return segments


def _inject_ctx(graph, ctx) -> None:
    """Inject ctx into all operator _fn instances in a compiled graph."""
    for handle in getattr(graph, "actor_handles", []):
        wrapper = getattr(handle, "_target", None)
        fn = getattr(wrapper, "_fn", None)
        if fn is not None and hasattr(fn, "ctx"):
            fn.ctx = ctx


def _submit_multi_pipeline(env, bridges: list) -> None:
    """Compile and run sub-pipelines independently.

    1. Pre-instantiate ALL registered services (shared across sub-graphs).
    2. Build a _LocalRuntimeContext and inject it into every operator.
    3. PipelineServiceSource sub-graphs: run in daemon threads with a loop
       that skips None (= "no pending request") rather than treating it as a
       stop signal (sage's built-in _run_source_thread would break on None).
    4. Main batch sub-graph: synchronous foreground execution.
    5. Close bridges → StopSignal drains through → listener threads exit.
    """
    import threading

    from sage.runtime.pipeline_compiler import PipelineCompiler
    from sage.runtime.local_backend import get_local_runtime_backend

    compiler = PipelineCompiler()
    adapter = get_local_runtime_backend()

    # ── Step 1: instantiate all services ────────────────────────────────────
    services: dict = {}
    for svc_name, factory in env.service_factories.items():
        instance = factory.create_service(ctx=None)
        if hasattr(instance, "setup"):
            instance.setup()
        services[svc_name] = instance

    ctx = _LocalRuntimeContext(services=services, name=env.name)

    # ── Step 2: compile each sub-pipeline and inject context ─────────────────
    segments = _partition_by_source(env.pipeline)

    listener_threads: list[threading.Thread] = []
    stop_events: list[threading.Event] = []
    batch_graphs = []

    def _run_listener(graph, stop_event: threading.Event) -> None:
        """Service-listener loop.

        sage's built-in _run_source_thread breaks when execute() returns None
        (no pending request).  We own this loop, so we *continue* on None and
        only stop on an explicit StopSignal or when stop_event is set.
        """
        t = graph.source_transformation
        fn = t.function_class(*t.function_args, **t.function_kwargs)
        while not stop_event.is_set():
            item = fn.execute()
            if item is None:
                continue  # no pending request – keep polling
            if type(item).__name__ == "StopSignal":
                break
            try:
                graph._execute_chain([item], graph.stage_ops)
            except Exception:
                import logging
                logging.getLogger(__name__).exception(
                    "Listener pipeline raised an exception; item discarded."
                )

    for seg in segments:
        graph = compiler.compile(seg, adapter)
        _inject_ctx(graph, ctx)

        fn_class = getattr(seg[0], "function_class", None)
        if fn_class is PipelineServiceSource:
            stop_event = threading.Event()
            stop_events.append(stop_event)
            t = threading.Thread(
                target=_run_listener,
                args=(graph, stop_event),
                daemon=True,
                name=f"tim-listener-{len(listener_threads)}",
            )
            t.start()
            listener_threads.append(t)
        else:
            batch_graphs.append(graph)

    # ── Step 3: run main batch pipeline(s) synchronously ────────────────────
    for graph in batch_graphs:
        graph.submit(autostop=True)

    # ── Step 4: signal listeners to stop, then wait ─────────────────────────
    for bridge in bridges:
        bridge.close()  # injects StopSignal into the bridge queue
    for evt in stop_events:
        evt.set()

    for t in listener_threads:
        t.join(timeout=120.0)


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

    # local.* 前缀使用本地 datastructure 层（无 sage 数据依赖），其余走 sage
    if services_type.startswith("local."):
        factory = TiMServiceFactory.create(services_type, config)
    else:
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

    # 启动并等待完成（三条子 pipeline 分别编译：service listener 为流式后台，主 pipeline 为批式前台）
    _submit_multi_pipeline(env, bridges=[insert_bridge, test_bridge])

    # 关闭过程日志
    process_logger.close()


if __name__ == "__main__":
    main()
