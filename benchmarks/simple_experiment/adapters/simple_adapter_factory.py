"""SimpleAdapterFactory — 将黑盒适配器包装为 Sage ServiceFactory

作用：把 BaseSimpleMemoryAdapter 的实现包装成 sage.runtime.ServiceFactory 接口，
使 simple_pipeline.py 可以与 memory_test_pipeline.py 使用完全一致的
env.register_service_factory() 调用方式。

设计对比 TiMServiceFactory：
    TiMServiceFactory.create()      → _TiMLocalProxy（本地 LSH 数据结构）
    SimpleAdapterFactory.create()   → _SimpleAdapterProxy（第三方记忆体黑盒）

用法（simple_pipeline.py 中）：
    factory = SimpleAdapterFactory.create("simple.mem0", config)
    env.register_service_factory("mem0", factory)
"""

from __future__ import annotations

from typing import Any

from sage.runtime.service_factory import ServiceFactory

from .adapter_registry import AdapterRegistry
from .base_adapter import BaseSimpleMemoryAdapter

# 触发所有适配器的自动注册
from . import mem0_adapter  # noqa: F401


class SimpleAdapterFactory:
    """黑盒适配器服务工厂。

    静态方法 create() 返回一个 sage ServiceFactory，其内部 service_class 是
    _SimpleAdapterProxy。sage 的调度层会在合适时机实例化 proxy 并调用 setup()。
    """

    @staticmethod
    def create(adapter_type: str, config: Any) -> ServiceFactory:
        """创建适配器服务工厂。

        Args:
            adapter_type: 服务类型字符串，支持两种格式：
                          "simple.mem0"（带命名空间前缀）或 "mem0"（直接名称）
            config:       RuntimeConfig 对象（支持 .get(key, default) 接口）

        Returns:
            ServiceFactory 实例，可直接传给 env.register_service_factory()
        """
        if "." in adapter_type:
            _, adapter_name = adapter_type.rsplit(".", 1)
        else:
            adapter_name = adapter_type

        # 从配置中读取适配器专属配置块
        adapter_config: dict[str, Any] = config.get(f"services.{adapter_name}", {})

        # 解析 user_id：区分 "auto"（每次运行生成唯一 ID）和 "task"（同 task 共享）
        user_id_scope: str = adapter_config.get("user_id_scope", "auto")
        task_id: str = config.get("task_id", "unknown")

        if user_id_scope == "auto":
            from benchmarks.experiment.utils import get_time_filename

            user_id = f"bench_{task_id}_{get_time_filename()}"
        else:
            user_id = f"bench_{task_id}"

        # 将解析好的 user_id 注入适配器配置（不修改原配置对象）
        resolved_config = dict(adapter_config)
        resolved_config["user_id"] = user_id

        class _SimpleAdapterProxy:
            """将 BaseSimpleMemoryAdapter 暴露为 Sage 服务接口。

            sage 在 pipeline 启动时调用 setup()，在 call_service 时直接调用
            add / search / clear / get_stats 方法。
            """

            def __init__(self, ctx: Any = None) -> None:
                self.ctx = ctx
                self._adapter_name = adapter_name
                self._config = resolved_config
                self._adapter: BaseSimpleMemoryAdapter | None = None

            def setup(self) -> None:
                self._adapter = AdapterRegistry.create(self._adapter_name, self._config)

            def teardown(self) -> None:
                self._adapter = None

            # ── 防御性私有方法 ─────────────────────────────────────────────
            def _require(self) -> BaseSimpleMemoryAdapter:
                if self._adapter is None:
                    raise RuntimeError(
                        "_SimpleAdapterProxy: adapter not initialized. "
                        "Call setup() first."
                    )
                return self._adapter

            # ── 公开服务方法（供 call_service 路由）──────────────────────
            def add(
                self,
                text: str,
                metadata: dict[str, Any] | None = None,
            ) -> str:
                return self._require().add(text, metadata)

            def search(
                self,
                query: str,
                top_k: int = 5,
            ) -> list[dict[str, Any]]:
                return self._require().search(query, top_k)

            def clear(self) -> None:
                self._require().clear()

            def get_stats(self) -> dict[str, Any]:
                return self._require().get_stats()

        return ServiceFactory(service_name=adapter_name, service_class=_SimpleAdapterProxy)
