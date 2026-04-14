"""TiMServiceFactory - 本地服务工厂适配器

作用：把本地 SimpleCollection + TiMTripleStoreService 包装成
sage.runtime.service_factory.ServiceFactory 接口，使 tim_pipeline.py
可以与 memory_test_pipeline.py 使用完全一致的 env.register_service_factory()
调用方式，但不依赖 sage 的数据结构层（UnifiedCollection / sage LSHHashService）。

设计对比 NeuromemServiceFactory：
    NeuromemServiceFactory.create() → NeuromemServiceProxy（依赖 sage 数据栈）
    TiMServiceFactory.create()      → _TiMLocalProxy（使用本地 SimpleCollection）

用法：
    # tim_pipeline.py 中
    factory = TiMServiceFactory.create("local.tim_triple_store", config)
    env.register_service_factory("tim_triple_store", factory)
"""

from __future__ import annotations

from typing import Any

from sage.runtime.service_factory import ServiceFactory

from .collection import SimpleCollection
from .lsh.tim_triple_store import TiMTripleStoreService
from .service_registry import MemoryServiceRegistry

# 触发 TiMTripleStoreService 注册（顺带也触发 LSHHashService）
from . import __init__ as _ds_init  # noqa: F401


class TiMServiceFactory:
    """TiM 本地服务工厂。

    静态方法 create() 返回一个 sage ServiceFactory，其内部 service_class 是
    _TiMLocalProxy。sage 的调度层会在合适时机调用：
        proxy = _TiMLocalProxy(ctx=ctx)
        proxy.setup()
        ...
        result = proxy.insert(...)
    """

    @staticmethod
    def create(service_name: str, config: Any) -> ServiceFactory:
        """创建 TiM 本地服务工厂。

        Args:
            service_name: 服务名称，如 "local.tim_triple_store" 或 "tim_triple_store"
            config:       RuntimeConfig 对象（支持 .get(key, default) 接口）

        Returns:
            ServiceFactory 实例，可直接传给 env.register_service_factory()
        """
        if "." in service_name:
            _, service_type = service_name.rsplit(".", 1)
        else:
            service_type = service_name

        # 从配置读取 Service 参数，配置块名与 service_type 一致
        service_config: dict[str, Any] = config.get(f"services.{service_type}", {})

        class _TiMLocalProxy:
            """TiM 本地代理 — 对应 sage NeuromemServiceProxy 的职责。

            sage 在 pipeline 启动时调用 setup()，在 call_service 时直接调用
            insert / retrieve / delete / get 方法。
            """

            def __init__(self, ctx: Any = None) -> None:  # noqa: ANN401
                self.ctx = ctx
                self._service_type = service_type
                self._service_config = service_config
                self._service: TiMTripleStoreService | None = None
                self._collection: SimpleCollection | None = None

            def setup(self) -> None:
                """初始化 SimpleCollection + TiMTripleStoreService。"""
                self._collection = SimpleCollection(name=self._service_type)
                self._service = MemoryServiceRegistry.create(
                    service_type=self._service_type,
                    collection=self._collection,
                    config=self._service_config,
                )

            def teardown(self) -> None:
                """释放资源（重置引用，由 GC 回收内存）。"""
                self._service = None
                self._collection = None

            # ── Service method forwarding ────────────────────────────────────

            def _require_service(self) -> TiMTripleStoreService:
                if self._service is None:
                    raise RuntimeError(
                        "TiMLocalProxy: service not initialized. Call setup() first."
                    )
                return self._service

            def insert(
                self,
                entry: str,
                vector: Any = None,
                metadata: dict[str, Any] | None = None,
                *,
                insert_mode: str = "passive",
                insert_params: dict[str, Any] | None = None,
            ) -> str:
                return self._require_service().insert(
                    entry,
                    vector=vector,
                    metadata=metadata,
                    insert_mode=insert_mode,
                    insert_params=insert_params,
                )

            def retrieve(
                self,
                query: str | None = None,
                vector: Any = None,
                metadata: dict[str, Any] | None = None,
                top_k: int = 5,
                hints: dict[str, Any] | None = None,
                threshold: float | None = None,
                **kwargs: Any,
            ) -> list[dict[str, Any]]:
                return self._require_service().retrieve(
                    query=query,
                    vector=vector,
                    metadata=metadata,
                    top_k=top_k,
                    hints=hints,
                    threshold=threshold,
                    **kwargs,
                )

            def delete(self, entry_id: str) -> bool:
                return self._require_service().delete(entry_id)

            def get(self, data_id: str) -> dict[str, Any] | None:
                return self._require_service().get(data_id)

            def get_recent(self, limit: int = 10) -> list[dict[str, Any]]:
                return self._require_service().get_recent(limit)

            def get_stats(self) -> dict[str, Any]:
                return self._require_service().get_stats()

        # 工厂中使用唯一的类名，避免 sage 序列化/哈希冲突
        _TiMLocalProxy.__name__ = f"_TiMLocalProxy_{service_type}"
        _TiMLocalProxy.__qualname__ = f"TiMServiceFactory.create.<locals>._TiMLocalProxy_{service_type}"

        return ServiceFactory(
            service_name=service_type,
            service_class=_TiMLocalProxy,
        )
