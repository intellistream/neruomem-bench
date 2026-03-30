"""数据集加载器工厂（开放接口版）

此工厂提供数据集加载器的统一创建入口。
实际的数据集适配器需要用户自行实现并注册。

典型用法：

    # 1. 实现自定义数据集适配器
    class MyDatasetAdapter(BaseDataLoader):
        @property
        def dataset_name(self) -> str:
            return "my_dataset"

        def get_dialog(self, task_id, session_x, dialog_y):
            # 实现具体的数据读取逻辑
            ...

        def get_evaluation(self, task_id, session_x, dialog_y):
            ...

        def sessions(self, task_id):
            ...

        def question_count(self, task_id):
            ...

        def dialog_count(self, task_id):
            ...

        def message_count(self, task_id):
            ...

        def statistics(self, task_id):
            ...

    # 2. 注册适配器
    DataLoaderFactory.register("my_dataset", MyDatasetAdapter)

    # 3. 使用工厂创建
    loader = DataLoaderFactory.create("my_dataset")
"""

from __future__ import annotations

from typing import Callable

from benchmarks.experiment.utils.dataloader.base import BaseDataLoader


class DataLoaderFactory:
    """数据集加载器工厂

    使用注册机制管理数据集适配器，支持用户自定义扩展。
    """

    _registry: dict[str, Callable[..., BaseDataLoader]] = {}

    @classmethod
    def register(cls, name: str, adapter_cls: Callable[..., BaseDataLoader]) -> None:
        """注册数据集适配器

        Args:
            name: 数据集名称标识
            adapter_cls: 适配器类（需继承 BaseDataLoader）

        Examples:
            >>> DataLoaderFactory.register("my_dataset", MyDatasetAdapter)
        """
        cls._registry[name] = adapter_cls

    @classmethod
    def create(cls, dataset: str, **kwargs) -> BaseDataLoader:
        """根据数据集名称创建 DataLoader

        Args:
            dataset: 数据集名称
            **kwargs: 传递给适配器构造函数的额外参数

        Returns:
            对应的 DataLoader 适配器实例

        Raises:
            ValueError: 不支持的数据集类型
        """
        if dataset not in cls._registry:
            available = sorted(cls._registry.keys()) if cls._registry else ["(无)"]
            raise ValueError(
                f"不支持的数据集: {dataset}. "
                f"已注册的数据集: {available}\n"
                f"请先使用 DataLoaderFactory.register() 注册适配器。"
            )

        return cls._registry[dataset](**kwargs)

    @classmethod
    def list_datasets(cls) -> list[str]:
        """列出所有已注册的数据集"""
        return sorted(cls._registry.keys())

    @classmethod
    def is_supported(cls, dataset: str) -> bool:
        """检查数据集是否已注册"""
        return dataset in cls._registry
