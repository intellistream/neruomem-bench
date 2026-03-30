"""运行时配置管理器

负责加载 YAML 配置并提供运行时参数访问接口
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Protocol

import yaml


class ConfigProtocol(Protocol):
    """配置对象协议"""

    def get(self, key: str) -> Any: ...


def get_required_config(config: ConfigProtocol, key: str, context: str = "") -> Any:
    """获取必需配置，缺失则报错"""
    value = config.get(key)
    if value is None:
        ctx = f" ({context})" if context else ""
        raise ValueError(f"缺少必需配置: {key}{ctx}")
    return value


class RuntimeConfig:
    """运行时配置类

    功能：
    1. 加载 YAML 配置文件
    2. 接收运行时参数（如 task_id）
    3. 提供统一的参数访问接口 config.get("key")
    """

    def __init__(self, config_path: str | None, **runtime_params):
        self.config_path = config_path
        self.runtime_params = runtime_params
        self._config: dict[str, Any] = {}
        if config_path is not None:
            self._load()

    def _load(self) -> None:
        config_file = Path(self.config_path)
        if not config_file.exists():
            print(f"❌ 配置文件不存在: {config_file}")
            sys.exit(1)

        try:
            with open(config_file) as f:
                self._config = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            print(f"❌ 配置文件格式错误: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ 加载配置文件失败: {e}")
            sys.exit(1)

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项，支持点号路径和运行时参数"""
        if key in self.runtime_params:
            return self.runtime_params[key]

        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        return value

    def set_runtime_param(self, key: str, value: Any) -> None:
        self.runtime_params[key] = value

    def get_full_config(self) -> dict[str, Any]:
        return {**self._config, **self.runtime_params}

    @staticmethod
    def create(config_path: str, **runtime_params) -> RuntimeConfig:
        return RuntimeConfig(config_path, **runtime_params)

    @staticmethod
    def load(config_path: str, task_id: str | None = None) -> RuntimeConfig:
        config = RuntimeConfig(config_path)

        if task_id:
            config.set_runtime_param("task_id", task_id)
        elif config.get("runtime.task_id"):
            config.set_runtime_param("task_id", config.get("runtime.task_id"))

        dataset = config.get("runtime.dataset")
        if not dataset:
            print("❌ 配置文件缺少 runtime.dataset 字段")
            sys.exit(1)
        config.set_runtime_param("dataset", dataset)

        return config
