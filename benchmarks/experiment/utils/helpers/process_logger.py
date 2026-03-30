"""过程日志管理器 - 记录实验过程中的详细日志

输出目录结构:
.sage/output/benchmarks/benchmark_memory/{dataset}/{memory_name}/{task_id}_{HHMMSS}/
    - memory_service.log
    - memory_qa.log
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from benchmarks.experiment.utils.helpers.path_finder import get_project_root


class ProcessLogger:
    """过程日志管理器（单例模式）"""

    _instance: ProcessLogger | None = None
    _initialized: bool = False

    def __new__(cls) -> ProcessLogger:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if ProcessLogger._initialized:
            return
        ProcessLogger._initialized = True

        self._output_dir: str | None = None
        self._service_log_path: str | None = None
        self._qa_log_path: str | None = None
        self._service_file = None
        self._qa_file = None

    def setup(self, dataset: str, memory_name: str, task_id: str) -> str:
        env_log_dir = os.environ.get("PROCESS_LOG_DIR")
        if env_log_dir:
            self._output_dir = env_log_dir
            os.makedirs(self._output_dir, exist_ok=True)
        else:
            project_root = get_project_root()
            timestamp = datetime.now(timezone.utc).strftime("%H%M%S")

            self._output_dir = os.path.join(
                project_root,
                ".sage/output/benchmarks/benchmark_memory",
                dataset,
                memory_name,
                f"{task_id}_{timestamp}",
            )
            os.makedirs(self._output_dir, exist_ok=True)

        self._service_log_path = os.path.join(self._output_dir, "memory_service.log")
        self._qa_log_path = os.path.join(self._output_dir, "memory_qa.log")

        self._service_file = open(self._service_log_path, "w", encoding="utf-8")  # noqa: SIM115
        self._qa_file = open(self._qa_log_path, "w", encoding="utf-8")  # noqa: SIM115

        self._service_file.write(
            f"# Memory Service Log - {datetime.now(timezone.utc).isoformat()}\n"
        )
        self._service_file.write(f"# Dataset: {dataset}, Task: {task_id}, Memory: {memory_name}\n")
        self._service_file.write("=" * 80 + "\n\n")
        self._service_file.flush()

        self._qa_file.write(f"# Memory QA Log - {datetime.now(timezone.utc).isoformat()}\n")
        self._qa_file.write(f"# Dataset: {dataset}, Task: {task_id}, Memory: {memory_name}\n")
        self._qa_file.write("=" * 80 + "\n\n")
        self._qa_file.flush()

        print(f"📂 过程日志目录: {self._output_dir}")

        return self._output_dir

    def log_service(self, operation: str, details: str) -> None:
        if self._service_file is None:
            return

        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
        self._service_file.write(f"[{timestamp}] [{operation}]\n")
        self._service_file.write(f"{details}\n")
        self._service_file.write("-" * 40 + "\n")
        self._service_file.flush()

    def log_qa(
        self,
        question_idx: int,
        question: str,
        answer: str,
        context: str | None = None,
        metadata: dict[str, Any] | None = None,
        full_prompt: str | None = None,
    ) -> None:
        if self._qa_file is None:
            return

        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        self._qa_file.write(f"[{timestamp}] Question #{question_idx}\n")
        self._qa_file.write(f"Q: {question}\n")
        self._qa_file.write(f"A: {answer}\n")

        if metadata:
            ref_answer = metadata.get("answer", "")
            if ref_answer:
                self._qa_file.write(f"Ref: {ref_answer}\n")
            category = metadata.get("category", "")
            if category:
                self._qa_file.write(f"Category: {category}\n")

        if full_prompt:
            self._qa_file.write(f"\n{full_prompt}\n")

        self._qa_file.write("\n" + "=" * 60 + "\n\n")
        self._qa_file.flush()

    def close(self) -> None:
        if self._service_file:
            self._service_file.close()
            self._service_file = None
        if self._qa_file:
            self._qa_file.close()
            self._qa_file = None

    @property
    def output_dir(self) -> str | None:
        return self._output_dir

    @classmethod
    def reset(cls) -> None:
        if cls._instance:
            cls._instance.close()
        cls._instance = None
        cls._initialized = False


process_logger = ProcessLogger()
