"""SimpleMemorySink — 将结果输出至 simple_benchmark_memory 目录

与 MemorySink 完全一致，唯一区别是输出路径前缀为
.sage/benchmarks/simple_benchmark_memory/ 而非 .sage/benchmarks/benchmark_memory/。
"""

from __future__ import annotations

import os

from benchmarks.experiment.libs.memory_sink import MemorySink
from benchmarks.experiment.utils import get_project_root, get_runtime_timestamp


class SimpleMemorySink(MemorySink):
    """MemorySink 子类，仅覆写输出目录。"""

    def __init__(self, config):
        # 调用父类 __init__（会设置 output_dir / output_file 等属性）
        super().__init__(config)

        # 覆写输出路径
        project_root = get_project_root()
        self.output_dir = os.path.join(
            project_root,
            f".sage/benchmarks/simple_benchmark_memory/{self.dataset}/{self.memory_name}",
        )
        os.makedirs(self.output_dir, exist_ok=True)

        runtime_stamp = get_runtime_timestamp()
        self.output_file = os.path.join(
            self.output_dir, f"{self.task_id}_{runtime_stamp}.json"
        )
        print(f"💾 [SimpleMemorySink] 输出文件: {self.output_file}")
