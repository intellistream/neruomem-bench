from __future__ import annotations

import sys

from sage.foundation import BatchFunction, CustomLogger, MapFunction, SinkFunction
from sage.runtime import LocalEnvironment, StopSignal
from sage.runtime.job_manager import JobManager


class OperatorSource(BatchFunction):
    """批处理数据源：逐条产生示例数据，耗尽后发送 StopSignal。"""

    def __init__(self) -> None:
        super().__init__()
        self._data = ["item_1", "item_2", "item_3"]
        self._index = 0

    def execute(self):
        if self._index >= len(self._data):
            return StopSignal("source-done")
        value = self._data[self._index]
        self._index += 1
        return value


class OperatorCaller(MapFunction):
    """映射变换：模拟调用外部服务（如 neuromem Service）。"""

    def execute(self, data: str) -> str:
        return f"processed_{data}"


class OperatorSink(SinkFunction):
    """数据汇聚：打印最终结果。"""

    def execute(self, data: str) -> None:
        print(f"[Sink] {data}")


def main() -> None:
    CustomLogger.disable_global_console_debug()

    JobManager().cleanup_all_jobs()

    env = LocalEnvironment(name="sage_pipeline")
    env.from_batch(OperatorSource).map(OperatorCaller).sink(OperatorSink)

    env.submit(autostop=True)


if __name__ == "__main__":
    main()
