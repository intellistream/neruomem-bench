"""Pipeline-as-Service 演示

演示如何将 Sage Pipeline 包装成同步服务端点，使外部调用方可以
向正在运行的 Pipeline 提交请求并同步获取处理结果。

核心机制（纯 Sage API 实现）：
    1. 双向 Queue Bridge：外部线程通过 bridge 投递请求，Pipeline Source 消费
    2. Pipeline 内部处理：Source → Map → Sink
    3. Sink 将结果写回 bridge 的响应 queue，外部线程同步获取

运行方式：
    python -m test.installation_validation.pipeline_as_service
"""

from __future__ import annotations

import queue
import sys
from dataclasses import dataclass
from typing import Any

from sage.foundation import CustomLogger, MapFunction, SinkFunction, SourceFunction
from sage.runtime import LocalEnvironment, StopSignal
from sage.runtime.job_manager import JobManager


# ============================================================================
# 轻量级 Bridge（纯 Python queue，不依赖外部模块）
# ============================================================================


@dataclass
class _Request:
    payload: dict[str, Any]
    reply: queue.Queue


class ServiceBridge:
    """双向 queue 桥接：外部调用方 ←→ Pipeline 内部算子"""

    def __init__(self):
        self._q: queue.Queue = queue.Queue()
        self._closed = False

    def send(self, payload: dict) -> queue.Queue:
        """外部调用方：提交请求，返回可阻塞等待响应的 queue"""
        if self._closed:
            raise RuntimeError("bridge closed")
        reply: queue.Queue = queue.Queue(maxsize=1)
        self._q.put(_Request(payload=dict(payload), reply=reply))
        return reply

    def recv(self, timeout: float = 0.1):
        """Pipeline Source 端：获取下一个请求（超时返回 None）"""
        if self._closed and self._q.empty():
            return StopSignal("bridge-closed")
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self):
        if not self._closed:
            self._closed = True
            self._q.put(StopSignal("bridge-closed"))


# ============================================================================
# Pipeline 算子
# ============================================================================


class ServiceSource(SourceFunction):
    """从 bridge 读取请求注入 Pipeline"""

    def __init__(self, bridge: ServiceBridge):
        super().__init__()
        self._bridge = bridge

    def execute(self, data=None):
        req = self._bridge.recv(timeout=0.05)
        if req is None:
            return None
        if isinstance(req, StopSignal):
            return req
        # 将 reply queue 随 payload 一起向下传递
        payload = dict(req.payload)
        payload["_reply"] = req.reply
        return payload


class UpperCaseMap(MapFunction):
    """示例处理：将 text 字段转为大写"""

    def execute(self, data):
        if data is None or isinstance(data, StopSignal):
            return data
        data["text"] = data.get("text", "").upper()
        data["processed"] = True
        return data


class ReplySink(SinkFunction):
    """将处理结果写回调用方的 reply queue"""

    def execute(self, data):
        if data is None or isinstance(data, StopSignal):
            return
        reply = data.pop("_reply", None)
        if reply is not None:
            reply.put(data, timeout=5.0)


# ============================================================================
# 主演示
# ============================================================================


def main() -> int:
    print("=" * 50)
    print("Pipeline-as-Service 演示")
    print("=" * 50)

    CustomLogger.disable_global_console_debug()
    JobManager().cleanup_all_jobs()

    bridge = ServiceBridge()

    # 预填充请求（Pipeline 启动前）
    messages = [
        {"text": "hello sage"},
        {"text": "pipeline as service"},
        {"text": "done"},
    ]
    reply_queues = [bridge.send(m) for m in messages]
    bridge.close()  # 所有请求已入队，关闭触发 StopSignal

    # 构建 Pipeline: Source → Map → Sink
    env = LocalEnvironment(name="pipeline_as_service")
    env.from_source(ServiceSource, bridge).map(UpperCaseMap).sink(ReplySink)
    env.submit(autostop=True)

    # 收集并验证结果
    for i, (msg, rq) in enumerate(zip(messages, reply_queues)):
        result = rq.get(timeout=5.0)
        expected = msg["text"].upper()
        assert result["text"] == expected
        assert result["processed"] is True
        print(f"  [{i+1}] '{msg['text']}' → '{result['text']}'")

    print("\n✅ Pipeline-as-Service 演示完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
