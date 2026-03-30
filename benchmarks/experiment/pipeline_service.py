"""Lightweight pipeline-as-service helpers for NeuroMem benchmarks."""

from __future__ import annotations

import queue
from dataclasses import dataclass
from typing import Any

from sage.foundation import SinkFunction, SourceFunction
from sage.runtime import BaseService, StopSignal


@dataclass
class PipelineRequest:
    """Bridge request carrying payload and reply queue."""

    payload: dict[str, Any]
    response_queue: queue.Queue[dict[str, Any]]


class PipelineBridge:
    """Bidirectional queue bridge between a service and a pipeline."""

    def __init__(self) -> None:
        self._requests: queue.Queue[PipelineRequest | StopSignal] = queue.Queue()
        self._closed = False

    def submit(self, payload: dict[str, Any]) -> queue.Queue[dict[str, Any]]:
        if self._closed:
            raise RuntimeError("Pipeline bridge is closed")

        response_queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
        self._requests.put(PipelineRequest(payload=dict(payload), response_queue=response_queue))
        return response_queue

    def next(self, timeout: float = 0.1):
        if self._closed and self._requests.empty():
            return StopSignal("pipeline-service-shutdown")

        try:
            return self._requests.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._requests.put(StopSignal("pipeline-service-shutdown"))


class PipelineService(BaseService):
    """Expose a pipeline as a synchronous service endpoint."""

    def __init__(self, bridge: PipelineBridge, request_timeout: float = 300.0):
        super().__init__()
        self._bridge = bridge
        self._request_timeout = request_timeout

    def process(self, message: dict[str, Any]):
        if message is None:
            raise ValueError("Pipeline service received an empty message")

        if message.get("command") == "shutdown":
            self._bridge.close()
            return {"status": "shutdown_requested"}

        try:
            response_queue = self._bridge.submit(message)
        except RuntimeError as exc:
            raise RuntimeError("Pipeline service is shutting down") from exc

        try:
            return response_queue.get(timeout=self._request_timeout)
        except queue.Empty as exc:
            raise TimeoutError("Pipeline service timed out waiting for a reply") from exc


class PipelineServiceSource(SourceFunction):
    """Read service requests from the bridge and inject them into the pipeline."""

    def __init__(self, bridge: PipelineBridge, poll_interval: float = 0.1):
        super().__init__()
        self._bridge = bridge
        self._poll_interval = poll_interval

    def execute(self, data=None):
        request = self._bridge.next(timeout=self._poll_interval)
        if request is None:
            return None
        if isinstance(request, StopSignal):
            return request

        payload = dict(request.payload)
        payload["_response_queue"] = request.response_queue
        return payload


class PipelineServiceSink(SinkFunction):
    """Publish the final pipeline result back to the waiting service caller."""

    def execute(self, payload: dict[str, Any] | StopSignal | None):
        if payload is None or isinstance(payload, StopSignal):
            return payload

        response_queue = payload.pop("_response_queue", None)
        if response_queue is not None:
            try:
                response_queue.put(payload, timeout=5.0)
            except queue.Full as exc:
                raise RuntimeError("Pipeline response queue is full") from exc
        return payload


__all__ = [
    "PipelineBridge",
    "PipelineRequest",
    "PipelineService",
    "PipelineServiceSource",
    "PipelineServiceSink",
]
