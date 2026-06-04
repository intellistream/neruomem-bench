from __future__ import annotations

import sys
import types
from pathlib import Path

_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

_workspace_root = Path(__file__).resolve().parents[3]
_neuromem_repo = _workspace_root / "neuromem"
if _neuromem_repo.exists():
    neuromem_repo_str = str(_neuromem_repo)
    if neuromem_repo_str not in sys.path:
        sys.path.append(neuromem_repo_str)

try:
    import sage.foundation  # type: ignore  # noqa: F401
except ModuleNotFoundError:
    sage_module = types.ModuleType("sage")
    foundation_module = types.ModuleType("sage.foundation")

    class _CompatMapFunction:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            self.logger = None

        def call_service(self, *args, **kwargs):
            del args, kwargs
            raise RuntimeError("inject call_service manually in tests")

    foundation_module.MapFunction = _CompatMapFunction
    sage_module.foundation = foundation_module
    sys.modules.setdefault("sage", sage_module)
    sys.modules["sage.foundation"] = foundation_module

from benchmarks.experiment.libs.pipeline_caller import PipelineCaller


class _DummyLoader:
    def sessions(self, task_id: str):
        del task_id
        return [(0, 1)]

    def message_count(self, task_id: str):
        del task_id
        return 2

    def dialog_count(self, task_id: str):
        del task_id
        return 1

    def question_count(self, task_id: str):
        del task_id
        return 10

    def get_evaluation(self, task_id: str, session_x: int, dialog_y: int):
        del task_id, session_x, dialog_y
        return []


def test_pipeline_caller_forwards_session_summary(monkeypatch):
    monkeypatch.setattr(
        "benchmarks.experiment.utils.dataloader.factory.DataLoaderFactory.create",
        lambda dataset, **kwargs: _DummyLoader(),
    )

    caller = PipelineCaller(
        {
            "dataset": "locomo",
            "task_id": "conv-30",
            "services.services_type": "partitional.online_continual_memory",
            "runtime.test_segments": 1,
            "runtime.memory_insert_verbose": False,
            "runtime.memory_test_verbose": False,
        }
    )

    captured: dict[str, object] = {}

    def _fake_call_service(service_name, *args, **kwargs):
        del service_name
        captured["args"] = args
        captured["kwargs"] = kwargs
        if kwargs.get("method") == "process":
            return {"stage_timings": {"pre_insert_ms": [], "memory_insert_ms": [], "post_insert_ms": []}}
        return []

    caller.call_service = _fake_call_service  # type: ignore[assignment]

    caller.execute(
        {
            "task_id": "conv-30",
            "session_id": 0,
            "dialog_id": 0,
            "dialogs": [{"speaker": "Jon", "text": "Hello"}],
            "dialog_len": 1,
            "packet_idx": 0,
            "total_packets": 1,
            "is_session_end": True,
            "session_summary": "Session summary text.",
        }
    )

    insert_payload = captured["args"][0]
    assert insert_payload["session_summary"] == "Session summary text."
