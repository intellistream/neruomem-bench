from __future__ import annotations

import json
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

    class _CompatBatchFunction:
        def __init__(self, *args, **kwargs):
            del args, kwargs

    foundation_module.BatchFunction = _CompatBatchFunction
    sage_module.foundation = foundation_module
    sys.modules.setdefault("sage", sage_module)
    sys.modules["sage.foundation"] = foundation_module

from benchmarks.experiment.libs.memory_source import MemorySource
from benchmarks.experiment.libs.pre_insert.base import PreInsertInput
from benchmarks.experiment.libs.pre_insert.none_action import NoneAction
from benchmarks.experiment.utils.dataloader.adapters.locomo_adapter import LocalLocomoAdapter


class _DummyLoader:
    def __init__(self):
        self._summaries = {0: "Session 1 summary text."}

    def sessions(self, task_id: str):
        del task_id
        return [(0, 1)]

    def message_count(self, task_id: str):
        del task_id
        return 2

    def dialog_count(self, task_id: str):
        del task_id
        return 1

    def get_dialog(self, task_id: str, session_x: int, dialog_y: int):
        del task_id, session_x, dialog_y
        return [{"speaker": "Alice", "text": "Hello"}, {"speaker": "Bob", "text": "World"}]

    def get_evaluation(self, task_id: str, session_x: int, dialog_y: int):
        del task_id, session_x, dialog_y
        return []

    def session_summary(self, task_id: str, session_x: int):
        del task_id
        return self._summaries.get(session_x, "")


def test_local_locomo_adapter_session_summary(tmp_path):
    data = [
        {
            "task_id": "demo",
            "sessions": [{"session_id": 0, "messages": [{"speaker": "A", "text": "hi"}, {"speaker": "B", "text": "ok"}]}],
            "questions": [],
            "session_summary": {"session_1_summary": "Session summary text."},
        }
    ]
    data_path = tmp_path / "locomo.json"
    data_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    adapter = LocalLocomoAdapter(data_file=str(data_path))
    assert adapter.session_summary("demo", 0) == "Session summary text."


def test_local_locomo_adapter_supports_sample_id_records(tmp_path):
    data = [
        {
            "sample_id": "conv-30",
            "sessions": [{"session_id": 0, "messages": [{"speaker": "A", "text": "hi"}, {"speaker": "B", "text": "ok"}]}],
            "questions": [],
            "session_summary": {"session_1_summary": "Official summary text."},
        }
    ]
    data_path = tmp_path / "locomo.json"
    data_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    adapter = LocalLocomoAdapter(data_file=str(data_path))
    assert adapter.session_summary("conv-30", 0) == "Official summary text."


def test_memory_source_attaches_session_summary(monkeypatch):
    dummy_loader = _DummyLoader()
    monkeypatch.setattr(
        "benchmarks.experiment.utils.dataloader.factory.DataLoaderFactory.create",
        lambda dataset, **kwargs: dummy_loader,
    )

    source = MemorySource({"dataset": "locomo", "task_id": "demo"})
    packet = source.execute()

    assert packet is not None
    assert packet["session_summary"] == "Session 1 summary text."


def test_none_action_adds_summary_entry():
    action = NoneAction({})
    output = action.execute(
        PreInsertInput(
            data={
                "dialogs": [{"speaker": "Alice", "text": "Hello"}],
                "session_summary": "Session summary text.",
            },
            config={},
            service_name="online_continual_memory",
        )
    )

    assert len(output.memory_entries) == 2
    assert output.memory_entries[0]["insert_method"] == "none"
    assert output.memory_entries[1]["insert_method"] == "session_summary"
    assert output.memory_entries[1]["insert_params"]["skip_parametric_update"] is True
    assert output.memory_entries[1]["text"] == "Session summary text."
