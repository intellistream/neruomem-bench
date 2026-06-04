from __future__ import annotations

from scripts.convert_locomo_official_to_local import convert_record


def test_convert_record_preserves_session_summary():
    record = {
        "sample_id": "conv-30",
        "conversation": {
            "session_1": [{"speaker": "Jon", "text": "Hello"}],
        },
        "qa": [],
        "session_summary": {"session_1_summary": "Summary text."},
        "event_summary": {"event_1": "Event text."},
    }

    converted = convert_record(record)

    assert converted["task_id"] == "conv-30"
    assert converted["session_summary"]["session_1_summary"] == "Summary text."
    assert converted["event_summary"]["event_1"] == "Event text."
