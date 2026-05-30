from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


_SESSION_KEY_RE = re.compile(r"^session_(\d+)$")
_DIA_ID_RE = re.compile(r"^D(\d+):(\d+)$")


def _sorted_session_items(conversation: dict[str, Any]) -> list[tuple[int, list[dict[str, Any]]]]:
    items: list[tuple[int, list[dict[str, Any]]]] = []
    for key, value in conversation.items():
        match = _SESSION_KEY_RE.match(key)
        if not match or not isinstance(value, list):
            continue
        items.append((int(match.group(1)), value))
    items.sort(key=lambda item: item[0])
    return items


def _parse_evidence_token(token: str) -> tuple[int, int] | None:
    match = _DIA_ID_RE.match(token.strip())
    if not match:
        return None
    session_num = int(match.group(1))
    dialog_num = int(match.group(2))
    return session_num - 1, dialog_num - 1


def _convert_questions(raw_questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    for item in raw_questions:
        evidences = item.get("evidence") or []
        visible_session = 0
        visible_dialog = 0

        parsed_positions = [pos for token in evidences if (pos := _parse_evidence_token(str(token)))]
        if parsed_positions:
            visible_session, visible_dialog = max(parsed_positions)

        questions.append(
            {
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "category": item.get("category"),
                "evidence": list(evidences),
                "visible_session": visible_session,
                "visible_dialog": visible_dialog,
            }
        )
    return questions


def convert_record(record: dict[str, Any]) -> dict[str, Any]:
    conversation = record.get("conversation") or {}
    sessions = []
    for session_idx, messages in _sorted_session_items(conversation):
        session_messages = [
            {
                "speaker": message.get("speaker", ""),
                "text": message.get("text", ""),
            }
            for message in messages
        ]
        sessions.append(
            {
                "session_id": session_idx - 1,
                "messages": session_messages,
            }
        )

    return {
        "task_id": record.get("sample_id", ""),
        "sessions": sessions,
        "questions": _convert_questions(record.get("qa") or []),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert official LoCoMo JSON into neuromem-bench local adapter format"
    )
    parser.add_argument("--input", required=True, help="Path to the official LoCoMo JSON file")
    parser.add_argument("--output", required=True, help="Path to the converted local-format JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    with input_path.open(encoding="utf-8") as f:
        raw_records = json.load(f)

    converted = [convert_record(record) for record in raw_records]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(
        f"Converted {len(converted)} records from {input_path} to {output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())