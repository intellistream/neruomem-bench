from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.evaluation.matched_locomo_online import (
    _reject_answer_material,
    freeze_inputs,
    score_predictions,
)

from benchmarks.experiment.libs.pipeline_caller import (
    build_inference_question_payload,
    inference_question_metadata,
)


def test_inference_metadata_is_invariant_to_gold_and_evidence():
    base = {
        "question_id": "q-7",
        "question": "Where?",
        "answer": "Paris",
        "evidence": ["turn-1"],
        "category": "location",
    }
    mutated = {
        **base,
        "answer": "SECRET-CHANGED-GOLD",
        "evidence": ["SECRET-CHANGED-EVIDENCE"],
    }
    assert inference_question_metadata(base, fallback_index=7) == {"question_id": "q-7"}
    assert inference_question_metadata(mutated, fallback_index=7) == {"question_id": "q-7"}
    common = {
        "task_id": "conv-x",
        "session_id": 2,
        "dialog_id": 9,
        "dialogs": [{"speaker": "A", "text": "history"}],
        "question_index": 7,
    }
    # Query, history, and every byte passed into retrieval/generation are
    # invariant when the scoring-only gold/evidence sidecar changes.
    assert build_inference_question_payload(question=base, **common) == (
        build_inference_question_payload(question=mutated, **common)
    )


def test_answer_material_is_rejected_recursively():
    with pytest.raises(ValueError, match="answer-bearing"):
        _reject_answer_material(
            {"questions": [{"question_id": "q1", "metadata": {"gold": "secret"}}]}
        )


def test_freeze_keeps_answers_out_of_execution_workload(tmp_path: Path):
    dataset = [
        {
            "task_id": "conv-test",
            "sessions": [
                {
                    "session_id": 0,
                    "messages": [{"speaker": "A", "text": "A blue bicycle."}],
                }
            ],
            "questions": [
                {
                    "question": "What color?",
                    "answer": "secret-azure-reference",
                    "evidence": ["d1"],
                    "category": "fact",
                }
            ],
        }
    ]
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
    import hashlib

    protocol = {
        "dataset": {
            "name": "test",
            "sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        },
        "task_id": "conv-test",
    }
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    manifest = freeze_inputs(
        protocol_path=protocol_path,
        dataset_path=dataset_path,
        output_dir=tmp_path / "frozen",
    )
    workload = json.loads((tmp_path / "frozen" / "workload.json").read_text(encoding="utf-8"))
    answer_key = json.loads((tmp_path / "frozen" / "answer_key.json").read_text(encoding="utf-8"))
    _reject_answer_material(workload)
    assert "secret-azure-reference" not in json.dumps(workload)
    assert answer_key["answers"][0]["reference_answer"] == ("secret-azure-reference")
    assert manifest["answer_isolated"] is True


def test_official_freeze_preserves_time_speaker_and_visual_context(tmp_path: Path):
    dataset = [
        {
            "sample_id": "conv-official",
            "conversation": {
                "speaker_a": "A",
                "speaker_b": "B",
                "session_1_date_time": "4:04 pm on 20 January, 2023",
                "session_1": [
                    {
                        "speaker": "A",
                        "dia_id": "D1:1",
                        "text": "I arrived yesterday.",
                        "blip_caption": "a red train",
                    }
                ],
            },
            "qa": [
                {
                    "question": "When?",
                    "answer": "19 January, 2023",
                    "category": 2,
                    "evidence": ["D1:1"],
                },
                {
                    "question": "A trap?",
                    "category": 5,
                    "adversarial_answer": "tomorrow",
                },
            ],
        }
    ]
    dataset_path = tmp_path / "official.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
    import hashlib

    protocol = {
        "dataset": {
            "name": "test",
            "sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        },
        "task_id": "conv-official",
        "scoring": {"included_categories": [1, 2, 3, 4]},
    }
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    manifest = freeze_inputs(
        protocol_path=protocol_path,
        dataset_path=dataset_path,
        output_dir=tmp_path / "frozen",
    )
    workload = json.loads((tmp_path / "frozen" / "workload.json").read_text(encoding="utf-8"))
    assert manifest["question_count"] == 1
    assert workload["messages"] == [
        {
            "message_id": "D1:1",
            "session_id": 0,
            "dialog_id": 0,
            "session_datetime": "4:04 pm on 20 January, 2023",
            "speaker": "A",
            "text": "I arrived yesterday.",
            "visual_context": "a red train",
        }
    ]
    assert workload["questions"][0]["question"] == "When?"
    assert "evidence" not in json.dumps(workload)
    assert "adversarial_answer" not in json.dumps(workload)


def test_scorer_requires_exact_paired_matrix_and_has_no_answer_rules(tmp_path: Path):
    answer_key = {
        "schema_version": "neuromem-matched-locomo-answer-key-v1",
        "answers": [
            {"question_id": "q1", "reference_answer": "The blue bicycle"},
            {"question_id": "q2", "reference_answer": ["Paris", "Paris, France"]},
        ],
    }
    records = []
    for question_id, prediction in (("q1", "blue bicycle"), ("q2", "Paris")):
        for policy in ("safe_static_k2", "adaptive_budget"):
            records.append(
                {
                    "question_id": question_id,
                    "policy": policy,
                    "prediction": prediction,
                }
            )
    predictions_path = tmp_path / "predictions.json"
    answer_path = tmp_path / "answers.json"
    output_path = tmp_path / "scores.json"
    predictions_path.write_text(json.dumps({"records": records}), encoding="utf-8")
    answer_path.write_text(json.dumps(answer_key), encoding="utf-8")
    report = score_predictions(
        predictions_path=predictions_path,
        answer_key_path=answer_path,
        output_path=output_path,
    )
    assert report["aggregates"]["safe_static_k2"]["exact_match"] == 1.0
    assert report["aggregates"]["adaptive_budget"]["token_f1"] == 1.0
    assert report["scorer"] == ("normalized exact match and whitespace-token F1; no task rules")
