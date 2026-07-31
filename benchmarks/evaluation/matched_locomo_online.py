"""Matched, answer-isolated LoCoMo real-online pilot.

This carrier is deliberately separate from the historical LoCoMo tables.  It
freezes an answer-free execution workload, runs two policies against one live
OpenAI-compatible model service, and scores only after predictions are sealed.
The pilot is not a full LoCoMo result and must not be promoted as such.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import string
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

FORBIDDEN_INFERENCE_KEYS = {
    "answer",
    "answers",
    "reference",
    "reference_answer",
    "gold",
    "ground_truth",
    "evidence",
}
POLICY_NAMES = ("safe_static_k2", "adaptive_budget")


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_new_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(_canonical_bytes(value))


def _reject_answer_material(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_INFERENCE_KEYS:
                raise ValueError(f"answer-bearing inference key at {path}.{key}")
            _reject_answer_material(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_answer_material(child, f"{path}[{index}]")


def _task_from_dataset(dataset_path: Path, task_id: str) -> dict[str, Any]:
    raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    matches = [
        row
        for row in raw
        if str(row.get("task_id") or row.get("sample_id")) == task_id
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one task {task_id}, found {len(matches)}")
    return matches[0]


def freeze_inputs(
    *,
    protocol_path: Path,
    dataset_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if _sha256_file(dataset_path) != protocol["dataset"]["sha256"]:
        raise ValueError("dataset SHA256 does not match the frozen protocol")
    task = _task_from_dataset(dataset_path, protocol["task_id"])

    messages = []
    for session in task["sessions"]:
        session_id = int(session["session_id"])
        for dialog_id, message in enumerate(session["messages"]):
            messages.append(
                {
                    "message_id": f"s{session_id:02d}-m{dialog_id:04d}",
                    "session_id": session_id,
                    "dialog_id": dialog_id,
                    "speaker": str(message.get("speaker", "Unknown")),
                    "text": str(message["text"]),
                }
            )

    questions = []
    answer_rows = []
    for index, question in enumerate(task["questions"], 1):
        question_id = f"{protocol['task_id']}-q{index:04d}"
        questions.append(
            {
                "question_id": question_id,
                "question": str(question["question"]),
                "history_checkpoint": len(messages),
            }
        )
        answer_rows.append(
            {
                "question_id": question_id,
                "reference_answer": question["answer"],
                "category": question.get("category"),
            }
        )

    workload = {
        "schema_version": "neuromem-matched-locomo-workload-v1",
        "dataset": protocol["dataset"]["name"],
        "task_id": protocol["task_id"],
        "messages": messages,
        "questions": questions,
    }
    _reject_answer_material(workload)
    answer_key = {
        "schema_version": "neuromem-matched-locomo-answer-key-v1",
        "dataset": protocol["dataset"]["name"],
        "task_id": protocol["task_id"],
        "answers": answer_rows,
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    workload_path = output_dir / "workload.json"
    answer_path = output_dir / "answer_key.json"
    _write_new_json(workload_path, workload)
    _write_new_json(answer_path, answer_key)
    manifest = {
        "schema_version": "neuromem-matched-locomo-freeze-v1",
        "protocol_path": str(protocol_path),
        "protocol_sha256": _sha256_file(protocol_path),
        "dataset_path": str(dataset_path),
        "dataset_sha256": _sha256_file(dataset_path),
        "workload_path": str(workload_path),
        "workload_sha256": _sha256_file(workload_path),
        "answer_key_path": str(answer_path),
        "answer_key_sha256": _sha256_file(answer_path),
        "message_count": len(messages),
        "question_count": len(questions),
        "answer_isolated": True,
    }
    _write_new_json(output_dir / "freeze_manifest.json", manifest)
    return manifest


def _policy_config(protocol: dict[str, Any], policy: str) -> dict[str, Any]:
    common = dict(protocol["memory"]["common"])
    budget = dict(protocol["memory"]["budget"])
    if policy == "safe_static_k2":
        budget["retrieval_slo_ms"] = None
    elif policy == "adaptive_budget":
        budget["retrieval_slo_ms"] = protocol["memory"]["adaptive_retrieval_slo_ms"]
    else:
        raise ValueError(f"unknown policy: {policy}")
    common["budget_controller"] = {"enabled": True, **budget}
    return common


def _create_service(protocol: dict[str, Any], policy: str):
    from sage.neuromem.memory_collection import NeuralMemoryCollection
    from sage.neuromem.services.partitional import OnlineContinualMemoryService

    config = _policy_config(protocol, policy)
    collection = NeuralMemoryCollection(name=f"matched-{policy}", config=config)
    return OnlineContinualMemoryService(collection, config)


def _json_request(url: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=_canonical_bytes(payload),
        headers={"Authorization": "Bearer EMPTY", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read()
        raise RuntimeError(f"HTTP {exc.code}: {body[:1000]!r}") from exc
    return json.loads(body)


def _answer_prompt(template: str, context: str, question: str) -> str:
    return template.replace("{context}", context).replace("{question}", question)


def run_online(
    *,
    protocol_path: Path,
    workload_path: Path,
    workload_sha256: str,
    api_base: str,
    model: str,
    tokenizer_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if _sha256_file(workload_path) != workload_sha256:
        raise ValueError("workload SHA256 mismatch")
    workload = json.loads(workload_path.read_text(encoding="utf-8"))
    _reject_answer_material(workload)
    if workload["task_id"] != protocol["task_id"]:
        raise ValueError("task ID mismatch")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        local_files_only=True,
        trust_remote_code=True,
    )
    services = {policy: _create_service(protocol, policy) for policy in POLICY_NAMES}
    for message in workload["messages"]:
        text = f"{message['speaker']}: {message['text']}"
        metadata = {
            "message_id": message["message_id"],
            "session_id": message["session_id"],
            "dialog_id": message["dialog_id"],
        }
        for policy in POLICY_NAMES:
            services[policy].insert(text, metadata=metadata)

    output_dir.mkdir(parents=True, exist_ok=False)
    raw_path = output_dir / "predictions.partial.jsonl"
    records: list[dict[str, Any]] = []
    request_index = 0
    started = time.time()
    with raw_path.open("x", encoding="utf-8") as raw_stream:
        for question_index, question in enumerate(workload["questions"]):
            pair_order = protocol["request_orders"][
                question_index % len(protocol["request_orders"])
            ]
            for policy in pair_order:
                service = services[policy]
                requested_top_k = int(protocol["policies"][policy]["requested_top_k"])
                retrieval_start = time.perf_counter()
                results = service.retrieve(
                    query=question["question"],
                    top_k=requested_top_k,
                )
                retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
                context = "\n".join(item["text"] for item in results)
                prompt = _answer_prompt(
                    protocol["generation"]["prompt_template"],
                    context,
                    question["question"],
                )
                messages = [{"role": "user", "content": prompt}]
                prompt_tokens = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                payload = {
                    "model": model,
                    "messages": messages,
                    "temperature": protocol["generation"]["temperature"],
                    "max_tokens": protocol["generation"]["max_tokens"],
                    "seed": protocol["generation"]["seed"],
                    "chat_template_kwargs": {"enable_thinking": False},
                }
                request_index += 1
                generation_start = time.perf_counter()
                response = _json_request(
                    f"{api_base.rstrip('/')}/chat/completions",
                    payload,
                    float(protocol["generation"]["request_timeout_s"]),
                )
                generation_ms = (time.perf_counter() - generation_start) * 1000
                choices = response.get("choices") or []
                if len(choices) != 1:
                    raise RuntimeError("model response does not contain exactly one choice")
                answer = str(choices[0].get("message", {}).get("content") or "")
                if not answer.strip():
                    raise RuntimeError("model returned an empty answer")
                record = {
                    "request_index": request_index,
                    "question_id": question["question_id"],
                    "policy": policy,
                    "requested_top_k": requested_top_k,
                    "applied_top_k": len(results),
                    "retrieved_ids": [item["metadata"]["message_id"] for item in results],
                    "retrieval_ms": retrieval_ms,
                    "prompt_sha256": _sha256_bytes(prompt.encode()),
                    "prompt_tokens": len(prompt_tokens),
                    "prediction": answer,
                    "generation_ms": generation_ms,
                    "usage": response.get("usage"),
                    "response_id": response.get("id"),
                }
                records.append(record)
                raw_stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                raw_stream.flush()

    expected = len(workload["questions"]) * len(POLICY_NAMES)
    if len(records) != expected:
        raise RuntimeError(f"incomplete prediction matrix: {len(records)} != {expected}")
    predictions = {
        "schema_version": "neuromem-matched-locomo-predictions-v1",
        "protocol_sha256": _sha256_file(protocol_path),
        "workload_sha256": workload_sha256,
        "model": model,
        "api_base": api_base,
        "started_unix_s": started,
        "finished_unix_s": time.time(),
        "records": records,
        "policy_snapshots": {
            policy: services[policy].budget_controller.snapshot()
            for policy in POLICY_NAMES
        },
    }
    prediction_path = output_dir / "predictions.json"
    _write_new_json(prediction_path, predictions)
    manifest = {
        "schema_version": "neuromem-matched-locomo-run-v1",
        "evidence_label": "real-online",
        "pilot_only": True,
        "protocol_sha256": _sha256_file(protocol_path),
        "workload_sha256": workload_sha256,
        "predictions_sha256": _sha256_file(prediction_path),
        "record_count": len(records),
        "question_count": len(workload["questions"]),
        "policies": list(POLICY_NAMES),
        "automatic_retry_count": 0,
        "answer_key_visible_to_runner": False,
    }
    _write_new_json(output_dir / "run_manifest.json", manifest)
    return manifest


_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)


def _normalize_answer(value: str) -> str:
    value = value.lower()
    value = "".join(character for character in value if character not in string.punctuation)
    value = _ARTICLES.sub(" ", value)
    return " ".join(value.split())


def _references(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _exact_match(prediction: str, reference: str) -> float:
    return float(_normalize_answer(prediction) == _normalize_answer(reference))


def _token_f1(prediction: str, reference: str) -> float:
    prediction_tokens = _normalize_answer(prediction).split()
    reference_tokens = _normalize_answer(reference).split()
    if not prediction_tokens or not reference_tokens:
        return float(prediction_tokens == reference_tokens)
    counts: dict[str, int] = {}
    for token in prediction_tokens:
        counts[token] = counts.get(token, 0) + 1
    common = 0
    for token in reference_tokens:
        if counts.get(token, 0) > 0:
            common += 1
            counts[token] -= 1
    if common == 0:
        return 0.0
    precision = common / len(prediction_tokens)
    recall = common / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def score_predictions(
    *,
    predictions_path: Path,
    answer_key_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    answer_key = json.loads(answer_key_path.read_text(encoding="utf-8"))
    answers = {row["question_id"]: row for row in answer_key["answers"]}
    records = predictions["records"]
    expected_keys = {
        (question_id, policy)
        for question_id in answers
        for policy in POLICY_NAMES
    }
    actual_keys = {(row["question_id"], row["policy"]) for row in records}
    if actual_keys != expected_keys or len(records) != len(expected_keys):
        raise ValueError("prediction matrix does not exactly match the answer key")

    scored = []
    for record in records:
        answer = answers[record["question_id"]]
        refs = _references(answer["reference_answer"])
        scored.append(
            {
                "question_id": record["question_id"],
                "policy": record["policy"],
                "exact_match": max(_exact_match(record["prediction"], ref) for ref in refs),
                "token_f1": max(_token_f1(record["prediction"], ref) for ref in refs),
                "category": answer.get("category"),
            }
        )
    aggregates = {}
    for policy in POLICY_NAMES:
        rows = [row for row in scored if row["policy"] == policy]
        aggregates[policy] = {
            "n": len(rows),
            "exact_match": statistics.fmean(row["exact_match"] for row in rows),
            "token_f1": statistics.fmean(row["token_f1"] for row in rows),
        }
    paired_f1 = [
        next(
            row["token_f1"]
            for row in scored
            if row["question_id"] == question_id and row["policy"] == "adaptive_budget"
        )
        - next(
            row["token_f1"]
            for row in scored
            if row["question_id"] == question_id and row["policy"] == "safe_static_k2"
        )
        for question_id in answers
    ]
    report = {
        "schema_version": "neuromem-matched-locomo-score-v1",
        "predictions_sha256": _sha256_file(predictions_path),
        "answer_key_sha256": _sha256_file(answer_key_path),
        "scorer": "normalized exact match and whitespace-token F1; no task rules",
        "aggregates": aggregates,
        "paired_token_f1_delta": {
            "n": len(paired_f1),
            "mean": statistics.fmean(paired_f1),
            "wins": sum(delta > 0 for delta in paired_f1),
            "ties": sum(math.isclose(delta, 0.0, abs_tol=1e-12) for delta in paired_f1),
            "losses": sum(delta < 0 for delta in paired_f1),
        },
        "rows": scored,
    }
    _write_new_json(output_path, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--protocol", type=Path, required=True)
    freeze.add_argument("--dataset", type=Path, required=True)
    freeze.add_argument("--output-dir", type=Path, required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--protocol", type=Path, required=True)
    run.add_argument("--workload", type=Path, required=True)
    run.add_argument("--workload-sha256", required=True)
    run.add_argument("--api-base", required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--tokenizer", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)

    score = subparsers.add_parser("score")
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--answer-key", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "freeze":
        result = freeze_inputs(
            protocol_path=args.protocol,
            dataset_path=args.dataset,
            output_dir=args.output_dir,
        )
    elif args.command == "run":
        result = run_online(
            protocol_path=args.protocol,
            workload_path=args.workload,
            workload_sha256=args.workload_sha256,
            api_base=args.api_base,
            model=args.model,
            tokenizer_path=args.tokenizer,
            output_dir=args.output_dir,
        )
    else:
        result = score_predictions(
            predictions_path=args.predictions,
            answer_key_path=args.answer_key,
            output_path=args.output,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
