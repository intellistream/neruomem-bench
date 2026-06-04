from __future__ import annotations

import json

import yaml

from benchmarks.evaluation.system_runtime_bench import run_system_benchmarks


def test_system_runtime_bench_produces_expected_sections(tmp_path):
    config_path = tmp_path / "fifo_system_eval.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "runtime": {
                    "memory_name": "fifo_system_eval",
                },
                "services": {
                    "services_type": "partitional.fifo_queue",
                    "fifo_queue": {
                        "max_capacity": 64,
                        "max_size": 64,
                        "retrieval_mode": "recent_first",
                        "retrieval_top_k": 3,
                        "telemetry_limit": 8,
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "summary.json"

    report = run_system_benchmarks(
        config_path,
        output_path=output_path,
        workers=[1, 2],
        initial_records=8,
        operations_per_worker=6,
        insert_every=3,
        retrieval_top_k=2,
        footprint_checkpoints=[2, 4],
        telemetry_limit_enabled=8,
    )

    assert report["service_type"] == "partitional.fifo_queue"
    assert report["service_name"] == "fifo_queue"
    assert len(report["concurrency"]) == 2
    assert report["concurrency"][0]["workers"] == 1
    assert report["concurrency"][1]["workers"] == 2
    assert report["footprint"][0]["entries"] == 2
    assert report["footprint"][1]["entries"] == 4

    disabled = report["observability"]["disabled"]
    enabled = report["observability"]["enabled"]
    assert disabled["telemetry"]["enabled"] is False
    assert enabled["telemetry"]["enabled"] is True
    assert "avg_latency_ms" in report["observability"]["delta"]

    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["service_name"] == "fifo_queue"