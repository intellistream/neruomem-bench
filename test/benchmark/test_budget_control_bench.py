from __future__ import annotations

import json

from benchmarks.evaluation.budget_control_bench import run_budget_control_benchmark


def test_budget_control_benchmark_is_reproducible_and_exports_report(tmp_path):
    output_path = tmp_path / "budget-control.json"
    kwargs = {
        "seeds": [7, 13],
        "operations_per_phase": 12,
        "latency_slo_ms": 24.0,
        "state_budget_entries": 4,
        "requested_top_k": 8,
        "observation_window": 4,
    }

    first = run_budget_control_benchmark(**kwargs, output_path=output_path)
    second = run_budget_control_benchmark(**kwargs)

    assert first["config"] == second["config"]
    assert [
        (trial["policy"], trial["seed"], trial["mean_top_k_by_phase"])
        for trial in first["trials"]
    ] == [
        (trial["policy"], trial["seed"], trial["mean_top_k_by_phase"])
        for trial in second["trials"]
    ]
    assert len(first["aggregates"]) == 5
    assert json.loads(output_path.read_text(encoding="utf-8"))["benchmark"] == (
        "controlled_phase_shift"
    )
    assert output_path.with_suffix(".manifest.json").exists()
    assert output_path.with_suffix(".environment.txt").exists()


def test_online_controller_trades_quality_for_fewer_budget_violations():
    report = run_budget_control_benchmark(
        seeds=[11, 23],
        operations_per_phase=40,
        latency_slo_ms=24.0,
        state_budget_entries=8,
        requested_top_k=8,
        observation_window=8,
    )
    aggregates = {row["policy"]: row for row in report["aggregates"]}

    online = aggregates["online_controller"]
    aggressive = aggregates["static_aggressive"]
    conservative = aggregates["static_conservative"]
    no_controller = aggregates["no_controller"]

    assert online["latency_violation_rate"]["mean"] < aggressive["latency_violation_rate"]["mean"]
    assert online["quality_proxy"]["mean"] > conservative["quality_proxy"]["mean"]
    assert online["state_violation_rate"]["mean"] == 0.0
    assert no_controller["state_violation_rate"]["mean"] > 0.0
