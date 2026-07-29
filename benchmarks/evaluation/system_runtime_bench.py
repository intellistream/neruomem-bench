"""ATC-facing runtime microbenchmarks for NeuroMem services.

This module measures three systems views over a single memory-service runtime:

1. Concurrency scaling under a mixed write/read workload.
2. Retained-state growth and process RSS as the collection grows.
3. Observability overhead with telemetry enabled vs disabled.

The benchmark intentionally bypasses the full LoCoMo model-serving stack. It is
meant to isolate the memory runtime boundary and export paper-ready metrics that
complement the end-to-end pipeline evaluation.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import math
import os
import sys
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
from typing import Any

import yaml

try:
    from sage.neuromem.services.neuromem_service_factory import NeuromemServiceFactory
except ModuleNotFoundError:
    benchmark_repo = Path(__file__).resolve().parents[2]
    source_workspace = (
        benchmark_repo.parent.parent
        if benchmark_repo.parent.name == "third_party"
        else benchmark_repo.parent / "neuromem"
    )
    candidate_repos = []
    if os.environ.get("NEUROMEM_ROOT"):
        candidate_repos.append(Path(os.environ["NEUROMEM_ROOT"]))
    candidate_repos.extend([source_workspace, Path.cwd().parent / "neuromem"])
    for candidate_repo in candidate_repos:
        if candidate_repo.exists():
            sys.path.insert(0, str(candidate_repo))
            sys.modules.pop("sage", None)
            sys.modules.pop("sage.neuromem", None)
            importlib.invalidate_caches()
            break
    from sage.neuromem.services.neuromem_service_factory import NeuromemServiceFactory


class ConfigView:
    """Small adapter that provides dotted-key access over a dict config."""

    def __init__(self, data: dict[str, Any]):
        self._data = data

    def get(self, key: str, default: Any = None) -> Any:
        current: Any = self._data
        for part in key.split("."):
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
        return current


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a NeuroMem service YAML config")
    parser.add_argument(
        "--workers",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8],
        help="Worker counts for the concurrency study",
    )
    parser.add_argument("--initial-records", type=int, default=128)
    parser.add_argument("--operations-per-worker", type=int, default=64)
    parser.add_argument(
        "--insert-every",
        type=int,
        default=4,
        help="Perform one insert every N operations; use 0 for retrieval-only workloads",
    )
    parser.add_argument("--retrieval-top-k", type=int, default=3)
    parser.add_argument(
        "--footprint-checkpoints",
        default="64,128,256,512",
        help="Comma-separated entry counts sampled in the footprint study",
    )
    parser.add_argument(
        "--telemetry-limit-enabled",
        type=int,
        default=100,
        help="Telemetry buffer size used for the enabled observability trial",
    )
    parser.add_argument("--output", default="", help="Optional JSON output path for the report")
    return parser.parse_args()


def load_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping")
    return data


def current_rss_mb() -> float:
    status_path = "/proc/self/status"
    if os.path.exists(status_path):
        with open(status_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) / 1024.0
    return 0.0


def percentile_ms(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    lower_value = ordered[lower_index]
    upper_value = ordered[upper_index]
    weight = position - lower_index
    return lower_value + (upper_value - lower_value) * weight


def jain_fairness(values: list[float]) -> float:
    if not values:
        return 0.0
    numerator = sum(values) ** 2
    denominator = len(values) * sum(value * value for value in values)
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def latency_summary(latencies_ms: list[float]) -> dict[str, float]:
    if not latencies_ms:
        return {
            "avg_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "max_ms": 0.0,
        }
    return {
        "avg_ms": sum(latencies_ms) / len(latencies_ms),
        "p50_ms": percentile_ms(latencies_ms, 0.50),
        "p95_ms": percentile_ms(latencies_ms, 0.95),
        "p99_ms": percentile_ms(latencies_ms, 0.99),
        "max_ms": max(latencies_ms),
    }


def sanitize_for_json(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_for_json(item) for item in value]
    return str(value)


def service_name_from_config(config_data: dict[str, Any]) -> str:
    services_type = config_data.get("services", {}).get("services_type")
    if not isinstance(services_type, str) or not services_type:
        raise ValueError("Missing services.services_type in config")
    return services_type.split(".")[-1]


def with_telemetry_limit(config_data: dict[str, Any], telemetry_limit: int) -> dict[str, Any]:
    updated = copy.deepcopy(config_data)
    service_name = service_name_from_config(updated)
    updated.setdefault("services", {}).setdefault(service_name, {})["telemetry_limit"] = telemetry_limit
    return updated


def build_service(config_data: dict[str, Any]):
    services_type = config_data.get("services", {}).get("services_type")
    config_view = ConfigView(config_data)
    factory = NeuromemServiceFactory.create(services_type, config_view)
    service = factory.create_service()
    service.setup()
    return service


def generate_records(count: int, prefix: str) -> list[str]:
    return [f"{prefix} record {index} token_{index % 17}" for index in range(count)]


def populate_service(service: Any, records: list[str]) -> None:
    for record in records:
        service.insert(record)


def run_worker(
    service: Any,
    *,
    worker_id: int,
    operations: int,
    insert_every: int,
    retrieval_top_k: int,
    query_pool: list[str],
) -> dict[str, Any]:
    latencies_ms: list[float] = []
    insert_count = 0
    retrieval_count = 0
    started = perf_counter()
    query_count = max(len(query_pool), 1)

    for operation_index in range(operations):
        operation_started = perf_counter()
        if insert_every > 0 and (operation_index + 1) % insert_every == 0:
            record = f"worker-{worker_id} live record {operation_index} topic_{operation_index % 13}"
            service.insert(record)
            insert_count += 1
        else:
            query = query_pool[(worker_id + operation_index) % query_count]
            service.retrieve(query=query, top_k=retrieval_top_k)
            retrieval_count += 1
        latencies_ms.append((perf_counter() - operation_started) * 1000.0)

    wall_time_s = perf_counter() - started
    return {
        "worker_id": worker_id,
        "wall_time_s": wall_time_s,
        "ops": operations,
        "insert_ops": insert_count,
        "retrieve_ops": retrieval_count,
        "throughput_ops_per_s": operations / wall_time_s if wall_time_s > 0 else 0.0,
        "latencies_ms": latencies_ms,
    }


def run_concurrency_study(
    config_data: dict[str, Any],
    *,
    worker_counts: list[int],
    initial_records: int,
    operations_per_worker: int,
    insert_every: int,
    retrieval_top_k: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    base_records = generate_records(initial_records, prefix="seed")

    for worker_count in worker_counts:
        service = build_service(copy.deepcopy(config_data))
        try:
            populate_service(service, base_records)
            started = perf_counter()
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                worker_reports = list(
                    executor.map(
                        lambda current_worker: run_worker(
                            service,
                            worker_id=current_worker,
                            operations=operations_per_worker,
                            insert_every=insert_every,
                            retrieval_top_k=retrieval_top_k,
                            query_pool=base_records,
                        ),
                        range(worker_count),
                    )
                )
            total_wall_s = perf_counter() - started
            all_latencies = [
                latency
                for worker_report in worker_reports
                for latency in worker_report["latencies_ms"]
            ]
            per_worker_throughput = [
                worker_report["throughput_ops_per_s"] for worker_report in worker_reports
            ]
            stats = service.get_stats()
            results.append(
                {
                    "workers": worker_count,
                    "total_ops": worker_count * operations_per_worker,
                    "total_wall_s": total_wall_s,
                    "throughput_ops_per_s": (
                        (worker_count * operations_per_worker) / total_wall_s if total_wall_s > 0 else 0.0
                    ),
                    "fairness_jain": jain_fairness(per_worker_throughput),
                    "latency_ms": latency_summary(all_latencies),
                    "worker_breakdown": [
                        {
                            "worker_id": worker_report["worker_id"],
                            "ops": worker_report["ops"],
                            "insert_ops": worker_report["insert_ops"],
                            "retrieve_ops": worker_report["retrieve_ops"],
                            "wall_time_s": worker_report["wall_time_s"],
                            "throughput_ops_per_s": worker_report["throughput_ops_per_s"],
                        }
                        for worker_report in worker_reports
                    ],
                    "telemetry": sanitize_for_json(stats.get("telemetry", {})),
                }
            )
        finally:
            service.teardown()

    return results


def run_footprint_study(
    config_data: dict[str, Any],
    *,
    checkpoints: list[int],
) -> list[dict[str, Any]]:
    if not checkpoints:
        return []

    sorted_checkpoints = sorted(set(checkpoints))
    records = generate_records(sorted_checkpoints[-1], prefix="footprint")
    service = build_service(copy.deepcopy(config_data))
    tracemalloc.start()
    samples: list[dict[str, Any]] = []
    inserted = 0

    try:
        for checkpoint in sorted_checkpoints:
            while inserted < checkpoint:
                service.insert(records[inserted])
                inserted += 1
            current_bytes, peak_bytes = tracemalloc.get_traced_memory()
            stats = service.get_stats()
            samples.append(
                {
                    "entries": checkpoint,
                    "rss_mb": current_rss_mb(),
                    "python_alloc_current_mb": current_bytes / (1024.0 * 1024.0),
                    "python_alloc_peak_mb": peak_bytes / (1024.0 * 1024.0),
                    "total_entries": stats.get("total_entries", checkpoint),
                    "index_count": stats.get("index_count", 0),
                    "storage": sanitize_for_json(stats.get("storage", {})),
                    "telemetry": sanitize_for_json(stats.get("telemetry", {})),
                }
            )
    finally:
        tracemalloc.stop()
        service.teardown()

    return samples


def run_observability_study(
    config_data: dict[str, Any],
    *,
    initial_records: int,
    operations: int,
    insert_every: int,
    retrieval_top_k: int,
    enabled_telemetry_limit: int,
) -> dict[str, Any]:
    base_records = generate_records(initial_records, prefix="observability")
    results: dict[str, Any] = {}

    for label, telemetry_limit in (("disabled", 0), ("enabled", enabled_telemetry_limit)):
        service = build_service(with_telemetry_limit(config_data, telemetry_limit))
        try:
            populate_service(service, base_records)
            worker_report = run_worker(
                service,
                worker_id=0,
                operations=operations,
                insert_every=insert_every,
                retrieval_top_k=retrieval_top_k,
                query_pool=base_records,
            )
            stats = service.get_stats()
            results[label] = {
                "ops": worker_report["ops"],
                "insert_ops": worker_report["insert_ops"],
                "retrieve_ops": worker_report["retrieve_ops"],
                "wall_time_s": worker_report["wall_time_s"],
                "throughput_ops_per_s": worker_report["throughput_ops_per_s"],
                "latency_ms": latency_summary(worker_report["latencies_ms"]),
                "telemetry": sanitize_for_json(stats.get("telemetry", {})),
            }
        finally:
            service.teardown()

    disabled_avg = results["disabled"]["latency_ms"]["avg_ms"]
    enabled_avg = results["enabled"]["latency_ms"]["avg_ms"]
    disabled_p95 = results["disabled"]["latency_ms"]["p95_ms"]
    enabled_p95 = results["enabled"]["latency_ms"]["p95_ms"]
    disabled_tp = results["disabled"]["throughput_ops_per_s"]
    enabled_tp = results["enabled"]["throughput_ops_per_s"]

    results["delta"] = {
        "avg_latency_ms": enabled_avg - disabled_avg,
        "p95_latency_ms": enabled_p95 - disabled_p95,
        "throughput_ops_per_s": enabled_tp - disabled_tp,
        "throughput_ratio": enabled_tp / disabled_tp if disabled_tp > 0 else 0.0,
    }
    return results


def run_all_studies(
    config_data: dict[str, Any],
    *,
    worker_counts: list[int],
    initial_records: int,
    operations_per_worker: int,
    insert_every: int,
    retrieval_top_k: int,
    footprint_checkpoints: list[int],
    enabled_telemetry_limit: int,
) -> dict[str, Any]:
    service_name = service_name_from_config(config_data)
    report = {
        "service_type": config_data.get("services", {}).get("services_type"),
        "service_name": service_name,
        "memory_name": config_data.get("runtime", {}).get("memory_name", service_name),
        "parameters": {
            "worker_counts": worker_counts,
            "initial_records": initial_records,
            "operations_per_worker": operations_per_worker,
            "insert_every": insert_every,
            "retrieval_top_k": retrieval_top_k,
            "footprint_checkpoints": footprint_checkpoints,
            "enabled_telemetry_limit": enabled_telemetry_limit,
        },
        "concurrency": run_concurrency_study(
            config_data,
            worker_counts=worker_counts,
            initial_records=initial_records,
            operations_per_worker=operations_per_worker,
            insert_every=insert_every,
            retrieval_top_k=retrieval_top_k,
        ),
        "footprint": run_footprint_study(config_data, checkpoints=footprint_checkpoints),
        "observability": run_observability_study(
            config_data,
            initial_records=initial_records,
            operations=operations_per_worker,
            insert_every=insert_every,
            retrieval_top_k=retrieval_top_k,
            enabled_telemetry_limit=enabled_telemetry_limit,
        ),
    }
    return sanitize_for_json(report)


def run_system_benchmarks(
    config_path: str | Path,
    *,
    output_path: str | Path | None = None,
    workers: list[int] | None = None,
    initial_records: int = 128,
    operations_per_worker: int = 64,
    insert_every: int = 4,
    retrieval_top_k: int = 3,
    footprint_checkpoints: list[int] | None = None,
    telemetry_limit_enabled: int = 100,
) -> dict[str, Any]:
    config_data = load_config(config_path)
    report = run_all_studies(
        config_data,
        worker_counts=workers or [1, 2, 4, 8],
        initial_records=initial_records,
        operations_per_worker=operations_per_worker,
        insert_every=insert_every,
        retrieval_top_k=retrieval_top_k,
        footprint_checkpoints=footprint_checkpoints or [64, 128, 256, 512],
        enabled_telemetry_limit=telemetry_limit_enabled,
    )

    if output_path is None:
        output_path = Path(".sage/output/benchmarks/system_runtime") / (
            f"{report['service_name']}_atc_system_eval.json"
        )
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    checkpoints = [int(item) for item in args.footprint_checkpoints.split(",") if item.strip()]
    report = run_system_benchmarks(
        args.config,
        output_path=args.output or None,
        workers=args.workers,
        initial_records=args.initial_records,
        operations_per_worker=args.operations_per_worker,
        insert_every=args.insert_every,
        retrieval_top_k=args.retrieval_top_k,
        footprint_checkpoints=checkpoints,
        telemetry_limit_enabled=args.telemetry_limit_enabled,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
