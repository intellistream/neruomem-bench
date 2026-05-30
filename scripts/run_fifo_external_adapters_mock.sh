#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_FILE="$BENCH_ROOT/benchmarks/experiment/config/fifo_external_adapters_mock.yaml"

TASK_ID="mock-01"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task_id) TASK_ID="$2"; shift 2 ;;
        *) echo "[WARN] Unknown argument: $1"; shift ;;
    esac
done

echo "========================================================================"
echo "  NeuroMem Benchmark - FIFO External Adapters Validation"
echo "========================================================================"
echo "  Config  : $CONFIG_FILE"
echo "  Task ID : $TASK_ID"
echo ""

cd "$BENCH_ROOT"
python -m test.installation_validation.external_strategy_adapter_pipeline \
    --config "$CONFIG_FILE" \
    --task_id "$TASK_ID"