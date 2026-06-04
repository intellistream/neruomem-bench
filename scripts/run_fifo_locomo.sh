#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_FILE="$BENCH_ROOT/benchmarks/experiment/config/fifo_locomo_pipeline.yaml"
NEUROMEM_ROOT="$(cd "$BENCH_ROOT/.." && pwd)/neuromem"
SAGE_ROOT="$(cd "$BENCH_ROOT/.." && pwd)/SAGE/src"
DATA_FILE="${LOCOMO_DATA_FILE:-$BENCH_ROOT/benchmarks/experiment/data/locomo/locomo10_local.json}"

TASK_IDS=("conv-mini-01")

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data_file) DATA_FILE="$2"; shift 2 ;;
        --task_id) TASK_IDS=("$2"); shift 2 ;;
        --tasks) read -r -a TASK_IDS <<< "$2"; shift 2 ;;
        *) echo "[WARN] Unknown argument: $1"; shift ;;
    esac
done

if [[ ! -f "$DATA_FILE" ]]; then
    echo "[ERROR] LoCoMo data file not found: $DATA_FILE"
    exit 1
fi

export LOCOMO_DATA_FILE="$DATA_FILE"

DATASET="locomo"
MEMORY_NAME="fifo_queue"
LOG_BASE_DIR="$BENCH_ROOT/.sage/output/benchmarks/benchmark_memory/$DATASET/$MEMORY_NAME"
mkdir -p "$LOG_BASE_DIR"

echo "========================================================================"
echo "  NeuroMem Benchmark — FIFO Queue on LoCoMo"
echo "========================================================================"
echo "  Config  : $CONFIG_FILE"
echo "  Data    : $LOCOMO_DATA_FILE"
echo "  Tasks   : ${TASK_IDS[*]}"
echo "  Log dir : $LOG_BASE_DIR"
echo ""

_check_service() {
    local label="$1" url="$2"
    if curl -sf --max-time 5 "$url" > /dev/null 2>&1; then
        echo "[OK]    $label is reachable at $url"
    else
        echo "[ERROR] $label is NOT reachable at $url"
        echo "        Please start it with the corresponding deploy script first."
        exit 1
    fi
}

_check_service "LLM   (Llama-3.1-8B)" "http://localhost:18000/v1/models"
echo ""

cd "$BENCH_ROOT"
export PYTHONPATH="$SAGE_ROOT:$NEUROMEM_ROOT${PYTHONPATH:+:$PYTHONPATH}"

for i in "${!TASK_IDS[@]}"; do
    TASK_ID="${TASK_IDS[$i]}"
    TASK_NUM=$((i + 1))
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_DIR="$LOG_BASE_DIR/${TASK_ID}_${TIMESTAMP}"
    mkdir -p "$LOG_DIR"
    LOG_FILE="$LOG_DIR/terminal.log"

    echo "────────────────────────────────────────────────────────────────────"
    echo "  Task $TASK_NUM/${#TASK_IDS[@]}: $TASK_ID"
    echo "  Log  : $LOG_FILE"
    echo ""

    python -m benchmarks.experiment.memory_test_pipeline \
        --config "$CONFIG_FILE" \
        --task_id "$TASK_ID" \
        2>&1 | tee "$LOG_FILE"

    EXIT_CODE=${PIPESTATUS[0]}
    if [[ $EXIT_CODE -eq 0 ]]; then
        echo ""
        echo "[OK]    Task $TASK_ID completed successfully."
    else
        echo ""
        echo "[ERROR] Task $TASK_ID exited with code $EXIT_CODE."
        echo "        Check log: $LOG_FILE"
    fi
    echo ""
done

echo "========================================================================"
echo "  All tasks done. Results: $LOG_BASE_DIR"
echo "========================================================================"