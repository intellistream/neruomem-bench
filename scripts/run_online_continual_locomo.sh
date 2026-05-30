#!/usr/bin/env bash
# run_online_continual_locomo.sh — Run online continual neural memory benchmark on LoCoMo
#
# Usage:
#   bash scripts/run_online_continual_locomo.sh [--task_id TASK_ID] [--tasks "id1 id2 ..."]
#
# Prerequisites:
#   1. conda activate neuromem
#   2. bash scripts/deploy/deploy_bge_m3.sh     (port 18001)
#   3. bash scripts/deploy/deploy_llama31_8b.sh (port 18000)
#
# Output:
#   .sage/output/benchmarks/benchmark_memory/locomo/online_continual_memory/<task_id>_<ts>/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_FILE="$BENCH_ROOT/benchmarks/experiment/config/online_continual_memory_locomo_pipeline.yaml"

TASK_IDS=("conv-mini-01")

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task_id) TASK_IDS=("$2"); shift 2 ;;
        --tasks) read -r -a TASK_IDS <<< "$2"; shift 2 ;;
        *) echo "[WARN] Unknown argument: $1"; shift ;;
    esac
done

DATASET="locomo"
MEMORY_NAME="online_continual_memory"
LOG_BASE_DIR="$BENCH_ROOT/.sage/output/benchmarks/benchmark_memory/$DATASET/$MEMORY_NAME"
mkdir -p "$LOG_BASE_DIR"

echo "========================================================================"
echo "  NeuroMem Benchmark — Online Continual Memory on LoCoMo"
echo "========================================================================"
echo "  Config  : $CONFIG_FILE"
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
_check_service "Embed (BGE-M3)" "http://localhost:18001/v1/models"
echo ""

cd "$BENCH_ROOT"

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