#!/usr/bin/env bash
# run_mem0_locomo.sh — Run mem0 black-box benchmark on LoCoMo dataset
#
# Usage:
#   bash scripts/run_mem0_locomo.sh [--task_id TASK_ID] [--tasks "id1 id2 ..."]
#
# Prerequisites:
#   1. conda activate neuromem
#   2. pip install mem0ai qdrant-client
#   3. docker run -p 6333:6333 qdrant/qdrant          (Qdrant vector store)
#   4. bash scripts/deploy/deploy_bge_m3.sh            (port 18001, Embedding)
#   5. bash scripts/deploy/deploy_llama31_8b.sh        (port 18000, LLM)
#   6. Register your LoCoMo DataLoader adapter:
#      DataLoaderFactory.register("locomo", YourLocomoLoader)
#
# Output:
#   .sage/output/benchmarks/benchmark_memory/locomo/mem0/<task_id>_<ts>/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_SCRIPT="$BENCH_ROOT/benchmarks/simple_experiment/simple_pipeline.py"
CONFIG_FILE="$BENCH_ROOT/benchmarks/simple_experiment/config/mem0_locomo.yaml"

# ── Defaults ─────────────────────────────────────────────────────────────────
TASK_IDS=("conv-mini-01")

# ── Argument parsing ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --task_id)  TASK_IDS=("$2"); shift 2 ;;
        --tasks)    read -r -a TASK_IDS <<< "$2"; shift 2 ;;
        *) echo "[WARN] Unknown argument: $1"; shift ;;
    esac
done

# ── Paths ─────────────────────────────────────────────────────────────────────
DATASET="locomo"
MEMORY_NAME="mem0"
LOG_BASE_DIR="$BENCH_ROOT/.sage/output/benchmarks/benchmark_memory/$DATASET/$MEMORY_NAME"
mkdir -p "$LOG_BASE_DIR"

echo "========================================================================"
echo "  NeuroMem Benchmark — mem0 (black-box)"
echo "========================================================================"
echo "  Config  : $CONFIG_FILE"
echo "  Tasks   : ${TASK_IDS[*]}"
echo "  Log dir : $LOG_BASE_DIR"
echo ""

# ── Pre-flight health checks ──────────────────────────────────────────────────
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
_check_service "LLM   (Llama-3.1-8B)"  "http://localhost:18000/v1/models"
_check_service "Embed (BGE-M3)"         "http://localhost:18001/v1/models"
_check_service "Qdrant"                 "http://localhost:6333/healthz"
echo ""

# ── Run tasks ─────────────────────────────────────────────────────────────────
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

    python -m benchmarks.simple_experiment.simple_pipeline \
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
