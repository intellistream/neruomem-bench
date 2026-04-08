#!/usr/bin/env bash
# deploy_bge_m3.sh — Deploy BAAI/bge-m3 as a vLLM embedding server
#
# Usage:
#   bash scripts/deploy/deploy_bge_m3.sh [OPTIONS]
#
# Options:
#   --host HOST                 Bind host           (default: 0.0.0.0)
#   --port PORT                 Bind port           (default: 18001)
#   --gpus DEVICES              CUDA_VISIBLE_DEVICES (default: 0, e.g. 0,1 or all)
#   --gpu-memory-utilization N  GPU memory fraction  (default: 0.4)
#   --tensor-parallel-size N    Number of GPUs       (default: 1)
#   --env ENV_NAME              Conda env name       (default: neuromem)
#   --hf-token TOKEN            HuggingFace token    (optional)
#   --                          Pass remaining args directly to vllm serve
#
# Model: BAAI/bge-m3 (served as OpenAI-compatible /v1/embeddings endpoint)
# Log  : <bench_root>/.sage/installation/install_plus_vllm_embed_<timestamp>.log

set -euo pipefail

# ── Resolve project root (two levels up from scripts/deploy/) ────────────────
DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "$DEPLOY_DIR/../.." && pwd)"
INSTALL_DIR="$BENCH_ROOT/scripts/installation"
LOG_DIR="$BENCH_ROOT/.sage/installation"

# ── Source i18n for conda messages ──────────────────────────────────────────
: "${LANG_CODE:=en}"
source "$INSTALL_DIR/i18n.sh"

# ── Load .env from project root (non-fatal if absent) ───────────────────────
[[ -f "$BENCH_ROOT/.env" ]] && set -a && source "$BENCH_ROOT/.env" && set +a

# ── Defaults ─────────────────────────────────────────────────────────────────
ENV_NAME="neuromem"
HOST="0.0.0.0"
PORT="${DEPLOY_EMBED_PORT:-18001}"
GPU_DEVICES="${DEPLOY_EMBED_GPUS:-0}"
GPU_MEM_UTIL="${DEPLOY_EMBED_GPU_UTIL:-0.4}"
TP_SIZE="1"
HF_TOKEN=""
EXTRA_ARGS=()
MODEL="BAAI/bge-m3"

# ── Argument parsing ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --env)                   ENV_NAME="$2";      shift 2 ;;
        --host)                  HOST="$2";           shift 2 ;;
        --port)                  PORT="$2";           shift 2 ;;
        --gpus)                  GPU_DEVICES="$2";   shift 2 ;;
        --gpu-memory-utilization) GPU_MEM_UTIL="$2"; shift 2 ;;
        --tensor-parallel-size)  TP_SIZE="$2";       shift 2 ;;
        --hf-token)              HF_TOKEN="$2";      shift 2 ;;
        --)                      shift; EXTRA_ARGS+=("$@"); break ;;
        *)                       EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# ── Step 1: Check conda ───────────────────────────────────────────────────────
source "$INSTALL_DIR/check_conda.sh"
check_conda

# ── Step 2: Enforce active conda environment ─────────────────────────────────
echo ""
echo "[INFO]  Checking active conda environment..."
ACTIVE_ENV="${CONDA_DEFAULT_ENV:-}"
if [[ "$ACTIVE_ENV" != "$ENV_NAME" ]]; then
    echo "[ERROR] This script must be run inside the '$ENV_NAME' conda environment."
    echo "        Current active environment: '${ACTIVE_ENV:-<none>}'"
    echo "        Please activate it first:"
    echo "          conda activate $ENV_NAME"
    exit 1
fi
echo "[OK]    Active environment is '$ENV_NAME'."

# ── Step 3: Check / install vLLM minimum dependencies ────────────────────────
echo ""
echo "[INFO]  Checking vLLM installation in '$ENV_NAME'..."

mkdir -p "$LOG_DIR"
INSTALL_LOG="$LOG_DIR/install_plus_vllm_embed_$(date +%Y%m%d_%H%M%S).log"

_check_vllm() {
    conda run --no-capture-output -n "$ENV_NAME" \
        python -c "import vllm" 2>/dev/null
}

if _check_vllm; then
    VLLM_VER="$(conda run --no-capture-output -n "$ENV_NAME" \
        python -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "unknown")"
    echo "[OK]    vllm found (version: $VLLM_VER)"
else
    echo "[INFO]  vLLM not found. Installing into '$ENV_NAME'..."
    echo "======================================" >> "$INSTALL_LOG"
    echo "install_plus_vllm_embed — $(date)"      >> "$INSTALL_LOG"
    echo "model: $MODEL"                          >> "$INSTALL_LOG"
    echo "======================================" >> "$INSTALL_LOG"

    conda run --no-capture-output -n "$ENV_NAME" \
        pip install "vllm" 2>&1 | tee -a "$INSTALL_LOG"

    if _check_vllm; then
        echo "[OK]    vLLM installed successfully." | tee -a "$INSTALL_LOG"
        echo "[INFO]  Log saved to: $INSTALL_LOG"
    else
        echo "[ERROR] vLLM installation failed. Check log: $INSTALL_LOG"
        exit 1
    fi
fi

# ── Step 4: Launch vLLM embedding server (background) ───────────────────────
RUN_LOG_DIR="$BENCH_ROOT/.sage/run"
mkdir -p "$RUN_LOG_DIR"
SERVER_LOG="$RUN_LOG_DIR/bge_m3.log"
PID_FILE="$RUN_LOG_DIR/bge_m3.pid"

echo ""
echo "[INFO]  Starting vLLM embedding server for $MODEL (background)"
echo "[INFO]  Endpoint : http://$HOST:$PORT/v1/embeddings"
echo "[INFO]  GPUs     : CUDA_VISIBLE_DEVICES=$GPU_DEVICES"
echo "[INFO]  Env      : $ENV_NAME"
echo "[INFO]  Server log: $SERVER_LOG"
echo "[INFO]  PID file  : $PID_FILE"
echo ""

# Build the command array
CMD_ARGS=(
    vllm serve "$MODEL"
    --host "$HOST"
    --port "$PORT"
    --gpu-memory-utilization "$GPU_MEM_UTIL"
    --tensor-parallel-size "$TP_SIZE"
)

# Append any extra passthrough arguments
[[ ${#EXTRA_ARGS[@]} -gt 0 ]] && CMD_ARGS+=("${EXTRA_ARGS[@]}")

# Set GPU visibility and HF token
export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"
echo "[INFO]  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
# Set HuggingFace mirror endpoint if configured
if [[ -n "${HF_ENDPOINT:-}" ]]; then
    export HF_ENDPOINT
    echo "[INFO]  HF_ENDPOINT=$HF_ENDPOINT"
fi
if [[ -n "$HF_TOKEN" ]]; then
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
    echo "[INFO]  HuggingFace token set."
fi

# Launch in background; rotate old log first
if [[ -f "$SERVER_LOG" ]]; then
    mv "$SERVER_LOG" "${SERVER_LOG%.log}_$(date +%Y%m%d_%H%M%S).log"
fi
nohup conda run --no-capture-output -n "$ENV_NAME" "${CMD_ARGS[@]}" \
    >> "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$PID_FILE"
echo "[OK]    Server launched (PID $SERVER_PID)."
echo "[INFO]  Tailing log (Ctrl-C to stop tailing — server keeps running):"
echo "        $SERVER_LOG"
echo ""
tail -f "$SERVER_LOG"
