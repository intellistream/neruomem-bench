#!/usr/bin/env bash
# deploy_llama31_8b.sh — Deploy meta-llama/Llama-3.1-8B-Instruct via vLLM
#
# Usage:
#   bash scripts/deploy/deploy_llama31_8b.sh [OPTIONS]
#
# Options:
#   --host HOST                 Bind host           (default: 0.0.0.0)
#   --port PORT                 Bind port           (default: 18000)
#   --gpus DEVICES              CUDA_VISIBLE_DEVICES (default: 0, e.g. 0,1 or all)
#   --gpu-memory-utilization N  GPU memory fraction  (default: 0.9)
#   --tensor-parallel-size N    Number of GPUs       (default: 1)
#   --dtype DTYPE               Model dtype          (default: auto)
#   --max-model-len N           Max context length   (default: 8192)
#   --env ENV_NAME              Conda env name       (default: neuromem)
#   --hf-token TOKEN            HuggingFace token    (REQUIRED for Llama gated repo)
#   --                          Pass remaining args directly to vllm serve
#
# Model: meta-llama/Llama-3.1-8B-Instruct (OpenAI-compatible /v1/chat/completions)
# Log  : <bench_root>/.sage/installation/install_plus_vllm_llm_<timestamp>.log
#
# NOTE: Accessing Llama-3.1 requires accepting the Meta license on HuggingFace
#       and providing a valid --hf-token.

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
PORT="${DEPLOY_LLM_PORT:-18000}"
GPU_DEVICES="${DEPLOY_LLM_GPUS:-0}"
GPU_MEM_UTIL="${DEPLOY_LLM_GPU_UTIL:-0.9}"
TP_SIZE="1"
DTYPE="auto"
MAX_MODEL_LEN="8192"
HF_TOKEN=""
EXTRA_ARGS=()
MODEL="meta-llama/Llama-3.1-8B-Instruct"

# ── Argument parsing ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --env)                   ENV_NAME="$2";      shift 2 ;;
        --host)                  HOST="$2";           shift 2 ;;
        --port)                  PORT="$2";           shift 2 ;;
        --gpus)                  GPU_DEVICES="$2";   shift 2 ;;
        --gpu-memory-utilization) GPU_MEM_UTIL="$2"; shift 2 ;;
        --tensor-parallel-size)  TP_SIZE="$2";       shift 2 ;;
        --dtype)                 DTYPE="$2";         shift 2 ;;
        --max-model-len)         MAX_MODEL_LEN="$2"; shift 2 ;;
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
INSTALL_LOG="$LOG_DIR/install_plus_vllm_llm_$(date +%Y%m%d_%H%M%S).log"

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
    echo "install_plus_vllm_llm — $(date)"        >> "$INSTALL_LOG"
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

# ── Step 4: Validate HF token presence ───────────────────────────────────────
if [[ -z "$HF_TOKEN" ]]; then
    # Fall back to env variable if already set in the shell or .env
    HF_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}"
fi
if [[ -z "$HF_TOKEN" ]]; then
    echo ""
    echo "[ERROR] No HuggingFace token found."
    echo "        Llama-3.1 is a gated model and requires authentication."
    echo ""
    echo "        Option 1 — set HUGGING_FACE_HUB_TOKEN in .env:"
    echo "          echo 'HUGGING_FACE_HUB_TOKEN=hf_xxx' >> $BENCH_ROOT/.env"
    echo ""
    echo "        Option 2 — pass via flag:"
    echo "          bash $0 --hf-token hf_xxx"
    echo ""
    echo "        You must first accept the Meta license at:"
    echo "          https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct"
    echo ""
    exit 1
fi

# ── Step 4.5: Pre-download model weights to local HF cache ─────────────────
# Run this before launching vLLM so the server starts from local cache without
# any network activity. huggingface-cli skips automatically if already cached.
export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
[[ -n "${HF_ENDPOINT:-}" ]]              && export HF_ENDPOINT
export HF_HUB_HTTP_TIMEOUT="${HF_HUB_HTTP_TIMEOUT:-120}"
export HF_HUB_DOWNLOAD_RETRY_COUNT="${HF_HUB_DOWNLOAD_RETRY_COUNT:-5}"

echo ""
echo "[INFO]  Pre-downloading model weights to local HF cache..."
echo "        (Already-cached files are skipped automatically)"
if conda run --no-capture-output -n "$ENV_NAME" \
        huggingface-cli download "$MODEL" --token "$HF_TOKEN" 2>&1 ; then
    echo "[OK]    Model weights ready in local cache."
else
    echo "[WARN]  Pre-download ended with errors."
    echo "        vLLM will attempt to load from cache anyway."
    echo "        If the model is not cached, startup may fail."
fi

# ── Step 4.9: Kill any lingering vLLM processes before startup ──────────────
# Failed or Ctrl-C’d deploys leave orphan processes that lock GPU memory.
# Kill them now so the new launch starts from a clean slate.
RUN_LOG_DIR="$BENCH_ROOT/.sage/run"
mkdir -p "$RUN_LOG_DIR"
PID_FILE_TMP="$RUN_LOG_DIR/llama31_8b.pid"

_cleanup_existing() {
    # 1. PID file
    if [[ -f "$PID_FILE_TMP" ]]; then
        local old_pid; old_pid="$(cat "$PID_FILE_TMP")"
        if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
            echo "[INFO]  Stopping existing server (PID $old_pid) before relaunch..."
            pkill -TERM -P "$old_pid" 2>/dev/null || true
            kill -TERM "$old_pid"   2>/dev/null || true
            sleep 3
            kill -0 "$old_pid" 2>/dev/null && kill -KILL "$old_pid" 2>/dev/null || true
            echo "[OK]    Old server stopped."
        fi
        rm -f "$PID_FILE_TMP"
    fi
    # 2. Any remaining vLLM processes matching model name on target port
    local stale_pids
    stale_pids="$(pgrep -f "vllm.*[Ll]lama.*$PORT|vllm.*$PORT.*[Ll]lama" 2>/dev/null || true)"
    if [[ -n "$stale_pids" ]]; then
        echo "[INFO]  Killing stale vLLM LLM processes: $stale_pids"
        echo "$stale_pids" | xargs -r kill -KILL 2>/dev/null || true
        sleep 2
        echo "[OK]    Stale processes removed."
    fi
}
_cleanup_existing

# ── Step 5: Launch vLLM LLM server (background) ─────────────────────────────
RUN_LOG_DIR="$BENCH_ROOT/.sage/run"
mkdir -p "$RUN_LOG_DIR"
SERVER_LOG="$RUN_LOG_DIR/llama31_8b.log"
PID_FILE="$RUN_LOG_DIR/llama31_8b.pid"

echo ""
echo "[INFO]  Starting vLLM LLM server for $MODEL (background)"
echo "[INFO]  Endpoint : http://$HOST:$PORT/v1/chat/completions"
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
    --dtype "$DTYPE"
    --max-model-len "$MAX_MODEL_LEN"
)

# Append any extra passthrough arguments
[[ ${#EXTRA_ARGS[@]} -gt 0 ]] && CMD_ARGS+=("${EXTRA_ARGS[@]}")

# Env vars already exported in Step 4.5; echo summary here
echo "[INFO]  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
[[ -n "${HF_ENDPOINT:-}" ]] && echo "[INFO]  HF_ENDPOINT=$HF_ENDPOINT"
[[ -n "$HF_TOKEN" ]]        && echo "[INFO]  HuggingFace token set."

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
