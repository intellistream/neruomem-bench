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
#   --env ENV_NAME|ENV_PATH     Conda env name/path  (default: neuromem)
#   --vllm-bin CMD              vLLM CLI command     (default: auto-detect vllm-hust/vllm)
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
VLLM_BIN=""
EXTRA_ARGS=()
MODEL="BAAI/bge-m3"

# ── Argument parsing ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --env)                   ENV_NAME="$2";      shift 2 ;;
        --vllm-bin)              VLLM_BIN="$2";      shift 2 ;;
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

CONDA_TARGET_FLAG="-n"
CONDA_TARGET_VALUE="$ENV_NAME"
if [[ "$ENV_NAME" == /* ]]; then
    CONDA_TARGET_FLAG="-p"
fi

# ── Step 2: Enforce active conda environment ─────────────────────────────────
echo ""
echo "[INFO]  Checking active conda environment..."
ACTIVE_ENV="${CONDA_DEFAULT_ENV:-}"
ACTIVE_PREFIX="${CONDA_PREFIX:-}"
if [[ "$CONDA_TARGET_FLAG" == "-p" ]]; then
    if [[ "$ACTIVE_PREFIX" != "$ENV_NAME" ]]; then
        echo "[ERROR] This script must be run inside the target conda environment path."
        echo "        Current active prefix: '${ACTIVE_PREFIX:-<none>}'"
        echo "        Expected prefix      : '$ENV_NAME'"
        echo "        Please activate it first:"
        echo "          conda activate $ENV_NAME"
        exit 1
    fi
    echo "[OK]    Active environment prefix is '$ENV_NAME'."
else
    if [[ "$ACTIVE_ENV" != "$ENV_NAME" ]]; then
        echo "[ERROR] This script must be run inside the '$ENV_NAME' conda environment."
        echo "        Current active environment: '${ACTIVE_ENV:-<none>}'"
        echo "        Please activate it first:"
        echo "          conda activate $ENV_NAME"
        exit 1
    fi
    echo "[OK]    Active environment is '$ENV_NAME'."
fi

IN_ACTIVE_TARGET_ENV="false"
if [[ "$CONDA_TARGET_FLAG" == "-p" && "$ACTIVE_PREFIX" == "$CONDA_TARGET_VALUE" ]]; then
    IN_ACTIVE_TARGET_ENV="true"
elif [[ "$CONDA_TARGET_FLAG" == "-n" && "$ACTIVE_ENV" == "$CONDA_TARGET_VALUE" ]]; then
    IN_ACTIVE_TARGET_ENV="true"
fi

run_in_target_env() {
    if [[ "$IN_ACTIVE_TARGET_ENV" == "true" ]]; then
        "$@"
    else
        conda run --no-capture-output "$CONDA_TARGET_FLAG" "$CONDA_TARGET_VALUE" "$@"
    fi
}

# ── Step 3: Check / install vLLM minimum dependencies ────────────────────────
echo ""
echo "[INFO]  Checking vLLM installation in '$ENV_NAME'..."

mkdir -p "$LOG_DIR"
INSTALL_LOG="$LOG_DIR/install_plus_vllm_embed_$(date +%Y%m%d_%H%M%S).log"

_check_vllm() {
    run_in_target_env python -c "import vllm" 2>/dev/null
}

if [[ -z "$VLLM_BIN" ]]; then
    if run_in_target_env bash -lc 'command -v vllm-hust >/dev/null 2>&1'; then
        VLLM_BIN="vllm-hust"
    else
        VLLM_BIN="vllm"
    fi
fi
echo "[INFO]  Using CLI: $VLLM_BIN"

if _check_vllm; then
    VLLM_VER="$(run_in_target_env \
        python -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "unknown")"
    echo "[OK]    vllm found (version: $VLLM_VER)"
else
    echo "[INFO]  vLLM not found. Installing into '$ENV_NAME'..."
    echo "======================================" >> "$INSTALL_LOG"
    echo "install_plus_vllm_embed — $(date)"      >> "$INSTALL_LOG"
    echo "model: $MODEL"                          >> "$INSTALL_LOG"
    echo "======================================" >> "$INSTALL_LOG"

    run_in_target_env \
        pip install "vllm" 2>&1 | tee -a "$INSTALL_LOG"

    if _check_vllm; then
        echo "[OK]    vLLM installed successfully." | tee -a "$INSTALL_LOG"
        echo "[INFO]  Log saved to: $INSTALL_LOG"
    else
        echo "[ERROR] vLLM installation failed. Check log: $INSTALL_LOG"
        exit 1
    fi
fi

# ── Step 4: Prepare local model files ───────────────────────────────────────
export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"
echo "[INFO]  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
export HF_HOME="$BENCH_ROOT/.sage/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE"
echo "[INFO]  HF_HOME=$HF_HOME"
if [[ -n "${HF_ENDPOINT:-}" ]]; then
    export HF_ENDPOINT
    echo "[INFO]  HF_ENDPOINT=$HF_ENDPOINT"
fi
if [[ -n "$HF_TOKEN" ]]; then
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
    echo "[INFO]  HuggingFace token set."
fi

MODEL_DIR="$BENCH_ROOT/.sage/models/bge-m3"
mkdir -p "$MODEL_DIR"

echo ""
echo "[INFO]  Pre-downloading embedding model to local directory..."
echo "        $MODEL_DIR"
DOWNLOAD_ARGS=(
    huggingface-cli download "$MODEL"
    --local-dir "$MODEL_DIR"
    --exclude ".DS_Store" "imgs/.DS_Store"
)
if [[ -n "$HF_TOKEN" ]]; then
    DOWNLOAD_ARGS+=(--token "$HF_TOKEN")
fi
if run_in_target_env "${DOWNLOAD_ARGS[@]}" 2>&1; then
    echo "[OK]    Embedding model ready at: $MODEL_DIR"
else
    echo "[WARN]  Pre-download ended with errors."
    echo "        Startup will continue using the local directory if usable."
fi

# ── Step 5: Launch vLLM embedding server (background) ───────────────────────
RUN_LOG_DIR="$BENCH_ROOT/.sage/run"
mkdir -p "$RUN_LOG_DIR"
SERVER_LOG="$RUN_LOG_DIR/bge_m3.log"
PID_FILE="$RUN_LOG_DIR/bge_m3.pid"

echo ""
echo "[INFO]  Starting vLLM embedding server for $MODEL (background)"
echo "[INFO]  Local model dir: $MODEL_DIR"
echo "[INFO]  Endpoint : http://$HOST:$PORT/v1/embeddings"
echo "[INFO]  GPUs     : CUDA_VISIBLE_DEVICES=$GPU_DEVICES"
echo "[INFO]  Env      : $ENV_NAME"
echo "[INFO]  CLI      : $VLLM_BIN"
echo "[INFO]  Server log: $SERVER_LOG"
echo "[INFO]  PID file  : $PID_FILE"
echo ""

CMD_ARGS=(
    "$VLLM_BIN" serve "$MODEL_DIR"
    --served-model-name "$MODEL"
    --host "$HOST"
    --port "$PORT"
    --gpu-memory-utilization "$GPU_MEM_UTIL"
    --tensor-parallel-size "$TP_SIZE"
)

[[ ${#EXTRA_ARGS[@]} -gt 0 ]] && CMD_ARGS+=("${EXTRA_ARGS[@]}")

LAUNCH_CMD=()
if [[ "$IN_ACTIVE_TARGET_ENV" == "true" ]]; then
    LAUNCH_CMD=("${CMD_ARGS[@]}")
else
    LAUNCH_CMD=(conda run --no-capture-output "$CONDA_TARGET_FLAG" "$CONDA_TARGET_VALUE" "${CMD_ARGS[@]}")
fi

# Launch in background; rotate old log first
if [[ -f "$SERVER_LOG" ]]; then
    mv "$SERVER_LOG" "${SERVER_LOG%.log}_$(date +%Y%m%d_%H%M%S).log"
fi
nohup "${LAUNCH_CMD[@]}" \
    >> "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$PID_FILE"
echo "[OK]    Server launched (PID $SERVER_PID)."
echo "[INFO]  Tailing log (Ctrl-C to stop tailing — server keeps running):"
echo "        $SERVER_LOG"
echo ""
tail -f "$SERVER_LOG"
