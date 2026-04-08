#!/usr/bin/env bash
# stop_llama31_8b.sh — Detect and stop the Llama-3.1-8B vLLM LLM server
#
# Usage:
#   bash scripts/deploy/stop_llama31_8b.sh [--port PORT]
#
# Detection order:
#   1. PID file at .sage/run/llama31_8b.pid
#   2. Fallback: process scan for 'vllm' + 'Llama-3.1' on the expected port

set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "$DEPLOY_DIR/../.." && pwd)"
RUN_LOG_DIR="$BENCH_ROOT/.sage/run"
PID_FILE="$RUN_LOG_DIR/llama31_8b.pid"

# Load .env for default port
[[ -f "$BENCH_ROOT/.env" ]] && set -a && source "$BENCH_ROOT/.env" && set +a
PORT="${DEPLOY_LLM_PORT:-18000}"

# Parse optional --port override
while [[ $# -gt 0 ]]; do
    case "$1" in
        --port) PORT="$2"; shift 2 ;;
        *) shift ;;
    esac
done

MODEL_TAG="Llama-3.1"
STOPPED=0

# ── Helper: kill a PID tree ───────────────────────────────────────────────────
_kill_pid() {
    local pid="$1"
    if kill -0 "$pid" 2>/dev/null; then
        echo "[INFO]  Sending SIGTERM to PID $pid and its children..."
        pkill -TERM -P "$pid" 2>/dev/null || true
        kill -TERM "$pid" 2>/dev/null || true
        local i=0
        while kill -0 "$pid" 2>/dev/null && (( i < 10 )); do
            sleep 1; (( i += 1 ))
        done
        if kill -0 "$pid" 2>/dev/null; then
            echo "[WARN]  Process $pid did not exit; sending SIGKILL..."
            pkill -KILL -P "$pid" 2>/dev/null || true
            kill -KILL "$pid" 2>/dev/null || true
        fi
        echo "[OK]    PID $pid stopped."
        STOPPED=1
    else
        echo "[WARN]  PID $pid is not running."
    fi
}

# ── Method 1: PID file ────────────────────────────────────────────────────────
if [[ -f "$PID_FILE" ]]; then
    SAVED_PID="$(cat "$PID_FILE")"
    echo "[INFO]  Found PID file: $PID_FILE (PID $SAVED_PID)"
    _kill_pid "$SAVED_PID"
    rm -f "$PID_FILE"
fi

# ── Method 2: Process scan (fallback / catch leaked workers) ─────────────────
echo "[INFO]  Scanning for residual '$MODEL_TAG' processes on port $PORT..."
PIDS="$(pgrep -f "vllm.*${MODEL_TAG}" 2>/dev/null || true)"
if [[ -n "$PIDS" ]]; then
    for pid in $PIDS; do
        echo "[INFO]  Found residual process PID $pid — stopping..."
        _kill_pid "$pid"
    done
fi

# ── Also check by port (catches ray/vllm worker processes) ───────────────────
if command -v fuser &>/dev/null; then
    PORT_PIDS="$(fuser "${PORT}/tcp" 2>/dev/null | tr ' ' '\n' | grep -v '^$' || true)"
    if [[ -n "$PORT_PIDS" ]]; then
        for pid in $PORT_PIDS; do
            echo "[INFO]  Process PID $pid is holding port $PORT — stopping..."
            _kill_pid "$pid"
        done
    fi
elif command -v lsof &>/dev/null; then
    PORT_PIDS="$(lsof -ti "tcp:${PORT}" 2>/dev/null || true)"
    if [[ -n "$PORT_PIDS" ]]; then
        for pid in $PORT_PIDS; do
            echo "[INFO]  Process PID $pid is holding port $PORT — stopping..."
            _kill_pid "$pid"
        done
    fi
fi

if [[ $STOPPED -eq 1 ]]; then
    echo ""
    echo "[OK]    meta-llama/Llama-3.1-8B-Instruct server stopped."
else
    echo ""
    echo "[INFO]  No running Llama-3.1-8B server found."
fi
exit 0
