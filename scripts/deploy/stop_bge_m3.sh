#!/usr/bin/env bash
# stop_bge_m3.sh — Detect and stop the BAAI/bge-m3 vLLM embedding server
#
# Usage:
#   bash scripts/deploy/stop_bge_m3.sh [--port PORT]
#
# Detection order:
#   1. PID file at .sage/run/bge_m3.pid
#   2. Fallback: process scan for 'vllm' + 'bge-m3' on the expected port

set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "$DEPLOY_DIR/../.." && pwd)"
RUN_LOG_DIR="$BENCH_ROOT/.sage/run"
PID_FILE="$RUN_LOG_DIR/bge_m3.pid"

# Load .env for default port
[[ -f "$BENCH_ROOT/.env" ]] && set -a && source "$BENCH_ROOT/.env" && set +a
PORT="${DEPLOY_EMBED_PORT:-18001}"

# Parse optional --port override
while [[ $# -gt 0 ]]; do
    case "$1" in
        --port) PORT="$2"; shift 2 ;;
        *) shift ;;
    esac
done

MODEL_TAG="bge-m3"
STOPPED=0

# ── Helper: kill a PID tree ───────────────────────────────────────────────────
_kill_pid() {
    local pid="$1"
    if kill -0 "$pid" 2>/dev/null; then
        echo "[INFO]  Sending SIGTERM to PID $pid and its children..."
        # Kill entire process group rooted at pid
        pkill -TERM -P "$pid" 2>/dev/null || true
        kill -TERM "$pid" 2>/dev/null || true
        # Wait up to 10 s for graceful shutdown
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
# Match any python/vllm process serving bge-m3 on the configured port
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
    echo "[OK]    BAAI/bge-m3 embedding server stopped."
else
    echo ""
    echo "[INFO]  No running BAAI/bge-m3 server found."
fi
exit 0
