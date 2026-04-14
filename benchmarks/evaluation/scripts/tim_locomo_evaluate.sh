#!/usr/bin/env bash
# tim_locomo_evaluate.sh — Evaluate TiM results on the LoCoMo dataset
#
# Usage:
#   bash tim_locomo_evaluate.sh [--output-dir OUTPUT_DIR]
#
# Examples:
#   bash tim_locomo_evaluate.sh
#   bash tim_locomo_evaluate.sh --output-dir /tmp/tim_output

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$(dirname "$EVAL_DIR")")"

OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir)
            OUTPUT_DIR="$2"; shift 2 ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Usage: $0 [--output-dir OUTPUT_DIR]" >&2
            exit 1 ;;
    esac
done

cd "$PROJECT_ROOT"

CMD=(python "$EVAL_DIR/analysis/round_analyzer.py" --config locomo --path TiM)
[[ -n "$OUTPUT_DIR" ]] && CMD+=(--output-dir "$OUTPUT_DIR")

echo "Running: ${CMD[*]}"
"${CMD[@]}"
