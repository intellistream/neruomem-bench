#!/usr/bin/env bash
# locomo_evaluate.sh — Run round_analyzer on all subdirectories under locomo/
#
# Usage:
#   bash locomo_evaluate.sh [--prefix PREFIX] [--output-dir OUTPUT_DIR]
#
# Examples:
#   bash locomo_evaluate.sh
#   bash locomo_evaluate.sh --prefix TiM
#   bash locomo_evaluate.sh --output-dir /tmp/locomo_output

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$(dirname "$EVAL_DIR")")"

PREFIX=""
OUTPUT_DIR=""

# Parse optional args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix)
            PREFIX="$2"; shift 2 ;;
        --output-dir)
            OUTPUT_DIR="$2"; shift 2 ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Usage: $0 [--prefix PREFIX] [--output-dir OUTPUT_DIR]" >&2
            exit 1 ;;
    esac
done

cd "$PROJECT_ROOT"

CMD=(python "$EVAL_DIR/analysis/round_analyzer.py" --config locomo --all)

[[ -n "$PREFIX" ]] && CMD+=(--prefix "$PREFIX")
[[ -n "$OUTPUT_DIR" ]] && CMD+=(--output-dir "$OUTPUT_DIR")

echo "Running: ${CMD[*]}"
"${CMD[@]}"
