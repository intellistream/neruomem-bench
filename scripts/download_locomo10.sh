#!/usr/bin/env bash
# download_locomo10.sh — Download the public LoCoMo release and convert it to neuromem-bench local format

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$BENCH_ROOT/benchmarks/experiment/data/locomo"
OFFICIAL_FILE="$DATA_DIR/locomo10_official.json"
LOCAL_FILE="$DATA_DIR/locomo10_local.json"
SOURCE_URL="https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"

mkdir -p "$DATA_DIR"

echo "========================================================================"
echo "  Download LoCoMo10"
echo "========================================================================"
echo "  Source : $SOURCE_URL"
echo "  Output : $OFFICIAL_FILE"
echo "  Local  : $LOCAL_FILE"
echo ""

curl -L --fail "$SOURCE_URL" -o "$OFFICIAL_FILE"

python "$BENCH_ROOT/scripts/convert_locomo_official_to_local.py" \
    --input "$OFFICIAL_FILE" \
    --output "$LOCAL_FILE"

echo ""
echo "[OK] Downloaded official LoCoMo10 and converted it to local adapter format."
echo "========================================================================"