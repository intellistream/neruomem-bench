#!/usr/bin/env bash
# neuromem-bench quickstart — interactive installer
# Usage: bash quickstart.sh
set -euo pipefail

BENCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="$BENCH_ROOT/scripts"
export BENCH_ROOT

# ============================================================
# Step 0: Language selection
# ============================================================
echo ""
echo "  1) 中文   2) English"
echo -n "  Language / 语言 [1/2, default 2]: "
read -r _lang_choice </dev/tty || _lang_choice=""
case "${_lang_choice:-2}" in
    1) export LANG_CODE=cn ;;
    *) export LANG_CODE=en ;;
esac

source "$SCRIPTS_DIR/i18n.sh"

echo ""
echo "  === $MSG_WELCOME ==="
echo ""

# ============================================================
# Step 0b: Environment name
# ============================================================
echo -n "  $MSG_ENV_PROMPT"
read -r _env_input </dev/tty || _env_input=""
export ENV_NAME="${_env_input:-neuromem}"
echo ""

# ============================================================
# Step 1: Check conda
# ============================================================
source "$SCRIPTS_DIR/check_conda.sh"
check_conda

# ============================================================
# Step 2+3: Create env + install
# ============================================================
source "$SCRIPTS_DIR/setup_env.sh"
setup_env

# ============================================================
# Step 4: Validate
# ============================================================
source "$SCRIPTS_DIR/validate.sh"
validate_install

echo ""
printf "  $MSG_DONE\n" "$ENV_NAME"
echo ""
