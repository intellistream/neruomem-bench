#!/usr/bin/env bash
# setup_env.sh — Step 2+3: create conda env (Python 3.11) and pip install -e .
# Expects: ENV_NAME, BENCH_ROOT (from quickstart.sh or caller)

set -euo pipefail

if [[ -z "${MSG_ENV_CREATING:-}" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    source "$SCRIPT_DIR/i18n.sh"
fi

setup_env() {
    local env_name="${ENV_NAME:-neuromem}"
    local bench_root="${BENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
    local log_dir="$bench_root/.sage/installation"
    local log_file="$log_dir/install_$(date +%Y%m%d_%H%M%S).log"

    mkdir -p "$log_dir"
    echo "[INFO]  $(printf "$MSG_LOG_DIR" "$log_dir")"
    echo "======================================" >> "$log_file"
    echo "neuromem-bench install — $(date)"      >> "$log_file"
    echo "======================================" >> "$log_file"

    # ---- create env if it does not exist ----
    if conda env list | awk '{print $1}' | grep -qx "$env_name"; then
        echo "[INFO]  $(printf "$MSG_ENV_EXISTS" "$env_name")" | tee -a "$log_file"
    else
        echo "[INFO]  $(printf "$MSG_ENV_CREATING" "$env_name")" | tee -a "$log_file"
        conda create -y -n "$env_name" python=3.11 2>&1 | tee -a "$log_file"
        echo "[OK]    $MSG_ENV_CREATED" | tee -a "$log_file"
    fi

    # ---- verify Python version inside env ----
    local py_ver
    py_ver="$(conda run --no-capture-output -n "$env_name" python --version 2>&1)"
    if [[ "$py_ver" != *"3.11"* ]]; then
        echo "[ERROR] $(printf "$MSG_PYTHON_MISMATCH" "$py_ver")" | tee -a "$log_file"
        return 1
    fi
    echo "[OK]    $(printf "$MSG_PYTHON_OK" "$py_ver")" | tee -a "$log_file"

    # ---- install from pyproject.toml (editable, real-time output + log) ----
    echo "[INFO]  $MSG_INSTALLING" | tee -a "$log_file"
    conda run --no-capture-output -n "$env_name" \
        pip install -e "$bench_root" 2>&1 | tee -a "$log_file"
    echo "[OK]    $MSG_INSTALL_OK" | tee -a "$log_file"

    export SETUP_ENV_NAME="$env_name"
    export INSTALL_LOG="$log_file"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    : "${LANG_CODE:=en}"
    : "${ENV_NAME:=neuromem}"
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    source "$SCRIPT_DIR/i18n.sh"
    source "$SCRIPT_DIR/check_conda.sh"
    check_conda
    setup_env
fi
