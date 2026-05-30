#!/usr/bin/env bash
# setup_env.sh — Step 2+3: create conda env (Python 3.11) and pip install -e .
# Expects: ENV_NAME, BENCH_ROOT (from quickstart.sh or caller)

set -euo pipefail

if [[ -z "${MSG_ENV_CREATING:-}" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    source "$SCRIPT_DIR/i18n.sh"
fi

_install_runtime_package() {
    local env_name="$1"
    local log_file="$2"
    local local_repo="$3"
    local pip_spec="$4"
    local label="$5"

    if [[ -f "$local_repo/pyproject.toml" ]]; then
        echo "[INFO]  Installing $label from local repo: $local_repo" | tee -a "$log_file"
        conda run --no-capture-output -n "$env_name" \
            pip install -e "$local_repo" 2>&1 | tee -a "$log_file"
    else
        echo "[INFO]  Installing $label from package index: $pip_spec" | tee -a "$log_file"
        conda run --no-capture-output -n "$env_name" \
            pip install "$pip_spec" 2>&1 | tee -a "$log_file"
    fi
}

_verify_runtime_imports() {
    local env_name="$1"
    local log_file="$2"

    echo "[INFO]  Verifying Sage/NeuroMem runtime imports" | tee -a "$log_file"
    conda run --no-capture-output -n "$env_name" python - <<'PY' 2>&1 | tee -a "$log_file"
import importlib

required = ["sage.foundation", "sage.runtime", "sage.neuromem"]
missing = []
for module_name in required:
    try:
        importlib.import_module(module_name)
        print(f"[OK]    import {module_name}")
    except Exception as exc:
        missing.append((module_name, f"{type(exc).__name__}: {exc}"))

if missing:
    for module_name, reason in missing:
        print(f"[ERROR] missing {module_name}: {reason}")
    raise SystemExit(1)
PY
}

setup_env() {
    local env_name="${ENV_NAME:-neuromem}"
    local bench_root="${BENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
    local workspace_root="$(cd "$bench_root/.." && pwd)"
    local sage_root="${SAGE_ROOT:-$workspace_root/SAGE}"
    local neuromem_root="${NEUROMEM_ROOT:-$workspace_root/neuromem}"
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

    # ---- install runtime dependencies explicitly so validation won't fail later ----
    _install_runtime_package "$env_name" "$log_file" "$sage_root" "isage" "SAGE runtime"
    _install_runtime_package "$env_name" "$log_file" "$neuromem_root" "isage-neuromem[full]" "NeuroMem"

    # ---- install from pyproject.toml (editable, real-time output + log) ----
    echo "[INFO]  $MSG_INSTALLING" | tee -a "$log_file"
    conda run --no-capture-output -n "$env_name" \
        pip install -e "$bench_root" 2>&1 | tee -a "$log_file"
    echo "[OK]    $MSG_INSTALL_OK" | tee -a "$log_file"

    _verify_runtime_imports "$env_name" "$log_file"

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
