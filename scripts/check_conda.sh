#!/usr/bin/env bash
# check_conda.sh — Step 1: verify conda is available (required)
# Exports: CONDA_BIN
# Exits with error if conda is not found.

set -euo pipefail

if [[ -z "${MSG_CONDA_OK:-}" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    source "$SCRIPT_DIR/i18n.sh"
fi

check_conda() {
    if ! command -v conda &>/dev/null; then
        echo "[ERROR] $(printf "$MSG_CONDA_MISSING")"
        return 1
    fi

    CONDA_BIN="$(command -v conda)"
    local conda_version
    conda_version="$(conda --version 2>&1)"
    echo "[OK]    $(printf "$MSG_CONDA_OK" "$conda_version")"

    export CONDA_BIN
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    : "${LANG_CODE:=en}"
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    source "$SCRIPT_DIR/i18n.sh"
    check_conda
fi
