#!/usr/bin/env bash
# validate.sh — Step 4: run sage_pipeline.py inside the target conda env
# Expects: ENV_NAME (or SETUP_ENV_NAME), BENCH_ROOT, INSTALL_LOG

set -euo pipefail

if [[ -z "${MSG_VALIDATING:-}" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    source "$SCRIPT_DIR/i18n.sh"
fi

validate_install() {
    local env_name="${SETUP_ENV_NAME:-${ENV_NAME:-neuromem}}"
    local bench_root="${BENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
    local script="$bench_root/test/installation_validation/sage_pipeline.py"
    local log_file="${INSTALL_LOG:-$bench_root/.sage/installation/validate_$(date +%Y%m%d_%H%M%S).log}"

    echo "[INFO]  $MSG_VALIDATING" | tee -a "$log_file"
    echo "--- validation: $(date) ---" >> "$log_file"

    if conda run --no-capture-output -n "$env_name" python "$script" 2>&1 | tee -a "$log_file"; then
        echo "[OK]    $MSG_VALIDATE_OK" | tee -a "$log_file"
        echo "" | tee -a "$log_file"
        echo "[INFO]  $(printf "$MSG_LOG_SAVED" "$log_file")"
    else
        echo "[ERROR] $MSG_VALIDATE_FAIL" | tee -a "$log_file"
        printf "        $MSG_ACTIVATE_HINT\n" "$env_name" | tee -a "$log_file"
        echo "[INFO]  $(printf "$MSG_LOG_SAVED" "$log_file")"
        return 1
    fi
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    : "${LANG_CODE:=en}"
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    source "$SCRIPT_DIR/i18n.sh"
    validate_install
fi
