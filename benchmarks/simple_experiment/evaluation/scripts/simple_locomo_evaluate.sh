#!/usr/bin/env bash
# ===========================================================================
# Simple Experiment — LoCoMo 评估脚本
#
# 分析 .sage/benchmarks/simple_benchmark_memory/locomo/ 下所有适配器结果
#
# 用法:
#   bash benchmarks/simple_experiment/evaluation/scripts/simple_locomo_evaluate.sh
#   bash benchmarks/simple_experiment/evaluation/scripts/simple_locomo_evaluate.sh mem0
# ===========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

cd "$PROJECT_ROOT"

if [ $# -gt 0 ]; then
    echo "▶ 分析指定适配器: $*"
    python -m benchmarks.simple_experiment.evaluation.analysis.round_analyzer \
        --config locomo --path "$@"
else
    echo "▶ 分析所有适配器"
    python -m benchmarks.simple_experiment.evaluation.analysis.round_analyzer \
        --config locomo --all
fi
