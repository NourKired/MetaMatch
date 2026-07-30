#!/usr/bin/env bash
set -euo pipefail

# Preliminary feature-selection benchmark used to justify the final compact configuration.
# It evaluates correlation pruning, mRMR/MI, RF/XGBoost importance, SHAP, forward/backward variants.

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
OUT="${OUT:-outputs/exp_occidata/reports/meeting_baselines_vs_metamatch/feature_selection_benchmark_methods_and_combinations_STAGE2_PROTOCOL}"

cd "$PROJECT_ROOT"
mkdir -p "$OUT"
SCRIPT_DIR="scripts"
if [ ! -d "$SCRIPT_DIR" ]; then SCRIPT_DIR="code/scripts"; fi
export PYTHONPATH="$PROJECT_ROOT/code/src:${PYTHONPATH:-}"

python "$SCRIPT_DIR/recompute_stage1_benchmark_with_stage2_protocol.py" \
  --output-root outputs/exp_occidata \
  --eval-train-sample 0 \
  --n-estimators 220 \
  --max-depth 4 \
  --learning-rate 0.05

python "$SCRIPT_DIR/feature_selection_stage2_add_missing_non_stepwise.py" \
  --output-root outputs/exp_occidata \
  --out-dir "$OUT"
