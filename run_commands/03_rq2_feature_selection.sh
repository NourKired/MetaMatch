#!/usr/bin/env bash
set -euo pipefail

# Main RQ2/RQ3 experiment: full 60 meta-features, reduced Pearson=0.85 + RF importance k=10,
# and family ablations with RandomForestClassifier(n_estimators=350, class_weight=balanced_subsample).

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
OUT="${OUT:-outputs/exp_occidata}"

cd "$PROJECT_ROOT"
SCRIPT_DIR="scripts"
if [ ! -d "$SCRIPT_DIR" ]; then SCRIPT_DIR="code/scripts"; fi
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

python "$SCRIPT_DIR/run_rq2_rq3_feature_ablation_rf350.py" \
  --output-root "$OUT" \
  --out-dir "$OUT/reports/meeting_baselines_vs_metamatch/rq2_rq3_ablation_feature_selection_rf350" \
  --seed 42 \
  --n-estimators 350

python "$SCRIPT_DIR/plot_rq2_rq3_results.py" \
  --out-dir "$OUT/reports/meeting_baselines_vs_metamatch/rq2_rq3_ablation_feature_selection_rf350"
