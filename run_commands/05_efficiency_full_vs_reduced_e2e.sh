#!/usr/bin/env bash
set -euo pipefail

# End-to-end feature construction + train + test timing for full and reduced MetaMatch.
# Use sample-rows=0 for the full original table content if memory/time permit.

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_ROOT"
SCRIPT_DIR="scripts"
if [ ! -d "$SCRIPT_DIR" ]; then SCRIPT_DIR="code/scripts"; fi
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

for CONFIG in full60 reduced10; do
  OUT="outputs/exp_occidata/reports/meeting_baselines_vs_metamatch/e2e_${CONFIG}"
  mkdir -p "$OUT"
  python "$SCRIPT_DIR/run_metamatch_full_or_reduced_e2e.py" \
    --config "$CONFIG" \
    --dataset-root valentine \
    --out-dir "$OUT" \
    --sample-rows 5000 \
    --feature-workers 8 \
    --folds 6 \
    --seed 42 \
    --n-estimators 350 \
    --n-jobs -1 \
    2>&1 | tee "$OUT/run.log"
done
