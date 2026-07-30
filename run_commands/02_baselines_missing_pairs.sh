#!/usr/bin/env bash
set -euo pipefail

# Recompute only missing baseline table pairs and store runtime.
# External baselines require their own package/model paths.

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
OUT="${OUT:-outputs/exp_occidata/reports/meeting_baselines_vs_metamatch/baseline_missing_pairs_completion}"

cd "$PROJECT_ROOT"
mkdir -p "$OUT"
SCRIPT_DIR="scripts"
if [ ! -d "$SCRIPT_DIR" ]; then SCRIPT_DIR="code/scripts"; fi
export PYTHONPATH="$PROJECT_ROOT/code/src:${PYTHONPATH:-}"

python "$SCRIPT_DIR/run_missing_baseline_pairs.py" \
  --source-output-root outputs/exp_occidata \
  --out-root "$OUT" \
  --baseline-workers 8 \
  --threshold 0.0 \
  2>&1 | tee "$OUT/run.log"
