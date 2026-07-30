#!/usr/bin/env bash
set -euo pipefail

# From the original MetaMatch repository root.
# This regenerates the RQ1 companion CSV tables and figures from stored scores.

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
OUT="${OUT:-outputs/exp_occidata}"

cd "$PROJECT_ROOT"
SCRIPT_DIR="scripts"
if [ ! -d "$SCRIPT_DIR" ]; then SCRIPT_DIR="code/scripts"; fi
export PYTHONPATH="$PROJECT_ROOT/code/src:${PYTHONPATH:-}"

python "$SCRIPT_DIR/build_rq1_companion_csvs.py" \
  --output-root "$OUT" \
  --out-dir "$OUT/reports/meeting_baselines_vs_metamatch/rq1_effectiveness_efficiency"

python "$SCRIPT_DIR/report_rq1_metaspace_vs_baselines_from_scores.py" \
  --output-root "$OUT" \
  --out-dir "$OUT/reports/meeting_baselines_vs_metamatch/rq1_effectiveness_efficiency"
