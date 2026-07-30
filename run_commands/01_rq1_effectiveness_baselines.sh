#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
OUT="${OUT:-outputs/exp_occidata}"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

python scripts/build_rq1_results.py \
  --output-root "$OUT" \
  --out-dir "$OUT/reports/meeting_baselines_vs_metamatch/rq1_effectiveness_efficiency"

python scripts/report_rq1_metamatch_vs_baselines_from_scores.py \
  --output-root "$OUT" \
  --out-dir "$OUT/reports/meeting_baselines_vs_metamatch/rq1_effectiveness_efficiency"
