#!/usr/bin/env bash
set -euo pipefail

# Re-run the paper notebooks after the CSV results have been generated.

cd "$(dirname "$0")/.."
source .venv/bin/activate

jupyter nbconvert --to notebook --execute notebooks/rq1_effectiveness_efficiency_baselines_metaspace.ipynb \
  --output rq1_effectiveness_efficiency_baselines_metaspace.executed.ipynb

jupyter nbconvert --to notebook --execute notebooks/feature_selection_metaspace_overview.ipynb \
  --output feature_selection_metaspace_overview.executed.ipynb
