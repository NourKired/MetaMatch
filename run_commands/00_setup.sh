#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3 -m pip install --upgrade poetry
poetry install

poetry run python -m ipykernel install --user --name metamatch --display-name "MetaMatch"
