#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .

python -m ipykernel install --user --name metamatch --display-name "MetaMatch"
