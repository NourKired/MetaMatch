from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import yaml

DEFAULT_VALENTINE_BASELINES = [
    "coma_schema",
    "coma_instance",
    "cupid",
    "similarity_flooding",
    "distribution_based",
]


# Mapping user-facing names to external adapters.
DEFAULT_EXTERNAL_BASELINES = [
    "ISResMat",
    "LLMATCH",
    "Magneto_ft_gpt",
    "Magneto_ft_no_gpt",
    "Magneto_no_ft_gpt",
    "Magneto_no_ft_no_gpt",
    "coma",
    "coma_pp",
    "coma_inst",
    "cupid_ext",
    "similarity_flooding_ext",
    "distribution_based_ext",
    "SMUTF",
]


def load_baseline_config(path: Path | None) -> Dict:
    if path is None or not path.exists():
        return {"external": {}}

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        return {"external": {}}

    if "external" not in data or not isinstance(data["external"], dict):
        data["external"] = {}

    return data
