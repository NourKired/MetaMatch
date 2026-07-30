from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd

from .io_utils import safe_write_table
from .modeling import compute_metrics


def save_scored_predictions(pred_df: pd.DataFrame, out_dir: Path, method_name: str, runtime_sec: float) -> Dict[str, float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_write_table(pred_df, out_dir / "predictions.parquet")

    metrics = compute_metrics(pred_df, score_col="score")
    payload = {
        "method": method_name,
        "runtime_sec": runtime_sec,
        "metrics": metrics,
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return metrics
