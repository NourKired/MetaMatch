from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .io_utils import safe_write_table

NON_FEATURE_COLUMNS = {
    "pair_id",
    "dataset",
    "relation_type",
    "category",
    "source_table",
    "target_table",
    "source_column",
    "target_column",
    "source_col_norm",
    "target_col_norm",
    "label",
    "feature_build_sec_pair_real",
}


@dataclass
class TrainResult:
    model_name: str
    fit_seconds: float
    predict_seconds: float
    metrics: Dict[str, float]
    predictions: pd.DataFrame


def feature_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    return [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]


def build_model(model_name: str, seed: int):
    """Legacy MetaMatch builder: RobustScaler + default hyperparameters."""
    if model_name == "logreg":
        return make_pipeline(
            RobustScaler(),
            LogisticRegression(random_state=seed, n_jobs=1),
        )
    if model_name == "rf":
        return make_pipeline(
            RobustScaler(),
            RandomForestClassifier(random_state=seed, n_jobs=1),
        )
    raise ValueError(f"Unknown model: {model_name}")


def _greedy_one_to_one(df: pd.DataFrame, score_col: str = "score") -> pd.Series:
    selected = pd.Series(0, index=df.index, dtype=int)
    for _, block in df.groupby("pair_id", sort=False):
        used_src = set()
        used_tgt = set()
        ordered = block.sort_values(score_col, ascending=False)
        for idx, row in ordered.iterrows():
            s = row["source_col_norm"]
            t = row["target_col_norm"]
            if s in used_src or t in used_tgt:
                continue
            used_src.add(s)
            used_tgt.add(t)
            selected.at[idx] = 1
    return selected


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def compute_metrics(df: pd.DataFrame, score_col: str = "score") -> Dict[str, float]:
    y_true = df["label"].to_numpy()
    y_score = df[score_col].to_numpy()

    threshold_pred = (y_score >= 0.5).astype(int)
    one_to_one_pred = _greedy_one_to_one(df, score_col=score_col).to_numpy()

    metrics = {
        "roc_auc": _safe_auc(y_true, y_score),
        "pr_auc": float(average_precision_score(y_true, y_score)) if len(y_true) else float("nan"),
        "precision_at_05": float(precision_score(y_true, threshold_pred, zero_division=0)),
        "recall_at_05": float(recall_score(y_true, threshold_pred, zero_division=0)),
        "f1_at_05": float(f1_score(y_true, threshold_pred, zero_division=0)),
        "precision_1to1": float(precision_score(y_true, one_to_one_pred, zero_division=0)),
        "recall_1to1": float(recall_score(y_true, one_to_one_pred, zero_division=0)),
        "f1_1to1": float(f1_score(y_true, one_to_one_pred, zero_division=0)),
        "n_rows": float(len(df)),
        "n_positive": float(df["label"].sum()),
    }
    return metrics


def train_and_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str = "rf",
    seed: int = 42,
) -> TrainResult:
    fcols = feature_columns(train_df)
    x_train = train_df[fcols].to_numpy()
    y_train = train_df["label"].to_numpy()
    x_test = test_df[fcols].to_numpy()

    model = build_model(model_name, seed=seed)

    t0 = time.perf_counter()
    model.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - t0

    t1 = time.perf_counter()
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(x_test)[:, 1]
    else:
        scores = model.decision_function(x_test)
    predict_seconds = time.perf_counter() - t1

    pred_df = test_df.copy()
    pred_df["score"] = scores

    metrics = compute_metrics(pred_df, score_col="score")

    return TrainResult(
        model_name=model_name,
        fit_seconds=fit_seconds,
        predict_seconds=predict_seconds,
        metrics=metrics,
        predictions=pred_df,
    )


def save_train_result(result: TrainResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_write_table(result.predictions, out_dir / "predictions.parquet")

    payload = {
        "model_name": result.model_name,
        "fit_seconds": result.fit_seconds,
        "predict_seconds": result.predict_seconds,
        "metrics": result.metrics,
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
