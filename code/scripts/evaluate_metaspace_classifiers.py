#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import LinearSVC

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from osirim_occidata.io_utils import safe_read_table, safe_write_table  # noqa: E402
from osirim_occidata.modeling import feature_columns  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate MetaSpace classifiers with threshold_opt_train per fold.")
    p.add_argument("--output-root", type=Path, default=Path("outputs/exp_occidata"))
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--classifiers", type=str, default="xgboost,rf,logreg,svm")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--feature-list", type=str, default="", help="Optional comma-separated features. Empty = all MetaSpace numeric features.")
    p.add_argument("--feature-set-json", type=Path, default=None, help="Optional JSON list of features.")
    p.add_argument("--feature-families", type=str, default="", help="Optional comma-separated prefixes, e.g. syn,cls,tda.")
    p.add_argument("--exclude-feature-substrings", type=str, default="", help="Optional comma-separated substrings to exclude.")
    p.add_argument("--expected-n-features", type=int, default=0, help="Fail if the selected feature count is different.")
    p.add_argument("--eval-train-sample", type=int, default=0)
    p.add_argument("--save-row-scores", action="store_true", help="Store row-level scores without labels.")
    return p.parse_args()


def read_table(base: Path, stem: str) -> pd.DataFrame:
    for name in (f"{stem}.parquet", f"{stem}.csv.gz", f"{stem}.csv"):
        p = base / name
        if p.exists():
            return safe_read_table(p)
    raise FileNotFoundError(f"{stem} not found in {base}")


def load_features(train_df: pd.DataFrame, args: argparse.Namespace) -> List[str]:
    if args.feature_set_json:
        features = list(json.loads(args.feature_set_json.read_text()))
    elif args.feature_list.strip():
        features = [x.strip() for x in args.feature_list.split(",") if x.strip()]
    else:
        features = feature_columns(train_df)

    if args.feature_families.strip():
        prefixes = tuple(f"{x.strip()}_" for x in args.feature_families.split(",") if x.strip())
        features = [f for f in features if f.startswith(prefixes)]

    if args.exclude_feature_substrings.strip():
        banned = [x.strip().lower() for x in args.exclude_feature_substrings.split(",") if x.strip()]
        features = [f for f in features if not any(b in f.lower() for b in banned)]

    if args.expected_n_features and len(features) != int(args.expected_n_features):
        raise ValueError(
            f"Expected {args.expected_n_features} features, got {len(features)}. "
            f"Selected features: {features}"
        )

    return features


def best_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if thresholds.size == 0:
        return 0.5
    p = precision[1:]
    r = recall[1:]
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    return float(thresholds[int(np.nanargmax(f1))])


def build_classifier(name: str, y_train: np.ndarray, seed: int, args: argparse.Namespace):
    """Build MetaMatch classifiers with RobustScaler for every model.

    The classifiers intentionally keep their library default hyperparameters.
    Only random_state/random_seed and n_jobs/thread_count are set for reproducibility
    and execution control; they do not change the model family itself.
    """
    if name == "xgboost":
        if XGBClassifier is None:
            raise RuntimeError("xgboost is not installed.")
        return make_pipeline(
            RobustScaler(),
            XGBClassifier(random_state=seed, n_jobs=1),
        )
    if name == "rf":
        return make_pipeline(
            RobustScaler(),
            RandomForestClassifier(random_state=seed, n_jobs=1),
        )
    if name == "catboost":
        if CatBoostClassifier is None:
            raise RuntimeError("catboost is not installed. Install it with: pip install catboost")
        return make_pipeline(
            RobustScaler(),
            CatBoostClassifier(random_seed=seed, thread_count=1, verbose=False),
        )
    if name == "logreg":
        return make_pipeline(
            RobustScaler(),
            LogisticRegression(random_state=seed, n_jobs=1),
        )
    if name == "svm":
        return make_pipeline(
            RobustScaler(),
            LinearSVC(random_state=seed),
        )
    raise ValueError(f"Unknown classifier: {name}")


def scores(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(model, "decision_function"):
        s = model.decision_function(x)
        return 1.0 / (1.0 + np.exp(-s))
    return model.predict(x)


def pair_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    for pair_id, g in df.groupby("pair_id", sort=False):
        y = g["label"].to_numpy().astype(int)
        pred = g["pred"].to_numpy().astype(int)
        rows.append(
            {
                "pair_id": pair_id,
                "n_rows": int(len(g)),
                "n_pos": int(y.sum()),
                "precision": float(precision_score(y, pred, zero_division=0)),
                "recall": float(recall_score(y, pred, zero_division=0)),
                "f1": float(f1_score(y, pred, zero_division=0)),
            }
        )
    return pd.DataFrame(rows)


def pred_at_ground_size(df: pd.DataFrame) -> pd.Series:
    pred = pd.Series(0, index=df.index, dtype=int)
    for _, g in df.groupby("pair_id", sort=False):
        k = int(g["label"].sum())
        if k <= 0:
            continue
        selected = g.sort_values("score", ascending=False, kind="mergesort").head(k).index
        pred.loc[selected] = 1
    return pred


def pair_metrics_ground_size(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["pred"] = pred_at_ground_size(tmp)
    return pair_metrics(tmp)


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    out_dir = args.out_dir or (output_root / "reports" / "meeting_baselines_vs_metamatch" / "metaspace_classifier_benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)

    classifiers = [x.strip() for x in args.classifiers.split(",") if x.strip()]
    fold_dirs = sorted((output_root / "folds").glob("fold_*"), key=lambda p: int(p.name.split("_")[-1]))
    if not fold_dirs:
        raise RuntimeError(f"No fold_* found in {output_root / 'folds'}")

    fold_rows: List[Dict] = []
    pair_rows: List[pd.DataFrame] = []
    score_rows: List[pd.DataFrame] = []

    for fold_dir in fold_dirs:
        fid = int(fold_dir.name.split("_")[-1])
        train = read_table(fold_dir, "train")
        test = read_table(fold_dir, "test")
        features = load_features(train, args)
        print(f"fold={fid} features={len(features)}", flush=True)

        fit_train = train
        if args.eval_train_sample and len(fit_train) > args.eval_train_sample:
            fit_train = fit_train.sample(n=args.eval_train_sample, random_state=args.seed + fid)

        x_fit = fit_train[features].to_numpy()
        y_fit = fit_train["label"].to_numpy().astype(int)
        x_thr = train[features].to_numpy()
        y_thr = train["label"].to_numpy().astype(int)
        x_test = test[features].to_numpy()
        y_test = test["label"].to_numpy().astype(int)

        for clf_name in classifiers:
            print(f"[fold {fid}] classifier={clf_name}", flush=True)
            model = build_classifier(clf_name, y_fit, args.seed + fid, args)
            t0 = time.perf_counter()
            model.fit(x_fit, y_fit)
            fit_sec = time.perf_counter() - t0
            train_score = scores(model, x_thr)
            test_score = scores(model, x_test)
            threshold = best_threshold(y_thr, train_score)
            pred = (test_score >= threshold).astype(int)

            fold_rows.append(
                {
                    "fold_id": fid,
                    "classifier": clf_name,
                    "n_features": len(features),
                    "threshold_opt_train": threshold,
                    "fit_seconds": fit_sec,
                    "precision": float(precision_score(y_test, pred, zero_division=0)),
                    "recall": float(recall_score(y_test, pred, zero_division=0)),
                    "f1": float(f1_score(y_test, pred, zero_division=0)),
                    "features": " | ".join(features),
                }
            )
            tmp = test[["pair_id", "label"]].copy()
            tmp["pred"] = pred
            tmp["score"] = test_score
            pm = pair_metrics(tmp)
            pm["fold_id"] = fid
            pm["classifier"] = clf_name
            pm["metric"] = "opt_train_threshold"
            pair_rows.append(pm)

            gm = pair_metrics_ground_size(tmp)
            gm["fold_id"] = fid
            gm["classifier"] = clf_name
            gm["metric"] = "ground_size"
            pair_rows.append(gm)

            if args.save_row_scores:
                id_cols = [
                    c
                    for c in [
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
                    ]
                    if c in test.columns
                ]
                scores_df = test[id_cols].copy()
                scores_df.insert(0, "fold_id", fid)
                scores_df.insert(1, "method", f"meta_space_{clf_name}")
                scores_df.insert(2, "classifier", clf_name)
                scores_df["score"] = test_score
                scores_df["threshold_opt_train"] = threshold
                scores_df["n_features"] = len(features)
                score_rows.append(scores_df)

    fold_df = pd.DataFrame(fold_rows)
    pair_df = pd.concat(pair_rows, ignore_index=True)
    summary = (
        pair_df.groupby(["classifier", "metric"], as_index=False)
        .agg(
            mean_f1_pairs=("f1", "mean"),
            std_f1_pairs=("f1", "std"),
            mean_precision_pairs=("precision", "mean"),
            mean_recall_pairs=("recall", "mean"),
            n_pair_ids_eval=("pair_id", "nunique"),
        )
        .sort_values("mean_f1_pairs", ascending=False)
    )

    fold_df.to_csv(out_dir / "metaspace_classifier_fold_metrics.csv", index=False)
    pair_df.to_csv(out_dir / "metaspace_classifier_pair_metrics.csv", index=False)
    summary.to_csv(out_dir / "metaspace_classifier_summary.csv", index=False)
    if args.save_row_scores and score_rows:
        score_df = pd.concat(score_rows, ignore_index=True)
        saved = safe_write_table(score_df, out_dir / "metaspace_classifier_row_scores.parquet")
        print(f"Saved: {saved}")
    print(f"Saved: {out_dir / 'metaspace_classifier_summary.csv'}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
