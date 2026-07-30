#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", type=Path, default=Path("outputs/exp_occidata"))
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/exp_occidata/reports/meeting_baselines_vs_metamatch/rq2_rq3_ablation_feature_selection_rf350"),
    )
    p.add_argument(
        "--selected-json",
        type=Path,
        default=Path(
            "outputs/exp_occidata/reports/meeting_baselines_vs_metamatch/"
            "feature_selection_final_pearson085_rf_topk10/"
            "selected_features_pearson085_random_forest_importance_k10.json"
        ),
    )
    p.add_argument("--n-estimators", type=int, default=350)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def read_table(base: Path, stem: str) -> pd.DataFrame:
    for name in (f"{stem}.parquet", f"{stem}.csv.gz", f"{stem}.csv"):
        p = base / name
        if p.exists():
            return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    raise FileNotFoundError(f"{stem} not found in {base}")


def all_paper_features(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS and pd.api.types.is_numeric_dtype(df[c])]
    return [
        c
        for c in cols
        if not c.startswith("spc_")
        and not c.startswith("nlp_")
        and "overlap" not in c.lower()
        and c != "cls_cosine_dist"
    ]


def pair_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pair_id, g in df.groupby("pair_id", sort=False):
        y = g["label"].to_numpy(dtype=int)
        pred = g["pred"].to_numpy(dtype=int)
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




def best_threshold_pair_f1_train(train_scored: pd.DataFrame) -> tuple[float, float]:
    y_true = train_scored["label"].to_numpy(dtype=int)
    if len(np.unique(y_true)) < 2:
        return 0.5, 0.0

    thresholds = np.linspace(0.0, 1.0, 1001)
    sum_f1 = np.zeros_like(thresholds, dtype=float)
    n_pairs = 0

    for _, g in train_scored.groupby("pair_id", sort=False):
        scores = g["score"].to_numpy(dtype=float)
        labels = g["label"].to_numpy(dtype=int)
        total_pos = int(labels.sum())
        if total_pos <= 0:
            continue

        order = np.argsort(scores)
        scores_asc = scores[order]
        labels_asc = labels[order]
        prefix_pos = np.r_[0, np.cumsum(labels_asc)]

        # idx = first position with score > threshold. Predictions are scores strictly greater than threshold.
        idx = np.searchsorted(scores_asc, thresholds, side="right")
        n_pred = len(scores_asc) - idx
        tp = total_pos - prefix_pos[idx]
        fp = n_pred - tp
        fn = total_pos - tp
        denom = 2 * tp + fp + fn
        f1 = np.divide(2 * tp, denom, out=np.zeros_like(thresholds, dtype=float), where=denom > 0)
        sum_f1 += f1
        n_pairs += 1

    if n_pairs == 0:
        return 0.5, 0.0
    mean_f1 = sum_f1 / n_pairs
    best_idx = int(np.nanargmax(mean_f1))
    return float(thresholds[best_idx]), float(mean_f1[best_idx])

def evaluate_config(
    name: str,
    features: list[str],
    fold_dirs: list[Path],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows = []
    pair_parts = []

    for fold_dir in fold_dirs:
        fold_id = int(fold_dir.name.split("_")[-1])
        train = read_table(fold_dir, "train")
        test = read_table(fold_dir, "test")

        x_train = train[features].to_numpy(dtype=np.float32)
        y_train = train["label"].to_numpy(dtype=int)
        x_test = test[features].to_numpy(dtype=np.float32)
        y_test = test["label"].to_numpy(dtype=int)

        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            class_weight="balanced_subsample",
            random_state=args.seed + fold_id,
            n_jobs=args.n_jobs,
        )

        t0 = time.perf_counter()
        model.fit(x_train, y_train)
        fit_sec = time.perf_counter() - t0

        t0 = time.perf_counter()
        train_score = model.predict_proba(x_train)[:, 1]
        train_score_sec = time.perf_counter() - t0

        train_scored = train[["pair_id", "label"]].copy()
        train_scored["score"] = train_score
        threshold, train_mean_f1_pair = best_threshold_pair_f1_train(train_scored)

        t0 = time.perf_counter()
        test_score = model.predict_proba(x_test)[:, 1]
        predict_sec = time.perf_counter() - t0

        pred = (test_score > threshold).astype(int)
        fold_rows.append(
            {
                "config": name,
                "fold_id": fold_id,
                "n_features": len(features),
                "threshold_opt_train": threshold,
                "train_mean_f1_pair_at_threshold": train_mean_f1_pair,
                "fit_sec": fit_sec,
                "train_score_sec": train_score_sec,
                "predict_sec": predict_sec,
                "precision": float(precision_score(y_test, pred, zero_division=0)),
                "recall": float(recall_score(y_test, pred, zero_division=0)),
                "f1": float(f1_score(y_test, pred, zero_division=0)),
                "features": " | ".join(features),
            }
        )

        tmp = test[["pair_id", "label"]].copy()
        tmp["pred"] = pred
        pm = pair_metrics(tmp)
        pm.insert(0, "config", name)
        pm.insert(1, "fold_id", fold_id)
        pm["n_features"] = len(features)
        pair_parts.append(pm)

    return pd.DataFrame(fold_rows), pd.concat(pair_parts, ignore_index=True)


def summarize(pair_df: pd.DataFrame, fold_df: pd.DataFrame) -> pd.DataFrame:
    pair_summary = (
        pair_df.groupby("config", as_index=False)
        .agg(
            mean_f1=("f1", "mean"),
            std_f1=("f1", "std"),
            mean_precision=("precision", "mean"),
            std_precision=("precision", "std"),
            mean_recall=("recall", "mean"),
            std_recall=("recall", "std"),
            n_pair_id=("pair_id", "nunique"),
            n_features=("n_features", "first"),
        )
    )
    times = (
        fold_df.groupby("config", as_index=False)
        .agg(
            total_fit_sec=("fit_sec", "sum"),
            total_train_score_sec=("train_score_sec", "sum"),
            total_predict_sec=("predict_sec", "sum"),
            mean_fit_sec_fold=("fit_sec", "mean"),
            mean_predict_sec_fold=("predict_sec", "mean"),
        )
    )
    out = pair_summary.merge(times, on="config", how="left")
    out["total_model_sec"] = out["total_fit_sec"] + out["total_train_score_sec"] + out["total_predict_sec"]
    return out.sort_values(["mean_f1", "n_features"], ascending=[False, True])


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fold_dirs = sorted((args.output_root / "folds").glob("fold_*"), key=lambda p: int(p.name.split("_")[-1]))
    if not fold_dirs:
        raise FileNotFoundError(args.output_root / "folds")

    first_train = read_table(fold_dirs[0], "train")
    full = all_paper_features(first_train)
    syn = [c for c in full if c.startswith("syn_")]
    cls = [c for c in full if c.startswith("cls_")]
    tda = [c for c in full if c.startswith("tda_")]

    if (len(full), len(syn), len(cls), len(tda)) != (60, 22, 7, 31):
        raise RuntimeError(f"Unexpected feature counts: full={len(full)} syn={len(syn)} cls={len(cls)} tda={len(tda)}")

    selected_payload = json.loads(args.selected_json.read_text())
    reduced = list(selected_payload["selected_features"])

    configs = {
        "full60": full,
        "syn_only": syn,
        "distance_only": cls,
        "topological_only": tda,
        "syn_distance": syn + cls,
        "syn_topological": syn + tda,
        "distance_topological": cls + tda,
        "pearson085_rf_importance_k10": reduced,
    }

    manifest_path = args.out_dir / "configs_manifest.json"
    manifest_path.write_text(json.dumps({k: v for k, v in configs.items()}, indent=2), encoding="utf-8")

    all_fold = []
    all_pair = []
    for i, (name, features) in enumerate(configs.items(), start=1):
        fold_path = args.out_dir / f"{name}_fold_metrics.csv"
        pair_path = args.out_dir / f"{name}_pair_metrics.csv"
        if fold_path.exists() and pair_path.exists() and not args.force:
            print(f"[{i}/{len(configs)}] skip {name}", flush=True)
            fold_df = pd.read_csv(fold_path)
            pair_df = pd.read_csv(pair_path)
        else:
            print(f"[{i}/{len(configs)}] eval {name} n_features={len(features)}", flush=True)
            fold_df, pair_df = evaluate_config(name, features, fold_dirs, args)
            fold_df.to_csv(fold_path, index=False)
            pair_df.to_csv(pair_path, index=False)
            print("checkpoint saved", flush=True)
        all_fold.append(fold_df)
        all_pair.append(pair_df)

        summary = summarize(pd.concat(all_pair, ignore_index=True), pd.concat(all_fold, ignore_index=True))
        summary.to_csv(args.out_dir / "rq2_rq3_summary_checkpoint.csv", index=False)

    fold_all = pd.concat(all_fold, ignore_index=True)
    pair_all = pd.concat(all_pair, ignore_index=True)
    summary = summarize(pair_all, fold_all)

    fold_all.to_csv(args.out_dir / "rq2_rq3_fold_metrics.csv", index=False)
    pair_all.to_csv(args.out_dir / "rq2_rq3_pair_metrics.csv", index=False)
    summary.to_csv(args.out_dir / "rq2_rq3_summary.csv", index=False)

    paper = summary.copy()
    for c in ["mean_f1", "std_f1", "mean_precision", "std_precision", "mean_recall", "std_recall"]:
        paper[c] = paper[c].map(lambda x: f"{x:.2f}")
    for c in ["total_fit_sec", "total_predict_sec", "total_model_sec"]:
        paper[c] = paper[c].map(lambda x: f"{x:.2f}")
    paper.to_csv(args.out_dir / "rq2_rq3_table_for_paper.csv", index=False)

    print(summary.to_string(index=False))
    print(f"Saved: {args.out_dir}")


if __name__ == "__main__":
    main()
