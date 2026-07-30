#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score

try:
    from scipy.stats import wilcoxon
except Exception:  # pragma: no cover
    wilcoxon = None

try:
    import plotly.express as px
except Exception:  # pragma: no cover
    px = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from metamatch.io_utils import safe_read_table, safe_write_table  # noqa: E402


KEY_COLS = ["pair_id", "source_col_norm", "target_col_norm"]

METHOD_LABELS = {
    "meta_space_xgboost": "MetaMatch XGB",
    "meta_space_rf": "MetaMatch RF",
    "meta_space_logreg": "MetaMatch Reg",
    "meta_space_svm": "MetaMatch SVM",
    "meta_space": "MetaMatch",
    "Magneto_ft_gpt": "Mag FT+GPT",
    "Magneto_ft_no_gpt": "Mag FT",
    "Magneto_no_ft_gpt": "Mag GPT",
    "Magneto_no_ft_no_gpt": "Mag Base",
    "similarity_flooding": "SimFlood",
    "similarity_flooding_ext": "SimFlood Ext",
    "distribution_based": "DistBased",
    "distribution_based_ext": "DistBased Ext",
    "coma_instance": "ComaInst",
    "coma_schema": "ComaSchema",
    "coma_pp": "ComaPP",
    "cupid_ext": "Cupid Ext",
    "LLMATCH": "LLMatch",
    "SMUTF": "SMUTF",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RQ1 report from stored scores: MetaMatch classifiers vs baselines.")
    p.add_argument("--output-root", type=Path, default=Path("outputs/exp_occidata"))
    p.add_argument("--scores-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--score-files",
        type=str,
        default="baseline_scores_only.parquet,metamatch_xgboost_60_scores_only.parquet",
    )
    p.add_argument("--primary-metric", choices=["f1_opt_train", "f1_ground_size"], default="f1_ground_size")
    return p.parse_args()


def read_existing(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return safe_read_table(path)
    for alt in (path.with_suffix(".csv.gz"), path.with_suffix(".csv")):
        if alt.exists():
            return safe_read_table(alt)
    return None


def load_labels(output_root: Path) -> pd.DataFrame:
    chunks: List[pd.DataFrame] = []
    for fold_dir in sorted((output_root / "folds").glob("fold_*"), key=lambda p: int(p.name.split("_")[-1])):
        fold_id = int(fold_dir.name.split("_")[-1])
        test = None
        for name in ("test.parquet", "test.csv.gz", "test.csv"):
            p = fold_dir / name
            if p.exists():
                test = safe_read_table(p)
                break
        if test is None:
            raise FileNotFoundError(f"Missing test file in {fold_dir}")
        out = test[["pair_id", "source_col_norm", "target_col_norm", "label"]].copy()
        out.insert(0, "fold_id", fold_id)
        chunks.append(out)
    labels = pd.concat(chunks, ignore_index=True)
    labels["pair_id"] = labels["pair_id"].astype(str)
    labels["source_col_norm"] = labels["source_col_norm"].astype(str).str.strip().str.lower()
    labels["target_col_norm"] = labels["target_col_norm"].astype(str).str.strip().str.lower()
    labels["label"] = labels["label"].astype(int)
    return labels


def normalize_scores(df: pd.DataFrame, source_file: str) -> pd.DataFrame:
    required = ["run_id", "fold_id", "method", "score"] + KEY_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"{source_file}: missing columns {missing}")
    out = df.copy()
    out["run_id"] = out["run_id"].astype(str)
    out["fold_id"] = out["fold_id"].astype(int)
    out["method"] = out["method"].astype(str)
    out["pair_id"] = out["pair_id"].astype(str)
    out["source_col_norm"] = out["source_col_norm"].astype(str).str.strip().str.lower()
    out["target_col_norm"] = out["target_col_norm"].astype(str).str.strip().str.lower()
    out["score"] = pd.to_numeric(out["score"], errors="coerce").fillna(0.0)
    out["score_source_file"] = source_file
    return out


def load_scores(args: argparse.Namespace) -> pd.DataFrame:
    chunks: List[pd.DataFrame] = []
    for name in [x.strip() for x in args.score_files.split(",") if x.strip()]:
        df = read_existing(args.scores_dir / name)
        if df is None:
            print(f"skip missing: {args.scores_dir / name}", flush=True)
            continue
        chunks.append(normalize_scores(df, name))
    if not chunks:
        raise RuntimeError("No score files found.")
    return pd.concat(chunks, ignore_index=True)


def best_threshold(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float, float, float]:
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return 0.5, 0.0, 0.0, 0.0
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if thresholds.size == 0:
        return 0.5, 0.0, 0.0, 0.0
    p = precision[1:]
    r = recall[1:]
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    idx = int(np.nanargmax(f1))
    return float(thresholds[idx]), float(p[idx]), float(r[idx]), float(f1[idx])


def pred_ground_size(df: pd.DataFrame) -> pd.Series:
    pred = pd.Series(0, index=df.index, dtype=int)
    for _, block in df.groupby("pair_id", sort=False):
        k = int(block["label"].sum())
        if k <= 0:
            continue
        pred.loc[block.sort_values("score", ascending=False, kind="mergesort").head(k).index] = 1
    return pred


def metrics(y: np.ndarray, pred: np.ndarray, prefix: str) -> Dict[str, float]:
    return {
        f"precision_{prefix}": float(precision_score(y, pred, zero_division=0)),
        f"recall_{prefix}": float(recall_score(y, pred, zero_division=0)),
        f"f1_{prefix}": float(f1_score(y, pred, zero_division=0)),
    }


def sign_test_pvalue(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return float("nan")
    k = min(wins, losses)
    p = 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, float(p))


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    labels = load_labels(args.output_root)
    scores = load_scores(args)
    df = scores.merge(labels, on=["fold_id"] + KEY_COLS, how="inner")
    if df.empty:
        raise RuntimeError("No score rows matched labels.")

    fold_rows: List[Dict] = []
    pair_rows: List[Dict] = []

    for (run_id, method), g_method in df.groupby(["run_id", "method"], sort=False):
        folds = sorted(g_method["fold_id"].unique().tolist())
        for fold_id in folds:
            train = g_method[g_method["fold_id"] != fold_id]
            test = g_method[g_method["fold_id"] == fold_id].copy()
            threshold, train_p, train_r, train_f1 = best_threshold(
                train["label"].to_numpy(dtype=int),
                train["score"].to_numpy(dtype=float),
            )
            test["pred_opt_train"] = (test["score"] >= threshold).astype(int)
            test["pred_ground_size"] = pred_ground_size(test)

            y = test["label"].to_numpy(dtype=int)
            row = {
                "run_id": run_id,
                "fold_id": int(fold_id),
                "method": method,
                "method_label": METHOD_LABELS.get(method, method),
                "threshold_opt_train": threshold,
                "train_precision_opt_train": train_p,
                "train_recall_opt_train": train_r,
                "train_f1_opt_train": train_f1,
                "n_rows": int(len(test)),
                "n_pair_id": int(test["pair_id"].nunique()),
            }
            row.update(metrics(y, test["pred_opt_train"].to_numpy(dtype=int), "opt_train"))
            row.update(metrics(y, test["pred_ground_size"].to_numpy(dtype=int), "ground_size"))
            fold_rows.append(row)

            for pair_id, block in test.groupby("pair_id", sort=False):
                yy = block["label"].to_numpy(dtype=int)
                prow = {
                    "run_id": run_id,
                    "fold_id": int(fold_id),
                    "pair_id": str(pair_id),
                    "method": method,
                    "method_label": METHOD_LABELS.get(method, method),
                    "threshold_opt_train": threshold,
                    "n_rows": int(len(block)),
                    "n_positive": int(block["label"].sum()),
                }
                prow.update(metrics(yy, block["pred_opt_train"].to_numpy(dtype=int), "opt_train"))
                prow.update(metrics(yy, block["pred_ground_size"].to_numpy(dtype=int), "ground_size"))
                pair_rows.append(prow)

    fold_df = pd.DataFrame(fold_rows)
    pair_df = pd.DataFrame(pair_rows)

    summary = (
        pair_df.groupby(["method", "method_label"], as_index=False)
        .agg(
            mean_f1_opt_train=("f1_opt_train", "mean"),
            std_f1_opt_train=("f1_opt_train", "std"),
            mean_precision_opt_train=("precision_opt_train", "mean"),
            mean_recall_opt_train=("recall_opt_train", "mean"),
            mean_f1_ground_size=("f1_ground_size", "mean"),
            std_f1_ground_size=("f1_ground_size", "std"),
            mean_precision_ground_size=("precision_ground_size", "mean"),
            mean_recall_ground_size=("recall_ground_size", "mean"),
            n_pair_id=("pair_id", "nunique"),
            n_runs=("run_id", "nunique"),
        )
        .sort_values(f"mean_{args.primary_metric}", ascending=False)
    )

    meta = summary[summary["method"].str.startswith("meta_space_")].copy()
    if meta.empty:
        meta = summary[summary["method"].eq("meta_space")].copy()
    best_method = str(meta.iloc[0]["method"]) if not meta.empty else str(summary.iloc[0]["method"])
    best_label = METHOD_LABELS.get(best_method, best_method)

    stats_rows: List[Dict] = []
    base_metric = args.primary_metric
    best_raw = pair_df[pair_df["method"].eq(best_method)][["run_id", "pair_id", base_metric]].rename(
        columns={base_metric: "best_metric"}
    )
    for method in sorted(pair_df["method"].unique().tolist()):
        if method == best_method:
            continue
        other_raw = pair_df[pair_df["method"].eq(method)][["run_id", "pair_id", base_metric]].rename(
            columns={base_metric: "other_metric"}
        )
        z = best_raw.merge(other_raw, on=["run_id", "pair_id"], how="inner")
        paired_on = "run_id+pair_id"
        if z.empty:
            best = best_raw.groupby("pair_id", as_index=False)["best_metric"].mean()
            other = other_raw.groupby("pair_id", as_index=False)["other_metric"].mean()
            z = best.merge(other, on="pair_id", how="inner")
            paired_on = "pair_id_mean_over_runs"
        if z.empty:
            continue
        delta = z["best_metric"] - z["other_metric"]
        wins = int((delta > 0).sum())
        ties = int((delta == 0).sum())
        losses = int((delta < 0).sum())
        p_wilcoxon = float("nan")
        if wilcoxon is not None and int((delta != 0).sum()) > 0:
            try:
                p_wilcoxon = float(wilcoxon(delta[delta != 0]).pvalue)
            except Exception:
                p_wilcoxon = float("nan")
        stats_rows.append(
            {
                "best_method": best_method,
                "best_label": best_label,
                "baseline_method": method,
                "baseline_label": METHOD_LABELS.get(method, method),
                "metric": base_metric,
                "paired_on": paired_on,
                "n_pairs_runs": int(len(z)),
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "winrate_half_ties": float((wins + 0.5 * ties) / len(z)),
                "mean_delta": float(delta.mean()),
                "median_delta": float(delta.median()),
                "wilcoxon_pvalue": p_wilcoxon,
                "sign_test_pvalue": sign_test_pvalue(wins, losses),
            }
        )
    stats_df = pd.DataFrame(stats_rows).sort_values("winrate_half_ties", ascending=False)

    saved = [
        safe_write_table(summary, args.out_dir / "rq1_summary_by_method.csv"),
        safe_write_table(fold_df, args.out_dir / "rq1_fold_metrics.csv"),
        safe_write_table(pair_df, args.out_dir / "rq1_pair_metrics.csv"),
        safe_write_table(stats_df, args.out_dir / "rq1_winrate_stats_vs_best_metamatch.csv"),
    ]

    if px is not None:
        fig = px.box(
            pair_df,
            x="method_label",
            y=base_metric,
            points="outliers",
            color="method_label",
            title=f"RQ1 {base_metric}: {best_label} vs baselines",
        )
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title=base_metric, height=720)
        html_path = args.out_dir / "rq1_boxplot.html"
        fig.write_html(html_path, include_plotlyjs="cdn")
        saved.append(html_path)

    for path in saved:
        print(f"saved: {path}")
    print("\nTop methods:")
    print(summary.head(20).to_string(index=False))
    print(f"\nBest MetaMatch classifier for {base_metric}: {best_label} ({best_method})")


if __name__ == "__main__":
    main()
