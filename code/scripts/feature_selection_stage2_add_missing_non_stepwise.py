#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from feature_selection_stage2_from_greedy_kendall34 import (  # noqa: E402
    evaluate_feature_set,
    load_base_features,
    load_folds,
    make_candidates,
    make_selection_sample,
    methods_catalog,
    redundancy_stats,
    write_html_report,
)

STEPWISE_METHODS = {"forward_stepwise_xgb_inner_valid", "backward_stepwise_xgb_inner_valid"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Append missing non-stepwise stage-2 feature-selection methods without overwriting existing stepwise results."
    )
    p.add_argument("--output-root", type=Path, default=Path("outputs/exp_occidata"))
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-n", type=int, default=120000)
    p.add_argument("--eval-train-sample", type=int, default=0)
    p.add_argument("--k-grid", type=str, default="5,8,10,12,15,20,25,30,34,38")
    p.add_argument("--max-stepwise-k", type=int, default=0)
    p.add_argument("--n-estimators", type=int, default=220)
    p.add_argument("--max-depth", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--base-source-summary", type=Path, default=None)
    p.add_argument("--base-source-method", type=str, default=None)
    p.add_argument("--base-source-hyperparams", type=str, default=None)
    p.add_argument("--base-method-name", type=str, default="base_greedy_pearson_corr_pruned")
    p.add_argument("--base-hyperparams", type=str, default='{"threshold": 0.85}')
    return p.parse_args()


def read_existing(out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_path = out_dir / "feature_selection_stage2_summary.csv"
    pair_path = out_dir / "feature_selection_stage2_pair_id_evaluation.csv"
    fold_path = out_dir / "feature_selection_stage2_fold_evaluation.csv"
    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    pair = pd.read_csv(pair_path) if pair_path.exists() else pd.DataFrame()
    fold = pd.read_csv(fold_path) if fold_path.exists() else pd.DataFrame()
    return summary, pair, fold


def done_keys(summary: pd.DataFrame) -> set[tuple[str, str]]:
    if summary.empty:
        return set()
    return set(zip(summary["method"].astype(str), summary["hyperparams"].astype(str)))


def save_all(out_dir: Path, summary: pd.DataFrame, pair: pd.DataFrame, fold: pd.DataFrame) -> None:
    summary = summary.sort_values("mean_f1_pairs", ascending=False, na_position="last")
    summary.to_csv(out_dir / "feature_selection_stage2_summary.csv", index=False)
    summary.to_csv(out_dir / "feature_selection_stage2_summary_STEPWISE_COMPLETED.csv", index=False)
    pair.to_csv(out_dir / "feature_selection_stage2_pair_id_evaluation.csv", index=False)
    fold.to_csv(out_dir / "feature_selection_stage2_fold_evaluation.csv", index=False)
    write_html_report(summary, methods_catalog(), out_dir / "feature_selection_stage2_report.html")
    write_html_report(summary, methods_catalog(), out_dir / "feature_selection_stage2_report_STEPWISE_COMPLETED.html")


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_df, pair_df, fold_df = read_existing(out_dir)
    done = done_keys(summary_df)
    print(f"existing evaluated configs: {len(done)}", flush=True)

    k_grid = [int(x) for x in args.k_grid.split(",") if x.strip()]
    base_features = load_base_features(output_root, args)
    print(f"base features: {len(base_features)}", flush=True)

    folds = load_folds(output_root)
    sample_df = make_selection_sample(folds, base_features, args.sample_n, args.seed)
    candidates = [
        c for c in make_candidates(base_features, sample_df, args, k_grid)
        if c.method not in STEPWISE_METHODS and c.features
    ]
    missing = []
    for c in candidates:
        hp = json.dumps(c.hyperparams, sort_keys=True)
        if (c.method, hp) not in done:
            missing.append(c)

    print(f"non-stepwise candidates total: {len(candidates)}", flush=True)
    print(f"missing to evaluate: {len(missing)}", flush=True)

    for i, c in enumerate(missing, start=1):
        hp = json.dumps(c.hyperparams, sort_keys=True)
        print(f"[{i}/{len(missing)}] {c.method} {hp} k={len(c.features)}", flush=True)
        summary, pair_rows, fold_rows = evaluate_feature_set(c.features, folds, args, model_kind="xgb")
        summary.update(
            {
                "method": c.method,
                "hyperparams": hp,
                "note": c.note or "Added by missing non-stepwise completion script.",
                **redundancy_stats(c.features, sample_df),
            }
        )
        summary_df = pd.concat([summary_df, pd.DataFrame([summary])], ignore_index=True)
        pair_rows["method"] = c.method
        pair_rows["hyperparams"] = hp
        fold_rows["method"] = c.method
        fold_rows["hyperparams"] = hp
        pair_df = pd.concat([pair_df, pair_rows], ignore_index=True)
        fold_df = pd.concat([fold_df, fold_rows], ignore_index=True)
        save_all(out_dir, summary_df, pair_df, fold_df)
        print("checkpoint saved", flush=True)

    save_all(out_dir, summary_df, pair_df, fold_df)
    print(f"done: {out_dir / 'feature_selection_stage2_report_STEPWISE_COMPLETED.html'}")


if __name__ == "__main__":
    main()
