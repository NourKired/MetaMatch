#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import List

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from feature_selection_stage2_from_greedy_kendall34 import (  # noqa: E402
    CandidateSet,
    evaluate_feature_set,
    family_quota,
    load_base_features,
    load_folds,
    make_selection_sample,
    methods_catalog,
    rank_shap,
    redundancy_stats,
    write_html_report,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Add only SHAP feature-selection results to a copy of the completed stage-2 report."
    )
    p.add_argument("--output-root", type=Path, default=Path("outputs/exp_occidata"))
    p.add_argument("--source-dir", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-n", type=int, default=120000)
    p.add_argument("--eval-train-sample", type=int, default=0)
    p.add_argument("--k-grid", type=str, default="5,8,10,12,15,20,25,30,34")
    p.add_argument("--n-estimators", type=int, default=220)
    p.add_argument("--max-depth", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--html-only", action="store_true", help="Only refresh CSV/HTML from existing results; do not compute SHAP.")
    p.add_argument("--base-source-summary", type=Path, default=None)
    p.add_argument("--base-source-method", type=str, default=None)
    p.add_argument("--base-source-hyperparams", type=str, default=None)
    return p.parse_args()


def default_source_dir(output_root: Path) -> Path:
    return (
        output_root
        / "reports"
        / "meeting_baselines_vs_metamatch"
        / "feature_selection_stage2_from_greedy_kendall34"
    )


def copy_once(source_dir: Path, out_dir: Path) -> None:
    if out_dir.exists():
        print(f"resume existing output copy: {out_dir}", flush=True)
        return
    print(f"copy source report directory:", flush=True)
    print(f"  from: {source_dir}", flush=True)
    print(f"  to:   {out_dir}", flush=True)
    shutil.copytree(source_dir, out_dir)


def read_existing(out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stepwise_summary = out_dir / "feature_selection_stage2_summary_STEPWISE_COMPLETED.csv"
    summary_path = stepwise_summary if stepwise_summary.exists() else out_dir / "feature_selection_stage2_summary.csv"
    pair_path = out_dir / "feature_selection_stage2_pair_id_evaluation.csv"
    fold_path = out_dir / "feature_selection_stage2_fold_evaluation.csv"
    return pd.read_csv(summary_path), pd.read_csv(pair_path), pd.read_csv(fold_path)


def full_reference_path(output_root: Path) -> Path:
    return (
        output_root
        / "reports"
        / "meeting_baselines_vs_metamatch"
        / "feature_selection_benchmark_methods_and_combinations_STAGE2_PROTOCOL"
        / "benchmark_summary_STAGE2_PROTOCOL.csv"
    )


def add_full_reference_if_available(output_root: Path, summary: pd.DataFrame) -> pd.DataFrame:
    if "method" in summary.columns and summary["method"].astype(str).eq("Base FULL MetaSpace").any():
        return summary
    p = full_reference_path(output_root)
    if not p.exists():
        print(f"FULL MetaSpace reference not found, skipped: {p}", flush=True)
        return summary
    full_df = pd.read_csv(p)
    full_df = full_df[full_df["method"].astype(str).eq("Base FULL MetaSpace")]
    if full_df.empty:
        print(f"FULL MetaSpace reference not found inside: {p}", flush=True)
        return summary
    row = full_df.iloc[[0]].copy()
    row["note"] = "Reference FULL MetaSpace added for comparison; already recomputed with stage-2 protocol."
    for col in summary.columns:
        if col not in row.columns:
            row[col] = pd.NA
    row = row[summary.columns]
    print("added Base FULL MetaSpace reference to summary", flush=True)
    return pd.concat([summary, row], ignore_index=True)


def done_keys(summary: pd.DataFrame) -> set[tuple[str, str]]:
    return set(zip(summary["method"].astype(str), summary["hyperparams"].astype(str)))


def save_all(out_dir: Path, summary: pd.DataFrame, pair: pd.DataFrame, fold: pd.DataFrame) -> None:
    summary = summary.sort_values("mean_f1_pairs", ascending=False, na_position="last")
    summary.to_csv(out_dir / "feature_selection_stage2_summary.csv", index=False)
    summary.to_csv(out_dir / "feature_selection_stage2_summary_STEPWISE_PLUS_SHAP.csv", index=False)
    summary.to_csv(out_dir / "feature_selection_stage2_summary_STEPWISE_COMPLETED.csv", index=False)
    pair.to_csv(out_dir / "feature_selection_stage2_pair_id_evaluation.csv", index=False)
    fold.to_csv(out_dir / "feature_selection_stage2_fold_evaluation.csv", index=False)
    write_html_report(summary, methods_catalog(), out_dir / "feature_selection_stage2_report_STEPWISE_PLUS_SHAP.html")
    write_html_report(summary, methods_catalog(), out_dir / "feature_selection_stage2_report_STEPWISE_COMPLETED.html")


def shap_candidates(sample_df: pd.DataFrame, base_features: list[str], args: argparse.Namespace) -> List[CandidateSet]:
    runtime_args = SimpleNamespace(
        seed=args.seed,
        eval_train_sample=args.eval_train_sample,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
    )
    rank = rank_shap(sample_df, base_features, runtime_args)
    if not rank:
        raise RuntimeError(
            "SHAP was not computed. Check that shap is installed with: "
            ".venv/bin/python3 -c \"import shap; print(shap.__version__)\""
        )
    k_grid = [int(x) for x in args.k_grid.split(",") if x.strip()]
    candidates: List[CandidateSet] = []
    for k in k_grid:
        candidates.append(CandidateSet("shap_xgboost_topk", {"k": int(k)}, rank[: min(int(k), len(rank))]))
    for k in k_grid:
        candidates.append(CandidateSet(f"family_quota_shap_xgboost_topk", {"k": int(k)}, family_quota(rank, int(k))))
    return candidates


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    source_dir = args.source_dir or default_source_dir(output_root)
    out_dir = args.out_dir or source_dir.with_name(source_dir.name + "_STEPWISE_PLUS_SHAP")
    source_dir = source_dir.resolve()
    out_dir = out_dir.resolve()

    copy_once(source_dir, out_dir)
    summary_df, pair_df, fold_df = read_existing(out_dir)
    summary_df = add_full_reference_if_available(output_root, summary_df)
    save_all(out_dir, summary_df, pair_df, fold_df)
    if args.html_only:
        print(f"html refreshed: {out_dir / 'feature_selection_stage2_report_STEPWISE_PLUS_SHAP.html'}")
        return

    done = done_keys(summary_df)
    print(f"existing evaluated configs in output copy: {len(done)}", flush=True)

    base_features = load_base_features(output_root, args)
    folds = load_folds(output_root)
    sample_df = make_selection_sample(folds, base_features, args.sample_n, args.seed)
    runtime_args = SimpleNamespace(
        seed=args.seed,
        eval_train_sample=args.eval_train_sample,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
    )

    candidates = shap_candidates(sample_df, base_features, args)
    missing = []
    for c in candidates:
        hp = json.dumps(c.hyperparams, sort_keys=True)
        if (c.method, hp) not in done:
            missing.append(c)

    print(f"SHAP candidates total: {len(candidates)}", flush=True)
    print(f"missing to evaluate: {len(missing)}", flush=True)

    for i, c in enumerate(missing, start=1):
        hp = json.dumps(c.hyperparams, sort_keys=True)
        print(f"[{i}/{len(missing)}] {c.method} {hp} k={len(c.features)}", flush=True)
        summary, pair_rows, fold_rows = evaluate_feature_set(c.features, folds, runtime_args, model_kind="xgb")
        summary.update(
            {
                "method": c.method,
                "hyperparams": hp,
                "note": "Added by SHAP-only script; source stepwise report was copied first.",
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
    print(f"done: {out_dir / 'feature_selection_stage2_report_STEPWISE_PLUS_SHAP.html'}")


if __name__ == "__main__":
    main()
