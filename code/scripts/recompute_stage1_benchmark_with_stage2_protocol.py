#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
sys.path.insert(0, str(ROOT))

from scripts.feature_selection_stage2_from_greedy_kendall34 import (  # noqa: E402
    evaluate_feature_set,
    family_counts,
    load_folds,
    methods_catalog,
    redundancy_stats,
    write_html_report,
)


REPORT_ROOT = Path("reports") / "meeting_baselines_vs_metamatch"
STAGE1_DIR = REPORT_ROOT / "feature_selection_benchmark_methods_and_combinations"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Recompute the first feature-selection benchmark with the exact stage-2 F1 protocol."
    )
    p.add_argument("--output-root", type=Path, default=Path("outputs/exp_occidata"))
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-train-sample", type=int, default=0)
    p.add_argument("--n-estimators", type=int, default=220)
    p.add_argument("--max-depth", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--limit", type=int, default=0, help="Debug only: evaluate first N configs.")
    return p.parse_args()


def split_features(value: object) -> list[str]:
    if value is None or pd.isna(value):
        return []
    return [x.strip() for x in str(value).split("|") if x.strip()]


def load_full_features(output_root: Path) -> list[str]:
    p = output_root / REPORT_ROOT / "feature_spearman_corr_58.json"
    obj = json.loads(p.read_text())
    return list(obj["features"])


def load_stage1_candidates(output_root: Path) -> pd.DataFrame:
    p = output_root / STAGE1_DIR / "candidate_feature_sets.csv"
    df = pd.read_csv(p)
    df = df[["method", "hyperparams", "n_selected", "selected"]].copy()
    df.insert(0, "source_protocol", "stage1_first_benchmark")
    return df


def add_full_reference(candidates: pd.DataFrame, full_features: list[str]) -> pd.DataFrame:
    row = {
        "source_protocol": "stage1_reference_full",
        "method": "Base FULL MetaSpace",
        "hyperparams": json.dumps({"n_features": len(full_features)}, sort_keys=True),
        "n_selected": len(full_features),
        "selected": " | ".join(full_features),
    }
    return pd.concat([pd.DataFrame([row]), candidates], ignore_index=True)


def existing_keys(summary_path: Path) -> set[tuple[str, str]]:
    if not summary_path.exists():
        return set()
    df = pd.read_csv(summary_path, usecols=["method", "hyperparams"])
    return set(zip(df["method"].astype(str), df["hyperparams"].astype(str)))


def existing_feature_cache(summary_path: Path) -> dict[str, tuple[str, str]]:
    if not summary_path.exists():
        return {}
    df = pd.read_csv(summary_path, usecols=["method", "hyperparams", "selected"])
    cache: dict[str, tuple[str, str]] = {}
    for row in df.itertuples(index=False):
        cache.setdefault(str(row.selected), (str(row.method), str(row.hyperparams)))
    return cache


def append_csv(path: Path, df: pd.DataFrame) -> None:
    df.to_csv(path, mode="a", index=False, header=not path.exists())


def clone_cached_result(
    summary_path: Path,
    pair_path: Path,
    fold_path: Path,
    cached_key: tuple[str, str],
    new_source: str,
    new_method: str,
    new_hyperparams: str,
) -> None:
    cached_method, cached_hyperparams = cached_key
    summary_df = pd.read_csv(summary_path)
    mask = summary_df["method"].astype(str).eq(cached_method) & summary_df["hyperparams"].astype(str).eq(cached_hyperparams)
    cached_summary = summary_df.loc[mask].iloc[0].copy()
    cached_summary["source_protocol"] = new_source
    cached_summary["method"] = new_method
    cached_summary["hyperparams"] = new_hyperparams
    cached_summary["note"] = "Copied from identical ordered feature set already recomputed with stage-2 protocol."
    append_csv(summary_path, pd.DataFrame([cached_summary]))

    if pair_path.exists():
        pair_df = pd.read_csv(pair_path)
        mask = pair_df["method"].astype(str).eq(cached_method) & pair_df["hyperparams"].astype(str).eq(cached_hyperparams)
        pair_clone = pair_df.loc[mask].copy()
        pair_clone["source_protocol"] = new_source
        pair_clone["method"] = new_method
        pair_clone["hyperparams"] = new_hyperparams
        append_csv(pair_path, pair_clone)

    if fold_path.exists():
        fold_df = pd.read_csv(fold_path)
        mask = fold_df["method"].astype(str).eq(cached_method) & fold_df["hyperparams"].astype(str).eq(cached_hyperparams)
        fold_clone = fold_df.loc[mask].copy()
        fold_clone["source_protocol"] = new_source
        fold_clone["method"] = new_method
        fold_clone["hyperparams"] = new_hyperparams
        append_csv(fold_path, fold_clone)


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    out_dir = args.out_dir or (
        output_root / REPORT_ROOT / "feature_selection_benchmark_methods_and_combinations_STAGE2_PROTOCOL"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "benchmark_summary_STAGE2_PROTOCOL.csv"
    pair_path = out_dir / "benchmark_pair_id_evaluation_STAGE2_PROTOCOL.csv"
    fold_path = out_dir / "benchmark_fold_evaluation_STAGE2_PROTOCOL.csv"
    html_path = out_dir / "feature_selection_first_benchmark_STAGE2_PROTOCOL.html"
    candidates_path = out_dir / "candidate_feature_sets_STAGE2_PROTOCOL.csv"

    candidates = add_full_reference(load_stage1_candidates(output_root), load_full_features(output_root))
    if args.limit:
        candidates = candidates.head(args.limit).copy()
    candidates.to_csv(candidates_path, index=False)

    folds = load_folds(output_root)
    full_features = load_full_features(output_root)
    sample_df = pd.concat(
        [v["train"][full_features + ["label", "pair_id"]] for v in folds.values()],
        ignore_index=True,
    )
    eval_args = SimpleNamespace(
        seed=args.seed,
        eval_train_sample=args.eval_train_sample,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
    )

    done = existing_keys(summary_path)
    feature_cache = existing_feature_cache(summary_path)
    total = len(candidates)
    for idx, row in candidates.iterrows():
        method = str(row["method"])
        hyperparams = str(row["hyperparams"])
        if (method, hyperparams) in done:
            print(f"[{idx + 1}/{total}] skip {method} {hyperparams}", flush=True)
            continue

        features = split_features(row["selected"])
        if not features:
            print(f"[{idx + 1}/{total}] empty {method} {hyperparams}", flush=True)
            continue
        feature_key = " | ".join(features)

        if feature_key in feature_cache:
            print(f"[{idx + 1}/{total}] copy cached {method} {hyperparams} k={len(features)}", flush=True)
            clone_cached_result(
                summary_path,
                pair_path,
                fold_path,
                feature_cache[feature_key],
                str(row["source_protocol"]),
                method,
                hyperparams,
            )
            done.add((method, hyperparams))
            current = pd.read_csv(summary_path).sort_values("mean_f1_pairs", ascending=False)
            write_html_report(current, methods_catalog(), html_path)
            continue

        print(f"[{idx + 1}/{total}] eval {method} {hyperparams} k={len(features)}", flush=True)
        summary, pair_df, fold_df = evaluate_feature_set(features, folds, eval_args, model_kind="xgb")
        summary.update(
            {
                "source_protocol": row["source_protocol"],
                "method": method,
                "hyperparams": hyperparams,
                "note": "Recomputed with stage-2 protocol: full train, XGBoost 220 trees, threshold_opt_train, F1 averaged by pair_id.",
                **redundancy_stats(features, sample_df),
            }
        )
        pair_df["source_protocol"] = row["source_protocol"]
        pair_df["method"] = method
        pair_df["hyperparams"] = hyperparams
        fold_df["source_protocol"] = row["source_protocol"]
        fold_df["method"] = method
        fold_df["hyperparams"] = hyperparams

        append_csv(summary_path, pd.DataFrame([summary]))
        append_csv(pair_path, pair_df)
        append_csv(fold_path, fold_df)

        current = pd.read_csv(summary_path).sort_values("mean_f1_pairs", ascending=False)
        write_html_report(current, methods_catalog(), html_path)
        done.add((method, hyperparams))
        feature_cache[feature_key] = (method, hyperparams)

    final = pd.read_csv(summary_path).sort_values("mean_f1_pairs", ascending=False)
    final.to_csv(summary_path, index=False)
    write_html_report(final, methods_catalog(), html_path)
    print(f"Done: {html_path}")


if __name__ == "__main__":
    main()
