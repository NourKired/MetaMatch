#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

from run_rq2_rq3_feature_ablation_rf350 import (
    all_paper_features,
    best_threshold_pair_f1_train,
    pair_metrics,
    read_table,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Repeat MetaMatch RF-350 train/test over fixed folds and fixed 60 meta-features."
    )
    p.add_argument("--output-root", type=Path, default=Path("outputs/exp_occidata"))
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "outputs/exp_occidata/reports/meeting_baselines_vs_metamatch/"
            "metamatch_rf350_full60_repeated_seeds"
        ),
    )
    p.add_argument("--seeds", type=str, default="42,43,44,45,46,47,48,49,50,51")
    p.add_argument("--n-estimators", type=int, default=350)
    p.add_argument("--n-jobs", type=int, default=-1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    fold_dirs = sorted(
        (args.output_root / "folds").glob("fold_*"),
        key=lambda p: int(p.name.split("_")[-1]),
    )
    if not fold_dirs:
        raise FileNotFoundError(args.output_root / "folds")

    first_train = read_table(fold_dirs[0], "train")
    features = all_paper_features(first_train)
    if len(features) != 60:
        raise RuntimeError(f"Expected 60 paper meta-features, got {len(features)}")

    fold_rows = []
    pair_parts = []

    for run_id, seed in enumerate(seeds, start=1):
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
                random_state=seed + fold_id,
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
                    "run_id": run_id,
                    "seed": seed,
                    "fold_id": fold_id,
                    "n_features": len(features),
                    "threshold_opt_train": threshold,
                    "train_mean_f1_pair_at_threshold": train_mean_f1_pair,
                    "fit_sec": fit_sec,
                    "train_score_sec": train_score_sec,
                    "predict_sec": predict_sec,
                    "precision": precision_score(y_test, pred, zero_division=0),
                    "recall": recall_score(y_test, pred, zero_division=0),
                    "f1": f1_score(y_test, pred, zero_division=0),
                }
            )

            tmp = test[["pair_id", "label"]].copy()
            tmp["pred"] = pred
            pm = pair_metrics(tmp)
            pm.insert(0, "run_id", run_id)
            pm.insert(1, "seed", seed)
            pm.insert(2, "fold_id", fold_id)
            pm["n_features"] = len(features)
            pair_parts.append(pm)

            print(
                f"run={run_id}/{len(seeds)} seed={seed} fold={fold_id} "
                f"threshold={threshold:.3f} pair_f1={pm['f1'].mean():.4f} "
                f"fit={fit_sec:.2f}s predict={predict_sec:.2f}s",
                flush=True,
            )

    fold_df = pd.DataFrame(fold_rows)
    pair_df = pd.concat(pair_parts, ignore_index=True)

    by_seed = (
        pair_df.groupby(["run_id", "seed"], as_index=False)
        .agg(
            mean_f1=("f1", "mean"),
            std_f1=("f1", "std"),
            mean_precision=("precision", "mean"),
            std_precision=("precision", "std"),
            mean_recall=("recall", "mean"),
            std_recall=("recall", "std"),
            n_pair_id=("pair_id", "nunique"),
        )
    )
    global_summary = by_seed.agg(
        mean_f1=("mean_f1", "mean"),
        std_f1_across_seeds=("mean_f1", "std"),
        mean_precision=("mean_precision", "mean"),
        std_precision_across_seeds=("mean_precision", "std"),
        mean_recall=("mean_recall", "mean"),
        std_recall_across_seeds=("mean_recall", "std"),
        n_runs=("run_id", "nunique"),
        n_pair_id=("n_pair_id", "max"),
    ).reset_index(drop=True)

    fold_df.to_csv(args.out_dir / "metamatch_rf350_full60_fold_metrics_all_seeds.csv", index=False)
    pair_df.to_csv(args.out_dir / "metamatch_rf350_full60_pair_metrics_all_seeds.csv", index=False)
    by_seed.to_csv(args.out_dir / "metamatch_rf350_full60_summary_by_seed.csv", index=False)
    global_summary.to_csv(args.out_dir / "metamatch_rf350_full60_summary_across_seeds.csv", index=False)

    print("\n=== Summary by seed ===")
    print(by_seed.to_string(index=False))
    print("\n=== Across-seed summary ===")
    print(global_summary.to_string(index=False))
    print(f"\nSaved: {args.out_dir}")


if __name__ == "__main__":
    main()
