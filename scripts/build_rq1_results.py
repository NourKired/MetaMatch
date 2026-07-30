#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/Users/nahawandkired/Documents/metamatch")
OUT = ROOT / "outputs/exp_occidata/reports/meeting_baselines_vs_metamatch/rq1_effectiveness_efficiency"
COMPANION = OUT / "paper_csv"
COMPANION.mkdir(parents=True, exist_ok=True)

PLOT_ORDER = [
    "MetaMatch",
    "LLMATCH",
    "SMUTF",
    "MagnetoFTGPT",
    "MagnetoFT",
    "MagnetoGPT",
    "Magneto",
    "ISResMat",
    "COMA++",
    "COMA Instance",
    "COMA",
    "Similarity Flooding",
    "Distribution Based",
    "Cupid",
]


def round_numeric(df: pd.DataFrame, digits: int = 2) -> pd.DataFrame:
    out = df.copy()
    for col in out.select_dtypes(include=[np.number]).columns:
        out[col] = out[col].round(digits)
    return out


def seconds_to_hours(sec: float | int | None) -> float:
    if sec is None or pd.isna(sec):
        return np.nan
    return float(sec) / 3600.0


def read_feature_build_seconds() -> float:
    p = OUT.parent / "metamatch_60_runtime_benchmark/runtime_estimate_summary_60features.json"
    with p.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    return float(obj["estimated_total_feature_build_sec_551"])


def read_recalibrated_rf_train_seconds() -> tuple[float, str]:
    one_fold = pd.read_csv(OUT / "metamatch_rf_one_fold_real_train_test_time.csv").iloc[0]
    fold_root = ROOT / "outputs/exp_occidata/folds"
    total_train_rows = 0
    for fold_dir in sorted(fold_root.glob("fold_*"), key=lambda p: int(p.name.split("_")[-1])):
        total_train_rows += len(pd.read_parquet(fold_dir / "train.parquet", columns=["label"]))
    sec_per_train_row = float(one_fold["fit_sec"]) / float(one_fold["n_train_rows"])
    estimated_all_folds = sec_per_train_row * total_train_rows
    note = (
        "Recalibrated from real fold_0 RF fit with n_jobs=-1 "
        f"({float(one_fold['fit_sec']):.2f}s for {int(one_fold['n_train_rows'])} train rows), "
        "scaled by total train rows over the 6 folds."
    )
    return estimated_all_folds, note


def build_table_metamatch_classifiers() -> None:
    src = pd.read_csv(OUT / "metamatch_classifiers_metrics_mean_std.csv")
    table = src.rename(
        columns={
            "method": "classifier",
            "f1_mean": "mean_f1_all_to_all",
            "f1_std": "std_f1_all_to_all",
            "precision_mean": "mean_precision_all_to_all",
            "precision_std": "std_precision_all_to_all",
            "recall_mean": "mean_recall_all_to_all",
            "recall_std": "std_recall_all_to_all",
        }
    )
    table["f1_pm_std"] = table.apply(lambda r: f"{r['mean_f1_all_to_all']:.2f} +/- {r['std_f1_all_to_all']:.2f}", axis=1)
    table["precision_pm_std"] = table.apply(
        lambda r: f"{r['mean_precision_all_to_all']:.2f} +/- {r['std_precision_all_to_all']:.2f}", axis=1
    )
    table["recall_pm_std"] = table.apply(
        lambda r: f"{r['mean_recall_all_to_all']:.2f} +/- {r['std_recall_all_to_all']:.2f}", axis=1
    )
    table = table[
        [
            "classifier",
            "mean_f1_all_to_all",
            "std_f1_all_to_all",
            "f1_pm_std",
            "mean_precision_all_to_all",
            "std_precision_all_to_all",
            "precision_pm_std",
            "mean_recall_all_to_all",
            "std_recall_all_to_all",
            "recall_pm_std",
            "n_pair_id",
        ]
    ]
    table = round_numeric(table, 2)
    table.to_csv(COMPANION / "table_1_metamatch_classifiers_all_to_all.csv", index=False)
    table.to_csv(OUT / "table_1_metamatch_classifiers_all_to_all.csv", index=False)


def build_winrate_csvs() -> None:
    by_pair = pd.read_csv(OUT / "effectiveness_baselines_metamatch_by_pair.csv")
    by_pair = by_pair[by_pair["method"].isin(PLOT_ORDER)].copy()
    by_pair["f1_all_to_all"] = pd.to_numeric(by_pair["f1_all_to_all"], errors="coerce")
    wide = (
        by_pair.dropna(subset=["f1_all_to_all"])
        .sort_values(["method", "pair_id", "f1_all_to_all"], ascending=[True, True, False])
        .drop_duplicates(["method", "pair_id"], keep="first")
        .pivot(index="pair_id", columns="method", values="f1_all_to_all")
    )
    methods = [m for m in PLOT_ORDER if m in wide.columns]
    win = pd.DataFrame(np.nan, index=methods, columns=methods, dtype=float)
    support = pd.DataFrame(0, index=methods, columns=methods, dtype=int)
    ties = pd.DataFrame(0, index=methods, columns=methods, dtype=int)
    long_rows = []
    for a in methods:
        for b in methods:
            if a == b:
                continue
            sub = wide[[a, b]].dropna()
            wins = int((sub[a] > sub[b]).sum())
            losses = int((sub[a] < sub[b]).sum())
            tie_count = int((sub[a] == sub[b]).sum())
            denom = wins + losses
            pct = np.nan if denom == 0 else 100.0 * wins / denom
            win.loc[a, b] = pct
            support.loc[a, b] = int(len(sub))
            ties.loc[a, b] = tie_count
            long_rows.append(
                {
                    "row_method": a,
                    "column_method": b,
                    "win_percent_excluding_ties": pct,
                    "wins": wins,
                    "losses": losses,
                    "ties": tie_count,
                    "common_pair_id": int(len(sub)),
                }
            )
    win = round_numeric(win, 2)
    long_df = round_numeric(pd.DataFrame(long_rows), 2)
    win.to_csv(OUT / "winrate_f1_all_to_all_no_ties_percent_matrix.csv")
    support.to_csv(OUT / "winrate_f1_all_to_all_common_support_matrix.csv")
    ties.to_csv(OUT / "winrate_f1_all_to_all_tie_count_matrix.csv")
    win.to_csv(COMPANION / "figure_2_winrate_f1_all_to_all_no_ties_percent_matrix.csv")
    long_df.to_csv(COMPANION / "figure_2_winrate_f1_all_to_all_no_ties_long.csv", index=False)


def build_f1_distribution_csv() -> None:
    cols = ["method", "pair_id", "f1_all_to_all", "precision_all_to_all", "recall_all_to_all", "source"]
    df = pd.read_csv(OUT / "effectiveness_baselines_metamatch_by_pair.csv")
    df = df[df["method"].isin(PLOT_ORDER)].copy()
    df = df[[c for c in cols if c in df.columns]]
    df["method_order"] = df["method"].map({m: i for i, m in enumerate(PLOT_ORDER)})
    df = df.sort_values(["method_order", "method", "pair_id"]).drop(columns=["method_order"])
    df = round_numeric(df, 2)
    df.to_csv(COMPANION / "figure_3_distribution_f1_all_to_all_by_pair.csv", index=False)
    df.to_csv(OUT / "figure_3_distribution_f1_all_to_all_by_pair.csv", index=False)


def build_runtime_distribution_csv() -> None:
    runtime = pd.read_csv(OUT / "efficiency_runtime_by_pair.csv")
    runtime["runtime_sec_pair"] = pd.to_numeric(runtime["runtime_sec_pair"], errors="coerce")
    runtime = runtime[runtime["method"].isin(PLOT_ORDER) & runtime["runtime_sec_pair"].notna()].copy()
    runtime = runtime[["method", "pair_id", "runtime_sec_pair"]].copy()
    runtime["runtime_min_pair"] = runtime["runtime_sec_pair"] / 60.0

    # Replace MetaMatch pair runtime by RF predict-only application time allocated by candidate rows.
    rf_summary = pd.read_csv(OUT / "metamatch_rf_predict_only_microbenchmark_summary.csv").iloc[0]
    rf_total_predict_sec = float(rf_summary["application_rf_predict_only_sec_all_folds"])
    row_scores = pd.read_parquet(OUT.parent / "metamatch_rf_60features_seed42/metamatch_rf_60features_seed42_row_scores.parquet")
    counts = row_scores.groupby("pair_id").size().reset_index(name="n_candidate_rows")
    counts["method"] = "MetaMatch"
    counts["runtime_sec_pair"] = rf_total_predict_sec * counts["n_candidate_rows"] / counts["n_candidate_rows"].sum()
    counts["runtime_min_pair"] = counts["runtime_sec_pair"] / 60.0
    metamatch_runtime = counts[["method", "pair_id", "runtime_sec_pair", "runtime_min_pair", "n_candidate_rows"]]

    runtime = runtime[runtime["method"] != "MetaMatch"].copy()
    runtime["n_candidate_rows"] = np.nan
    out = pd.concat([metamatch_runtime, runtime], ignore_index=True, sort=False)
    out["method_order"] = out["method"].map({m: i for i, m in enumerate(PLOT_ORDER)})
    out = out.sort_values(["method_order", "method", "pair_id"]).drop(columns=["method_order"])
    out = round_numeric(out, 2)
    out.to_csv(COMPANION / "figure_4_distribution_application_runtime_by_pair.csv", index=False)
    out.to_csv(OUT / "figure_4_distribution_application_runtime_by_pair.csv", index=False)


def build_efficiency_split_table() -> None:
    runtime_pair = pd.read_csv(OUT / "efficiency_runtime_by_pair.csv")
    runtime_pair["runtime_sec_pair"] = pd.to_numeric(runtime_pair["runtime_sec_pair"], errors="coerce")
    runtime_pair = runtime_pair[runtime_pair["method"].isin(PLOT_ORDER)].copy()

    feature_build_sec = read_feature_build_seconds()
    rf_train_sec, rf_train_note = read_recalibrated_rf_train_seconds()
    rf_predict_summary = pd.read_csv(OUT / "metamatch_rf_predict_only_microbenchmark_summary.csv").iloc[0]
    metamatch_application_sec = float(rf_predict_summary["application_rf_predict_only_sec_all_folds"])

    smutf_summary = pd.read_csv(OUT / "smutf_xgboost_only_microbenchmark_summary.csv").iloc[0]
    smutf_application_sec = float(smutf_summary["estimated_predict_loaded_sec_551"])
    smutf_finetuning = pd.read_csv(OUT / "smutf_finetuning_time_estimated_from_model_timestamps.csv")
    smutf_xgboost_finetuning_sec = float(
        smutf_finetuning.loc[
            smutf_finetuning["scenario"].eq("complet-complet"),
            "training_sec_estimated_from_timestamps",
        ].iloc[0]
    )

    magneto_estimate_path = OUT.parent / "magneto_training_time_estimate/magneto_training_time_estimate_magneto_estimated.json"
    with magneto_estimate_path.open("r", encoding="utf-8") as f:
        magneto_estimate = json.load(f)
    magneto_ft_prelim_sec_5_categories = float(magneto_estimate["estimated_total_sec"]) * 5

    rows = []
    for method in PLOT_ORDER:
        observed = runtime_pair.loc[runtime_pair["method"] == method, "runtime_sec_pair"].dropna()
        preliminary_sec = 0.0
        preliminary_computation = "/"
        preliminary_note = "/"
        application_sec = np.nan
        application_source = "not available"
        n_runtime = 0

        if len(observed) > 0:
            application_sec = float(observed.mean() * 551)
            application_source = "mean pair runtime x 551"
            n_runtime = int(runtime_pair.loc[runtime_pair["method"] == method, "pair_id"].nunique())

        if method == "MetaMatch":
            preliminary_sec = feature_build_sec + rf_train_sec
            preliminary_computation = "60 meta-features construction + RF training"
            preliminary_note = rf_train_note
            application_sec = metamatch_application_sec
            application_source = "RF predict_proba only on all test rows over 6 folds"
            n_runtime = 551
        elif method == "SMUTF":
            preliminary_sec = smutf_xgboost_finetuning_sec
            preliminary_computation = "XGBoost fine-tuning"
            preliminary_note = "Local SMUTF complet-complet fine-tuning duration estimated from model timestamps."
            application_sec = smutf_application_sec
            application_source = "16 official XGBoost models predict-only over 551 pairs"
            n_runtime = 551
        elif method in {"MagnetoFT", "MagnetoFTGPT"}:
            preliminary_sec = magneto_ft_prelim_sec_5_categories
            preliminary_computation = "fine-tuning"
            preliminary_note = "Estimated from measured CPU batches: one category model x 5 categories."
        elif method == "LLMATCH":
            preliminary_sec = np.nan
            preliminary_note = "External/precomputed cost not logged."
        elif method in {"Magneto", "MagnetoGPT"}:
            preliminary_sec = np.nan
            preliminary_note = "No fine-tuning in this run; original checkpoint cost not logged."

        known_total = (0 if pd.isna(preliminary_sec) else preliminary_sec) + (0 if pd.isna(application_sec) else application_sec)
        rows.append(
            {
                "method": method,
                "preliminary_computation": preliminary_computation,
                "preliminary_time_sec": preliminary_sec,
                "preliminary_time_hours": seconds_to_hours(preliminary_sec),
                "application_time_sec_551_pairs": application_sec,
                "application_time_hours_551_pairs": seconds_to_hours(application_sec),
                "known_total_time_sec_551_pairs": known_total,
                "known_total_time_hours_551_pairs": seconds_to_hours(known_total),
                "n_runtime_pairs": n_runtime,
                "application_source": application_source,
                "preliminary_note": preliminary_note,
            }
        )

    table = pd.DataFrame(rows)
    table = round_numeric(table, 2)
    table.to_csv(OUT / "table_efficiency_preliminary_application_total_551_pairs.csv", index=False)
    table.to_csv(COMPANION / "table_2_efficiency_preliminary_application_total_551_pairs.csv", index=False)

    # Keep the microbenchmark summary consistent with the corrected RF train estimate.
    corrected_summary = pd.read_csv(OUT / "metamatch_rf_predict_only_microbenchmark_summary.csv")
    corrected_summary["preliminary_rf_train_sec_all_folds"] = rf_train_sec
    corrected_summary["preliminary_total_sec"] = feature_build_sec + rf_train_sec
    corrected_summary["note"] = (
        "RF predict-only measured on all test rows over 6 folds; RF train time is "
        "recalibrated from real fold_0 n_jobs=-1 fit and scaled by train rows."
    )
    corrected_summary["rf_train_time_note"] = rf_train_note
    corrected_summary = round_numeric(corrected_summary, 2)
    corrected_summary.to_csv(OUT / "metamatch_rf_predict_only_microbenchmark_summary.csv", index=False)
    corrected_summary.to_csv(COMPANION / "table_2a_metamatch_rf_time_components.csv", index=False)


def main() -> None:
    build_table_metamatch_classifiers()
    build_winrate_csvs()
    build_f1_distribution_csv()
    build_runtime_distribution_csv()
    build_efficiency_split_table()
    print(f"Saved paper CSVs in: {COMPANION}")


if __name__ == "__main__":
    main()
