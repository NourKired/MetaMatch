#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


METHODS = [
    ("LLMATCH", "LLMATCH"),
    ("Magneto_ft_gpt", "MagnetoFTGPT"),
    ("Magneto_ft_no_gpt", "MagnetoFT"),
    ("Magneto_no_ft_gpt", "MagnetoGPT"),
    ("Magneto_no_ft_no_gpt", "Magneto"),
    ("ISResMat", "ISResMat"),
    ("coma_pp", "COMA++"),
    ("coma", "COMA"),
    ("similarity_flooding", "Similarity Flooding"),
    ("distribution_based", "Distribution Based"),
    ("cupid", "Cupid"),
]


def read_runtime_sec(path: Path) -> float:
    obj = json.loads(path.read_text())
    value = obj.get("runtime_sec", obj.get("runtime", {}).get("wall_time_sec"))
    if value is None:
        raise AssertionError(f"Missing runtime_sec in {path}")
    return float(value)


def assert_close(name: str, actual: float, expected: float, tol: float = 0.02) -> None:
    if abs(actual - expected) > tol:
        raise AssertionError(f"{name}: actual={actual:.6f}, expected={expected:.6f}, diff={actual - expected:.6f}")


def extract_latex_rows(main_tex: Path) -> dict[str, tuple[float, float, float]]:
    text = main_tex.read_text()
    table_match = re.search(
        r"\\caption\{Efficiency summary.*?\\midrule\s*(.*?)\\bottomrule",
        text,
        flags=re.S,
    )
    if not table_match:
        raise AssertionError("Efficiency table not found in main.tex")
    rows = {}
    for raw in table_match.group(1).splitlines():
        raw = raw.strip()
        if not raw or "&" not in raw:
            continue
        raw = raw.rstrip("\\").strip()
        parts = [p.strip() for p in raw.split("&")]
        if len(parts) != 4:
            raise AssertionError(f"Unexpected efficiency row format: {raw}")
        rows[parts[0]] = tuple(float(x) for x in parts[1:])
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=Path("/Users/nahawandkired/Documents/metamatch"))
    ap.add_argument("--paper-root", type=Path, default=Path("/Users/nahawandkired/Documents/vldb"))
    args = ap.parse_args()

    project = args.project_root
    report = project / "outputs/exp_occidata/reports/meeting_baselines_vs_metamatch"
    rq1 = report / "rq1_effectiveness_efficiency"
    results = project / "outputs/exp_occidata/results"

    paper_csv = rq1 / "table_efficiency_preliminary_features_train_application_predict.csv"
    if not paper_csv.exists():
        raise FileNotFoundError(paper_csv)
    paper = pd.read_csv(paper_csv)

    meta = pd.read_csv(rq1 / "metaspace_rf_predict_only_microbenchmark_summary.csv").iloc[0]
    compact = pd.read_csv(
        report / "rq2_rq3_ablation_feature_selection_rf350/rq3_full_vs_compact_efficiency_end_to_end_current_impl.csv"
    )
    smutf_folds = pd.read_csv(rq1 / "smutf_runtime_fold_amortized.csv")
    smutf_pred = pd.read_csv(rq1 / "smutf_xgboost_only_microbenchmark_summary.csv").iloc[0]
    smutf_bench = pd.read_csv(report / "smutf_real_benchmark_complet_complet/smutf_real_benchmark_measured_steps.csv")

    expected = {}
    expected["Full MetaSpace"] = (
        float(meta["preliminary_feature_build_sec_551"]) + float(meta["preliminary_rf_train_sec_all_folds"]),
        float(meta["application_rf_predict_only_sec_all_folds"]),
    )
    compact_row = compact.loc[compact["feature_set"].eq("Compact")].iloc[0]
    compact_predict = float(meta["application_rf_predict_only_sec_all_folds"]) * (10.0 / 60.0)
    compact_train = float(compact_row["rf_train_test_sec"]) - compact_predict
    expected["Reduced MetaSpace"] = (
        float(compact_row["feature_construction_sec_551"]) + compact_train,
        compact_predict,
    )

    smutf_runtime_total = float(smutf_folds["runtime_sec_fold"].sum())
    smutf_predict = float(smutf_pred["estimated_predict_loaded_sec_551"])
    smutf_feature_runtime = smutf_runtime_total - smutf_predict
    smutf_finetune = float(smutf_bench.loc[smutf_bench["step"].eq("finetuning_total"), "elapsed_sec"].iloc[0])
    expected["SMUTF"] = (smutf_feature_runtime + smutf_finetune, smutf_predict)

    for raw, label in METHODS:
        expected[label] = (0.0, sum(read_runtime_sec(p) for p in results.glob(f"fold_*/{raw}/task_info.json")))

    old = pd.read_csv(rq1 / "table_efficiency_training_runtime_total_551_pairs_with_reduced.csv")
    for label in ["MagnetoFTGPT", "MagnetoFT"]:
        if label in set(old["method"]):
            col = "training_or_preliminary_sec" if "training_or_preliminary_sec" in old.columns else "preliminary_or_training_time_sec"
            prelim = float(old.loc[old["method"].eq(label), col].iloc[0])
            expected[label] = (prelim, expected[label][1])

    audit_rows = []
    for method, (prelim, app) in expected.items():
        row = paper.loc[paper["method"].eq(method)]
        if len(row) != 1:
            raise AssertionError(f"Expected one CSV row for {method}, found {len(row)}")
        row = row.iloc[0]
        expected_total = prelim + app
        assert_close(f"{method} preliminary_time_sec", float(row["preliminary_time_sec"]), round(prelim, 2))
        assert_close(f"{method} application_time_sec", float(row["application_time_sec"]), round(app, 2))
        assert_close(f"{method} total_time_sec", float(row["total_time_sec"]), round(expected_total, 2))
        audit_rows.append(
            {
                "method": method,
                "expected_preliminary_time_sec": round(prelim, 2),
                "csv_preliminary_time_sec": float(row["preliminary_time_sec"]),
                "expected_application_time_sec": round(app, 2),
                "csv_application_time_sec": float(row["application_time_sec"]),
                "expected_total_time_sec": round(expected_total, 2),
                "csv_total_time_sec": float(row["total_time_sec"]),
            }
        )

    latex_rows = extract_latex_rows(args.paper_root / "main.tex")
    for row in audit_rows:
        method = row["method"]
        if method not in latex_rows:
            raise AssertionError(f"{method} missing from main.tex efficiency table")
        latex_prelim, latex_app, latex_total = latex_rows[method]
        assert_close(f"{method} LaTeX preliminary", latex_prelim, row["csv_preliminary_time_sec"])
        assert_close(f"{method} LaTeX application", latex_app, row["csv_application_time_sec"])
        assert_close(f"{method} LaTeX total", latex_total, row["csv_total_time_sec"])

    fig_pdf = args.paper_root / "images/fig_rq4_total_runtime_bar.pdf"
    fig_png = args.paper_root / "images/fig_rq4_total_runtime_bar.png"
    if not fig_pdf.exists() or fig_pdf.stat().st_size < 1000:
        raise AssertionError(f"Runtime figure PDF missing or too small: {fig_pdf}")
    if not fig_png.exists() or fig_png.stat().st_size < 1000:
        raise AssertionError(f"Runtime figure PNG missing or too small: {fig_png}")

    audit = pd.DataFrame(audit_rows).sort_values("csv_total_time_sec", ascending=False)
    out = rq1 / "audit_efficiency_results.csv"
    audit.to_csv(out, index=False)
    print("AUDIT PASSED")
    print(audit.to_string(index=False))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
