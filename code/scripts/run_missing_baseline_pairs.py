#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from osirim_occidata.baselines import run_external_baseline, run_valentine_baseline
from osirim_occidata.io_utils import read_json, read_jsonl, safe_read_table, write_json
from osirim_occidata.evaluator import save_scored_predictions


DISPLAY_TO_TASK_METHOD = {
    "LLMATCH": "LLMATCH",
    "SMUTF": "SMUTF",
    "MagnetoFTGPT": "Magneto_ft_gpt",
    "MagnetoFT": "Magneto_ft_no_gpt",
    "MagnetoGPT": "Magneto_no_ft_gpt",
    "Magneto": "Magneto_no_ft_no_gpt",
    "ISResMat": "ISResMat",
    "COMA++": "coma_pp",
    "COMA Instance": "coma_instance",
    "COMA": "coma",
    "Similarity Flooding": "similarity_flooding",
    "Distribution Based": "distribution_based",
    "Cupid": "cupid",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run only baseline table pairs missing from the RQ1 pair-level CSV.")
    p.add_argument("--source-output-root", type=Path, default=Path("outputs/exp_occidata"))
    p.add_argument(
        "--rq1-pair-csv",
        type=Path,
        default=Path("outputs/exp_occidata/reports/meeting_baselines_vs_metamatch/rq1_effectiveness_efficiency/effectiveness_baselines_metaspace_by_pair.csv"),
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=Path("outputs/exp_occidata/reports/meeting_baselines_vs_metamatch/baseline_missing_pairs_completion"),
    )
    p.add_argument("--methods", type=str, default="", help="Comma-separated display names. Default: all baseline methods with missing pairs.")
    p.add_argument("--baseline-workers", type=int, default=1)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--smutf-python",
        type=Path,
        default=Path("/Users/nahawandkired/Documents/Recherche/baselines/SMUTF-official/.venv_paper/bin/python3"),
    )
    p.add_argument(
        "--smutf-pythonpath",
        type=Path,
        default=Path("/Users/nahawandkired/Documents/Recherche/baselines"),
    )
    p.add_argument(
        "--smutf-model-dir",
        type=Path,
        default=Path("/Users/nahawandkired/Documents/Recherche/baselines/SMUTF-official/model/2022-04-12-12-06-32"),
    )
    p.add_argument(
        "--magneto-python",
        type=Path,
        default=Path("/Users/nahawandkired/Documents/Recherche/magneto_matcher/.venv/bin/python3"),
    )
    p.add_argument("--magneto-package-root", type=Path, default=Path("/Users/nahawandkired/Documents/Recherche/magneto_matcher/algorithms/magneto"))
    p.add_argument("--magneto-model-dir", type=Path, default=Path("/Users/nahawandkired/Documents/Recherche/magneto_matcher/models"))
    return p.parse_args()


def write_checkpoints(out_root: Path, manifest_rows: list[dict], pair_metric_parts: list[pd.DataFrame], runtime_rows: list[dict]) -> None:
    pd.DataFrame(manifest_rows).to_csv(out_root / "missing_pairs_manifest.csv", index=False)
    if pair_metric_parts:
        pair_metrics = pd.concat(pair_metric_parts, ignore_index=True)
    else:
        pair_metrics = pd.DataFrame()
    pair_metrics.to_csv(out_root / "completion_pair_metrics_at_threshold.csv", index=False)
    pd.DataFrame(runtime_rows).to_csv(out_root / "completion_runtime.csv", index=False)


def pair_metrics_at_threshold(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    work = df.copy()
    work["score"] = pd.to_numeric(work["score"], errors="coerce").fillna(0.0)
    work["label"] = work["label"].astype(int)
    work["pred"] = (work["score"] > threshold).astype(int)
    for pair_id, g in work.groupby("pair_id", sort=False):
        y = g["label"].to_numpy(int)
        pred = g["pred"].to_numpy(int)
        rows.append({
            "pair_id": str(pair_id),
            "precision_all_to_all": precision_score(y, pred, zero_division=0),
            "recall_all_to_all": recall_score(y, pred, zero_division=0),
            "f1_all_to_all": f1_score(y, pred, zero_division=0),
            "n_rows": len(g),
            "n_positive": int(g["label"].sum()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    source_root = args.source_output_root.resolve()
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    exp = read_json(source_root / "experiment.json")
    tasks = read_jsonl(source_root / "tasks.jsonl")
    task_by_method_fold = {(str(t["method"]), int(t["fold_id"])): t for t in tasks}

    rq1 = pd.read_csv(args.rq1_pair_csv)
    rq1["pair_id"] = rq1["pair_id"].astype(str)
    existing_by_method = {
        method: set(g["pair_id"].dropna().astype(str))
        for method, g in rq1.groupby("method", sort=False)
        if method != "MetaSpace"
    }

    expected_parts = []
    for fold_dir in sorted((source_root / "folds").glob("fold_*"), key=lambda p: int(p.name.split("_")[-1])):
        fold_id = int(fold_dir.name.split("_")[-1])
        test = safe_read_table(fold_dir / "test.parquet" if (fold_dir / "test.parquet").exists() else fold_dir / "test.csv.gz")
        tmp = test[["pair_id"]].drop_duplicates().copy()
        tmp["pair_id"] = tmp["pair_id"].astype(str)
        tmp["fold_id"] = fold_id
        expected_parts.append(tmp)
    expected = pd.concat(expected_parts, ignore_index=True).drop_duplicates(["fold_id", "pair_id"])
    expected_pairs = set(expected["pair_id"])

    requested = [m.strip() for m in args.methods.split(",") if m.strip()]
    if not requested:
        requested = [
            m for m in DISPLAY_TO_TASK_METHOD
            if len(expected_pairs - existing_by_method.get(m, set())) > 0
        ]

    manifest_rows = []
    pair_metric_parts = []
    runtime_rows = []

    for display_method in requested:
        task_method = DISPLAY_TO_TASK_METHOD.get(display_method, display_method)
        missing = sorted(expected_pairs - existing_by_method.get(display_method, set()))
        if not missing:
            print(f"skip complete: {display_method}", flush=True)
            continue

        missing_by_fold = expected[expected["pair_id"].isin(missing)].groupby("fold_id")["pair_id"].apply(list).to_dict()
        print(f"{display_method}: missing {len(missing)} pair_id", flush=True)

        for fold_id, pair_ids in sorted(missing_by_fold.items()):
            task = task_by_method_fold.get((task_method, int(fold_id)))
            if task is None:
                print(f"  skip fold_{fold_id}: no task for {task_method}", flush=True)
                continue

            fold_dir = source_root / "folds" / f"fold_{fold_id}"
            test = safe_read_table(fold_dir / "test.parquet" if (fold_dir / "test.parquet").exists() else fold_dir / "test.csv.gz")
            test = test[test["pair_id"].astype(str).isin(set(pair_ids))].copy()
            test["pair_id"] = test["pair_id"].astype(str)
            n_pair = test["pair_id"].nunique()
            n_rows = len(test)

            manifest_rows.append({
                "method": display_method,
                "task_method": task_method,
                "fold_id": fold_id,
                "n_missing_pair_id": n_pair,
                "n_candidate_rows": n_rows,
            })

            method_dir = out_root / "results" / f"fold_{fold_id}" / task_method
            if args.dry_run:
                print(f"  dry-run fold_{fold_id}: {n_pair} pairs | {n_rows} rows", flush=True)
                continue

            existing_pred = method_dir / "predictions.parquet"
            existing_info = method_dir / "completion_info.json"
            if not args.overwrite and existing_pred.exists() and existing_info.exists():
                pred_df = safe_read_table(existing_pred)
                info = read_json(existing_info)
                runtime = float(info.get("runtime_sec", 0.0))

                pm = pair_metrics_at_threshold(pred_df, threshold=args.threshold)
                pm.insert(0, "method", display_method)
                pm.insert(1, "task_method", task_method)
                pm.insert(2, "fold_id", fold_id)
                pm["runtime_sec_task"] = runtime
                pm["runtime_sec_pair_approx"] = float(runtime / n_pair) if n_pair else None
                pm["source"] = str(existing_pred)
                pair_metric_parts.append(pm)

                runtime_rows.append({
                    "method": display_method,
                    "task_method": task_method,
                    "fold_id": fold_id,
                    "n_missing_pair_id": int(n_pair),
                    "n_candidate_rows": int(n_rows),
                    "runtime_sec": runtime,
                    "runtime_sec_pair_approx": float(runtime / n_pair) if n_pair else None,
                })
                write_checkpoints(out_root, manifest_rows, pair_metric_parts, runtime_rows)
                print(f"  skip existing fold_{fold_id}: {n_pair} pairs", flush=True)
                continue

            t0 = time.perf_counter()
            if task["family"] == "valentine":
                out = run_valentine_baseline(
                    method=task_method,
                    test_df=test,
                    dataset_root=Path(exp["dataset_root"]),
                    max_workers=min(max(args.baseline_workers, 1), 128),
                )
                runtime = time.perf_counter() - t0
                pred_df = out.predictions
                runtime_payload = {"status": "ok", "wall_time_sec": runtime, "failures": out.failures}
            elif task["family"] == "external":
                backend_cmd = task.get("backend_cmd")
                if task_method == "SMUTF":
                    backend_cmd = (
                        f"PYTHONNOUSERSITE=1 "
                        f"PYTHONPATH={shlex.quote(str(args.smutf_pythonpath))} "
                        f"{shlex.quote(str(args.smutf_python))} "
                        f"scripts/external_baselines/smutf_real_runner.py "
                        f"--pair-manifest {{pair_manifest}} "
                        f"--test-manifest {{test_manifest}} "
                        f"--out {{predictions}} "
                        f"--model-dir {shlex.quote(str(args.smutf_model_dir))}"
                    )
                elif task_method.startswith("Magneto"):
                    backend_cmd = (
                        f"PYTHONNOUSERSITE=1 "
                        f"PYTHONPATH={shlex.quote(str(args.magneto_package_root))} "
                        f"{shlex.quote(str(args.magneto_python))} "
                        f"scripts/external_baselines/magneto_real_runner.py "
                        f"--variant {shlex.quote(task_method)} "
                        f"--magneto-package-root {shlex.quote(str(args.magneto_package_root))} "
                        f"--magneto-model-dir {shlex.quote(str(args.magneto_model_dir))} "
                        f"--encoding-mode header_values_repeat "
                        f"--strict-variant 1 "
                        f"--pair-manifest {{pair_manifest}} "
                        f"--test-manifest {{test_manifest}} "
                        f"--out {{predictions}}"
                    )
                pred_df, runtime_payload = run_external_baseline(
                    baseline_name=task_method,
                    test_df=test,
                    dataset_root=Path(exp["dataset_root"]),
                    fold_dir=fold_dir,
                    output_dir=method_dir,
                    command_template=task.get("command_template"),
                    backend_cmd=backend_cmd,
                    predictions_path=task.get("predictions_path"),
                    required_env=task.get("required_env", []),
                    timeout_sec=int(task.get("timeout_sec", 86400)),
                    baseline_workers=args.baseline_workers,
                )
                runtime = float(runtime_payload.get("wall_time_sec", time.perf_counter() - t0))
            else:
                print(f"  skip fold_{fold_id}: unsupported family {task['family']}", flush=True)
                continue

            metrics = save_scored_predictions(pred_df, method_dir, method_name=task_method, runtime_sec=runtime)
            write_json(method_dir / "completion_info.json", {
                "display_method": display_method,
                "task_method": task_method,
                "fold_id": fold_id,
                "n_missing_pair_id": int(n_pair),
                "n_candidate_rows": int(n_rows),
                "threshold_for_pair_csv": args.threshold,
                "runtime_sec": float(runtime),
                "runtime_sec_pair_approx": float(runtime / n_pair) if n_pair else None,
                "metrics": metrics,
                "runtime_payload": runtime_payload,
            })

            pm = pair_metrics_at_threshold(pred_df, threshold=args.threshold)
            pm.insert(0, "method", display_method)
            pm.insert(1, "task_method", task_method)
            pm.insert(2, "fold_id", fold_id)
            pm["runtime_sec_task"] = float(runtime)
            pm["runtime_sec_pair_approx"] = float(runtime / n_pair) if n_pair else None
            pm["source"] = str(method_dir / "predictions.parquet")
            pair_metric_parts.append(pm)

            runtime_rows.append({
                "method": display_method,
                "task_method": task_method,
                "fold_id": fold_id,
                "n_missing_pair_id": int(n_pair),
                "n_candidate_rows": int(n_rows),
                "runtime_sec": float(runtime),
                "runtime_sec_pair_approx": float(runtime / n_pair) if n_pair else None,
            })
            write_checkpoints(out_root, manifest_rows, pair_metric_parts, runtime_rows)
            print(f"  done fold_{fold_id}: {n_pair} pairs | {runtime:.2f}s", flush=True)

    write_checkpoints(out_root, manifest_rows, pair_metric_parts, runtime_rows)
    print("saved:", out_root / "missing_pairs_manifest.csv")
    print("saved:", out_root / "completion_pair_metrics_at_threshold.csv")
    print("saved:", out_root / "completion_runtime.csv")


if __name__ == "__main__":
    main()
