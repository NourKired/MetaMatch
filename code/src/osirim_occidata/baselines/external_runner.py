from __future__ import annotations

import os
import json
import shlex
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from ..catalog import discover_table_pairs
from .valentine_runner import run_valentine_baseline


REQUIRED_COLUMNS = {"pair_id", "source_col_norm", "target_col_norm", "score"}
VALENTINE_COMPAT_ALIASES = {
    "coma": "coma_schema",
    "coma_pp": "coma_instance",
    "coma_inst": "coma_instance",
    "cupid_ext": "cupid",
    "similarity_flooding_ext": "similarity_flooding",
    "distribution_based_ext": "distribution_based",
}


def _load_predictions(pred_path: Path) -> pd.DataFrame:
    if pred_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(pred_path)
    else:
        df = pd.read_csv(pred_path)

    # Normalize common alternative column names.
    rename_map = {}
    if "source_column" in df.columns and "source_col_norm" not in df.columns:
        rename_map["source_column"] = "source_col_norm"
    if "target_column" in df.columns and "target_col_norm" not in df.columns:
        rename_map["target_column"] = "target_col_norm"
    if "similarity" in df.columns and "score" not in df.columns:
        rename_map["similarity"] = "score"
    if rename_map:
        df = df.rename(columns=rename_map)

    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in baseline predictions {pred_path}: {sorted(missing)}")

    out = df[["pair_id", "source_col_norm", "target_col_norm", "score"]].copy()
    out["pair_id"] = out["pair_id"].astype(str)
    out["source_col_norm"] = out["source_col_norm"].astype(str).str.lower().str.strip()
    out["target_col_norm"] = out["target_col_norm"].astype(str).str.lower().str.strip()
    out["score"] = pd.to_numeric(out["score"], errors="coerce").fillna(0.0)
    return out


def _merge_scores(test_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    merged = test_df.merge(
        pred_df,
        how="left",
        on=["pair_id", "source_col_norm", "target_col_norm"],
        suffixes=("", "_ext"),
    )
    merged["score"] = pd.to_numeric(merged["score"], errors="coerce").fillna(0.0)
    return merged


def _write_pair_manifest(
    dataset_root: Path,
    test_df: pd.DataFrame,
    out_path: Path,
) -> Path:
    pairs = discover_table_pairs(dataset_root)
    by_id = {p.pair_id: p for p in pairs}
    requested = sorted(test_df["pair_id"].astype(str).unique().tolist())

    rows = []
    for pid in requested:
        p = by_id.get(pid)
        if p is None:
            continue
        rows.append(
            {
                "pair_id": p.pair_id,
                "dataset": p.dataset,
                "relation_type": p.relation_type,
                "category": p.category,
                "source_csv": str(p.source_csv.resolve()),
                "target_csv": str(p.target_csv.resolve()),
                "mapping_json": str(p.mapping_json.resolve()),
                "source_table_name": p.source_table_name,
                "target_table_name": p.target_table_name,
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    return out_path


def run_external_baseline(
    baseline_name: str,
    test_df: pd.DataFrame,
    dataset_root: Path,
    fold_dir: Path,
    output_dir: Path,
    command_template: Optional[str] = None,
    backend_cmd: Optional[str] = None,
    predictions_path: Optional[str] = None,
    required_env: Optional[list[str]] = None,
    timeout_sec: int = 86400,
    baseline_workers: int = 1,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)

    test_manifest_path = output_dir / "test_manifest.csv"
    test_df[["pair_id", "source_col_norm", "target_col_norm", "label"]].drop_duplicates().to_csv(test_manifest_path, index=False)
    pair_manifest_path = _write_pair_manifest(dataset_root=dataset_root, test_df=test_df, out_path=output_dir / "pair_manifest.jsonl")

    predicted_path = Path(predictions_path) if predictions_path else (output_dir / "predictions.csv")

    runtime = {
        "baseline": baseline_name,
        "command_executed": None,
        "return_code": None,
        "wall_time_sec": 0.0,
        "stderr": "",
        "stdout": "",
    }

    required_env = required_env or []
    missing_env = [name for name in required_env if not os.environ.get(name)]
    if missing_env:
        out = test_df.copy()
        out["score"] = 0.0
        runtime["status"] = "missing_env"
        runtime["stderr"] = f"Missing required environment variables: {', '.join(missing_env)}"
        return out, runtime

    # Compatibility aliases executed in-process with Valentine.
    if baseline_name in VALENTINE_COMPAT_ALIASES and command_template is None:
        mapped = VALENTINE_COMPAT_ALIASES[baseline_name]
        t0 = time.perf_counter()
        out = run_valentine_baseline(
            method=mapped,
            test_df=test_df,
            dataset_root=dataset_root,
            max_workers=min(max(baseline_workers, 1), 128),
        )
        runtime["wall_time_sec"] = time.perf_counter() - t0
        runtime["status"] = "ok"
        runtime["mapped_method"] = mapped
        return out.predictions, runtime

    if command_template:
        cmd = command_template.format(
            fold_dir=str(fold_dir),
            output_dir=str(output_dir),
            test_manifest=str(test_manifest_path),
            pair_manifest=str(pair_manifest_path),
            baseline_name=str(baseline_name),
            predictions=str(predicted_path),
        )
        runtime["command_executed"] = cmd

        t0 = time.perf_counter()
        proc = subprocess.run(
            cmd,
            shell=True,
            text=True,
            capture_output=True,
            timeout=timeout_sec,
        )
        runtime["wall_time_sec"] = time.perf_counter() - t0
        runtime["return_code"] = proc.returncode
        runtime["stdout"] = proc.stdout[-4000:]
        runtime["stderr"] = proc.stderr[-4000:]

        if proc.returncode != 0:
            out = test_df.copy()
            out["score"] = 0.0
            runtime["status"] = "command_failed"
            return out, runtime
    elif backend_cmd:
        # Standard bridge for external baselines requiring their own runner.
        cmd = backend_cmd.format(
            fold_dir=str(fold_dir),
            output_dir=str(output_dir),
            test_manifest=str(test_manifest_path),
            pair_manifest=str(pair_manifest_path),
            baseline_name=str(baseline_name),
            predictions=str(predicted_path),
        )
        runtime["command_executed"] = cmd

        t0 = time.perf_counter()
        proc = subprocess.run(
            cmd,
            shell=True,
            text=True,
            capture_output=True,
            timeout=timeout_sec,
        )
        runtime["wall_time_sec"] = time.perf_counter() - t0
        runtime["return_code"] = proc.returncode
        runtime["stdout"] = proc.stdout[-4000:]
        runtime["stderr"] = proc.stderr[-4000:]

        if proc.returncode != 0:
            out = test_df.copy()
            out["score"] = 0.0
            runtime["status"] = "command_failed"
            return out, runtime
    else:
        out = test_df.copy()
        out["score"] = 0.0
        runtime["status"] = "missing_backend"
        runtime["stderr"] = (
            "No command_template/backend_cmd provided for external baseline. "
            f"Provide config for method={baseline_name}."
        )
        return out, runtime

    if not predicted_path.exists():
        out = test_df.copy()
        out["score"] = 0.0
        runtime["status"] = "missing_predictions"
        return out, runtime

    pred_df = _load_predictions(predicted_path)
    out = _merge_scores(test_df, pred_df)
    runtime["status"] = "ok"
    return out, runtime
