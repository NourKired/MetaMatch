#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pandas as pd


def copy_smutf_source(src: Path, work_parent: Path) -> Path:
    pkg = work_parent / "schema_matching"
    if pkg.exists():
        shutil.rmtree(pkg)
    ignore = shutil.ignore_patterns(
        ".git",
        "__pycache__",
        ".venv*",
        "model",
        "Input",
        "Test Data",
        "*.zip",
    )
    shutil.copytree(src, pkg, ignore=ignore)
    cached_model = Path("/Users/nahawandkired/.cache/torch/sentence_transformers/sentence-transformers_paraphrase-multilingual-mpnet-base-v2")
    if cached_model.exists():
        init_py = pkg / "init.py"
        init_py.write_text(
            "from sentence_transformers import SentenceTransformer\n"
            "print(\"schema_matching|Loading local sentence transformer cache...\")\n"
            f"model = SentenceTransformer({str(cached_model)!r})\n"
            "print(\"schema_matching|Done loading sentence transformer\")\n",
            encoding="utf-8",
        )
    return pkg


def run_step(cmd: list[str], cwd: Path, env: dict[str, str], log_path: Path) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        rc = proc.wait()
    elapsed = time.perf_counter() - start
    if rc != 0:
        raise RuntimeError(f"Command failed with code {rc}: {' '.join(cmd)}")
    return elapsed


def write_pair_for_application(pair_manifest: Path, test_manifest: Path, out_dir: Path, pair_rank: str, sample_rows: int) -> dict:
    manifest = {}
    with pair_manifest.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                manifest[str(row["pair_id"])] = row

    test_df = pd.read_csv(test_manifest)
    sizes = test_df.groupby("pair_id").size().sort_values()
    if pair_rank == "small":
        pair_id = str(sizes.index[0])
    elif pair_rank == "large":
        pair_id = str(sizes.index[-1])
    else:
        pair_id = str(sizes.index[len(sizes) // 2])

    meta = manifest[pair_id]
    src = pd.read_csv(meta["source_csv"], nrows=sample_rows, low_memory=False)
    tgt = pd.read_csv(meta["target_csv"], nrows=sample_rows, low_memory=False)

    pair_dir = out_dir / f"application_{pair_rank}_{pair_id.replace('/', '_').replace(':', '_')}"
    if pair_dir.exists():
        shutil.rmtree(pair_dir)
    pair_dir.mkdir(parents=True, exist_ok=True)
    src.to_csv(pair_dir / "Table1.csv", index=False)
    tgt.to_csv(pair_dir / "Table2.csv", index=False)
    return {
        "pair_id": pair_id,
        "pair_dir": pair_dir,
        "n_src_cols": int(src.shape[1]),
        "n_tgt_cols": int(tgt.shape[1]),
        "candidate_pairs": int(src.shape[1] * tgt.shape[1]),
        "sample_rows": int(sample_rows),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smutf-root", type=Path, default=Path("/Users/nahawandkired/Documents/Recherche/baselines/SMUTF-official"))
    ap.add_argument("--python", type=Path, default=Path("/Users/nahawandkired/Documents/Recherche/baselines/SMUTF-official/.venv_paper/bin/python3"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/exp_occidata/reports/meeting_baselines_vs_metamatch/smutf_real_benchmark_complet_complet"))
    ap.add_argument("--results-root", type=Path, default=Path("outputs/exp_occidata/results"))
    ap.add_argument("--fold-id", type=int, default=0)
    ap.add_argument("--pair-rank", choices=["small", "median", "large"], default="median")
    ap.add_argument("--sample-rows", type=int, default=5000)
    ap.add_argument("--skip-finetune", action="store_true")
    args = ap.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    work_parent = out_dir / "work"
    work_parent.mkdir(parents=True, exist_ok=True)
    pkg = copy_smutf_source(args.smutf_root, work_parent)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(work_parent)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    rows = []
    if not args.skip_finetune:
        feature_sec = run_step(
            [str(args.python), "-u", "-m", "schema_matching.relation_features"],
            cwd=pkg,
            env=env,
            log_path=out_dir / "logs" / "01_relation_features.log",
        )
        train_sec = run_step(
            [str(args.python), "-u", "-m", "schema_matching.train"],
            cwd=pkg,
            env=env,
            log_path=out_dir / "logs" / "02_train.log",
        )
    else:
        feature_sec = 0.0
        train_sec = 0.0

    model_dirs = sorted((pkg / "model").glob("*"), key=lambda p: p.stat().st_mtime)
    model_dir = model_dirs[-1] if model_dirs else args.smutf_root / "model" / "2022-04-12-12-06-32"

    pair_info = write_pair_for_application(
        pair_manifest=args.results_root / f"fold_{args.fold_id}" / "SMUTF" / "pair_manifest.jsonl",
        test_manifest=args.results_root / f"fold_{args.fold_id}" / "SMUTF" / "test_manifest.csv",
        out_dir=out_dir,
        pair_rank=args.pair_rank,
        sample_rows=args.sample_rows,
    )
    app_sec = run_step(
        [
            str(args.python),
            "-u",
            "-m",
            "schema_matching.cal_column_similarity",
            "-p",
            str(pair_info["pair_dir"]),
            "-m",
            str(model_dir),
            "-s",
            "many-to-many",
        ],
        cwd=pkg,
        env=env,
        log_path=out_dir / "logs" / "03_application.log",
    )

    rows.append(
        {
            "benchmark": "smutf_complet_complet_real",
            "feature_extraction_train_data_sec": feature_sec,
            "xgboost_train_sec": train_sec,
            "finetune_total_sec": feature_sec + train_sec,
            "application_one_pair_sec": app_sec,
            "application_pair_rank": args.pair_rank,
            "application_pair_id": pair_info["pair_id"],
            "application_candidate_pairs": pair_info["candidate_pairs"],
            "application_sample_rows": pair_info["sample_rows"],
            "application_sec_per_candidate": app_sec / max(1, pair_info["candidate_pairs"]),
            "model_dir": str(model_dir),
            "work_package_dir": str(pkg),
        }
    )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "smutf_real_benchmark_summary.csv", index=False)
    (out_dir / "smutf_real_benchmark_summary.json").write_text(
        json.dumps(rows[0], indent=2),
        encoding="utf-8",
    )
    print("\n=== SMUTF real benchmark summary ===")
    print(df.to_string(index=False))
    print(f"\nSaved: {out_dir}")


if __name__ == "__main__":
    main()
