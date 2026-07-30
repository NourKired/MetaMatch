#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run baselines + MetaMatch several times per fold/method in isolated output roots.")
    p.add_argument("--source-output-root", type=Path, default=Path("outputs/exp_occidata"))
    p.add_argument("--repeats-root", type=Path, default=Path("outputs/exp_occidata_repeats"))
    p.add_argument("--n-repeats", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", choices=["rf", "logreg"], default="rf")
    p.add_argument("--task-family", choices=["all", "meta", "valentine", "external"], default="all")
    p.add_argument("--max-parallel", type=int, default=8)
    p.add_argument("--baseline-workers", type=int, default=1)
    p.add_argument("--python", type=Path, default=Path(".venv/bin/python3"))
    p.add_argument("--dataset-root-override", type=Path, default=None)
    p.add_argument("--methods", type=str, default="", help="Comma-separated method names to run.")
    p.add_argument("--skip-completed", action="store_true", help="Resume by skipping tasks with predictions.parquet and metrics.json.")
    return p.parse_args()


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    if src.is_dir():
        os.symlink(src.resolve(), dst, target_is_directory=True)
    else:
        os.symlink(src.resolve(), dst)


def prepare_repeat_root(source: Path, repeat_root: Path, dataset_root_override: Path | None = None) -> None:
    repeat_root.mkdir(parents=True, exist_ok=True)
    for name in ["tasks.jsonl", "folds", "metaspace.parquet"]:
        src = source / name
        if src.exists():
            link_or_copy(src, repeat_root / name)
    exp_src = source / "experiment.json"
    exp_dst = repeat_root / "experiment.json"
    if exp_src.exists() and not exp_dst.exists():
        if dataset_root_override is None:
            link_or_copy(exp_src, exp_dst)
        else:
            payload = json.loads(exp_src.read_text(encoding="utf-8"))
            payload["dataset_root"] = str(dataset_root_override.resolve())
            exp_dst.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    # Keep reports/results local to the repeat root.
    (repeat_root / "reports").mkdir(exist_ok=True)


def run(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}. See {log_path}")


def main() -> None:
    args = parse_args()
    source = args.source_output_root.resolve()
    repeats_root = args.repeats_root.resolve()
    py = str(args.python.resolve() if args.python.exists() else args.python)
    repeats_root.mkdir(parents=True, exist_ok=True)

    for i in range(args.n_repeats):
        run_id = f"run_{i + 1:02d}"
        repeat_root = repeats_root / run_id
        seed = args.seed + i
        prepare_repeat_root(source, repeat_root, args.dataset_root_override)
        log = repeat_root / "repeat.log"
        print(f"[{i + 1}/{args.n_repeats}] {run_id} seed={seed} -> {repeat_root}", flush=True)

        run_all_cmd = [
            py,
            "scripts/run_all_local.py",
            "--output-root",
            str(repeat_root),
            "--task-family",
            args.task_family,
            "--model",
            args.model,
            "--seed",
            str(seed),
            "--max-parallel",
            str(args.max_parallel),
            "--baseline-workers",
            str(args.baseline_workers),
            "--methods",
            args.methods,
        ]
        if args.skip_completed:
            run_all_cmd.append("--skip-completed")
        run(run_all_cmd, log)
        run([py, "scripts/aggregate_results.py", "--output-root", str(repeat_root)], log)
        run([py, "scripts/export_fold_pair_results.py", "--output-root", str(repeat_root), "--expected-pairs", "551"], log)
        selected_methods = {m.strip() for m in args.methods.split(",") if m.strip()}
        should_compute_meta = args.task_family in {"all", "meta"} and (
            not selected_methods or "meta_space" in selected_methods
        )
        if should_compute_meta:
            run([py, "scripts/compute_meta_space_opt_train_threshold.py", "--output-root", str(repeat_root), "--model", args.model, "--seed", str(seed)], log)

    print(f"Done repeats: {repeats_root}")


if __name__ == "__main__":
    main()
