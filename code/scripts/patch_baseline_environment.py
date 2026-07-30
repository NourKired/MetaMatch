#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Patch experiment.json/tasks.jsonl to remove stale cluster baseline environments.")
    p.add_argument("--output-root", type=Path, default=Path("outputs/exp_occidata"))
    p.add_argument("--baseline-config", type=Path, default=Path("configs/baselines.yaml"))
    p.add_argument("--dataset-root", type=Path, default=None)
    p.add_argument("--backup", action="store_true", help="Write .bak copies before patching.")
    return p.parse_args()


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    exp_path = output_root / "experiment.json"
    tasks_path = output_root / "tasks.jsonl"

    if not exp_path.exists():
        raise FileNotFoundError(exp_path)
    if not tasks_path.exists():
        raise FileNotFoundError(tasks_path)
    if not args.baseline_config.exists():
        raise FileNotFoundError(args.baseline_config)

    cfg = yaml.safe_load(args.baseline_config.read_text(encoding="utf-8")) or {}
    external_cfg = cfg.get("external", {}) if isinstance(cfg, dict) else {}

    if args.backup:
        exp_path.with_suffix(".json.bak").write_text(exp_path.read_text(encoding="utf-8"), encoding="utf-8")
        tasks_path.with_suffix(".jsonl.bak").write_text(tasks_path.read_text(encoding="utf-8"), encoding="utf-8")

    exp = json.loads(exp_path.read_text(encoding="utf-8"))
    if args.dataset_root is not None:
        exp["dataset_root"] = str(args.dataset_root.resolve())
    exp["output_root"] = str(output_root.resolve())
    exp["tasks_path"] = str(tasks_path.resolve())
    meta_path = output_root / "metaspace.parquet"
    if meta_path.exists():
        exp["metaspace_path"] = str(meta_path.resolve())
    exp_path.write_text(json.dumps(exp, indent=2), encoding="utf-8")

    rows = read_jsonl(tasks_path)
    patched = 0
    for row in rows:
        if row.get("family") != "external":
            continue
        method = str(row.get("method", ""))
        conf = external_cfg.get(method)
        if not isinstance(conf, dict):
            continue
        replacements = {
            "command_template": conf.get("command_template"),
            "backend_cmd": conf.get("backend_cmd"),
            "predictions_path": conf.get("predictions_path"),
            "required_env": conf.get("required_env", []),
            "timeout_sec": int(conf.get("timeout_sec", 86400)),
        }
        for key, new_val in replacements.items():
            if row.get(key) != new_val:
                row[key] = new_val
                patched += 1

    write_jsonl(tasks_path, rows)
    print(f"patched_tasks_fields={patched}")
    print(f"experiment={exp_path}")
    print(f"tasks={tasks_path}")
    if args.dataset_root is not None:
        print(f"dataset_root={args.dataset_root.resolve()}")


if __name__ == "__main__":
    main()
