#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Dict

import pandas as pd


def _norm(s: str) -> str:
    return str(s).strip().lower()


def _read_pair_manifest(path: Path) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[str(row["pair_id"])] = row
    return out


def _call_schema_matching(table1_pth: str, table2_pth: str, model_dir: Path | None = None):
    original_argv = sys.argv[:]
    try:
        # The official SMUTF module parses argparse at import time.
        # Hide this runner's CLI flags while importing it.
        sys.argv = ["cal_column_similarity.py"]
        from schema_matching.cal_column_similarity import schema_matching  # type: ignore
    finally:
        sys.argv = original_argv

    try:
        kwargs = {"strategy": "many-to-many"}
        if model_dir is not None:
            kwargs["model_pth"] = str(model_dir)
        return schema_matching(table1_pth, table2_pth, **kwargs)
    except TypeError:
        try:
            return schema_matching(table1_pth, table2_pth)
        except TypeError:
            # Older variants accept a folder path.
            return schema_matching(str(Path(table1_pth).parent))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Real SMUTF runner")
    p.add_argument("--pair-manifest", required=True, type=Path)
    p.add_argument("--test-manifest", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--sample-rows", type=int, default=5000)
    p.add_argument("--model-dir", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    test_df = pd.read_csv(args.test_manifest)
    test_df["pair_id"] = test_df["pair_id"].astype(str)
    test_df["source_col_norm"] = test_df["source_col_norm"].astype(str).str.strip().str.lower()
    test_df["target_col_norm"] = test_df["target_col_norm"].astype(str).str.strip().str.lower()

    pair_manifest = _read_pair_manifest(args.pair_manifest)
    outputs = []

    for pair_id, group in test_df.groupby("pair_id", sort=False):
        meta = pair_manifest.get(str(pair_id))
        if meta is None:
            for _, r in group.iterrows():
                outputs.append(
                    {
                        "pair_id": str(pair_id),
                        "source_col_norm": r["source_col_norm"],
                        "target_col_norm": r["target_col_norm"],
                        "score": 0.0,
                    }
                )
            continue

        src_df = pd.read_csv(meta["source_csv"], nrows=args.sample_rows, low_memory=False)
        tgt_df = pd.read_csv(meta["target_csv"], nrows=args.sample_rows, low_memory=False)

        with tempfile.TemporaryDirectory(prefix="smutf_pair_") as tmp:
            tmp_path = Path(tmp)
            src_df.to_csv(tmp_path / "Table1.csv", index=False)
            tgt_df.to_csv(tmp_path / "Table2.csv", index=False)

            try:
                df_pred, _, _ = _call_schema_matching(
                    str(tmp_path / "Table1.csv"),
                    str(tmp_path / "Table2.csv"),
                    model_dir=args.model_dir,
                )
            except Exception:
                df_pred = pd.DataFrame()

        score_lookup: Dict[tuple[str, str], float] = {}
        if isinstance(df_pred, pd.DataFrame) and len(df_pred):
            row_keys = {_norm(str(r)): r for r in df_pred.index}
            col_keys = {_norm(str(c)): c for c in df_pred.columns}

            for s_norm, src_idx in row_keys.items():
                for t_norm, tgt_col in col_keys.items():
                    try:
                        score_lookup[(s_norm, t_norm)] = float(df_pred.loc[src_idx, tgt_col])
                    except Exception:
                        continue

            # Some implementations might transpose (rows=tgt, cols=src). Add reverse lookup.
            for t_norm, tgt_idx in row_keys.items():
                for s_norm, src_col in col_keys.items():
                    k = (s_norm, t_norm)
                    if k in score_lookup:
                        continue
                    try:
                        score_lookup[k] = float(df_pred.loc[tgt_idx, src_col])
                    except Exception:
                        continue

        for _, r in group.iterrows():
            s = str(r["source_col_norm"])
            t = str(r["target_col_norm"])
            outputs.append(
                {
                    "pair_id": str(pair_id),
                    "source_col_norm": s,
                    "target_col_norm": t,
                    "score": float(score_lookup.get((s, t), 0.0)),
                }
            )

    out_df = pd.DataFrame(outputs, columns=["pair_id", "source_col_norm", "target_col_norm", "score"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"rows={len(out_df)} out={args.out}")


if __name__ == "__main__":
    main()
