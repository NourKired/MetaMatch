#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metamatch.catalog import discover_table_pairs, load_mapping  # noqa: E402
from metamatch.features import (  # noqa: E402
    _build_column_text,
    _get_encoder,
    _get_topological_api,
    _load_df,
    _normalize_name,
)
from metamatch.meta_features.classical import CLASSICAL_FEATURES, compute_classical_features  # noqa: E402
from metamatch.meta_features.syntax import SYNTAX_FEATURES, compute_syntax_features  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Benchmark MetaMatch 60-feature construction on representative table pairs "
            "and extrapolate the total runtime over all pairs."
        )
    )
    p.add_argument("--dataset-root", type=Path, default=Path("valentine"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/exp_occidata/reports/meeting_baselines_vs_metamatch/metamatch_60_runtime_benchmark"))
    p.add_argument("--sample-rows", type=int, default=5000)
    p.add_argument("--max-values-text", type=int, default=20)
    p.add_argument("--max-chars-per-value", type=int, default=80)
    p.add_argument("--max-total-chars-text", type=int, default=2000)
    p.add_argument("--embedding-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--embedding-device", type=str, default="cpu")
    p.add_argument("--embedding-batch-size", type=int, default=32)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--disable-transformer-embeddings", action="store_true")
    p.add_argument("--n-bins", type=int, default=10)
    p.add_argument("--extra-largest", type=int, default=3)
    p.add_argument("--max-pairs-to-time", type=int, default=15)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def read_pair_size(pair, sample_rows: int) -> Dict[str, object]:
    src = _load_df(pair.source_csv, sample_rows=sample_rows)
    tgt = _load_df(pair.target_csv, sample_rows=sample_rows)
    return {
        "pair_id": pair.pair_id,
        "dataset": pair.dataset,
        "relation_type": pair.relation_type,
        "source_csv": str(pair.source_csv),
        "target_csv": str(pair.target_csv),
        "mapping_json": str(pair.mapping_json),
        "n_src_cols": int(len(src.columns)),
        "n_tgt_cols": int(len(tgt.columns)),
        "n_src_rows_sampled": int(len(src)),
        "n_tgt_rows_sampled": int(len(tgt)),
        "candidate_pairs": int(len(src.columns) * len(tgt.columns)),
    }


def choose_representatives(size_df: pd.DataFrame, n_bins: int, extra_largest: int, max_pairs: int) -> pd.DataFrame:
    out = size_df.copy()
    out["log_candidate_pairs"] = np.log10(out["candidate_pairs"].clip(lower=1))
    n_unique = out["candidate_pairs"].nunique()
    q = min(max(1, n_bins), n_unique)
    out["size_bin"] = pd.qcut(out["candidate_pairs"].rank(method="first"), q=q, labels=False, duplicates="drop")

    selected_idx = set()
    for _, g in out.groupby("size_bin", sort=True):
        med = float(g["candidate_pairs"].median())
        idx = (g["candidate_pairs"] - med).abs().sort_values().index[0]
        selected_idx.add(int(idx))

    selected_idx.update(out.nsmallest(1, "candidate_pairs").index.astype(int).tolist())
    selected_idx.update(out.nlargest(extra_largest, "candidate_pairs").index.astype(int).tolist())

    selected = out.loc[sorted(selected_idx)].copy()
    if len(selected) > max_pairs:
        keep = set(selected.nlargest(extra_largest, "candidate_pairs").index.astype(int).tolist())
        keep.update(selected.nsmallest(1, "candidate_pairs").index.astype(int).tolist())
        rest = selected.loc[[i for i in selected.index if i not in keep]].copy()
        remaining = max(0, max_pairs - len(keep))
        if remaining:
            keep.update(rest.sort_values(["size_bin", "candidate_pairs"]).head(remaining).index.astype(int).tolist())
        selected = selected.loc[sorted(keep)].copy()

    out["is_timed_representative"] = out.index.isin(selected.index)
    return out


def build_pair_60_features(pair, args: argparse.Namespace) -> pd.DataFrame:
    t0 = time.perf_counter()
    src_df = _load_df(pair.source_csv, sample_rows=args.sample_rows)
    tgt_df = _load_df(pair.target_csv, sample_rows=args.sample_rows)

    gt = load_mapping(pair.mapping_json)
    gt_pairs = {(m.source_column.strip().lower(), m.target_column.strip().lower()) for m in gt}

    src_cols = list(src_df.columns)
    tgt_cols = list(tgt_df.columns)

    src_texts = [
        _build_column_text(c, src_df[c], args.max_values_text, args.max_chars_per_value, args.max_total_chars_text)
        for c in src_cols
    ]
    tgt_texts = [
        _build_column_text(c, tgt_df[c], args.max_values_text, args.max_chars_per_value, args.max_total_chars_text)
        for c in tgt_cols
    ]

    encoder = _get_encoder(
        model_name=args.embedding_model,
        device=args.embedding_device,
        batch_size=args.embedding_batch_size,
        max_tokens=args.max_tokens,
        use_transformer_embeddings=not args.disable_transformer_embeddings,
        fallback_dim=384,
    )
    all_vectors, all_token_mats = encoder.encode_texts(src_texts + tgt_texts)

    src_desc = {}
    for i, col in enumerate(src_cols):
        src_desc[col] = {
            "norm": _normalize_name(col),
            "text": src_texts[i],
            "vec": all_vectors[i],
            "tok": all_token_mats[i],
        }

    tgt_desc = {}
    offset = len(src_cols)
    for j, col in enumerate(tgt_cols):
        idx = offset + j
        tgt_desc[col] = {
            "norm": _normalize_name(col),
            "text": tgt_texts[j],
            "vec": all_vectors[idx],
            "tok": all_token_mats[idx],
        }

    topo_features, check_tda_available, compute_topological = _get_topological_api()
    if not check_tda_available():
        raise RuntimeError("TDA dependencies unavailable: install ripser, persim, gudhi.")

    selected_tda_features = [f for f in topo_features if "overlap" not in f.lower()]
    feature_cols = list(SYNTAX_FEATURES) + list(CLASSICAL_FEATURES) + selected_tda_features
    if len(feature_cols) != 61:
        raise RuntimeError(f"Expected exactly 61 feature columns, got {len(feature_cols)}")

    tda_diagram_cache = {}
    tda_entity_cache = {}
    tda_disk_cache_dir = str(Path.home() / ".cache" / "osirim_tda_runtime_61")

    rows: List[Dict[str, object]] = []
    for src_col in src_cols:
        s = src_desc[src_col]
        for tgt_col in tgt_cols:
            t = tgt_desc[tgt_col]
            feats: Dict[str, float] = {}
            feats.update(compute_syntax_features(s["text"], t["text"]))
            feats.update(compute_classical_features(s["vec"], t["vec"]))
            feats.update(
                compute_topological(
                    s["tok"],
                    t["tok"],
                    diagram_cache=tda_diagram_cache,
                    entity_cache=tda_entity_cache,
                    disk_cache_dir=tda_disk_cache_dir,
                    src_key=f"{pair.pair_id}::{s['norm']}",
                    tgt_key=f"{pair.pair_id}::{t['norm']}",
                )
            )

            row = {
                "pair_id": pair.pair_id,
                "source_column": src_col,
                "target_column": tgt_col,
                "label": 1 if (s["norm"], t["norm"]) in gt_pairs else 0,
            }
            for col in feature_cols:
                row[col] = float(feats.get(col, 0.0))
            rows.append(row)

    df = pd.DataFrame(rows)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["feature_build_sec_pair_61"] = float(time.perf_counter() - t0)
    return df


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import sentence_transformers  # noqa: F401

        transformer_embeddings_available = True
    except Exception:
        transformer_embeddings_available = False

    pairs = discover_table_pairs(args.dataset_root)
    pair_by_id = {p.pair_id: p for p in pairs}
    if not pairs:
        raise RuntimeError(f"No table pairs found under {args.dataset_root}")

    size_path = out_dir / "all_pair_size_grid.csv"
    if args.resume and size_path.exists():
        size_df = pd.read_csv(size_path)
    else:
        rows = [read_pair_size(pair, args.sample_rows) for pair in pairs]
        size_df = pd.DataFrame(rows).sort_values("candidate_pairs").reset_index(drop=True)
        size_df = choose_representatives(size_df, args.n_bins, args.extra_largest, args.max_pairs_to_time)
        size_df.to_csv(size_path, index=False)

    selected = size_df[size_df["is_timed_representative"]].copy()
    timing_path = out_dir / "representative_pair_timings_60features.csv"
    timed = pd.read_csv(timing_path) if args.resume and timing_path.exists() else pd.DataFrame()
    done = set(timed["pair_id"].astype(str)) if len(timed) and "pair_id" in timed else set()

    timing_rows = timed.to_dict("records") if len(timed) else []
    selected = selected.drop_duplicates("pair_id").reset_index(drop=True)
    for i, row in selected.iterrows():
        pair_id = str(row["pair_id"])
        if pair_id in done:
            print(f"[skip] {i + 1}/{len(selected)} {pair_id}", flush=True)
            continue
        print(
            f"[time] {i + 1}/{len(selected)} p={int(row['candidate_pairs'])} "
            f"{pair_id}",
            flush=True,
        )
        pair = pair_by_id[pair_id]
        t0 = time.perf_counter()
        feat_df = build_pair_60_features(pair, args)
        elapsed = time.perf_counter() - t0
        measured = float(feat_df["feature_build_sec_pair_61"].iloc[0])
        timing_rows.append(
            {
                "pair_id": pair_id,
                "dataset": row["dataset"],
                "relation_type": row["relation_type"],
                "size_bin": int(row["size_bin"]),
                "n_src_cols": int(row["n_src_cols"]),
                "n_tgt_cols": int(row["n_tgt_cols"]),
                "candidate_pairs": int(row["candidate_pairs"]),
                "n_rows_out": int(len(feat_df)),
                "n_features": 61,
                "feature_build_sec_pair_61": measured,
                "outer_elapsed_sec": float(elapsed),
                "sec_per_candidate": measured / max(1, int(row["candidate_pairs"])),
            }
        )
        pd.DataFrame(timing_rows).to_csv(timing_path, index=False)
        done.add(pair_id)
        print(f"checkpoint saved: {timing_path}", flush=True)

    timings = pd.DataFrame(timing_rows)
    if timings.empty:
        raise RuntimeError("No timings available.")

    # If a run was interrupted after re-timing the same pair, keep a stable
    # averaged measurement for that representative instead of double-counting it.
    timings = (
        timings.groupby(["pair_id", "dataset", "relation_type", "size_bin", "n_src_cols", "n_tgt_cols", "candidate_pairs", "n_rows_out", "n_features"], as_index=False)
        .agg(
            feature_build_sec_pair_61=("feature_build_sec_pair_61", "mean"),
            outer_elapsed_sec=("outer_elapsed_sec", "mean"),
            sec_per_candidate=("sec_per_candidate", "mean"),
        )
    )
    timings.to_csv(timing_path, index=False)

    reps = timings.sort_values("candidate_pairs").copy()
    all_df = size_df.copy()
    all_df["assigned_rep_pair_id"] = None
    all_df["estimated_feature_build_sec_pair_61"] = np.nan
    all_df["estimated_sec_per_candidate"] = np.nan
    for idx, row in all_df.iterrows():
        nearest = reps.iloc[(np.log1p(reps["candidate_pairs"]) - math.log1p(float(row["candidate_pairs"]))).abs().argmin()]
        all_df.at[idx, "assigned_rep_pair_id"] = nearest["pair_id"]
        all_df.at[idx, "estimated_sec_per_candidate"] = float(nearest["sec_per_candidate"])
        all_df.at[idx, "estimated_feature_build_sec_pair_61"] = float(row["candidate_pairs"]) * float(nearest["sec_per_candidate"])

    estimate_path = out_dir / "all_pair_runtime_estimate_60features.csv"
    all_df.to_csv(estimate_path, index=False)

    by_rep = (
        all_df.groupby("assigned_rep_pair_id", as_index=False)
        .agg(
            n_pairs_represented=("pair_id", "count"),
            total_candidate_pairs=("candidate_pairs", "sum"),
            estimated_total_sec=("estimated_feature_build_sec_pair_61", "sum"),
            mean_estimated_sec_pair=("estimated_feature_build_sec_pair_61", "mean"),
        )
        .merge(reps[["pair_id", "feature_build_sec_pair_61", "candidate_pairs", "sec_per_candidate"]], left_on="assigned_rep_pair_id", right_on="pair_id", how="left")
        .drop(columns=["pair_id"], errors="ignore")
        .rename(columns={"candidate_pairs": "rep_candidate_pairs"})
    )
    by_rep.to_csv(out_dir / "runtime_extrapolation_by_representative_60features.csv", index=False)

    summary = {
        "n_total_pairs": int(len(all_df)),
        "n_timed_representatives": int(len(reps)),
        "n_features_measured": 61,
        "feature_definition": "SYNTAX_FEATURES + CLASSICAL_FEATURES + TOPOLOGICAL_FEATURES excluding overlap; no spectral features",
        "sample_rows": int(args.sample_rows),
        "embedding_model": args.embedding_model,
        "requested_transformer_embeddings": bool(not args.disable_transformer_embeddings),
        "actual_transformer_embeddings_available": bool(transformer_embeddings_available),
        "actual_embedding_note": (
            "sentence_transformers available; transformer embeddings can be used"
            if transformer_embeddings_available and not args.disable_transformer_embeddings
            else "sentence_transformers unavailable or disabled; benchmark used hashed fallback embeddings"
        ),
        "estimated_total_feature_build_sec_551": float(all_df["estimated_feature_build_sec_pair_61"].sum()),
        "estimated_mean_feature_build_sec_pair_61": float(all_df["estimated_feature_build_sec_pair_61"].mean()),
        "estimated_median_feature_build_sec_pair_61": float(all_df["estimated_feature_build_sec_pair_61"].median()),
        "measured_min_sec": float(reps["feature_build_sec_pair_61"].min()),
        "measured_max_sec": float(reps["feature_build_sec_pair_61"].max()),
        "measured_mean_sec": float(reps["feature_build_sec_pair_61"].mean()),
        "outputs": {
            "all_pair_size_grid": str(size_path),
            "representative_timings": str(timing_path),
            "all_pair_estimate": str(estimate_path),
            "by_representative": str(out_dir / "runtime_extrapolation_by_representative_60features.csv"),
        },
    }
    (out_dir / "runtime_estimate_summary_60features.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
