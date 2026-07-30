#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_metaspace_61_feature_runtime import (  # noqa: E402
    _build_column_text,
    _get_encoder,
    _load_df,
    _normalize_name,
    choose_representatives,
    discover_table_pairs,
    read_pair_size,
)
from osirim_occidata.meta_features.classical import compute_classical_features  # noqa: E402
from osirim_occidata.meta_features.syntax import (  # noqa: E402
    char_ngrams,
    common_suffix_len,
    cosine_counts,
    jaccard,
    jaro_winkler,
    lcs_length,
    normalize_text,
    tokenize,
)
from ripser import ripser  # noqa: E402


COMPACT_FEATURES = [
    "syn_cosine_bigrams",
    "syn_jaccard_trigrams",
    "syn_jaccard_tokens",
    "syn_jaro_winkler",
    "cls_euclidean",
    "syn_lcs_ratio",
    "cls_chebyshev",
    "syn_len_b",
    "syn_common_suffix_ratio",
    "tda_h0_entropy_combined",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, default=Path("valentine"))
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "outputs/exp_occidata/reports/meeting_baselines_vs_metamatch/"
            "metaspace_compact10_runtime_benchmark"
        ),
    )
    p.add_argument("--sample-rows", type=int, default=5000)
    p.add_argument("--n-bins", type=int, default=10)
    p.add_argument("--extra-largest", type=int, default=2)
    p.add_argument("--max-pairs-to-time", type=int, default=12)
    p.add_argument("--max-values-text", type=int, default=20)
    p.add_argument("--max-chars-per-value", type=int, default=80)
    p.add_argument("--max-total-chars-text", type=int, default=2000)
    p.add_argument("--embedding-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--embedding-device", type=str, default="cpu")
    p.add_argument("--embedding-batch-size", type=int, default=32)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--disable-transformer-embeddings", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def h0_entropy_combined(tok_a: np.ndarray, tok_b: np.ndarray) -> float:
    if tok_a is None or tok_b is None or tok_a.size == 0 or tok_b.size == 0:
        return 0.0
    points = np.vstack([tok_a, tok_b])
    if len(points) < 2:
        return 0.0
    h0 = ripser(points, maxdim=0, distance_matrix=False)["dgms"][0]
    finite = h0[np.isfinite(h0[:, 1])]
    if finite.size == 0:
        return 0.0
    lifetimes = finite[:, 1] - finite[:, 0]
    lifetimes = lifetimes[lifetimes > 0]
    total = lifetimes.sum()
    if total <= 0:
        return 0.0
    probs = lifetimes / total
    return float(-(probs * np.log(probs)).sum())


def compute_compact_syntax(a: str, b: str) -> Dict[str, float]:
    raw_a, raw_b = a or "", b or ""
    na, nb = normalize_text(raw_a), normalize_text(raw_b)
    maxlen = max(len(na), len(nb), 1)
    tok_a, tok_b = tokenize(na), tokenize(nb)
    big_a, big_b = char_ngrams(na, 2), char_ngrams(nb, 2)
    tri_a, tri_b = char_ngrams(na, 3), char_ngrams(nb, 3)
    return {
        "syn_cosine_bigrams": cosine_counts(big_a, big_b),
        "syn_jaccard_trigrams": jaccard(tri_a, tri_b),
        "syn_jaccard_tokens": jaccard(tok_a, tok_b),
        "syn_jaro_winkler": jaro_winkler(na, nb),
        "syn_lcs_ratio": lcs_length(na, nb) / maxlen,
        "syn_len_b": float(len(nb)),
        "syn_common_suffix_ratio": common_suffix_len(na, nb) / maxlen,
    }


def build_pair_compact_features(pair, args: argparse.Namespace) -> pd.DataFrame:
    t0 = time.perf_counter()
    src_df = _load_df(pair.source_csv, sample_rows=args.sample_rows)
    tgt_df = _load_df(pair.target_csv, sample_rows=args.sample_rows)

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

    src_desc: Dict[str, Dict[str, object]] = {}
    for i, col in enumerate(src_cols):
        src_desc[col] = {
            "norm": _normalize_name(col),
            "text": src_texts[i],
            "vec": all_vectors[i],
            "tok": all_token_mats[i],
        }
    tgt_desc: Dict[str, Dict[str, object]] = {}
    offset = len(src_cols)
    for j, col in enumerate(tgt_cols):
        idx = offset + j
        tgt_desc[col] = {
            "norm": _normalize_name(col),
            "text": tgt_texts[j],
            "vec": all_vectors[idx],
            "tok": all_token_mats[idx],
        }

    rows: List[Dict[str, object]] = []
    for src_col in src_cols:
        s = src_desc[src_col]
        for tgt_col in tgt_cols:
            t = tgt_desc[tgt_col]
            feats = {}
            feats.update(compute_compact_syntax(str(s["text"]), str(t["text"])))
            feats.update(compute_classical_features(s["vec"], t["vec"]))
            feats["tda_h0_entropy_combined"] = h0_entropy_combined(s["tok"], t["tok"])
            row = {"pair_id": pair.pair_id, "source_column": src_col, "target_column": tgt_col}
            for feat in COMPACT_FEATURES:
                row[feat] = float(feats.get(feat, 0.0))
            rows.append(row)

    out = pd.DataFrame(rows)
    out["feature_build_sec_pair_compact10"] = float(time.perf_counter() - t0)
    return out


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = discover_table_pairs(args.dataset_root)
    pair_by_id = {p.pair_id: p for p in pairs}
    size_path = out_dir / "all_pair_size_grid.csv"
    if args.resume and size_path.exists():
        size_df = pd.read_csv(size_path)
    else:
        size_df = pd.DataFrame([read_pair_size(p, args.sample_rows) for p in pairs])
        size_df = size_df.sort_values("candidate_pairs").reset_index(drop=True)
        size_df = choose_representatives(size_df, args.n_bins, args.extra_largest, args.max_pairs_to_time)
        size_df.to_csv(size_path, index=False)

    timing_path = out_dir / "representative_pair_timings_compact10.csv"
    timed = pd.read_csv(timing_path) if args.resume and timing_path.exists() else pd.DataFrame()
    done = set(timed["pair_id"].astype(str)) if len(timed) and "pair_id" in timed else set()
    rows = timed.to_dict("records") if len(timed) else []

    selected = size_df[size_df["is_timed_representative"]].copy()
    for _, row in selected.iterrows():
        pair_id = str(row["pair_id"])
        if pair_id in done:
            continue
        pair = pair_by_id[pair_id]
        t_outer = time.perf_counter()
        feat_df = build_pair_compact_features(pair, args)
        measured = float(feat_df["feature_build_sec_pair_compact10"].iloc[0])
        payload = {
            "pair_id": pair_id,
            "dataset": row["dataset"],
            "relation_type": row["relation_type"],
            "size_bin": int(row["size_bin"]),
            "n_src_cols": int(row["n_src_cols"]),
            "n_tgt_cols": int(row["n_tgt_cols"]),
            "candidate_pairs": int(row["candidate_pairs"]),
            "n_rows_out": int(len(feat_df)),
            "n_features": len(COMPACT_FEATURES),
            "feature_build_sec_pair_compact10": measured,
            "outer_elapsed_sec": float(time.perf_counter() - t_outer),
            "sec_per_candidate": measured / max(int(row["candidate_pairs"]), 1),
        }
        rows.append(payload)
        pd.DataFrame(rows).to_csv(timing_path, index=False)
        print(f"timed {pair_id}: {measured:.2f}s")

    reps = pd.DataFrame(rows)
    all_df = size_df.copy()
    all_df["assigned_rep_pair_id"] = None
    all_df["estimated_feature_build_sec_pair_compact10"] = np.nan
    all_df["estimated_sec_per_candidate"] = np.nan
    for idx, row in all_df.iterrows():
        nearest = reps.iloc[(reps["candidate_pairs"] - row["candidate_pairs"]).abs().argmin()]
        all_df.at[idx, "assigned_rep_pair_id"] = nearest["pair_id"]
        all_df.at[idx, "estimated_sec_per_candidate"] = float(nearest["sec_per_candidate"])
        all_df.at[idx, "estimated_feature_build_sec_pair_compact10"] = (
            float(row["candidate_pairs"]) * float(nearest["sec_per_candidate"])
        )
    all_df.to_csv(out_dir / "all_pair_runtime_estimate_compact10.csv", index=False)

    summary = {
        "n_total_pairs": int(all_df["pair_id"].nunique()),
        "n_timed_representatives": int(len(reps)),
        "n_features_measured": len(COMPACT_FEATURES),
        "feature_definition": "compact10 selected features with optimized h0 entropy combined only",
        "estimated_total_feature_build_sec_551": float(all_df["estimated_feature_build_sec_pair_compact10"].sum()),
        "estimated_mean_feature_build_sec_pair_compact10": float(all_df["estimated_feature_build_sec_pair_compact10"].mean()),
        "estimated_median_feature_build_sec_pair_compact10": float(all_df["estimated_feature_build_sec_pair_compact10"].median()),
        "measured_min_sec": float(reps["feature_build_sec_pair_compact10"].min()),
        "measured_max_sec": float(reps["feature_build_sec_pair_compact10"].max()),
        "measured_mean_sec": float(reps["feature_build_sec_pair_compact10"].mean()),
    }
    (out_dir / "runtime_estimate_summary_compact10.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
