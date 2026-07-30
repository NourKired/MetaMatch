#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from benchmark_metamatch_60_feature_runtime import (  # noqa: E402
    _build_column_text,
    _get_encoder,
    _load_df,
    _normalize_name,
)
from benchmark_metamatch_reduced10_feature_runtime import (  # noqa: E402
    COMPACT_FEATURES,
    compute_reduced_syntax,
    h0_entropy_combined,
)
from metamatch.catalog import discover_table_pairs  # noqa: E402
from metamatch.config import load_baseline_config  # noqa: E402
from metamatch.features import build_full_metamatch  # noqa: E402
from metamatch.io_utils import safe_write_table, write_json, write_jsonl  # noqa: E402
from metamatch.meta_features.classical import compute_classical_features  # noqa: E402
from metamatch.splits import make_repeated_splits_70_30, materialize_fold_frames  # noqa: E402
from metamatch.catalog import load_mapping  # noqa: E402
from run_rq2_rq3_feature_ablation_rf350 import (  # noqa: E402
    all_paper_features,
    best_threshold_pair_f1_train,
    pair_metrics,
)


NON_FEATURE_COLUMNS = {
    "pair_id", "dataset", "relation_type", "category",
    "source_table", "target_table", "source_column", "target_column",
    "source_col_norm", "target_col_norm", "label", "feature_build_sec_pair_real",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="End-to-end MetaMatch run: build embeddings/features, split, train RF, learn threshold on train, test."
    )
    p.add_argument("--config", choices=["full60", "reduced10"], required=True)
    p.add_argument("--dataset-root", type=Path, default=Path("valentine"))
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--sample-rows", type=int, default=5000)
    p.add_argument("--max-values", type=int, default=200)
    p.add_argument("--max-values-text", type=int, default=20)
    p.add_argument("--max-chars-per-value", type=int, default=80)
    p.add_argument("--max-total-chars-text", type=int, default=2000)
    p.add_argument("--embedding-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--embedding-device", type=str, default="cpu")
    p.add_argument("--embedding-batch-size", type=int, default=32)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--disable-transformer-embeddings", action="store_true")
    p.add_argument("--feature-workers", type=int, default=8)
    p.add_argument("--folds", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-estimators", type=int, default=350)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--max-pairs", type=int, default=0, help="0 = all discovered pairs")
    return p.parse_args()


def build_pair_reduced_labeled(pair, args: argparse.Namespace) -> pd.DataFrame:
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
            feats.update(compute_reduced_syntax(str(s["text"]), str(t["text"])))
            feats.update(compute_classical_features(s["vec"], t["vec"]))
            feats["tda_h0_entropy_combined"] = h0_entropy_combined(s["tok"], t["tok"])
            row = {
                "pair_id": pair.pair_id,
                "dataset": pair.dataset,
                "relation_type": pair.relation_type,
                "category": pair.category,
                "source_table": pair.source_table_name,
                "target_table": pair.target_table_name,
                "source_column": src_col,
                "target_column": tgt_col,
                "source_col_norm": s["norm"],
                "target_col_norm": t["norm"],
                "label": 1 if (s["norm"], t["norm"]) in gt_pairs else 0,
            }
            for feat in COMPACT_FEATURES:
                row[feat] = float(feats.get(feat, 0.0))
            rows.append(row)

    out = pd.DataFrame(rows)
    out["feature_build_sec_pair_real"] = float(time.perf_counter() - t0)
    return out


def build_reduced_metamatch(pairs, args: argparse.Namespace) -> pd.DataFrame:
    if args.feature_workers <= 1:
        frames = [build_pair_reduced_labeled(p, args) for p in tqdm(pairs, desc="reduced meta-features")]
    else:
        frames = []
        with ThreadPoolExecutor(max_workers=args.feature_workers) as ex:
            futs = {ex.submit(build_pair_reduced_labeled, p, args): p.pair_id for p in pairs}
            for fut in tqdm(as_completed(futs), total=len(futs), desc="reduced meta-features"):
                frames.append(fut.result())
    df = pd.concat(frames, ignore_index=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


def feature_columns_for_config(df: pd.DataFrame, config: str) -> List[str]:
    if config == "full60":
        return all_paper_features(df)
    return list(COMPACT_FEATURES)


def evaluate_rf(metamatch_df: pd.DataFrame, fold_manifests: List[dict], features: List[str], args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    fold_rows = []
    pair_parts = []
    fit_total = 0.0
    train_score_total = 0.0
    predict_total = 0.0

    for fm in fold_manifests:
        fold_dir = Path(fm["fold_dir"])
        fold_id = int(fold_dir.name.split("_")[-1])
        train = pd.read_parquet(fold_dir / "train.parquet")
        test = pd.read_parquet(fold_dir / "test.parquet")

        x_train = train[features].to_numpy(dtype=np.float32)
        y_train = train["label"].to_numpy(dtype=int)
        x_test = test[features].to_numpy(dtype=np.float32)
        y_test = test["label"].to_numpy(dtype=int)

        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            class_weight="balanced_subsample",
            random_state=args.seed + fold_id,
            n_jobs=args.n_jobs,
        )

        print(f"fold={fold_id} | train={len(train)} | test={len(test)} | features={len(features)}", flush=True)
        t0 = time.perf_counter()
        model.fit(x_train, y_train)
        fit_sec = time.perf_counter() - t0
        fit_total += fit_sec

        t0 = time.perf_counter()
        train_score = model.predict_proba(x_train)[:, 1]
        train_score_sec = time.perf_counter() - t0
        train_score_total += train_score_sec

        train_scored = train[["pair_id", "label"]].copy()
        train_scored["score"] = train_score
        threshold, train_mean_f1_pair = best_threshold_pair_f1_train(train_scored)

        t0 = time.perf_counter()
        test_score = model.predict_proba(x_test)[:, 1]
        predict_sec = time.perf_counter() - t0
        predict_total += predict_sec

        pred = (test_score > threshold).astype(int)
        tmp = test[["pair_id", "label"]].copy()
        tmp["pred"] = pred
        pm = pair_metrics(tmp)
        pm["fold_id"] = fold_id
        pm["config"] = args.config
        pm["classifier"] = "rf"
        pair_parts.append(pm)

        fold_rows.append(
            {
                "config": args.config,
                "fold_id": fold_id,
                "n_features": len(features),
                "threshold_opt_pairf1_train": threshold,
                "train_mean_f1_pair_at_threshold": train_mean_f1_pair,
                "fit_sec": fit_sec,
                "train_score_sec": train_score_sec,
                "predict_sec": predict_sec,
                "precision": float(precision_score(y_test, pred, zero_division=0)),
                "recall": float(recall_score(y_test, pred, zero_division=0)),
                "f1": float(f1_score(y_test, pred, zero_division=0)),
            }
        )

    pair_df = pd.concat(pair_parts, ignore_index=True)
    fold_df = pd.DataFrame(fold_rows)
    timing = {
        "rf_fit_sec_total": fit_total,
        "rf_train_score_sec_total": train_score_total,
        "rf_predict_sec_total": predict_total,
        "rf_train_test_sec_total": fit_total + train_score_total + predict_total,
    }
    return fold_df, pair_df, timing


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t_total = time.perf_counter()

    pairs = discover_table_pairs(args.dataset_root)
    if args.max_pairs > 0:
        pairs = pairs[: args.max_pairs]
    if not pairs:
        raise RuntimeError(f"No table pairs found in {args.dataset_root}")

    t_features = time.perf_counter()
    if args.config == "full60":
        metamatch_df = build_full_metamatch(
            pairs,
            sample_rows=args.sample_rows,
            max_values=args.max_values,
            max_workers=args.feature_workers,
            max_values_text=args.max_values_text,
            max_chars_per_value=args.max_chars_per_value,
            max_total_chars_text=args.max_total_chars_text,
            embedding_model=args.embedding_model,
            embedding_device=args.embedding_device,
            embedding_batch_size=args.embedding_batch_size,
            max_tokens=args.max_tokens,
            use_transformer_embeddings=not args.disable_transformer_embeddings,
            use_topological=True,
        )
    else:
        metamatch_df = build_reduced_metamatch(pairs, args)
    feature_build_sec = time.perf_counter() - t_features

    metamatch_path = safe_write_table(metamatch_df, args.out_dir / "metamatch.parquet")
    features = feature_columns_for_config(metamatch_df, args.config)
    write_json(args.out_dir / "selected_features.json", {"config": args.config, "n_features": len(features), "features": features})

    t_split = time.perf_counter()
    folds = make_repeated_splits_70_30(metamatch_df, n_splits=args.folds, seed=args.seed, test_size=0.30)
    fold_manifests = materialize_fold_frames(metamatch_df, folds, out_dir=args.out_dir / "folds")
    split_sec = time.perf_counter() - t_split

    fold_df, pair_df, rf_timing = evaluate_rf(metamatch_df, fold_manifests, features, args)
    fold_df.to_csv(args.out_dir / "fold_metrics.csv", index=False)
    pair_df.to_csv(args.out_dir / "pair_metrics.csv", index=False)

    summary = (
        pair_df.groupby(["config", "classifier"], as_index=False)
        .agg(
            mean_f1=("f1", "mean"),
            std_f1=("f1", "std"),
            mean_precision=("precision", "mean"),
            std_precision=("precision", "std"),
            mean_recall=("recall", "mean"),
            std_recall=("recall", "std"),
            n_pair_id=("pair_id", "nunique"),
        )
    )
    summary.to_csv(args.out_dir / "summary_pair_metrics.csv", index=False)

    timing = {
        "config": args.config,
        "n_pairs": int(metamatch_df["pair_id"].nunique()),
        "n_rows": int(len(metamatch_df)),
        "n_features": int(len(features)),
        "metamatch_path": str(metamatch_path),
        "feature_build_sec": float(feature_build_sec),
        "split_materialize_sec": float(split_sec),
        **{k: float(v) for k, v in rf_timing.items()},
        "total_e2e_sec": float(time.perf_counter() - t_total),
        "feature_workers": int(args.feature_workers),
        "folds": int(args.folds),
        "seed": int(args.seed),
        "n_estimators": int(args.n_estimators),
        "threshold_rule": "strict score > threshold; threshold learned on train by maximizing mean F1 over train pair_id",
    }
    write_json(args.out_dir / "timing_summary.json", timing)
    write_json(args.out_dir / "run_config.json", vars(args))

    print("\n=== Summary over pair_id ===")
    print(summary.round(4).to_string(index=False))
    print("\n=== Timing ===")
    print(json.dumps(timing, indent=2))


if __name__ == "__main__":
    main()
