#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from rapidfuzz import fuzz


RX_TOKEN = re.compile(r"[A-Za-z0-9]+")


def _norm(s: str) -> str:
    return str(s).strip().lower()


def _tokens(s: str) -> List[str]:
    return RX_TOKEN.findall(_norm(s))


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return float(len(sa & sb) / len(sa | sb))


SEM_GROUPS: Dict[str, set[str]] = {
    "id": {"id", "identifier", "code", "key", "pk"},
    "name": {"name", "label", "title"},
    "desc": {"description", "comment", "text", "detail", "summary"},
    "time": {"date", "time", "timestamp", "year", "month", "day"},
    "loc": {"city", "country", "state", "address", "zip", "postal"},
    "person": {"first", "last", "fname", "lname", "surname", "given"},
    "qty": {"amount", "price", "cost", "total", "qty", "quantity", "number", "count"},
}


def _semantic_name_score(src: str, tgt: str) -> float:
    ts, tt = set(_tokens(src)), set(_tokens(tgt))
    if not ts or not tt:
        return 0.0
    best = 0.0
    for vocab in SEM_GROUPS.values():
        a = len(ts & vocab) > 0
        b = len(tt & vocab) > 0
        if a and b:
            best = 1.0
            break
    return best


@dataclass
class ColStats:
    is_numeric: bool
    nonnull_ratio: float
    unique_ratio: float
    mean: float
    std: float
    top_values: set[str]


def _build_col_stats(series: pd.Series, max_top_values: int = 30) -> ColStats:
    n = max(len(series), 1)
    nn = series.dropna()
    nonnull_ratio = float(len(nn) / n)

    if len(nn) == 0:
        return ColStats(False, nonnull_ratio, 0.0, 0.0, 0.0, set())

    uniq = float(min(nn.nunique(dropna=True) / max(len(nn), 1), 1.0))
    as_num = pd.to_numeric(nn, errors="coerce")
    num_ratio = float(as_num.notna().mean())
    is_numeric = num_ratio >= 0.8

    if is_numeric:
        v = as_num.dropna().astype(float)
        mean = float(v.mean()) if len(v) else 0.0
        std = float(v.std()) if len(v) > 1 else 0.0
    else:
        mean = 0.0
        std = 0.0

    top_vals = (
        nn.astype(str)
        .str.strip()
        .str.lower()
        .replace("", np.nan)
        .dropna()
        .value_counts()
        .head(max_top_values)
        .index.tolist()
    )
    return ColStats(
        is_numeric=is_numeric,
        nonnull_ratio=nonnull_ratio,
        unique_ratio=uniq,
        mean=mean,
        std=std,
        top_values=set(top_vals),
    )


def _dtype_sim(a: ColStats, b: ColStats) -> float:
    if a.is_numeric and b.is_numeric:
        return 1.0
    if (not a.is_numeric) and (not b.is_numeric):
        return 1.0
    return 0.2


def _numeric_sim(a: ColStats, b: ColStats) -> float:
    if not (a.is_numeric and b.is_numeric):
        return 0.0
    pooled = abs(a.std) + abs(b.std) + 1e-9
    mean_dist = abs(a.mean - b.mean) / pooled
    std_ratio = abs(math.log((abs(a.std) + 1e-9) / (abs(b.std) + 1e-9)))
    return float(math.exp(-(mean_dist + 0.5 * std_ratio)))


def _top_values_overlap(a: ColStats, b: ColStats) -> float:
    return _jaccard(a.top_values, b.top_values)


def _schema_name_score(src_col: str, tgt_col: str) -> float:
    s = _norm(src_col)
    t = _norm(tgt_col)
    exact = 1.0 if s == t else 0.0
    ratio = float(fuzz.token_sort_ratio(s, t) / 100.0)
    jac = _jaccard(_tokens(s), _tokens(t))
    return 0.5 * ratio + 0.3 * jac + 0.2 * exact


def _instance_score(a: ColStats, b: ColStats) -> float:
    return (
        0.35 * _top_values_overlap(a, b)
        + 0.35 * _numeric_sim(a, b)
        + 0.20 * (1.0 - abs(a.nonnull_ratio - b.nonnull_ratio))
        + 0.10 * (1.0 - abs(a.unique_ratio - b.unique_ratio))
    )


def _clip(v: float) -> float:
    return float(max(0.0, min(1.0, v)))


def _score_method(method: str, src_col: str, tgt_col: str, a: ColStats, b: ColStats) -> float:
    schema = _schema_name_score(src_col, tgt_col)
    semantic = _semantic_name_score(src_col, tgt_col)
    instance = _instance_score(a, b)
    dtype = _dtype_sim(a, b)

    m = method.lower()
    if m == "isresmat":
        return _clip(0.25 * schema + 0.65 * instance + 0.10 * dtype)

    if m == "smutf":
        return _clip(0.70 * schema + 0.15 * semantic + 0.15 * instance)

    if m == "llmatch":
        # Proxy "LLM-like": stronger semantic and schema weighting.
        return _clip(0.35 * schema + 0.45 * semantic + 0.20 * instance)

    if m.startswith("magneto_"):
        base = _clip(0.35 * schema + 0.35 * instance + 0.20 * semantic + 0.10 * dtype)
        has_ft = "ft" in m and "no_ft" not in m
        has_gpt = "gpt" in m and "no_gpt" not in m

        if has_ft:
            # Proxy fine-tuning effect: sharpen high-confidence candidates.
            base = _clip(base ** 0.9)
        if has_gpt:
            # Proxy GPT effect: semantic rescue for near-miss schema matches.
            base = _clip(0.7 * base + 0.3 * max(base, semantic))
        return base

    return _clip(schema)


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


def _build_norm_col_map(columns: Iterable[str]) -> Dict[str, str]:
    mp: Dict[str, str] = {}
    for c in columns:
        n = _norm(c)
        if n not in mp:
            mp[n] = str(c)
    return mp


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Proxy runners for external baselines")
    p.add_argument("--method", required=True, type=str)
    p.add_argument("--pair-manifest", required=True, type=Path)
    p.add_argument("--test-manifest", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--sample-rows", type=int, default=5000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    method = str(args.method)

    test_df = pd.read_csv(args.test_manifest)
    test_df["pair_id"] = test_df["pair_id"].astype(str)
    test_df["source_col_norm"] = test_df["source_col_norm"].astype(str).str.strip().str.lower()
    test_df["target_col_norm"] = test_df["target_col_norm"].astype(str).str.strip().str.lower()

    pair_manifest = _read_pair_manifest(args.pair_manifest)

    outputs: List[Dict[str, object]] = []
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

        src_map = _build_norm_col_map(src_df.columns)
        tgt_map = _build_norm_col_map(tgt_df.columns)

        src_stats_cache: Dict[str, ColStats] = {}
        tgt_stats_cache: Dict[str, ColStats] = {}

        for _, r in group.iterrows():
            s_norm = str(r["source_col_norm"])
            t_norm = str(r["target_col_norm"])
            s_col = src_map.get(s_norm)
            t_col = tgt_map.get(t_norm)
            if s_col is None or t_col is None:
                score = 0.0
            else:
                if s_norm not in src_stats_cache:
                    src_stats_cache[s_norm] = _build_col_stats(src_df[s_col])
                if t_norm not in tgt_stats_cache:
                    tgt_stats_cache[t_norm] = _build_col_stats(tgt_df[t_col])
                score = _score_method(
                    method=method,
                    src_col=s_norm,
                    tgt_col=t_norm,
                    a=src_stats_cache[s_norm],
                    b=tgt_stats_cache[t_norm],
                )

            outputs.append(
                {
                    "pair_id": str(pair_id),
                    "source_col_norm": s_norm,
                    "target_col_norm": t_norm,
                    "score": _clip(float(score)),
                }
            )

    out_df = pd.DataFrame(outputs, columns=["pair_id", "source_col_norm", "target_col_norm", "score"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"method={method} rows={len(out_df)} out={args.out}")


if __name__ == "__main__":
    main()

