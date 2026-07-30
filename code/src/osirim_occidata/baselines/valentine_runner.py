from __future__ import annotations

import importlib
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd
from tqdm import tqdm

from ..catalog import discover_table_pairs
from ..types import TablePair


@dataclass
class BaselineOutput:
    baseline_name: str
    predictions: pd.DataFrame
    failures: List[Dict[str, str]]


def _load_valentine_modules():
    valentine = importlib.import_module("valentine")
    algorithms = importlib.import_module("valentine.algorithms")
    return valentine, algorithms


def _make_matcher(method: str, algorithms):
    m = method.lower()

    if m == "coma_schema":
        try:
            return algorithms.Coma(use_instances=False)
        except TypeError:
            return algorithms.Coma(max_n=0, strategy="COMA_OPT")

    if m == "coma_instance":
        try:
            return algorithms.Coma(use_instances=True)
        except TypeError:
            return algorithms.Coma(max_n=0, strategy="COMA_OPT_INST")

    if m == "cupid":
        return algorithms.Cupid()

    if m == "similarity_flooding":
        return algorithms.SimilarityFlooding()

    if m == "distribution_based":
        return algorithms.DistributionBased()

    raise ValueError(f"Unsupported Valentine baseline: {method}")


def _extract_src_tgt(key: object) -> Tuple[str, str] | None:
    if hasattr(key, "source_column") and hasattr(key, "target_column"):
        return (str(key.source_column).strip().lower(), str(key.target_column).strip().lower())

    if isinstance(key, tuple) and len(key) == 2:
        left, right = key

        if hasattr(left, "column_name") and hasattr(right, "column_name"):
            return (str(left.column_name).strip().lower(), str(right.column_name).strip().lower())

        if isinstance(left, tuple) and isinstance(right, tuple) and len(left) >= 2 and len(right) >= 2:
            return (str(left[1]).strip().lower(), str(right[1]).strip().lower())

    return None


def _run_single_pair(
    method: str,
    source_csv: str,
    target_csv: str,
    source_table_name: str,
    target_table_name: str,
) -> Dict[Tuple[str, str], float]:
    valentine, algorithms = _load_valentine_modules()
    matcher = _make_matcher(method, algorithms)

    src_df = pd.read_csv(source_csv, low_memory=False)
    tgt_df = pd.read_csv(target_csv, low_memory=False)

    try:
        matches = valentine.valentine_match([src_df, tgt_df], matcher)
    except Exception:
        matches = valentine.valentine_match(src_df, tgt_df, matcher)

    score_map: Dict[Tuple[str, str], float] = {}
    for key, score in matches.items():
        pair = _extract_src_tgt(key)
        if pair is None:
            continue
        try:
            s = float(score)
        except Exception:
            continue
        old = score_map.get(pair, 0.0)
        if s > old:
            score_map[pair] = s
    return score_map


def _score_pair_rows(
    method: str,
    pair: TablePair,
    rows: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, str] | None]:
    try:
        scores = _run_single_pair(
            method=method,
            source_csv=str(pair.source_csv),
            target_csv=str(pair.target_csv),
            source_table_name=pair.source_table_name,
            target_table_name=pair.target_table_name,
        )
        out = rows.copy()
        out["score"] = [scores.get((s, t), 0.0) for s, t in zip(out["source_col_norm"], out["target_col_norm"]) ]
        return out, None
    except Exception as exc:
        out = rows.copy()
        out["score"] = 0.0
        return (
            out,
            {
                "pair_id": pair.pair_id,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=5),
            },
        )


def run_valentine_baseline(
    method: str,
    test_df: pd.DataFrame,
    dataset_root: Path,
    max_workers: int = 1,
) -> BaselineOutput:
    pairs = discover_table_pairs(dataset_root)
    by_id = {p.pair_id: p for p in pairs}

    test_pairs = sorted(test_df["pair_id"].unique().tolist())
    jobs = []
    for pid in test_pairs:
        pair = by_id.get(pid)
        if pair is None:
            continue
        rows = test_df[test_df["pair_id"] == pid].copy()
        jobs.append((pair, rows))

    results: List[pd.DataFrame] = []
    failures: List[Dict[str, str]] = []

    if max_workers <= 1:
        for pair, rows in tqdm(jobs, desc=f"baseline:{method}"):
            pred, err = _score_pair_rows(method, pair, rows)
            results.append(pred)
            if err:
                failures.append(err)
    else:
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                fut_map = {
                    ex.submit(_score_pair_rows, method, pair, rows): pair.pair_id
                    for pair, rows in jobs
                }
                for fut in tqdm(as_completed(fut_map), total=len(fut_map), desc=f"baseline:{method}"):
                    pred, err = fut.result()
                    results.append(pred)
                    if err:
                        failures.append(err)
        except (PermissionError, OSError):
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                fut_map = {
                    ex.submit(_score_pair_rows, method, pair, rows): pair.pair_id
                    for pair, rows in jobs
                }
                for fut in tqdm(as_completed(fut_map), total=len(fut_map), desc=f"baseline:{method}-thread"):
                    pred, err = fut.result()
                    results.append(pred)
                    if err:
                        failures.append(err)

    if not results:
        pred_df = test_df.copy()
        pred_df["score"] = 0.0
    else:
        pred_df = pd.concat(results, ignore_index=True)

    return BaselineOutput(baseline_name=method, predictions=pred_df, failures=failures)
