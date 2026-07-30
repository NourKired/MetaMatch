from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit

from .io_utils import safe_write_table
from .types import FoldSplit


def _strata_label(pair_meta: pd.DataFrame) -> pd.Series:
    return pair_meta["dataset"].astype(str) + "::" + pair_meta["relation_type"].astype(str)


def make_pair_folds(
    metaspace_df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
) -> List[FoldSplit]:
    pair_meta = (
        metaspace_df[["pair_id", "dataset", "relation_type"]]
        .drop_duplicates(subset=["pair_id"])
        .reset_index(drop=True)
    )

    y = _strata_label(pair_meta)
    min_count = y.value_counts().min() if len(y) else 0

    if min_count >= n_splits:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(pair_meta["pair_id"], y)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(pair_meta["pair_id"])

    folds: List[FoldSplit] = []
    for fold_id, (train_idx, test_idx) in enumerate(split_iter):
        train_ids = pair_meta.iloc[train_idx]["pair_id"].tolist()
        test_ids = pair_meta.iloc[test_idx]["pair_id"].tolist()
        folds.append(FoldSplit(fold_id=fold_id, train_pair_ids=train_ids, test_pair_ids=test_ids))

    return folds


def make_repeated_splits_70_30(
    metaspace_df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
    test_size: float = 0.30,
    max_attempts: int = 500,
) -> List[FoldSplit]:
    pair_meta = (
        metaspace_df[["pair_id", "dataset", "relation_type"]]
        .drop_duplicates(subset=["pair_id"])
        .reset_index(drop=True)
    )
    y = _strata_label(pair_meta)

    all_pair_ids = pair_meta["pair_id"].tolist()
    uncovered = set(all_pair_ids)
    folds: List[FoldSplit] = []

    # We sample repeated stratified 70/30 splits until every pair appears in test at least once.
    # If full coverage is not achieved quickly, we fill remaining folds with deterministic partitions.
    for attempt in range(max_attempts):
        if len(folds) >= n_splits:
            break
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=seed + attempt,
        )
        train_idx, test_idx = next(splitter.split(pair_meta["pair_id"], y))
        test_ids = pair_meta.iloc[test_idx]["pair_id"].tolist()
        train_ids = pair_meta.iloc[train_idx]["pair_id"].tolist()

        # Prefer splits that increase test coverage.
        gain = len(uncovered.intersection(test_ids))
        if gain == 0 and uncovered:
            continue

        folds.append(
            FoldSplit(
                fold_id=len(folds),
                train_pair_ids=train_ids,
                test_pair_ids=test_ids,
            )
        )
        uncovered.difference_update(test_ids)
        if not uncovered and len(folds) >= n_splits:
            break

    # Guarantee exactly n_splits folds.
    while len(folds) < n_splits:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold_id, (train_idx, test_idx) in enumerate(kf.split(pair_meta["pair_id"])):
            if len(folds) >= n_splits:
                break
            train_ids = pair_meta.iloc[train_idx]["pair_id"].tolist()
            test_ids = pair_meta.iloc[test_idx]["pair_id"].tolist()
            folds.append(
                FoldSplit(
                    fold_id=len(folds),
                    train_pair_ids=train_ids,
                    test_pair_ids=test_ids,
                )
            )
        break

    # Reindex fold ids.
    normalized = []
    for i, f in enumerate(folds[:n_splits]):
        normalized.append(FoldSplit(fold_id=i, train_pair_ids=f.train_pair_ids, test_pair_ids=f.test_pair_ids))
    return normalized


def _test_positive_signatures(test_df: pd.DataFrame) -> Set[Tuple[str, str]]:
    pos = test_df[test_df["label"] == 1]
    return set(zip(pos["source_col_norm"], pos["target_col_norm"]))


def filter_train_leakage(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    test_sigs = _test_positive_signatures(test_df)
    if not test_sigs:
        return train_df

    train_pos_mask = train_df["label"] == 1
    train_pos_sigs = list(zip(train_df.loc[train_pos_mask, "source_col_norm"], train_df.loc[train_pos_mask, "target_col_norm"]))

    to_drop = [sig in test_sigs for sig in train_pos_sigs]
    if not any(to_drop):
        return train_df

    keep_pos = ~pd.Series(to_drop, index=train_df.index[train_pos_mask])
    keep_mask = pd.Series(True, index=train_df.index)
    keep_mask.loc[train_df.index[train_pos_mask]] = keep_pos.values
    return train_df.loc[keep_mask].reset_index(drop=True)


def materialize_fold_frames(
    metaspace_df: pd.DataFrame,
    folds: Sequence[FoldSplit],
    out_dir: Path,
) -> List[Dict[str, Path]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifests: List[Dict[str, Path]] = []

    for fold in folds:
        fold_dir = out_dir / f"fold_{fold.fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_df = metaspace_df[metaspace_df["pair_id"].isin(fold.train_pair_ids)].copy()
        test_df = metaspace_df[metaspace_df["pair_id"].isin(fold.test_pair_ids)].copy()

        train_df = filter_train_leakage(train_df, test_df)

        train_path = safe_write_table(train_df, fold_dir / "train.parquet")
        test_path = safe_write_table(test_df, fold_dir / "test.parquet")
        split_path = fold_dir / "split.json"
        split_path.write_text(
            pd.Series(
                {
                    "fold_id": fold.fold_id,
                    "n_train_pairs": len(set(train_df["pair_id"])),
                    "n_test_pairs": len(set(test_df["pair_id"])),
                    "n_train_rows": len(train_df),
                    "n_test_rows": len(test_df),
                    "train_pair_ids": fold.train_pair_ids,
                    "test_pair_ids": fold.test_pair_ids,
                }
            ).to_json(indent=2),
            encoding="utf-8",
        )

        manifests.append({"fold_dir": fold_dir, "train": train_path, "test": test_path, "split": split_path})

    return manifests
