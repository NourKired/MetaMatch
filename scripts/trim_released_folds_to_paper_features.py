#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FOLDS_DIR = PROJECT_ROOT / "data" / "folds"

CONTEXT_COLUMNS = [
    "pair_id",
    "dataset",
    "relation_type",
    "category",
    "source_table",
    "target_table",
    "source_column",
    "target_column",
    "source_col_norm",
    "target_col_norm",
    "label",
]


def paper_meta_features(df: pd.DataFrame) -> list[str]:
    candidates = [
        c
        for c in df.columns
        if c not in CONTEXT_COLUMNS
        and c != "feature_build_sec_pair_real"
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    return [
        c
        for c in candidates
        if not c.startswith("spc_")
        and not c.startswith("nlp_")
        and "overlap" not in c.lower()
        and c != "cls_cosine_dist"
    ]


def main() -> None:
    paths = sorted(FOLDS_DIR.glob("fold_*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet fold files found under {FOLDS_DIR}")

    expected_features: list[str] | None = None
    for path in paths:
        df = pd.read_parquet(path)
        features = paper_meta_features(df)
        if expected_features is None:
            expected_features = features
            if len(expected_features) != 60:
                raise RuntimeError(f"Expected 60 paper meta-features, found {len(expected_features)}")
        elif features != expected_features:
            raise RuntimeError(f"Feature mismatch in {path}")

        keep = [c for c in CONTEXT_COLUMNS if c in df.columns] + expected_features
        df.loc[:, keep].to_parquet(path, index=False)
        print(f"{path}: {len(keep)} columns = {len(expected_features)} meta-features + {len(keep) - len(expected_features)} context columns")


if __name__ == "__main__":
    main()
