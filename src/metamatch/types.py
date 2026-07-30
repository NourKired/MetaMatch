from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


@dataclass(frozen=True)
class ColumnMatch:
    source_table: str
    source_column: str
    target_table: str
    target_column: str

    def signature(self) -> Tuple[str, str]:
        return (self.source_column.strip().lower(), self.target_column.strip().lower())


@dataclass(frozen=True)
class TablePair:
    pair_id: str
    dataset: str
    relation_type: str
    category: str
    source_csv: Path
    target_csv: Path
    mapping_json: Path
    source_table_name: str
    target_table_name: str


@dataclass(frozen=True)
class FoldSplit:
    fold_id: int
    train_pair_ids: List[str]
    test_pair_ids: List[str]
