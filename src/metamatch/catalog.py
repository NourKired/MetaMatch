from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

from .types import ColumnMatch, TablePair

KNOWN_RELATIONS = {
    "joinable": "Joinable",
    "semantically-joinable": "Semantically-Joinable",
    "unionable": "Unionable",
    "view-unionable": "View-Unionable",
    "view -unionable": "View-Unionable",
}


def _norm_relation(value: str) -> str:
    key = value.strip().lower()
    return KNOWN_RELATIONS.get(key, value.strip())


def _pair_id_from_mapping(root_dir: Path, mapping_path: Path) -> str:
    rel = mapping_path.relative_to(root_dir)
    pair_dir = rel.parent
    return str(pair_dir).replace("/", "__")


def discover_table_pairs(root_dir: Path) -> List[TablePair]:
    root_dir = root_dir.resolve()
    pairs: List[TablePair] = []

    for mapping_path in sorted(root_dir.rglob("*_mapping.json")):
        pair_dir = mapping_path.parent
        source_candidates = sorted(pair_dir.glob("*_source.csv"))
        target_candidates = sorted(pair_dir.glob("*_target.csv"))

        if len(source_candidates) != 1 or len(target_candidates) != 1:
            continue

        rel_parts = mapping_path.relative_to(root_dir).parts
        dataset = rel_parts[0] if len(rel_parts) >= 1 else "Unknown"
        relation_type = _norm_relation(rel_parts[1]) if len(rel_parts) >= 2 else "Unknown"

        if relation_type not in KNOWN_RELATIONS.values():
            relation_type = "Entity-Matching"

        pair_id = _pair_id_from_mapping(root_dir, mapping_path)
        source_name = source_candidates[0].stem.removesuffix("_source")
        target_name = target_candidates[0].stem.removesuffix("_target")

        pairs.append(
            TablePair(
                pair_id=pair_id,
                dataset=dataset,
                relation_type=relation_type,
                category=dataset,
                source_csv=source_candidates[0],
                target_csv=target_candidates[0],
                mapping_json=mapping_path,
                source_table_name=source_name,
                target_table_name=target_name,
            )
        )

    return pairs


def load_mapping(mapping_json: Path) -> List[ColumnMatch]:
    with mapping_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    matches_raw: Iterable[Dict]
    if isinstance(payload, dict):
        matches_raw = payload.get("matches", [])
    elif isinstance(payload, list):
        matches_raw = payload
    else:
        matches_raw = []

    matches: List[ColumnMatch] = []
    for item in matches_raw:
        if not isinstance(item, dict):
            continue
        src_col = item.get("source_column")
        tgt_col = item.get("target_column")
        if not src_col or not tgt_col:
            continue
        matches.append(
            ColumnMatch(
                source_table=str(item.get("source_table", "source")),
                source_column=str(src_col),
                target_table=str(item.get("target_table", "target")),
                target_column=str(tgt_col),
            )
        )
    return matches
