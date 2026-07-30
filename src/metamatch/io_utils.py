from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def read_jsonl(path: Path) -> List[Dict]:
    out: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def save_runtime_csv(rows: List[Dict], path: Path) -> None:
    if not rows:
        return
    pd.DataFrame(rows).to_csv(path, index=False)


def safe_write_table(df: pd.DataFrame, preferred_path: Path) -> Path:
    preferred_path.parent.mkdir(parents=True, exist_ok=True)
    if preferred_path.suffix.lower() == ".parquet":
        try:
            df.to_parquet(preferred_path, index=False)
            return preferred_path
        except Exception:
            fallback = preferred_path.with_suffix(".csv.gz")
            df.to_csv(fallback, index=False, compression="gzip")
            return fallback
    df.to_csv(preferred_path, index=False)
    return preferred_path


def safe_read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".gz", ".csv"} or path.name.endswith(".csv.gz"):
        return pd.read_csv(path)
    # Default fallback
    return pd.read_csv(path)
