#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _norm(s: str) -> str:
    return str(s).strip().lower()


def _extract_src_tgt(key: object) -> Optional[Tuple[str, str]]:
    if hasattr(key, "source_column") and hasattr(key, "target_column"):
        return (_norm(str(key.source_column)), _norm(str(key.target_column)))

    if isinstance(key, tuple) and len(key) == 2:
        left, right = key

        if hasattr(left, "column_name") and hasattr(right, "column_name"):
            return (_norm(str(left.column_name)), _norm(str(right.column_name)))

        if isinstance(left, tuple) and isinstance(right, tuple) and len(left) >= 2 and len(right) >= 2:
            return (_norm(str(left[1])), _norm(str(right[1])))

    return None


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Real Magneto runner")
    p.add_argument("--pair-manifest", required=True, type=Path)
    p.add_argument("--test-manifest", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--variant", required=True, type=str)
    p.add_argument("--sample-rows", type=int, default=5000)
    p.add_argument(
        "--magneto-package-root",
        type=Path,
        default=Path(
            os.environ.get(
                "MAGNETO_PACKAGE_ROOT",
                str(PROJECT_ROOT / "third_party" / "magneto-matcher" / "algorithms" / "magneto"),
            )
        ),
    )
    p.add_argument(
        "--magneto-model-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "MAGNETO_MODEL_DIR",
                str(PROJECT_ROOT / "third_party" / "magneto-matcher" / "models"),
            )
        ),
    )
    p.add_argument(
        "--magneto-ft-pattern",
        type=str,
        default=os.environ.get("MAGNETO_FT_PATTERN", "mpnet-{category}-complet-complet-exact_semantic-64-0.5.pth"),
        help="Filename template for fine-tuned checkpoints (supports {category}, {scenario}).",
    )
    p.add_argument(
        "--encoding-mode",
        type=str,
        default=os.environ.get("MAGNETO_ENCODING_MODE", "header_values_repeat"),
    )
    p.add_argument(
        "--strict-variant",
        type=int,
        default=1,
        help="If 1, fail when requested FT/GPT knobs are not actually supported/applied by Magneto API.",
    )
    return p.parse_args()


def _canonical_model_category(category: str, dataset: str = "") -> str:
    cat = str(category).strip().lower().replace(" ", "_").replace("-", "_")
    ds = str(dataset).strip().lower().replace(" ", "_").replace("-", "_")

    aliases = {
        "open_data": "opendata",
        "open_data_": "opendata",
        "opendata": "opendata",
        "tpc_di": "tpc",
        "tpcdi": "tpc",
        "tpc": "tpc",
        "chembl": "chembl",
        "magellan": "magellan",
        "wikidata": "wikidata",
        # Wikidata one-level benchmark category often appears as musicians.
        "musicians": "wikidata",
    }

    if cat in aliases:
        return aliases[cat]
    if ds in aliases:
        return aliases[ds]
    return cat


def _infer_benchmark_category(pair_id: str, category: str) -> str:
    cat = str(category).strip()
    leaf = str(pair_id).strip().lower()
    if leaf.endswith("_joinable"):
        return "Joinable"
    if leaf.endswith("_semjoinable"):
        return "Semantically-Joinable"
    if leaf.endswith("_unionable"):
        return "Unionable"
    if leaf.endswith("_viewunion"):
        return "View-Unionable"
    return cat


def _scenario_for_pair(pair_id: str, category: str) -> str:
    return "complet-complet"


def _resolve_ft_model(model_dir: Path, category: str, pattern: str, dataset: str = "", scenario: str = "complet-complet") -> Optional[Path]:
    cat = _canonical_model_category(category=category, dataset=dataset)
    if not cat:
        return None

    preferred = model_dir / pattern.format(category=cat, scenario=scenario)
    if preferred.exists():
        return preferred

    candidates = sorted(model_dir.glob(f"mpnet-{cat}-*.pth"))
    if not candidates:
        return None

    # deterministic preference among available checkpoints
    priority = [
        f"{scenario}-exact_semantic-64-0.5",
        "header_values_repeat-semantic-64-0.5",
        "header_values_repeat-exact_semantic-64-0.5",
        "complet-complet-exact_semantic-64-0.5",
        "schema-complet-exact_semantic-64-0.5",
        "schema-instance-exact_semantic-64-0.5",
        "instance-complet-exact_semantic-64-0.5",
        "schema-schema-exact_semantic-64-0.5",
    ]
    for key in priority:
        for c in candidates:
            if key in c.name:
                return c
    return candidates[0]


def _variant_flags(variant: str) -> Tuple[bool, bool]:
    v = variant.lower()
    use_gpt = "gpt" in v and "no_gpt" not in v
    use_ft = "ft" in v and "no_ft" not in v
    return use_ft, use_gpt


def _build_magneto(
    variant: str,
    ft_model_path: Optional[Path] = None,
    strict_variant: bool = True,
    encoding_mode: str = "header_values_default",
):
    # Import from local package root.
    from magneto import Magneto  # type: ignore

    use_ft, use_gpt = _variant_flags(variant)
    kwargs: Dict[str, object] = {}

    # Match the third-party benchmark API exactly:
    # - MagnetoFT      -> Magneto(encoding_mode=..., embedding_model=...)
    # - MagnetoGPT     -> Magneto(use_bp_reranker=False, use_gpt_reranker=True)
    # - MagnetoFTGPT   -> Magneto(encoding_mode=..., embedding_model=..., use_bp_reranker=False, use_gpt_reranker=True)
    if use_ft:
        if ft_model_path is None:
            if strict_variant:
                raise RuntimeError("FT variant requested but no matching checkpoint was found for this category.")
        else:
            kwargs["encoding_mode"] = encoding_mode
            kwargs["embedding_model"] = str(ft_model_path)

    if use_gpt:
        kwargs["use_bp_reranker"] = False
        kwargs["use_gpt_reranker"] = True

    if strict_variant and use_gpt and not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Variant requests GPT but OPENAI_API_KEY is missing.")

    try:
        matcher = Magneto(**kwargs)
    except Exception as e:
        if strict_variant:
            raise RuntimeError(f"Magneto init failed for variant={variant} kwargs={kwargs}: {e}") from e
        matcher = Magneto()

    debug = {
        "variant": variant,
        "use_ft": use_ft,
        "use_gpt": use_gpt,
        "encoding_mode": encoding_mode,
        "ft_model_path": str(ft_model_path) if ft_model_path else None,
        "constructor_kwargs": kwargs,
    }
    return matcher, debug


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.magneto_package_root))

    test_df = pd.read_csv(args.test_manifest)
    test_df["pair_id"] = test_df["pair_id"].astype(str)
    test_df["source_col_norm"] = test_df["source_col_norm"].astype(str).str.strip().str.lower()
    test_df["target_col_norm"] = test_df["target_col_norm"].astype(str).str.strip().str.lower()

    pair_manifest = _read_pair_manifest(args.pair_manifest)
    matcher_cache: Dict[Tuple[str, str], Tuple[object, Dict[str, object]]] = {}
    debug_records: Dict[Tuple[str, str], Dict[str, object]] = {}

    outputs = []
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
        category = _infer_benchmark_category(str(pair_id), str(meta.get("category", "")).strip())
        dataset = str(meta.get("dataset", "")).strip().lower()
        scenario = _scenario_for_pair(str(pair_id), category)

        ft_model_path: Optional[Path] = None
        canonical_category = _canonical_model_category(category=category, dataset=dataset)
        if "ft" in args.variant.lower() and "no_ft" not in args.variant.lower():
            ft_model_path = _resolve_ft_model(
                model_dir=args.magneto_model_dir,
                category=category,
                pattern=args.magneto_ft_pattern,
                dataset=dataset,
                scenario=scenario,
            )

        cache_key = (args.variant, str(ft_model_path) if ft_model_path else "")
        cached = matcher_cache.get(cache_key)
        if cached is None:
            matcher, dbg = _build_magneto(
                args.variant,
                ft_model_path=ft_model_path,
                strict_variant=bool(args.strict_variant),
                encoding_mode=args.encoding_mode,
            )
            dbg["category_raw"] = category
            dbg["dataset_raw"] = dataset
            dbg["category_canonical"] = canonical_category
            dbg["scenario"] = scenario
            matcher_cache[cache_key] = (matcher, dbg)
            debug_records[cache_key] = dbg
        else:
            matcher, _ = cached

        try:
            matches = matcher.get_matches(src_df, tgt_df)
        except Exception:
            if bool(args.strict_variant):
                raise
            matches = {}

        score_map: Dict[Tuple[str, str], float] = {}
        if hasattr(matches, "items"):
            for k, v in matches.items():
                pair = _extract_src_tgt(k)
                if pair is None:
                    continue
                try:
                    s = float(v)
                except Exception:
                    continue
                if s > score_map.get(pair, 0.0):
                    score_map[pair] = s

        for _, r in group.iterrows():
            s = str(r["source_col_norm"])
            t = str(r["target_col_norm"])
            outputs.append(
                {
                    "pair_id": str(pair_id),
                    "source_col_norm": s,
                    "target_col_norm": t,
                    "score": float(score_map.get((s, t), 0.0)),
                }
            )

    out_df = pd.DataFrame(outputs, columns=["pair_id", "source_col_norm", "target_col_norm", "score"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    debug_path = args.out.with_name(f"{args.out.stem}_magneto_debug.json")
    debug_payload = {
        "variant": args.variant,
        "strict_variant": bool(args.strict_variant),
        "magneto_model_dir": str(args.magneto_model_dir),
        "records": list(debug_records.values()),
    }
    debug_path.write_text(json.dumps(debug_payload, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"variant={args.variant} rows={len(out_df)} out={args.out}")


if __name__ == "__main__":
    main()
