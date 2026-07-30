from __future__ import annotations

import hashlib
import re
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from .catalog import load_mapping
from .meta_features.classical import CLASSICAL_FEATURES, compute_classical_features
from .meta_features.spectral import SPECTRAL_FEATURES, compute_spectral_features_pair
from .meta_features.syntax import SYNTAX_FEATURES, compute_syntax_features
from .types import TablePair


_RX_TOKEN = re.compile(r"[A-Za-z0-9]+")


def _normalize_name(value: str) -> str:
    return str(value).strip().lower()


def _load_df(path: Path, sample_rows: int | None = None) -> pd.DataFrame:
    if sample_rows is not None and sample_rows > 0:
        return pd.read_csv(path, nrows=sample_rows, low_memory=False)
    return pd.read_csv(path, low_memory=False)


def _truncate_text(s: str, max_len: int) -> str:
    if len(s) <= max_len:
        return s
    return s[:max_len]


def _build_column_text(
    col_name: str,
    series: pd.Series,
    max_values_text: int = 20,
    max_chars_per_value: int = 80,
    max_total_chars: int = 2000,
) -> str:
    non_null = series.dropna()
    if len(non_null) == 0:
        return str(col_name)

    as_str = non_null.astype(str).str.strip()
    as_str = as_str[as_str != ""]
    if len(as_str) == 0:
        return str(col_name)

    values = as_str.value_counts().head(max_values_text).index.tolist()
    values = [_truncate_text(v.replace("\n", " ").replace("\r", " "), max_chars_per_value) for v in values]

    text = f"{col_name} || " + " | ".join(values)
    return _truncate_text(text, max_total_chars)


def _hash_to_seed(text: str) -> int:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    return int(digest, 16)


def _hashed_vector(text: str, dim: int = 384) -> np.ndarray:
    seed = _hash_to_seed(text)
    rng = np.random.default_rng(seed)
    return rng.standard_normal(dim).astype(np.float32)


def _fallback_token_matrix(text: str, dim: int = 384, max_tokens: int = 64) -> np.ndarray:
    toks = _RX_TOKEN.findall(str(text).lower())[:max_tokens]
    if not toks:
        return np.zeros((0, dim), dtype=np.float32)
    mats = [_hashed_vector(tok, dim=dim) for tok in toks]
    return np.vstack(mats).astype(np.float32)


class ColumnTextEncoder:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 32,
        max_tokens: int = 64,
        use_transformer_embeddings: bool = True,
        fallback_dim: int = 384,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.use_transformer_embeddings = use_transformer_embeddings
        self.fallback_dim = fallback_dim

        self.model = None
        self.tokenizer = None
        self.backend = "fallback"
        self.embedding_dim = fallback_dim

        if use_transformer_embeddings:
            if model_name in {"sapbert", "sentence-sapbert", "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"}:
                self._init_sapbert()
            else:
                self._init_sentence_transformer()

    def _init_sentence_transformer(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.backend = "sentence_transformer"
            if hasattr(self.model, "get_sentence_embedding_dimension"):
                self.embedding_dim = int(self.model.get_sentence_embedding_dimension())
            elif hasattr(self.model, "get_embedding_dimension"):
                self.embedding_dim = int(self.model.get_embedding_dimension())
        except Exception as exc:
            warnings.warn(
                f"sentence-transformers unavailable ({type(exc).__name__}: {exc}). Falling back to hashed embeddings.",
                RuntimeWarning,
            )
            self.model = None
            self.backend = "fallback"

    def _init_sapbert(self) -> None:
        try:
            import torch  # type: ignore
            from transformers import AutoModel, AutoTokenizer  # type: ignore

            model_path = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModel.from_pretrained(model_path).to(self.device)
            self.model.eval()
            self.embedding_dim = int(getattr(self.model.config, "hidden_size", self.fallback_dim))
            self.backend = "sapbert"
            self._torch = torch
        except Exception as exc:
            warnings.warn(
                f"SapBERT unavailable ({type(exc).__name__}: {exc}). Falling back to hashed embeddings.",
                RuntimeWarning,
            )
            self.model = None
            self.tokenizer = None
            self.backend = "fallback"

    def _encode_sapbert(self, texts: List[str]) -> Tuple[np.ndarray, List[np.ndarray]]:
        torch = self._torch
        vectors: List[np.ndarray] = []
        token_mats: List[np.ndarray] = []

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_tokens,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.no_grad():
                out = self.model(**encoded)
            hidden = out.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            vectors.append(pooled.detach().cpu().numpy().astype(np.float32))

            hidden_np = hidden.detach().cpu().numpy().astype(np.float32)
            mask_np = encoded["attention_mask"].detach().cpu().numpy()
            for i in range(hidden_np.shape[0]):
                n_tok = int(mask_np[i].sum())
                token_mats.append(hidden_np[i, :n_tok, :].astype(np.float32))

        return np.vstack(vectors).astype(np.float32), token_mats

    def encode_texts(self, texts: List[str]) -> Tuple[np.ndarray, List[np.ndarray]]:
        if self.model is None:
            vectors = np.vstack([_hashed_vector(t, dim=self.embedding_dim) for t in texts]).astype(np.float32)
            token_mats = [_fallback_token_matrix(t, dim=self.embedding_dim, max_tokens=self.max_tokens) for t in texts]
            return vectors, token_mats

        if self.backend == "sapbert":
            return self._encode_sapbert(texts)

        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)

        token_mats: List[np.ndarray] = []
        try:
            token_raw = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                output_value="token_embeddings",
                convert_to_numpy=False,
            )

            for tok in token_raw:
                if hasattr(tok, "detach"):
                    arr = tok.detach().cpu().numpy()
                else:
                    arr = np.asarray(tok)
                arr = arr.astype(np.float32)
                if self.max_tokens and self.max_tokens > 0 and arr.shape[0] > self.max_tokens:
                    arr = arr[: self.max_tokens]
                token_mats.append(arr)
        except Exception:
            token_mats = [_fallback_token_matrix(t, dim=vectors.shape[1], max_tokens=self.max_tokens) for t in texts]

        return vectors, token_mats


_ENCODER_CACHE: Dict[Tuple[str, str, int, int, bool, int], ColumnTextEncoder] = {}


def _get_topological_api():
    from .meta_features.topological import (
        TOPOLOGICAL_FEATURES,
        check_tda_available,
        compute_topological_features,
    )

    return TOPOLOGICAL_FEATURES, check_tda_available, compute_topological_features


def _get_encoder(
    model_name: str,
    device: str,
    batch_size: int,
    max_tokens: int,
    use_transformer_embeddings: bool,
    fallback_dim: int,
) -> ColumnTextEncoder:
    key = (model_name, device, batch_size, max_tokens, use_transformer_embeddings, fallback_dim)
    if key not in _ENCODER_CACHE:
        _ENCODER_CACHE[key] = ColumnTextEncoder(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            max_tokens=max_tokens,
            use_transformer_embeddings=use_transformer_embeddings,
            fallback_dim=fallback_dim,
        )
    return _ENCODER_CACHE[key]


def _spectral_feature_names() -> List[str]:
    base_names = [f.replace("spc_", "") for f in SPECTRAL_FEATURES]
    names: List[str] = []
    for prefix in ("spc_src", "spc_tgt", "spc_combined"):
        names.extend([f"{prefix}_{b}" for b in base_names])
    return names


def _pair_feature_columns(use_topological: bool) -> List[str]:
    cols = list(SYNTAX_FEATURES) + list(CLASSICAL_FEATURES) + _spectral_feature_names()
    if use_topological:
        topo_features, _, _ = _get_topological_api()
        cols += list(topo_features)
    return cols


def build_pair_candidate_matrix(
    pair: TablePair,
    sample_rows: int | None = 5000,
    max_values: int = 200,
    max_values_text: int = 20,
    max_chars_per_value: int = 80,
    max_total_chars_text: int = 2000,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    embedding_device: str = "cpu",
    embedding_batch_size: int = 32,
    max_tokens: int = 64,
    use_transformer_embeddings: bool = True,
    use_topological: bool = True,
    fallback_embedding_dim: int = 384,
) -> pd.DataFrame:
    t_pair_start = time.perf_counter()
    src_df = _load_df(pair.source_csv, sample_rows=sample_rows)
    tgt_df = _load_df(pair.target_csv, sample_rows=sample_rows)

    gt = load_mapping(pair.mapping_json)
    gt_pairs = {(m.source_column.strip().lower(), m.target_column.strip().lower()) for m in gt}

    src_cols = list(src_df.columns)
    tgt_cols = list(tgt_df.columns)

    src_texts = [
        _build_column_text(
            c,
            src_df[c],
            max_values_text=max_values_text,
            max_chars_per_value=max_chars_per_value,
            max_total_chars=max_total_chars_text,
        )
        for c in src_cols
    ]
    tgt_texts = [
        _build_column_text(
            c,
            tgt_df[c],
            max_values_text=max_values_text,
            max_chars_per_value=max_chars_per_value,
            max_total_chars=max_total_chars_text,
        )
        for c in tgt_cols
    ]

    encoder = _get_encoder(
        model_name=embedding_model,
        device=embedding_device,
        batch_size=embedding_batch_size,
        max_tokens=max_tokens,
        use_transformer_embeddings=use_transformer_embeddings,
        fallback_dim=fallback_embedding_dim,
    )

    all_texts = src_texts + tgt_texts
    all_vectors, all_token_mats = encoder.encode_texts(all_texts)

    src_desc = {}
    for i, col in enumerate(src_cols):
        src_desc[col] = {
            "norm": _normalize_name(col),
            "text": src_texts[i],
            "vec": all_vectors[i],
            "tok": all_token_mats[i],
        }

    tgt_desc = {}
    offset = len(src_cols)
    for j, col in enumerate(tgt_cols):
        idx = offset + j
        tgt_desc[col] = {
            "norm": _normalize_name(col),
            "text": tgt_texts[j],
            "vec": all_vectors[idx],
            "tok": all_token_mats[idx],
        }

    compute_topological = None
    tda_diagram_cache = None
    tda_entity_cache = None
    tda_disk_cache_dir = None
    if use_topological:
        topo_features, check_tda_available, compute_topological = _get_topological_api()
        if not check_tda_available():
            raise RuntimeError(
                "Topological features are required but TDA dependencies are unavailable. "
                "Install: ripser, persim, gudhi"
            )
        tda_diagram_cache = {}
        tda_entity_cache = {}
        tda_disk_cache_dir = str(Path.home() / ".cache" / "osirim_tda")

    feat_cols = _pair_feature_columns(use_topological=use_topological)

    rows: List[Dict[str, object]] = []
    for src_col in src_cols:
        s = src_desc[src_col]
        for tgt_col in tgt_cols:
            t = tgt_desc[tgt_col]

            label = 1 if (s["norm"], t["norm"]) in gt_pairs else 0

            feats = {}
            feats.update(compute_syntax_features(s["text"], t["text"]))
            feats.update(compute_classical_features(s["vec"], t["vec"]))
            feats.update(compute_spectral_features_pair(s["tok"], t["tok"]))
            if use_topological and compute_topological is not None:
                src_key = f"{pair.pair_id}::{s['norm']}"
                tgt_key = f"{pair.pair_id}::{t['norm']}"
                feats.update(
                    compute_topological(
                        s["tok"],
                        t["tok"],
                        diagram_cache=tda_diagram_cache,
                        entity_cache=tda_entity_cache,
                        disk_cache_dir=tda_disk_cache_dir,
                        src_key=src_key,
                        tgt_key=tgt_key,
                    )
                )

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
                "label": label,
            }
            for c in feat_cols:
                row[c] = float(feats.get(c, 0.0))

            rows.append(row)

    out_df = pd.DataFrame(rows)
    # Real wall-clock time spent to build meta-features for this table pair.
    out_df["feature_build_sec_pair_real"] = float(time.perf_counter() - t_pair_start)
    return out_df


def _pair_worker(args: Tuple[TablePair, Dict[str, object]]) -> pd.DataFrame:
    pair, kwargs = args
    return build_pair_candidate_matrix(pair, **kwargs)


def build_full_metaspace(
    pairs: Sequence[TablePair],
    sample_rows: int | None = 5000,
    max_values: int = 200,
    max_workers: int = 1,
    max_values_text: int = 20,
    max_chars_per_value: int = 80,
    max_total_chars_text: int = 2000,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    embedding_device: str = "cpu",
    embedding_batch_size: int = 32,
    max_tokens: int = 64,
    use_transformer_embeddings: bool = True,
    use_topological: bool = True,
    fallback_embedding_dim: int = 384,
) -> pd.DataFrame:
    kwargs = {
        "sample_rows": sample_rows,
        "max_values": max_values,
        "max_values_text": max_values_text,
        "max_chars_per_value": max_chars_per_value,
        "max_total_chars_text": max_total_chars_text,
        "embedding_model": embedding_model,
        "embedding_device": embedding_device,
        "embedding_batch_size": embedding_batch_size,
        "max_tokens": max_tokens,
        "use_transformer_embeddings": use_transformer_embeddings,
        "use_topological": use_topological,
        "fallback_embedding_dim": fallback_embedding_dim,
    }

    if max_workers <= 1:
        frames = [build_pair_candidate_matrix(p, **kwargs) for p in tqdm(pairs, desc="meta-features")]
    else:
        jobs = [(p, kwargs) for p in pairs]
        frames: List[pd.DataFrame] = []
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                fut_map = {ex.submit(_pair_worker, job): job[0].pair_id for job in jobs}
                for fut in tqdm(as_completed(fut_map), total=len(fut_map), desc="meta-features"):
                    frames.append(fut.result())
        except (PermissionError, OSError):
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                fut_map = {ex.submit(_pair_worker, job): job[0].pair_id for job in jobs}
                for fut in tqdm(as_completed(fut_map), total=len(fut_map), desc="meta-features-thread"):
                    frames.append(fut.result())

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df
