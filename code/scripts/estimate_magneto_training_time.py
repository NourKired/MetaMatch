#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Quickly estimate Magneto fine-tuning time from a few measured batches."
    )
    p.add_argument("--magneto-finetune-dir", type=Path, required=True)
    p.add_argument("--synthetic-json", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--dataset-name", default="unknown")
    p.add_argument("--model-type", default="mpnet")
    p.add_argument("--serialization", default="header_values_repeat")
    p.add_argument("--augmentation", default="exact_semantic")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--margin", type=float, default=0.5)
    p.add_argument("--measure-batches", type=int, default=8)
    p.add_argument("--measure-eval-batches", type=int, default=4)
    p.add_argument("--synthetic-classes", type=int, default=1000)
    p.add_argument("--synthetic-items-per-class", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _clean_element(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _detect_column_type(values: list[str]) -> str:
    if not values:
        return "unknown"
    numeric = 0
    for value in values:
        try:
            float(value)
            numeric += 1
        except Exception:
            pass
    return "numeric" if numeric >= max(1, len(values) // 2) else "text"


def _sample_values(values: list[str], n: int = 10) -> list[str]:
    cleaned = [_clean_element(v) for v in values if str(v).strip()]
    return cleaned[:n]


def make_synthetic_training_data(n_classes: int, items_per_class: int) -> dict:
    data = {}
    for class_id in range(n_classes):
        columns = {}
        for item_id in range(items_per_class):
            name = f"column_{class_id}_{item_id}"
            columns[name] = [
                f"value {class_id} {item_id} {j}"
                for j in range(10)
            ]
        data[f"class_{class_id}"] = {"original": columns}
    return data


class LocalMagnetoDataset:
    """Minimal copy of Magneto finetune CustomDataset without importing valentine."""

    def __init__(self, data: dict, tokenizer, serialization: str, augmentation: str):
        self.tokenizer = tokenizer
        self.serialization = serialization
        self.labels: list[int] = []
        self.items: list[tuple[str, list[str], int]] = []
        self.cls_token = tokenizer.cls_token or ""
        self.sep_token = tokenizer.sep_token or ""
        self.eos_token = tokenizer.eos_token or ""

        class_id = 0
        for _, categories in data.items():
            for aug_type, columns in categories.items():
                if aug_type in augmentation or aug_type == "original":
                    for column_name, values in columns.items():
                        processed_column_name = (
                            column_name.rsplit("_", 1)[0] if aug_type == "exact" else column_name
                        )
                        tokens = _sample_values(list(values), n=10)
                        self.items.append((processed_column_name, tokens, class_id))
                        self.labels.append(class_id)
            class_id += 1

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[str, int]:
        header, values, class_id = self.items[idx]
        data_type = _detect_column_type(values)
        tokens = [str(token) for token in values]
        return self._serialize(header, data_type, tokens), class_id

    def _serialize(self, header: str, data_type: str, tokens: list[str]) -> str:
        sep = self.sep_token
        if self.serialization == "header_values_default":
            return f"{self.cls_token}{header}{sep}{data_type}{sep}{sep.join(tokens)}"
        if self.serialization == "header_values_prefix":
            return f"{self.cls_token}header:{header}{sep}datatype:{data_type}{sep}values:{', '.join(tokens)}"
        if self.serialization == "header_only":
            return f"{self.cls_token}{header}{self.eos_token}"
        if self.serialization == "header_values_verbose":
            return f"{self.cls_token}Column: {header}{sep}Type: {data_type}{sep}Values: {sep.join(tokens)}{sep}"
        # Magneto train.py default.
        repeated_header = sep.join([header] * 5)
        return f"{self.cls_token}{repeated_header}{sep}{data_type}{sep}{sep.join(tokens)}"


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Magneto's finetune/dataset.py imports both local modules such as
    # train_utils and package modules such as algorithms.magneto...
    sys.path.insert(0, str(args.magneto_finetune_dir))
    sys.path.insert(0, str(args.magneto_finetune_dir.parent))
    for parent in args.magneto_finetune_dir.resolve().parents:
        if (parent / "algorithms" / "magneto").exists():
            sys.path.insert(0, str(parent))
            break

    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from sentence_transformers import SentenceTransformer, losses

    from train_utils import BalancedBatchSampler, sentence_transformer_map
    from transformers import AutoTokenizer

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.synthetic_json is not None:
        with args.synthetic_json.open("r", encoding="utf-8") as f:
            data = json.load(f)
        synthetic_json_source = str(args.synthetic_json)
    else:
        data = make_synthetic_training_data(args.synthetic_classes, args.synthetic_items_per_class)
        synthetic_json_source = "generated_by_script"

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(sentence_transformer_map[args.model_type])
    dataset = LocalMagnetoDataset(
        data,
        tokenizer=tokenizer,
        serialization=args.serialization,
        augmentation=args.augmentation,
    )
    labels = dataset.labels
    sampler = BalancedBatchSampler(labels, batch_size=args.batch_size, n_samples_per_class=2)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=lambda x: ([d[0] for d in x], [d[1] for d in x]),
    )
    dataset_build_sec = time.perf_counter() - t0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(sentence_transformer_map[args.model_type])
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.5)
    loss_fn = losses.BatchHardTripletLoss(
        model=model,
        margin=args.margin,
        distance_metric=losses.BatchHardTripletLossDistanceFunction.cosine_distance,
    )

    n_batches_per_epoch = len(loader)
    n_train_measure = min(args.measure_batches, n_batches_per_epoch)
    if n_train_measure <= 0:
        raise RuntimeError("No batches available. Check synthetic data and batch size.")

    batch_times = []
    for i, batch in enumerate(loader):
        if i >= n_train_measure:
            break
        b0 = time.perf_counter()
        texts, batch_labels = batch
        batch_labels = torch.tensor(batch_labels, dtype=torch.float, device=device)
        optimizer.zero_grad()
        sentence_features = model.tokenize(texts)
        sentence_features = [{k: v.to(device) for k, v in sentence_features.items()}]
        loss = loss_fn(sentence_features, batch_labels)
        loss.backward()
        optimizer.step()
        batch_times.append(time.perf_counter() - b0)

    train_sec_per_batch = float(np.mean(batch_times))

    model.eval()
    eval_batch_times = []
    n_eval_measure = min(args.measure_eval_batches, n_batches_per_epoch)
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_eval_measure:
                break
            e0 = time.perf_counter()
            texts, _ = batch
            _ = model.encode(texts, convert_to_tensor=True, device=device)
            eval_batch_times.append(time.perf_counter() - e0)

    eval_sec_per_batch = float(np.mean(eval_batch_times)) if eval_batch_times else 0.0

    estimated_train_loop_sec = train_sec_per_batch * n_batches_per_epoch * args.epochs
    estimated_eval_encode_sec = eval_sec_per_batch * n_batches_per_epoch * args.epochs
    estimated_total_sec = dataset_build_sec + estimated_train_loop_sec + estimated_eval_encode_sec

    result = {
        "dataset_name": args.dataset_name,
        "synthetic_json": synthetic_json_source,
        "synthetic_classes_arg": int(args.synthetic_classes),
        "synthetic_items_per_class_arg": int(args.synthetic_items_per_class),
        "device": str(device),
        "model_type": args.model_type,
        "serialization": args.serialization,
        "augmentation": args.augmentation,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "n_items": len(dataset),
        "n_classes": int(len(np.unique(labels))),
        "n_batches_per_epoch": int(n_batches_per_epoch),
        "measured_train_batches": int(n_train_measure),
        "measured_eval_batches": int(n_eval_measure),
        "dataset_build_sec": float(dataset_build_sec),
        "train_sec_per_batch_mean": train_sec_per_batch,
        "eval_encode_sec_per_batch_mean": eval_sec_per_batch,
        "estimated_train_loop_sec": float(estimated_train_loop_sec),
        "estimated_eval_encode_sec": float(estimated_eval_encode_sec),
        "estimated_total_sec": float(estimated_total_sec),
        "estimated_total_min": float(estimated_total_sec / 60),
        "estimated_total_hours": float(estimated_total_sec / 3600),
        "note": (
            "Estimate from a few measured batches. It approximates Magneto train.py: "
            "training loop plus per-epoch embedding evaluation; checkpoint save overhead is not included."
        ),
    }

    out_json = args.out_dir / f"magneto_training_time_estimate_{args.dataset_name}.json"
    out_csv = args.out_dir / f"magneto_training_time_estimate_{args.dataset_name}.csv"
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    import pandas as pd

    pd.DataFrame([result]).to_csv(out_csv, index=False)
    print(json.dumps(result, indent=2))
    print(f"Saved:\\n{out_json}\\n{out_csv}")


if __name__ == "__main__":
    main()
