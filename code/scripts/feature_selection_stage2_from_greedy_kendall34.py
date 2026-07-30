#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, f_classif, mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


FAMILIES = ("syn", "cls", "tda")
BASE_SET_KEY = 'best_f1_among_independent__greedy_kendall__{"threshold": 0.7}'


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 2 feature selection from the 34 greedy_kendall threshold=0.7 features."
    )
    p.add_argument("--output-root", type=Path, default=Path("outputs/exp_occidata"))
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-n", type=int, default=120000)
    p.add_argument("--eval-train-sample", type=int, default=0)
    p.add_argument("--k-grid", type=str, default="5,8,10,12,15,20,25,30,34")
    p.add_argument("--run-evaluation", action="store_true")
    p.add_argument("--max-stepwise-k", type=int, default=15)
    p.add_argument("--n-estimators", type=int, default=220)
    p.add_argument("--max-depth", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--base-source-summary", type=Path, default=None)
    p.add_argument("--base-source-method", type=str, default=None)
    p.add_argument("--base-source-hyperparams", type=str, default=None)
    p.add_argument("--base-method-name", type=str, default="base_greedy_kendall_corr_pruned")
    p.add_argument("--base-hyperparams", type=str, default='{"threshold": 0.7}')
    return p.parse_args()


def fam(feature: str) -> str:
    return feature.split("_", 1)[0] if "_" in feature else "other"


def family_counts(features: Sequence[str]) -> Dict[str, int]:
    return {f"n_{prefix}": sum(str(x).startswith(prefix + "_") for x in features) for prefix in FAMILIES}


def read_table(base: Path, stem: str) -> pd.DataFrame:
    from osirim_occidata.io_utils import safe_read_table

    for name in (f"{stem}.parquet", f"{stem}.csv.gz", f"{stem}.csv"):
        p = base / name
        if p.exists():
            return safe_read_table(p)
    raise FileNotFoundError(f"{stem} in {base}")


def load_folds(output_root: Path) -> Dict[int, Dict[str, pd.DataFrame]]:
    folds = {}
    for fold_dir in sorted((output_root / "folds").glob("fold_*"), key=lambda p: int(p.name.split("_")[-1])):
        fid = int(fold_dir.name.split("_")[-1])
        folds[fid] = {"train": read_table(fold_dir, "train"), "test": read_table(fold_dir, "test")}
    if not folds:
        raise RuntimeError(f"No folds found in {output_root / 'folds'}")
    return folds


def parse_json_obj(value: str | None) -> dict:
    if not value:
        return {}
    return json.loads(value)


def same_hyperparams(left: object, right: object) -> bool:
    try:
        return parse_json_obj(str(left)) == parse_json_obj(str(right))
    except Exception:
        return str(left) == str(right)


def split_feature_string(value: object) -> List[str]:
    if value is None or pd.isna(value):
        return []
    return [x.strip() for x in str(value).split(" | ") if x.strip()]


def load_base_features(output_root: Path, args: argparse.Namespace | None = None) -> List[str]:
    if args is not None and args.base_source_method:
        summary_path = args.base_source_summary or (
            output_root
            / "reports"
            / "meeting_baselines_vs_metamatch"
            / "feature_selection_benchmark_methods_and_combinations_STAGE2_PROTOCOL"
            / "benchmark_summary_STAGE2_PROTOCOL.csv"
        )
        df = pd.read_csv(summary_path)
        mask = df["method"].astype(str).eq(args.base_source_method)
        if args.base_source_hyperparams:
            mask = mask & df["hyperparams"].map(lambda x: same_hyperparams(x, args.base_source_hyperparams))
        rows = df.loc[mask]
        if rows.empty:
            raise KeyError(
                f"Missing base source method={args.base_source_method!r} "
                f"hyperparams={args.base_source_hyperparams!r} in {summary_path}"
            )
        features = split_feature_string(rows.iloc[0]["selected"])
        if not features:
            raise ValueError(f"Selected feature list is empty in {summary_path}")
        return features

    p = (
        output_root
        / "reports"
        / "meeting_baselines_vs_metamatch"
        / "feature_correlation_strategic_pruning"
        / "final_correlation_pruning_sets.json"
    )
    obj = json.loads(p.read_text())
    if BASE_SET_KEY not in obj:
        raise KeyError(f"Missing {BASE_SET_KEY} in {p}")
    return list(obj[BASE_SET_KEY])


def make_selection_sample(folds: Dict[int, Dict[str, pd.DataFrame]], features: Sequence[str], sample_n: int, seed: int) -> pd.DataFrame:
    df = pd.concat([v["train"][list(features) + ["label", "pair_id"]] for v in folds.values()], ignore_index=True)
    if sample_n and len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=seed)
    return df.reset_index(drop=True)


def best_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if thresholds.size == 0:
        return 0.5
    p = precision[1:]
    r = recall[1:]
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    return float(thresholds[int(np.nanargmax(f1))])


def build_xgb(seed: int, y_train: np.ndarray, args: argparse.Namespace):
    try:
        from xgboost import XGBClassifier
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("xgboost is not installed.") from exc

    n_pos = max(1, int(y_train.sum()))
    n_neg = max(1, int(len(y_train) - n_pos))
    return XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=seed,
        n_jobs=1,
        scale_pos_weight=n_neg / n_pos,
    )


def build_logistic(kind: str, seed: int):
    if kind == "l1":
        clf = LogisticRegression(penalty="l1", solver="saga", C=0.5, max_iter=5000, n_jobs=1, random_state=seed)
    elif kind == "elasticnet":
        clf = LogisticRegression(
            penalty="elasticnet", solver="saga", l1_ratio=0.5, C=0.5, max_iter=5000, n_jobs=1, random_state=seed
        )
    else:
        clf = LogisticRegression(penalty="l2", solver="lbfgs", C=1.0, max_iter=3000, n_jobs=1, random_state=seed)
    return make_pipeline(RobustScaler(), clf)


def predict_scores(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    return model.decision_function(x)


def pair_metrics(df: pd.DataFrame, y_pred_col: str = "pred") -> pd.DataFrame:
    rows = []
    for pair_id, g in df.groupby("pair_id", sort=False):
        y = g["label"].to_numpy().astype(int)
        pred = g[y_pred_col].to_numpy().astype(int)
        rows.append(
            {
                "pair_id": pair_id,
                "precision": precision_score(y, pred, zero_division=0),
                "recall": recall_score(y, pred, zero_division=0),
                "f1": f1_score(y, pred, zero_division=0),
                "n_rows": len(g),
                "n_positive": int(y.sum()),
            }
        )
    return pd.DataFrame(rows)


def evaluate_feature_set(
    features: Sequence[str],
    folds: Dict[int, Dict[str, pd.DataFrame]],
    args: argparse.Namespace,
    model_kind: str = "xgb",
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    fold_rows = []
    pair_rows = []
    features = list(features)
    for fid, obj in folds.items():
        train = obj["train"]
        test = obj["test"]
        fit_train = train
        if args.eval_train_sample and len(fit_train) > args.eval_train_sample:
            fit_train = fit_train.sample(n=args.eval_train_sample, random_state=args.seed + fid)
        x_train = fit_train[features].to_numpy()
        y_train = fit_train["label"].to_numpy().astype(int)
        x_threshold = train[features].to_numpy()
        y_threshold = train["label"].to_numpy().astype(int)
        x_test = test[features].to_numpy()
        y_test = test["label"].to_numpy().astype(int)
        model = build_xgb(args.seed + fid, y_train, args) if model_kind == "xgb" else build_logistic(model_kind, args.seed + fid)
        model.fit(x_train, y_train)
        train_score = predict_scores(model, x_threshold)
        test_score = predict_scores(model, x_test)
        threshold = best_threshold(y_threshold, train_score)
        pred = (test_score >= threshold).astype(int)
        eval_df = test[["pair_id", "label"]].copy()
        eval_df["pred"] = pred
        eval_df["score"] = test_score
        pr = pair_metrics(eval_df)
        pr["fold_id"] = fid
        pair_rows.append(pr)
        fold_rows.append(
            {
                "fold_id": fid,
                "threshold_opt_train": threshold,
                "precision": precision_score(y_test, pred, zero_division=0),
                "recall": recall_score(y_test, pred, zero_division=0),
                "f1": f1_score(y_test, pred, zero_division=0),
            }
        )
    pair_df = pd.concat(pair_rows, ignore_index=True)
    fold_df = pd.DataFrame(fold_rows)
    summary = {
        "n_selected": len(features),
        "selected": " | ".join(features),
        "mean_precision_pairs": float(pair_df["precision"].mean()),
        "std_precision_pairs": float(pair_df["precision"].std(ddof=0)),
        "mean_recall_pairs": float(pair_df["recall"].mean()),
        "std_recall_pairs": float(pair_df["recall"].std(ddof=0)),
        "mean_f1_pairs": float(pair_df["f1"].mean()),
        "std_f1_pairs": float(pair_df["f1"].std(ddof=0)),
        "n_pair_ids_eval": int(pair_df["pair_id"].nunique()),
        "mean_f1_folds": float(fold_df["f1"].mean()),
        "std_f1_folds": float(fold_df["f1"].std(ddof=0)),
        **family_counts(features),
    }
    return summary, pair_df, fold_df


def split_inner(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    idx_train, idx_val = next(splitter.split(df, groups=df["pair_id"]))
    return df.iloc[idx_train].copy(), df.iloc[idx_val].copy()


def inner_score(features: Sequence[str], df: pd.DataFrame, args: argparse.Namespace, model_kind: str = "xgb") -> float:
    train, val = split_inner(df, args.seed)
    x_train = train[list(features)].to_numpy()
    y_train = train["label"].to_numpy().astype(int)
    x_val = val[list(features)].to_numpy()
    y_val = val["label"].to_numpy().astype(int)
    model = build_xgb(args.seed, y_train, args) if model_kind == "xgb" else build_logistic(model_kind, args.seed)
    model.fit(x_train, y_train)
    threshold = best_threshold(y_train, predict_scores(model, x_train))
    pred = (predict_scores(model, x_val) >= threshold).astype(int)
    return float(f1_score(y_val, pred, zero_division=0))


def rank_mi(df: pd.DataFrame, features: Sequence[str], seed: int) -> List[str]:
    x = df[list(features)].fillna(df[list(features)].median()).to_numpy()
    y = df["label"].to_numpy().astype(int)
    scores = mutual_info_classif(x, y, random_state=seed)
    return [f for _, f in sorted(zip(scores, features), reverse=True)]


def rank_anova(df: pd.DataFrame, features: Sequence[str]) -> List[str]:
    x = df[list(features)].fillna(df[list(features)].median()).to_numpy()
    y = df["label"].to_numpy().astype(int)
    scores, _ = f_classif(x, y)
    scores = np.nan_to_num(scores, nan=-np.inf)
    return [f for _, f in sorted(zip(scores, features), reverse=True)]


def rank_auc(df: pd.DataFrame, features: Sequence[str]) -> List[str]:
    y = df["label"].to_numpy().astype(int)
    rows = []
    for f in features:
        x = df[f].fillna(df[f].median()).to_numpy()
        if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
            auc = 0.5
        else:
            auc = roc_auc_score(y, x)
        rows.append((abs(float(auc) - 0.5), f))
    return [f for _, f in sorted(rows, reverse=True)]


def rank_l1_like(df: pd.DataFrame, features: Sequence[str], seed: int, kind: str) -> List[str]:
    x = df[list(features)].fillna(df[list(features)].median()).to_numpy()
    y = df["label"].to_numpy().astype(int)
    model = build_logistic(kind, seed)
    model.fit(x, y)
    coef = np.abs(model.named_steps["logisticregression"].coef_[0])
    return [f for _, f in sorted(zip(coef, features), reverse=True)]


def rank_forest(df: pd.DataFrame, features: Sequence[str], seed: int) -> List[str]:
    x = df[list(features)].fillna(df[list(features)].median()).to_numpy()
    y = df["label"].to_numpy().astype(int)
    model = RandomForestClassifier(n_estimators=400, max_depth=None, min_samples_leaf=2, n_jobs=1, random_state=seed)
    model.fit(x, y)
    return [f for _, f in sorted(zip(model.feature_importances_, features), reverse=True)]


def rank_xgb(df: pd.DataFrame, features: Sequence[str], args: argparse.Namespace) -> List[str]:
    x = df[list(features)].fillna(df[list(features)].median()).to_numpy()
    y = df["label"].to_numpy().astype(int)
    model = build_xgb(args.seed, y, args)
    model.fit(x, y)
    return [f for _, f in sorted(zip(model.feature_importances_, features), reverse=True)]


def rank_permutation(df: pd.DataFrame, features: Sequence[str], args: argparse.Namespace, base_model: str = "xgb") -> List[str]:
    train, val = split_inner(df, args.seed)
    x_train = train[list(features)].fillna(train[list(features)].median()).to_numpy()
    y_train = train["label"].to_numpy().astype(int)
    x_val = val[list(features)].fillna(train[list(features)].median()).to_numpy()
    y_val = val["label"].to_numpy().astype(int)
    model = build_xgb(args.seed, y_train, args) if base_model == "xgb" else build_logistic("l2", args.seed)
    model.fit(x_train, y_train)
    result = permutation_importance(model, x_val, y_val, scoring="f1", n_repeats=8, random_state=args.seed, n_jobs=1)
    return [f for _, f in sorted(zip(result.importances_mean, features), reverse=True)]


def rank_shap(df: pd.DataFrame, features: Sequence[str], args: argparse.Namespace) -> List[str]:
    try:
        import shap
    except Exception:
        return []
    x = df[list(features)].fillna(df[list(features)].median()).to_numpy()
    y = df["label"].to_numpy().astype(int)
    model = build_xgb(args.seed, y, args)
    model.fit(x, y)
    values = shap.TreeExplainer(model).shap_values(x)
    values = values[1] if isinstance(values, list) else values
    scores = np.abs(values).mean(axis=0)
    return [f for _, f in sorted(zip(scores, features), reverse=True)]


def rank_rfe(df: pd.DataFrame, features: Sequence[str], seed: int) -> List[str]:
    x = df[list(features)].fillna(df[list(features)].median()).to_numpy()
    y = df["label"].to_numpy().astype(int)
    estimator = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=3000, n_jobs=1, random_state=seed)
    model = make_pipeline(RobustScaler(), RFE(estimator, n_features_to_select=1, step=1))
    model.fit(x, y)
    ranking = model.named_steps["rfe"].ranking_
    return [f for _, f in sorted(zip(ranking, features))]


def rank_boruta_like(df: pd.DataFrame, features: Sequence[str], args: argparse.Namespace, n_iter: int = 12) -> List[str]:
    rng = np.random.default_rng(args.seed)
    x_real = df[list(features)].fillna(df[list(features)].median()).to_numpy()
    y = df["label"].to_numpy().astype(int)
    hits = dict.fromkeys(features, 0)
    for i in range(n_iter):
        shadow = np.apply_along_axis(rng.permutation, 0, x_real)
        x = np.hstack([x_real, shadow])
        names = list(features) + [f"shadow_{f}" for f in features]
        model = build_xgb(args.seed + i, y, args)
        model.fit(x, y)
        imp = dict(zip(names, model.feature_importances_))
        cutoff = max(imp[f"shadow_{f}"] for f in features)
        for f in features:
            hits[f] += int(imp[f] > cutoff)
    return [f for f, _ in sorted(hits.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)]


def rank_mrmr(df: pd.DataFrame, features: Sequence[str], seed: int, penalty: float = 1.0) -> List[str]:
    relevance = dict(zip(features, mutual_info_classif(df[list(features)].fillna(0), df["label"].astype(int), random_state=seed)))
    corr = df[list(features)].corr(method="spearman").abs().fillna(0.0)
    selected = []
    remaining = list(features)
    while remaining:
        def score(f: str) -> float:
            red = float(corr.loc[f, selected].mean()) if selected else 0.0
            return float(relevance[f] - penalty * red)
        best = max(remaining, key=score)
        selected.append(best)
        remaining.remove(best)
    return selected


def forward_stepwise(df: pd.DataFrame, pool: Sequence[str], args: argparse.Namespace, max_k: int) -> List[List[str]]:
    selected: List[str] = []
    path = []
    total_steps = min(max_k, len(pool))
    for step in range(total_steps):
        remaining = [f for f in pool if f not in selected]
        print(f"[stepwise forward] step {step + 1}/{total_steps} | selected={len(selected)} | candidates={len(remaining)}", flush=True)
        candidates = []
        for j, f in enumerate(remaining, start=1):
            print(f"  [forward {step + 1}/{total_steps}] candidate {j}/{len(remaining)}: +{f}", flush=True)
            candidates.append((inner_score(selected + [f], df, args), f))
        if not candidates:
            break
        best_score, best = max(candidates, key=lambda x: x[0])
        selected.append(best)
        path.append(list(selected))
        print(f"[stepwise forward] selected +{best} | inner_f1={best_score:.6f} | k={len(selected)}", flush=True)
    return path


def backward_stepwise(df: pd.DataFrame, pool: Sequence[str], args: argparse.Namespace, min_k: int) -> List[List[str]]:
    selected = list(pool)
    path = [list(selected)]
    total_steps = max(0, len(selected) - min_k)
    step = 0
    while len(selected) > min_k:
        step += 1
        print(f"[stepwise backward] step {step}/{total_steps} | selected={len(selected)} | candidates={len(selected)}", flush=True)
        candidates = []
        for j, drop in enumerate(selected, start=1):
            print(f"  [backward {step}/{total_steps}] candidate {j}/{len(selected)}: -{drop}", flush=True)
            candidates.append((inner_score([f for f in selected if f != drop], df, args), drop))
        best_score, drop = max(candidates, key=lambda x: x[0])
        selected.remove(drop)
        path.append(list(selected))
        print(f"[stepwise backward] removed -{drop} | inner_f1={best_score:.6f} | k={len(selected)}", flush=True)
    return path


def family_quota(rank: Sequence[str], k: int) -> List[str]:
    quotas = {p: max(1, int(round(k / len(FAMILIES)))) for p in FAMILIES}
    selected = []
    for p in FAMILIES:
        selected.extend([f for f in rank if fam(f) == p][: quotas[p]])
    for f in rank:
        if len(selected) >= k:
            break
        if f not in selected:
            selected.append(f)
    return list(dict.fromkeys(selected))[:k]


@dataclass
class CandidateSet:
    method: str
    hyperparams: dict
    features: List[str]
    note: str = ""


def make_candidates(base_features: Sequence[str], df: pd.DataFrame, args: argparse.Namespace, k_grid: Sequence[int]) -> List[CandidateSet]:
    base_method_name = getattr(args, "base_method_name", "base_greedy_kendall_corr_pruned")
    base_hyperparams = parse_json_obj(getattr(args, "base_hyperparams", '{"threshold": 0.7}'))
    candidates: List[CandidateSet] = [CandidateSet(base_method_name, base_hyperparams, list(base_features))]
    rankers: Dict[str, Callable[[], List[str]]] = {
        "mi_topk": lambda: rank_mi(df, base_features, args.seed),
        "anova_f_topk": lambda: rank_anova(df, base_features),
        "univariate_auc_topk": lambda: rank_auc(df, base_features),
        "l1_logistic_topk": lambda: rank_l1_like(df, base_features, args.seed, "l1"),
        "elasticnet_logistic_topk": lambda: rank_l1_like(df, base_features, args.seed, "elasticnet"),
        "random_forest_importance_topk": lambda: rank_forest(df, base_features, args.seed),
        "xgboost_importance_topk": lambda: rank_xgb(df, base_features, args),
        "permutation_importance_xgb_topk": lambda: rank_permutation(df, base_features, args, "xgb"),
        "shap_xgboost_topk": lambda: rank_shap(df, base_features, args),
        "rfe_logistic_topk": lambda: rank_rfe(df, base_features, args.seed),
        "boruta_like_xgb_topk": lambda: rank_boruta_like(df, base_features, args),
        "mrmr_topk_penalty_05": lambda: rank_mrmr(df, base_features, args.seed, 0.5),
        "mrmr_topk_penalty_10": lambda: rank_mrmr(df, base_features, args.seed, 1.0),
        "mrmr_topk_penalty_20": lambda: rank_mrmr(df, base_features, args.seed, 2.0),
    }
    computed_ranks: Dict[str, List[str]] = {}
    for method, fn in rankers.items():
        rank = fn()
        if not rank:
            candidates.append(CandidateSet(method, {"status": "skipped"}, [], "Optional dependency missing or method skipped."))
            continue
        computed_ranks[method] = rank
        for k in k_grid:
            candidates.append(CandidateSet(method, {"k": int(k)}, rank[: min(int(k), len(rank))]))
        if method in {"mi_topk", "xgboost_importance_topk", "shap_xgboost_topk", "permutation_importance_xgb_topk"}:
            for k in k_grid:
                candidates.append(CandidateSet(f"family_quota_{method}", {"k": int(k)}, family_quota(rank, int(k))))

    if computed_ranks:
        vote_scores: Dict[str, float] = dict.fromkeys(base_features, 0.0)
        for rank in computed_ranks.values():
            for pos, f in enumerate(rank):
                vote_scores[f] += 1.0 / (1 + pos)
        vote_rank = [f for f, _ in sorted(vote_scores.items(), key=lambda kv: kv[1], reverse=True)]
        for k in k_grid:
            candidates.append(CandidateSet("rank_voting_ensemble_topk", {"k": int(k)}, vote_rank[: min(int(k), len(vote_rank))]))

    fwd_path = forward_stepwise(df, base_features, args, max_k=min(args.max_stepwise_k, max(k_grid)))
    for subset in fwd_path:
        if len(subset) in k_grid or len(subset) <= args.max_stepwise_k:
            candidates.append(CandidateSet("forward_stepwise_xgb_inner_valid", {"k": len(subset)}, subset))

    bwd_path = backward_stepwise(df, base_features, args, min_k=min(k_grid))
    for subset in bwd_path:
        if len(subset) in k_grid:
            candidates.append(CandidateSet("backward_stepwise_xgb_inner_valid", {"k": len(subset)}, subset))

    # Hybrid examples: prefilter with MI, then rank remaining with SHAP/permutation/voting when available.
    mi_rank = computed_ranks.get("mi_topk", list(base_features))
    for prefilter in (12, 20, 28):
        sub_pool = mi_rank[: min(prefilter, len(mi_rank))]
        sub_df = df[list(sub_pool) + ["label", "pair_id"]]
        for final_k in [k for k in k_grid if k <= len(sub_pool)]:
            candidates.append(
                CandidateSet("hybrid_mi_prefilter_mrmr", {"prefilter": prefilter, "k": int(final_k)}, rank_mrmr(sub_df, sub_pool, args.seed, 1.0)[:final_k])
            )
    dedup = {}
    for c in candidates:
        if not c.features:
            key = (c.method, json.dumps(c.hyperparams, sort_keys=True), "")
        else:
            key = (c.method, json.dumps(c.hyperparams, sort_keys=True), "|".join(c.features))
        dedup[key] = c
    return list(dedup.values())


def redundancy_stats(features: Sequence[str], sample_df: pd.DataFrame) -> dict:
    if len(features) < 2:
        return {"max_abs_spearman": 0.0, "median_abs_spearman": 0.0, "n_pairs_abs_spearman_ge_0_5": 0, "n_pairs_abs_spearman_ge_0_7": 0}
    vals = sample_df[list(features)].corr(method="spearman").abs().where(np.triu(np.ones((len(features), len(features))), 1).astype(bool)).stack()
    return {
        "max_abs_spearman": float(vals.max()) if len(vals) else 0.0,
        "median_abs_spearman": float(vals.median()) if len(vals) else 0.0,
        "n_pairs_abs_spearman_ge_0_5": int((vals >= 0.5).sum()),
        "n_pairs_abs_spearman_ge_0_7": int((vals >= 0.7).sum()),
    }


def write_html_report(summary_df: pd.DataFrame, methods_df: pd.DataFrame, output_html: Path) -> None:
    payload = {
        "summary": summary_df.fillna("").to_dict("records"),
        "methods": methods_df.fillna("").to_dict("records"),
    }
    js = json.dumps(payload, ensure_ascii=False)
    html_text = f"""<!doctype html><html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Feature selection stage 2</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif;margin:0;color:#172033;background:#fff;line-height:1.28}}main{{max-width:1320px;margin:auto;padding:16px}}h1{{font-size:26px;margin:0 0 8px}}h2{{font-size:19px;margin:20px 0 8px;border-top:1px solid #d9deea;padding-top:12px}}h3{{font-size:15px;margin:12px 0 5px}}p,li{{font-size:12px;margin:4px 0}}code{{background:#eef2f7;border-radius:3px;padding:1px 3px}}.dash{{border:1px solid #d9deea;border-radius:7px;padding:10px;margin:8px 0;background:#fff}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:8px}}.card{{border:1px solid #d9deea;border-radius:6px;background:#f6f8fb;padding:8px}}.controls{{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:7px}}select{{font-size:12px;padding:3px 6px;max-width:760px}}.metric{{display:grid;grid-template-columns:repeat(6,1fr);gap:5px;margin:6px 0}}.kpi{{border:1px solid #d9deea;background:#f6f8fb;border-radius:5px;padding:6px}}.kpi b{{display:block;font-size:16px}}.row{{display:grid;grid-template-columns:1.35fr .65fr;gap:10px;align-items:start}}.chips{{display:flex;flex-wrap:wrap;gap:3px;border:1px solid #d9deea;border-radius:5px;padding:5px;max-height:180px;overflow:auto}}.chip{{font-size:10px;padding:2px 5px;border-radius:999px;color:white}}.syn{{background:#2f9e44}}.cls{{background:#1c7ed6}}.tda{{background:#f08c00}}.other{{background:#777}}svg{{width:100%;height:auto;border:1px solid #d9deea;border-radius:5px;background:white}}table{{width:100%;border-collapse:collapse;font-size:11px}}th,td{{border-bottom:1px solid #eef2f7;padding:4px 6px;text-align:left;vertical-align:top}}th{{background:#f6f8fb;position:sticky;top:0}}.tablebox{{max-height:420px;overflow:auto;border:1px solid #d9deea;border-radius:5px}}.muted{{color:#647084}}.best{{color:#d62828;font-weight:700;margin:4px 0 7px}}.small{{font-size:11px;color:#647084}}
</style></head><body><main>
<h1>Feature selection - étape 2 depuis greedy_kendall threshold=0.7</h1>
<p class="muted">Point de départ: 34 features déjà nettoyées par corrélation. Protocole prévu: threshold_opt_train appris sur le train de chaque fold, appliqué au test, puis métriques moyennes sur les 551 paires de tables.</p>
<h2>1. Méthodes testées</h2><div id="methodCards" class="grid"></div>
<h2>2. Dashboard interactif</h2><div class="dash"><div class="controls"><label>Méthode <select id="methodSelect"></select></label><label>Hyperparamètre / k <select id="paramSelect"></select></label><label>Top F1 <select id="topN"><option value="20">20</option><option value="30" selected>30</option><option value="50">50</option><option value="9999">Tout</option></select></label></div><div id="best" class="best"></div><div id="metrics" class="metric"></div><div class="row"><div><h3>Features sélectionnées</h3><div id="chips" class="chips"></div></div><div><h3>Répartition par famille</h3><svg id="famSvg" viewBox="0 0 460 170"></svg></div></div><h3>Comparaison F1</h3><p class="small">Barres triées par F1 moyen sur les 551 paires de tables. La meilleure configuration est en rouge.</p><svg id="barSvg" viewBox="0 0 1420 720"></svg></div>
<h2>3. Table complète</h2><div class="tablebox" id="table"></div>
<script>const DATA={js};
function esc(s){{return String(s??'').replace(/[&<>"']/g,m=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[m]));}}
function fam(f){{return String(f).split('_')[0];}}
function color(f){{return {{syn:'#2f9e44',cls:'#1c7ed6',tda:'#f08c00'}}[f]||'#777';}}
function fmt(x){{x=Number(x); return Number.isFinite(x)?x.toFixed(2):'';}}
function fmtCell(k,x){{const n=Number(x); if(!Number.isFinite(n)) return x??''; if(k==='n_selected'||k==='n_pair_ids_eval'||k==='n_syn'||k==='n_cls'||k==='n_tda'||String(k).startsWith('n_pairs_')) return String(Math.round(n)); if(String(k).includes('f1')||String(k).includes('precision')||String(k).includes('recall')||String(k).includes('spearman')) return n.toFixed(2); return n.toFixed(2);}}
function fmtPct1(x){{const n=Number(x); if(!Number.isFinite(n)) return x??''; return (n*100).toFixed(1)+'%';}}
function E(n,a,t){{const e=document.createElementNS('http://www.w3.org/2000/svg',n); for(const k in a)e.setAttribute(k,a[k]); if(t!=null)e.textContent=t; return e;}}
function hpObj(r){{try{{return JSON.parse(r.hyperparams||'{{}}');}}catch(e){{return {{}};}}}}
function paramLabel(r){{const h=hpObj(r); const parts=Object.keys(h).map(k=>k+'='+h[k]); return (parts.length?parts.join(', '):r.hyperparams)+' | features='+fmtCell('n_selected',r.n_selected)+' | F1='+fmtCell('mean_f1_pairs',r.mean_f1_pairs);}}
function label(r){{return r.method+' | '+r.hyperparams+' | k='+fmtCell('n_selected',r.n_selected);}}
function shortLabel(r){{return r.method+' | '+r.hyperparams+' | k='+fmtCell('n_selected',r.n_selected);}}
function renderCards(){{document.getElementById('methodCards').innerHTML=DATA.methods.map(r=>'<div class="card"><h3>'+esc(r.method)+'</h3><p>'+esc(r.explanation)+'</p><p><b>Hyperparamètres:</b> '+esc(r.hyperparams)+'</p></div>').join('');}}
function methodRows(){{const m=document.getElementById('methodSelect').value; return DATA.summary.filter(r=>r.method===m).sort((a,b)=>Number(b.mean_f1_pairs||0)-Number(a.mean_f1_pairs||0));}}
function renderOptions(){{DATA.summary.sort((a,b)=>Number(b.mean_f1_pairs||0)-Number(a.mean_f1_pairs||0)); const ms=document.getElementById('methodSelect'); const best=DATA.summary[0]; const methods=Array.from(new Set(DATA.summary.map(r=>r.method))).sort(); ms.innerHTML=methods.map(m=>'<option value="'+esc(m)+'">'+esc(m)+'</option>').join(''); ms.value=best.method; renderParamOptions();}}
function renderParamOptions(){{const ps=document.getElementById('paramSelect'); const rows=methodRows(); ps.innerHTML=rows.map(r=>'<option value="'+DATA.summary.indexOf(r)+'">'+esc(paramLabel(r))+'</option>').join(''); if(rows.length) ps.value=String(DATA.summary.indexOf(rows[0]));}}
function currentRow(){{const idx=Number(document.getElementById('paramSelect').value); return DATA.summary[idx] || methodRows()[0] || DATA.summary[0];}}
function render(){{const r=currentRow(); const best=DATA.summary[0]; document.getElementById('best').textContent='Meilleure config globale: '+label(best)+' | F1='+fmtCell('mean_f1_pairs',best.mean_f1_pairs); document.getElementById('metrics').innerHTML=['mean_f1_pairs','std_f1_pairs','mean_precision_pairs','mean_recall_pairs','n_selected','n_pair_ids_eval'].map(k=>'<div class="kpi"><b>'+esc(fmtCell(k,r[k]))+'</b>'+k+'</div>').join(''); const feats=String(r.selected||'').split(' | ').filter(Boolean); document.getElementById('chips').innerHTML=feats.map(f=>'<span class="chip '+fam(f)+'">'+esc(f)+'</span>').join(''); renderFam(feats); renderBars();}}
function renderFam(feats){{const svg=document.getElementById('famSvg'); svg.innerHTML=''; ['syn','cls','tda'].forEach((p,i)=>{{const n=feats.filter(f=>fam(f)==p).length; svg.appendChild(E('text',{{x:18,y:35+i*42,'font-size':15,fill:'#172033','font-weight':700}},p)); svg.appendChild(E('rect',{{x:80,y:18+i*42,width:320,height:22,fill:'#eef2f7',rx:3}})); svg.appendChild(E('rect',{{x:80,y:18+i*42,width:Math.max(1,n*14),height:22,fill:color(p),rx:3}})); svg.appendChild(E('text',{{x:410,y:35+i*42,'font-size':14,fill:'#172033'}},n));}});}}
function renderBars(){{const svg=document.getElementById('barSvg'); svg.innerHTML=''; const n=Number(document.getElementById('topN').value)||30; const rows=DATA.summary.slice(0,n); const h=Math.max(720,42+rows.length*22); svg.setAttribute('viewBox','0 0 1420 '+h); const max=Math.max(...rows.map(r=>Number(r.mean_f1_pairs)||0),.01); rows.forEach((r,i)=>{{const y=30+i*22; const v=Number(r.mean_f1_pairs)||0; const w=600*v/max; const isBest=i===0; svg.appendChild(E('text',{{x:10,y:y+12,'font-size':10,fill:'#172033'}},shortLabel(r).slice(0,88))); svg.appendChild(E('rect',{{x:610,y:y,width:600,height:13,fill:'#eef2f7',rx:2}})); svg.appendChild(E('rect',{{x:610,y:y,width:w,height:13,fill:isBest?'#d62828':'#1c7ed6',rx:2}})); svg.appendChild(E('text',{{x:1230,y:y+12,'font-size':11,fill:isBest?'#d62828':'#172033','font-weight':isBest?700:400}},fmtPct1(v))); svg.appendChild(E('text',{{x:1290,y:y+12,'font-size':10,fill:'#647084'}},'k='+fmtCell('n_selected',r.n_selected)));}});}}
function renderTable(){{const cols=['method','hyperparams','n_selected','mean_f1_pairs','std_f1_pairs','mean_precision_pairs','mean_recall_pairs','n_syn','n_cls','n_tda','max_abs_spearman','selected']; document.getElementById('table').innerHTML='<table><thead><tr>'+cols.map(c=>'<th>'+c+'</th>').join('')+'</tr></thead><tbody>'+DATA.summary.map(r=>'<tr>'+cols.map(c=>'<td>'+esc(fmtCell(c,r[c])||'')+'</td>').join('')+'</tr>').join('')+'</tbody></table>';}}
document.getElementById('methodSelect').addEventListener('change',()=>{{renderParamOptions(); render();}}); document.getElementById('paramSelect').addEventListener('change',render); document.getElementById('topN').addEventListener('change',renderBars); renderCards(); renderOptions(); render(); renderTable();
</script></main></body></html>"""
    output_html.write_text(html_text)


def methods_catalog() -> pd.DataFrame:
    rows = [
        ("base_greedy_kendall_corr_pruned", "threshold=0.7", "Référence de départ: les 34 features restantes après suppression stratégique des corrélations."),
        ("base_greedy_pearson_corr_pruned", "threshold=0.85", "Référence de départ: features restantes après pruning greedy Pearson."),
        ("mi_topk", "k", "Filtre supervisé: garde les features avec la plus grande information mutuelle avec le label."),
        ("anova_f_topk", "k", "Filtre supervisé linéaire: score F entre chaque feature et le label."),
        ("univariate_auc_topk", "k", "Filtre univarié: garde les features qui séparent le mieux match/non-match en AUC."),
        ("l1_logistic_topk", "k,C", "Embedded: la régression logistique L1 pousse des coefficients à zéro."),
        ("elasticnet_logistic_topk", "k,C,l1_ratio", "Embedded: compromis L1/L2, moins brutal que L1."),
        ("random_forest_importance_topk", "k,n_estimators", "Embedded non linéaire: importance moyenne des arbres Random Forest."),
        ("xgboost_importance_topk", "k,n_estimators,max_depth,learning_rate", "Embedded non linéaire: importance des splits XGBoost."),
        ("permutation_importance_xgb_topk", "k,n_repeats", "Wrapper: mesure la baisse de F1 quand on mélange une feature."),
        ("shap_xgboost_topk", "k", "Explicabilité: moyenne des valeurs absolues SHAP par feature."),
        ("rfe_logistic_topk", "k", "Recursive Feature Elimination: retire progressivement les features les moins utiles."),
        ("boruta_like_xgb_topk", "k,n_iter", "Compare chaque feature à des features shadow aléatoires."),
        ("mrmr_topk", "k,penalty", "Maximum relevance minimum redundancy: MI avec label moins redondance Spearman."),
        ("forward_stepwise_xgb_inner_valid", "k,max_stepwise_k", "Wrapper: ajoute à chaque étape la feature qui améliore le plus le F1 validation interne."),
        ("backward_stepwise_xgb_inner_valid", "k,min_k", "Wrapper: part des 34 features et retire une feature à la fois."),
        ("family_quota_*", "k", "Variante qui force une représentation syn/cls/tda dans le top-k."),
        ("rank_voting_ensemble_topk", "k", "Agrège les votes des rankers pour trouver les features stables."),
        ("hybrid_mi_prefilter_mrmr", "prefilter,k", "Préfiltre MI puis mRMR pour combiner utilité label et faible redondance."),
    ]
    return pd.DataFrame(rows, columns=["method", "hyperparams", "explanation"])


def planned_candidate_grid(base_features: Sequence[str], k_grid: Sequence[int], args: argparse.Namespace | None = None) -> pd.DataFrame:
    base_method_name = getattr(args, "base_method_name", "base_greedy_kendall_corr_pruned") if args is not None else "base_greedy_kendall_corr_pruned"
    base_hyperparams = parse_json_obj(getattr(args, "base_hyperparams", '{"threshold": 0.7}') if args is not None else '{"threshold": 0.7}')
    rows = [
        {
            "method": base_method_name,
            "hyperparams": json.dumps(base_hyperparams, sort_keys=True),
            "n_selected": len(base_features),
            "selected": " | ".join(base_features),
            "note": "Référence de départ déjà disponible.",
            **family_counts(base_features),
        }
    ]
    methods_with_k = [
        "mi_topk",
        "anova_f_topk",
        "univariate_auc_topk",
        "l1_logistic_topk",
        "elasticnet_logistic_topk",
        "random_forest_importance_topk",
        "xgboost_importance_topk",
        "permutation_importance_xgb_topk",
        "shap_xgboost_topk",
        "rfe_logistic_topk",
        "boruta_like_xgb_topk",
        "mrmr_topk_penalty_05",
        "mrmr_topk_penalty_10",
        "mrmr_topk_penalty_20",
        "family_quota_mi_topk",
        "family_quota_xgboost_importance_topk",
        "family_quota_shap_xgboost_topk",
        "family_quota_permutation_importance_xgb_topk",
        "rank_voting_ensemble_topk",
        "forward_stepwise_xgb_inner_valid",
        "backward_stepwise_xgb_inner_valid",
    ]
    for method in methods_with_k:
        for k in k_grid:
            if k <= len(base_features):
                rows.append(
                    {
                        "method": method,
                        "hyperparams": json.dumps({"k": int(k)}, sort_keys=True),
                        "n_selected": int(k),
                        "selected": "",
                        "note": "Plan seulement: features calculées pendant --run-evaluation.",
                        "n_syn": "",
                        "n_cls": "",
                        "n_tda": "",
                    }
                )
    for prefilter in (12, 20, 28):
        for k in k_grid:
            if k <= prefilter <= len(base_features):
                rows.append(
                    {
                        "method": "hybrid_mi_prefilter_mrmr",
                        "hyperparams": json.dumps({"prefilter": prefilter, "k": int(k)}, sort_keys=True),
                        "n_selected": int(k),
                        "selected": "",
                        "note": "Plan seulement: features calculées pendant --run-evaluation.",
                        "n_syn": "",
                        "n_cls": "",
                        "n_tda": "",
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    out_dir = args.out_dir or (
        output_root
        / "reports"
        / "meeting_baselines_vs_metamatch"
        / "feature_selection_stage2_from_greedy_kendall34"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    k_grid = [int(x) for x in args.k_grid.split(",") if x.strip()]

    base_features = load_base_features(output_root, args)

    methods_df = methods_catalog()
    methods_df.to_csv(out_dir / "methods_catalog.csv", index=False)
    (out_dir / "base_features.json").write_text(json.dumps(base_features, indent=2))

    if not args.run_evaluation:
        candidate_df = planned_candidate_grid(base_features, k_grid, args)
        candidate_df.to_csv(out_dir / "candidate_feature_sets_PLAN_ONLY.csv", index=False)
        write_html_report(candidate_df, methods_df, out_dir / "feature_selection_stage2_report_PLAN_ONLY.html")
        print(f"Fast plan written to {out_dir}")
        print("No model was trained. Run again with --run-evaluation to compute F1/precision/recall on the 551 pair_id.")
        return

    folds = load_folds(output_root)
    sample_df = make_selection_sample(folds, base_features, args.sample_n, args.seed)

    candidates = make_candidates(base_features, sample_df, args, k_grid)
    candidate_df = pd.DataFrame(
        [
            {
                "method": c.method,
                "hyperparams": json.dumps(c.hyperparams, sort_keys=True),
                "n_selected": len(c.features),
                "selected": " | ".join(c.features),
                "note": c.note,
                **family_counts(c.features),
            }
            for c in candidates
        ]
    )
    candidate_df.to_csv(out_dir / "candidate_feature_sets.csv", index=False)

    summary_rows = []
    pair_rows = []
    fold_rows = []
    for i, c in enumerate(candidates, start=1):
        if not c.features:
            continue
        print(f"[{i}/{len(candidates)}] {c.method} {c.hyperparams} k={len(c.features)}", flush=True)
        summary, pair_df, fold_df = evaluate_feature_set(c.features, folds, args, model_kind="xgb")
        summary.update(
            {
                "method": c.method,
                "hyperparams": json.dumps(c.hyperparams, sort_keys=True),
                "note": c.note,
                **redundancy_stats(c.features, sample_df),
            }
        )
        summary_rows.append(summary)
        pair_df["method"] = c.method
        pair_df["hyperparams"] = json.dumps(c.hyperparams, sort_keys=True)
        fold_df["method"] = c.method
        fold_df["hyperparams"] = json.dumps(c.hyperparams, sort_keys=True)
        pair_rows.append(pair_df)
        fold_rows.append(fold_df)

    summary_df = pd.DataFrame(summary_rows).sort_values("mean_f1_pairs", ascending=False)
    summary_df.to_csv(out_dir / "feature_selection_stage2_summary.csv", index=False)
    pd.concat(pair_rows, ignore_index=True).to_csv(out_dir / "feature_selection_stage2_pair_id_evaluation.csv", index=False)
    pd.concat(fold_rows, ignore_index=True).to_csv(out_dir / "feature_selection_stage2_fold_evaluation.csv", index=False)
    write_html_report(summary_df, methods_df, out_dir / "feature_selection_stage2_report.html")
    print(f"Done: {out_dir / 'feature_selection_stage2_report.html'}")


if __name__ == "__main__":
    main()
