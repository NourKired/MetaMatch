#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import LinearSVC

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None


NON_FEATURE_COLUMNS = {
    "pair_id", "dataset", "relation_type", "category",
    "source_table", "target_table", "source_column", "target_column",
    "source_col_norm", "target_col_norm", "label", "feature_build_sec_pair_real",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", type=Path, default=Path("outputs/exp_occidata"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/exp_occidata/reports/meeting_baselines_vs_metamatch/threshold_strategies_rf350"))
    p.add_argument("--selected-json", type=Path, default=Path("outputs/exp_occidata/reports/meeting_baselines_vs_metamatch/feature_selection_final_pearson085_rf_topk10/selected_features_pearson085_random_forest_importance_k10.json"))
    p.add_argument("--config", choices=["full60", "compact10", "syn_only", "distance_only", "topological_only", "syn_distance", "syn_topological", "distance_topological"], default="full60")
    p.add_argument("--n-estimators", type=int, default=350)
    p.add_argument("--classifiers", type=str, default="rf")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=-1)
    return p.parse_args()


def read_table(base: Path, stem: str) -> pd.DataFrame:
    for name in (f"{stem}.parquet", f"{stem}.csv.gz", f"{stem}.csv"):
        p = base / name
        if p.exists():
            return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    raise FileNotFoundError(stem)


def full60_features(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS and pd.api.types.is_numeric_dtype(df[c])]
    return [c for c in cols if not c.startswith("spc_") and "overlap" not in c.lower() and c != "cls_cosine_dist"]


def threshold_for_max_f1(y: np.ndarray, s: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return 0.5
    p, r, t = precision_recall_curve(y, s)
    if len(t) == 0:
        return 0.5
    f = np.divide(2 * p[1:] * r[1:], p[1:] + r[1:], out=np.zeros_like(t), where=(p[1:] + r[1:]) > 0)
    return float(np.nextafter(t[int(np.nanargmax(f))], -np.inf))


def pair_stats(scores: np.ndarray) -> dict[str, float]:
    qs = np.quantile(scores, [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    return {
        "n": float(len(scores)),
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "q10": float(qs[0]),
        "q25": float(qs[1]),
        "q50": float(qs[2]),
        "q75": float(qs[3]),
        "q90": float(qs[4]),
        "q95": float(qs[5]),
        "q99": float(qs[6]),
        "max_minus_median": float(np.max(scores) - qs[2]),
        "mean_top5": float(np.mean(np.sort(scores)[-max(1, int(np.ceil(0.05 * len(scores)))):])),
    }


def pair_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pid, g in df.groupby("pair_id", sort=False):
        st = pair_stats(g["score"].to_numpy())
        st["pair_id"] = pid
        rows.append(st)
    return pd.DataFrame(rows)


def topk_by_pair(df: pd.DataFrame, k_by_pair: dict[str, int]) -> pd.Series:
    pred = pd.Series(0, index=df.index, dtype=int)
    for pid, g in df.groupby("pair_id", sort=False):
        k = int(k_by_pair.get(pid, 0))
        k = max(0, min(k, len(g)))
        if k == 0:
            continue
        selected = g.sort_values("score", ascending=False, kind="mergesort").head(k).index
        pred.loc[selected] = 1
    return pred


def quantile_by_pair(df: pd.DataFrame, q_by_pair: dict[str, float]) -> pd.Series:
    pred = pd.Series(0, index=df.index, dtype=int)
    for pid, g in df.groupby("pair_id", sort=False):
        q = float(np.clip(q_by_pair.get(pid, 0.95), 0.0, 1.0))
        tau = float(np.quantile(g["score"], q))
        pred.loc[g.index] = (g["score"] > tau).astype(int)
    return pred


def shifted_pair_tau(df: pd.DataFrame, tau_by_pair: dict[str, float], shift: float) -> pd.Series:
    tau = df["pair_id"].map(tau_by_pair).to_numpy(float)
    return pd.Series((df["score"].to_numpy(float) > (tau + shift)).astype(int), index=df.index)


def mean_pair_f1_fast(df: pd.DataFrame, pred: pd.Series) -> float:
    tmp = df[["pair_id", "label"]].copy()
    tmp["pred"] = pred.to_numpy(dtype=int)
    tmp["tp"] = ((tmp["label"] == 1) & (tmp["pred"] == 1)).astype(int)
    tmp["fp"] = ((tmp["label"] == 0) & (tmp["pred"] == 1)).astype(int)
    tmp["fn"] = ((tmp["label"] == 1) & (tmp["pred"] == 0)).astype(int)
    grouped = tmp.groupby("pair_id", sort=False)[["tp", "fp", "fn"]].sum()
    tp = grouped["tp"].to_numpy(float)
    fp = grouped["fp"].to_numpy(float)
    fn = grouped["fn"].to_numpy(float)
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros(len(denom), dtype=float), where=denom > 0)
    return float(np.mean(f1))


def best_tau_shift_on_train(train_scored: pd.DataFrame, tau_by_pair: dict[str, float]) -> float:
    scores = train_scored["score"].to_numpy()
    span = float(np.nanpercentile(scores, 95) - np.nanpercentile(scores, 5))
    if not np.isfinite(span) or span <= 0:
        span = 1.0
    grid = np.linspace(-0.35 * span, 0.20 * span, 24)
    best_shift, best_f1 = 0.0, -1.0
    for shift in grid:
        pred = shifted_pair_tau(train_scored, tau_by_pair, float(shift))
        f = mean_pair_f1_fast(train_scored, pred)
        if f > best_f1:
            best_shift, best_f1 = float(shift), float(f)
    return best_shift


def pair_metrics(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    rows = []
    for pid, g in df.groupby("pair_id", sort=False):
        y = g["label"].to_numpy(dtype=int)
        pred = g[pred_col].to_numpy(dtype=int)
        rows.append({
            "pair_id": pid,
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0),
        })
    return pd.DataFrame(rows)


def build_classifier(name: str, args: argparse.Namespace, seed: int):
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=args.n_estimators,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=args.n_jobs,
        )
    if name == "xgboost":
        if XGBClassifier is None:
            raise RuntimeError("xgboost is not installed")
        return make_pipeline(
            RobustScaler(),
            XGBClassifier(
                n_estimators=220,
                max_depth=4,
                learning_rate=0.05,
                random_state=seed,
                n_jobs=1,
                eval_metric="logloss",
            ),
        )
    if name == "catboost":
        if CatBoostClassifier is None:
            raise RuntimeError("catboost is not installed")
        return make_pipeline(
            RobustScaler(),
            CatBoostClassifier(
                iterations=220,
                depth=4,
                learning_rate=0.05,
                random_seed=seed,
                thread_count=1,
                verbose=False,
            ),
        )
    if name == "logreg":
        return make_pipeline(
            RobustScaler(),
            LogisticRegression(random_state=seed, max_iter=1000, n_jobs=1),
        )
    if name == "svm":
        return make_pipeline(RobustScaler(), LinearSVC(random_state=seed))
    raise ValueError(name)


def get_scores(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(model, "decision_function"):
        s = model.decision_function(x)
        return 1.0 / (1.0 + np.exp(-s))
    return model.predict(x)


def best_quantile_on_train(train_scored: pd.DataFrame) -> float:
    grid = np.r_[np.linspace(0.50, 0.95, 10), [0.97, 0.98, 0.99, 0.995]]
    best_q, best_f1 = 0.95, -1.0
    for q in grid:
        pred = pd.Series(0, index=train_scored.index, dtype=int)
        for _, g in train_scored.groupby("pair_id", sort=False):
            tau = float(np.quantile(g["score"], q))
            pred.loc[g.index] = (g["score"] > tau).astype(int)
        tmp = train_scored[["pair_id", "label"]].copy()
        tmp["pred"] = pred
        f = mean_pair_f1_fast(tmp, tmp["pred"])
        if f > best_f1:
            best_q, best_f1 = float(q), float(f)
    return best_q


def apply_quantile(df: pd.DataFrame, q: float) -> pd.Series:
    pred = pd.Series(0, index=df.index, dtype=int)
    for _, g in df.groupby("pair_id", sort=False):
        tau = float(np.quantile(g["score"], q))
        pred.loc[g.index] = (g["score"] > tau).astype(int)
    return pred


def apply_pair_stat_threshold(df: pd.DataFrame, mode: str, z: float) -> pd.Series:
    pred = pd.Series(0, index=df.index, dtype=int)
    for _, g in df.groupby("pair_id", sort=False):
        s = g["score"].to_numpy(float)
        if mode == "mean_std":
            tau = float(np.mean(s) + z * np.std(s))
        elif mode == "median_iqr":
            q25, q50, q75 = np.quantile(s, [0.25, 0.5, 0.75])
            tau = float(q50 + z * (q75 - q25))
        else:
            raise ValueError(mode)
        pred.loc[g.index] = (g["score"] > tau).astype(int)
    return pred


def best_pair_stat_threshold_on_train(train_scored: pd.DataFrame, mode: str) -> float:
    grid = np.linspace(-1.0, 4.0, 31)
    best_z, best_f1 = 0.0, -1.0
    for z in grid:
        pred = apply_pair_stat_threshold(train_scored, mode, float(z))
        f = mean_pair_f1_fast(train_scored, pred)
        if f > best_f1:
            best_z, best_f1 = float(z), float(f)
    return best_z


def apply_gap_elbow(df: pd.DataFrame, max_frac: float, min_k: int = 1) -> pd.Series:
    pred = pd.Series(0, index=df.index, dtype=int)
    for _, g in df.groupby("pair_id", sort=False):
        s = g["score"].to_numpy(float)
        if len(s) <= min_k:
            k = len(s)
        else:
            order = np.argsort(-s, kind="mergesort")
            sorted_s = s[order]
            max_k = max(min_k + 1, int(np.ceil(max_frac * len(sorted_s))))
            max_k = min(max_k, len(sorted_s) - 1)
            gaps = sorted_s[:max_k] - sorted_s[1:max_k + 1]
            k = int(np.argmax(gaps) + 1)
            k = max(min_k, k)
        selected = g.iloc[np.argsort(-s, kind="mergesort")[:k]].index
        pred.loc[selected] = 1
    return pred


def best_gap_frac_on_train(train_scored: pd.DataFrame) -> float:
    grid = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]
    best_frac, best_f1 = 0.05, -1.0
    for frac in grid:
        pred = apply_gap_elbow(train_scored, frac)
        f = mean_pair_f1_fast(train_scored, pred)
        if f > best_f1:
            best_frac, best_f1 = float(frac), float(f)
    return best_frac


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fold_dirs = sorted((args.output_root / "folds").glob("fold_*"), key=lambda p: int(p.name.split("_")[-1]))
    first = read_table(fold_dirs[0], "train")
    full = full60_features(first)
    syn = [c for c in full if c.startswith("syn_")]
    cls = [c for c in full if c.startswith("cls_")]
    tda = [c for c in full if c.startswith("tda_")]
    if args.config == "full60":
        features = full
    elif args.config == "compact10":
        features = json.loads(args.selected_json.read_text())["selected_features"]
    elif args.config == "syn_only":
        features = syn
    elif args.config == "distance_only":
        features = cls
    elif args.config == "topological_only":
        features = tda
    elif args.config == "syn_distance":
        features = syn + cls
    elif args.config == "syn_topological":
        features = syn + tda
    elif args.config == "distance_topological":
        features = cls + tda
    else:
        raise ValueError(args.config)

    fold_rows, pair_parts = [], []
    classifiers = [x.strip() for x in args.classifiers.split(",") if x.strip()]
    for clf_name in classifiers:
        for fold_dir in fold_dirs:
            fid = int(fold_dir.name.split("_")[-1])
            print(f"fold={fid} config={args.config} classifier={clf_name} n_features={len(features)}", flush=True)
            train = read_table(fold_dir, "train")
            test = read_table(fold_dir, "test")
            model = build_classifier(clf_name, args, args.seed + fid)
            model.fit(train[features].to_numpy(np.float32), train["label"].to_numpy(int))

            train_scored = train[["pair_id", "label"]].copy()
            test_scored = test[["pair_id", "label"]].copy()
            train_scored["score"] = get_scores(model, train[features].to_numpy(np.float32))
            test_scored["score"] = get_scores(model, test[features].to_numpy(np.float32))

            global_tau = threshold_for_max_f1(train_scored["label"].to_numpy(int), train_scored["score"].to_numpy())

            pair_opt = []
            for pid, g in train_scored.groupby("pair_id", sort=False):
                tau = threshold_for_max_f1(g["label"].to_numpy(int), g["score"].to_numpy())
                scores = g["score"].to_numpy()
                pair_opt.append({
                    "pair_id": pid,
                    "tau": tau,
                    "opt_q": float(np.mean(scores <= tau)),
                    "k_true": int(g["label"].sum()),
                    "k_rate_true": float(g["label"].mean()),
                })
            pair_opt = pd.DataFrame(pair_opt)
            median_tau = float(pair_opt["tau"].median())

            stats_train = pair_table(train_scored).merge(pair_opt, on="pair_id")
            stats_test = pair_table(test_scored)
            xcols = [c for c in stats_train.columns if c not in {"pair_id", "tau", "opt_q", "k_true", "k_rate_true"}]
            reg = RandomForestRegressor(n_estimators=200, random_state=args.seed + fid, n_jobs=args.n_jobs, min_samples_leaf=5)
            reg.fit(stats_train[xcols], stats_train["tau"])
            pred_tau_by_pair = dict(zip(stats_test["pair_id"], reg.predict(stats_test[xcols])))
            pred_tau_train_by_pair = dict(zip(stats_train["pair_id"], reg.predict(stats_train[xcols])))
            tau_shift = best_tau_shift_on_train(train_scored, pred_tau_train_by_pair)

            q_reg = RandomForestRegressor(n_estimators=200, random_state=args.seed + fid + 1000, n_jobs=args.n_jobs, min_samples_leaf=5)
            q_reg.fit(stats_train[xcols], stats_train["opt_q"])
            pred_q_by_pair = dict(zip(stats_test["pair_id"], np.clip(q_reg.predict(stats_test[xcols]), 0.0, 1.0)))

            k_reg = RandomForestRegressor(n_estimators=200, random_state=args.seed + fid + 2000, n_jobs=args.n_jobs, min_samples_leaf=5)
            k_reg.fit(stats_train[xcols], np.log1p(stats_train["k_true"].to_numpy(float)))
            pred_k_by_pair = dict(zip(stats_test["pair_id"], np.maximum(0, np.rint(np.expm1(k_reg.predict(stats_test[xcols])))).astype(int)))

            k_rate_reg = RandomForestRegressor(n_estimators=200, random_state=args.seed + fid + 3000, n_jobs=args.n_jobs, min_samples_leaf=5)
            k_rate_reg.fit(stats_train[xcols], stats_train["k_rate_true"])
            pred_k_rate = np.clip(k_rate_reg.predict(stats_test[xcols]), 0.0, 1.0)
            pred_k_rate_by_pair = dict(zip(stats_test["pair_id"], np.maximum(0, np.rint(pred_k_rate * stats_test["n"].to_numpy())).astype(int)))

            q = best_quantile_on_train(train_scored)
            z_mean_std = best_pair_stat_threshold_on_train(train_scored, "mean_std")
            z_median_iqr = best_pair_stat_threshold_on_train(train_scored, "median_iqr")
            gap_frac = best_gap_frac_on_train(train_scored)

            preds = {
                "global_opt_train": (test_scored["score"] > global_tau).astype(int),
                "median_pair_tau_train": (test_scored["score"] > median_tau).astype(int),
                "pair_tau_regressor": shifted_pair_tau(test_scored, pred_tau_by_pair, 0.0),
                "pair_tau_regressor_shifted": shifted_pair_tau(test_scored, pred_tau_by_pair, tau_shift),
                "pair_quantile_train": apply_quantile(test_scored, q),
                "pair_quantile_regressor": quantile_by_pair(test_scored, pred_q_by_pair),
                "pair_topk_count_regressor": topk_by_pair(test_scored, pred_k_by_pair),
                "pair_topk_rate_regressor": topk_by_pair(test_scored, pred_k_rate_by_pair),
                "pair_mean_std_train": apply_pair_stat_threshold(test_scored, "mean_std", z_mean_std),
                "pair_median_iqr_train": apply_pair_stat_threshold(test_scored, "median_iqr", z_median_iqr),
                "pair_gap_elbow_train": apply_gap_elbow(test_scored, gap_frac),
            }

            for name, pred in preds.items():
                tmp = test_scored[["pair_id", "label"]].copy()
                tmp["pred"] = pred.to_numpy(dtype=int) if hasattr(pred, "to_numpy") else np.asarray(pred, dtype=int)
                pm = pair_metrics(tmp, "pred")
                pm.insert(0, "fold_id", fid)
                pm.insert(1, "config", args.config)
                pm.insert(2, "classifier", clf_name)
                pm.insert(3, "strategy", name)
                pair_parts.append(pm)
                fold_rows.append({
                    "fold_id": fid,
                    "config": args.config,
                    "classifier": clf_name,
                    "strategy": name,
                    "global_tau": global_tau,
                    "median_tau": median_tau,
                    "quantile": q,
                    "tau_shift": tau_shift,
                    "z_mean_std": z_mean_std,
                    "z_median_iqr": z_median_iqr,
                    "gap_frac": gap_frac,
                    "mean_pair_f1_fold": pm["f1"].mean(),
                    "mean_pair_precision_fold": pm["precision"].mean(),
                    "mean_pair_recall_fold": pm["recall"].mean(),
                })

    pair_df = pd.concat(pair_parts, ignore_index=True)
    fold_df = pd.DataFrame(fold_rows)
    summary = pair_df.groupby(["config", "classifier", "strategy"], as_index=False).agg(
        mean_f1=("f1", "mean"),
        std_f1=("f1", "std"),
        mean_precision=("precision", "mean"),
        std_precision=("precision", "std"),
        mean_recall=("recall", "mean"),
        std_recall=("recall", "std"),
        n_pair_id=("pair_id", "nunique"),
    ).sort_values("mean_f1", ascending=False)
    suffix = args.config if classifiers == ["rf"] else f"{args.config}_{'_'.join(classifiers)}"
    pair_df.to_csv(args.out_dir / f"{suffix}_threshold_strategy_pair_metrics.csv", index=False)
    fold_df.to_csv(args.out_dir / f"{suffix}_threshold_strategy_fold_metrics.csv", index=False)
    summary.to_csv(args.out_dir / f"{suffix}_threshold_strategy_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
