#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare MetaSpace/MetaMatch against baselines with boxplots, winrate and Wilcoxon tests.")
    p.add_argument("--output-root", type=Path, default=Path("outputs/exp_occidata"))
    p.add_argument("--meta-pair-csv", type=Path, default=None)
    p.add_argument("--meta-method-name", type=str, default="meta_space@opt_train")
    p.add_argument("--metric", type=str, default="f1")
    p.add_argument("--top-baselines", type=int, default=20)
    return p.parse_args()


def boxplot(df: pd.DataFrame, metric_col: str, out_png: Path) -> None:
    import matplotlib.pyplot as plt

    order = df.groupby("method")[metric_col].mean().sort_values(ascending=False).index.tolist()
    data = [df.loc[df["method"].eq(m), metric_col].dropna().to_numpy() for m in order]
    width = max(10, 0.34 * len(order))
    fig, ax = plt.subplots(figsize=(width, 5.2))
    ax.boxplot(data, labels=order, showfliers=False)
    ax.set_ylabel(metric_col)
    ax.set_title("Performance par paire de tables")
    ax.tick_params(axis="x", rotation=70, labelsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    report_dir = args.output_root / "reports" / "meeting_baselines_vs_metamatch"
    report_dir.mkdir(parents=True, exist_ok=True)

    baselines_path = report_dir / "pair_results_by_fold_all_methods.csv"
    if not baselines_path.exists():
        raise FileNotFoundError(f"Missing {baselines_path}. Run export_fold_pair_results.py first.")
    base = pd.read_csv(baselines_path)
    base = base.rename(
        columns={
            "metric_f1_all2all": "f1",
            "metric_precision_all2all": "precision",
            "metric_recall_all2all": "recall",
        }
    )
    base = base[["fold_id", "pair_id", "method", "f1", "precision", "recall", "runtime_sec_pair_approx"]].copy()

    if args.meta_pair_csv:
        meta_raw = pd.read_csv(args.meta_pair_csv)
        meta = meta_raw.rename(columns={"classifier": "method"})
        if "method" not in meta.columns:
            meta["method"] = args.meta_method_name
        meta["method"] = meta["method"].map(lambda x: f"metaspace_{x}" if str(x) not in {args.meta_method_name} else str(x))
        meta = meta[["fold_id", "pair_id", "method", "f1", "precision", "recall"]].copy()
        meta["runtime_sec_pair_approx"] = np.nan
        df = pd.concat([base, meta], ignore_index=True)
    else:
        df = base.copy()

    metric = args.metric
    mean_rank = df.groupby("method")[metric].mean().sort_values(ascending=False)
    keep = mean_rank.head(args.top_baselines).index.tolist()
    if args.meta_pair_csv:
        keep = sorted(set(keep).union(set(df[df["method"].str.startswith("metaspace_", na=False)]["method"].unique())))
    plot_df = df[df["method"].isin(keep)].copy()

    boxplot(plot_df, metric, report_dir / f"boxplot_{metric}_metaspace_vs_baselines_pair_id.png")

    meta_methods = [m for m in df["method"].dropna().unique() if str(m).startswith("metaspace_") or str(m) == args.meta_method_name]
    rows = []
    for meta_method in meta_methods:
        mdf = df[df["method"].eq(meta_method)][["fold_id", "pair_id", metric]].rename(columns={metric: "meta"})
        for baseline in sorted([m for m in df["method"].dropna().unique() if m != meta_method]):
            bdf = df[df["method"].eq(baseline)][["fold_id", "pair_id", metric]].rename(columns={metric: "base"})
            z = mdf.merge(bdf, on=["fold_id", "pair_id"], how="inner").dropna()
            if len(z) == 0:
                continue
            delta = z["meta"] - z["base"]
            try:
                p_value = float(wilcoxon(delta).pvalue) if (delta != 0).any() else 1.0
            except Exception:
                p_value = np.nan
            rows.append(
                {
                    "meta_method": meta_method,
                    "baseline": baseline,
                    "n": int(len(z)),
                    "wins": int((delta > 0).sum()),
                    "ties": int((delta == 0).sum()),
                    "losses": int((delta < 0).sum()),
                    "winrate_half_ties": float(((delta > 0).sum() + 0.5 * (delta == 0).sum()) / len(z)),
                    "mean_delta": float(delta.mean()),
                    "median_delta": float(delta.median()),
                    "wilcoxon_p": p_value,
                }
            )

    stats = pd.DataFrame(rows).sort_values(["meta_method", "winrate_half_ties"], ascending=[True, False])
    df.to_csv(report_dir / "performance_pair_metrics_with_metaspace_classifiers.csv", index=False)
    mean_rank.reset_index(name=f"mean_{metric}").to_csv(report_dir / f"performance_mean_{metric}_by_method.csv", index=False)
    stats.to_csv(report_dir / f"winrate_stats_{metric}_metaspace_vs_baselines.csv", index=False)
    print(f"Saved: {report_dir / 'performance_pair_metrics_with_metaspace_classifiers.csv'}")
    print(f"Saved: {report_dir / f'boxplot_{metric}_metaspace_vs_baselines_pair_id.png'}")
    print(f"Saved: {report_dir / f'winrate_stats_{metric}_metaspace_vs_baselines.csv'}")


if __name__ == "__main__":
    main()
