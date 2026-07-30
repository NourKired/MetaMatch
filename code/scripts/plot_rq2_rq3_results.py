#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


OUT = Path("outputs/exp_occidata/reports/meeting_baselines_vs_metamatch/rq2_rq3_ablation_feature_selection_rf350")

COLORS = {
    "full60": "#4C78A8",
    "syn_only": "#59A14F",
    "distance_only": "#4E79A7",
    "topological_only": "#F28E2B",
    "syn_distance": "#8CD17D",
    "syn_topological": "#F1CE63",
    "distance_topological": "#B07AA1",
    "pearson085_rf_importance_k10": "#E15759",
}

LABELS = {
    "full60": "Full",
    "syn_only": "Syntactic",
    "distance_only": "Distance-based",
    "topological_only": "Topological",
    "syn_distance": "Syn. + Dist.",
    "syn_topological": "Syn. + Topo.",
    "distance_topological": "Dist. + Topo.",
    "pearson085_rf_importance_k10": "Compact k=10",
}


def save_bar(df: pd.DataFrame, configs: list[str], path: Path, y: str = "mean_f1", err: str = "std_f1") -> None:
    plot = df.set_index("config").loc[configs].reset_index()
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    x = range(len(plot))
    ax.bar(
        x,
        plot[y],
        yerr=plot[err],
        capsize=3,
        color=[COLORS[c] for c in plot["config"]],
        edgecolor="#1F2937",
        linewidth=0.7,
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels([LABELS[c] for c in plot["config"]], rotation=25, ha="right")
    ax.set_ylabel("F1 all-to-all")
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", alpha=0.25)
    for i, row in plot.iterrows():
        ax.text(i, min(1.0, row[y] + 0.035), f"{row[y]:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(OUT / "rq2_rq3_summary.csv")

    save_bar(
        df,
        ["full60", "syn_only", "distance_only", "topological_only", "syn_distance", "syn_topological", "distance_topological"],
        OUT / "fig_rq2_family_ablation_f1.png",
    )
    save_bar(
        df,
        ["full60", "pearson085_rf_importance_k10"],
        OUT / "fig_rq3_full_vs_compact_f1.png",
    )

    compact = df[df["config"].isin(["full60", "pearson085_rf_importance_k10"])].copy()
    compact["method"] = compact["config"].map(LABELS)
    compact[
        [
            "method",
            "n_features",
            "mean_f1",
            "std_f1",
            "mean_precision",
            "std_precision",
            "mean_recall",
            "std_recall",
            "total_fit_sec",
            "total_predict_sec",
            "total_model_sec",
        ]
    ].to_csv(OUT / "rq3_full_vs_compact_for_paper.csv", index=False)
    print("saved plots and paper CSVs")


if __name__ == "__main__":
    main()
