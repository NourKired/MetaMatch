# MetaMatch

Public repository for **MetaMatch: A Topology-Aware Meta-Space for Schema Matching**.

This repository contains the implementation and the paper-ready experimental artifacts used in the article. It is intentionally kept reduced: only the source code, reproduction commands, CSV results, and figures used in the paper are included.

## Repository Structure

| Path | Content |
|---|---|
| `src/metamatch/` | MetaMatch implementation: meta-feature extraction, splits, modeling, evaluation, and baseline utilities. |
| `scripts/` | Reproduction and reporting scripts for the paper experiments. |
| `run_commands/` | Shell commands used to reproduce each experimental block. |
| `results/` | Paper-ready CSV tables and figure data. |
| `figures/` | Figures used in the article. |
| `pyproject.toml` | Python project metadata and dependencies. |

## Experimental Protocol

- Benchmark: Valentine complete-complete table pairs.
- Candidate space: many-to-many schema matching; every source attribute is paired with every target attribute.
- Evaluation: six folds built at table-pair level to avoid leakage between train and test table pairs.
- MetaMatch representation: 60 meta-features: syntactic, distance-based, and topological.
- Main classifier: Random Forest with 350 trees and `class_weight="balanced_subsample"`.
- Thresholding: the decision threshold is selected on the training split of each fold and then applied unchanged to the test split.
- Main metric: pair-level F1, with precision and recall reported alongside it.

## Paper Results

### RQ1: Effectiveness and Baselines

- `results/rq1_effectiveness_baselines/table_1_metamatch_classifiers_all_to_all.csv`
- `results/rq1_effectiveness_baselines/table_1_effectiveness_baselines_summary.csv`
- `results/rq1_effectiveness_baselines/rq1_metamatch_vs_baselines_paired_stat_tests_paper.csv`
- `results/rq1_effectiveness_baselines/figure_2_effectiveness_baselines_metamatch_by_pair.csv`
- `results/rq1_effectiveness_baselines/figure_2_winrate_f1_all_to_all_no_ties_percent_matrix.csv`
- `results/rq1_effectiveness_baselines/figure_rq1_effectiveness_by_group_pair_values.csv`

### RQ2: Meta-feature Selection

- `results/rq2_meta_feature_selection/selected_features_pearson085_random_forest_importance_k10.csv`
- `results/rq2_meta_feature_selection/rq2_rq3_table_for_paper.csv`

### RQ3: Efficiency

- `results/rq3_efficiency/table_efficiency_preliminary_application_total_551_pairs_with_reduced.csv`

### RQ4: Family Ablation

- `results/rq4_family_ablation/family_ablation_complete_vs_reduced_for_paper.csv`
- `results/rq4_family_ablation/rq2_family_ablation_for_paper.csv`

## Figures

The `figures/` directory contains only figures used in the article:

- `framework_overview.*`
- `topology_vr_h0_h1_h2_metamatch.*`
- `plot_distribution_baseline.*`
- `plot_winrate_f1_all_to_all_no_ties_matplotlib_percent_heatmap.*`
- `plot_effectiveness_by_relation_distribution.*`
- `plot_effectiveness_by_dataset_distribution.*`
- `fig_rq4_total_runtime_bar.*`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Reproduction Commands

```bash
bash run_commands/01_rq1_effectiveness_baselines.sh
bash run_commands/03_rq2_feature_selection.sh
bash run_commands/05_efficiency_full_vs_reduced_e2e.sh
```

Some external baselines require their official packages and checkpoints. Private local paths, API keys, generated candidate matrices, and large intermediate files are not included.
