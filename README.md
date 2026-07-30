# MetaMatch

**MetaMatch** is a topology-aware schema matching framework for complete tabular data. It represents each candidate source-target attribute pair with a meta-feature vector combining syntactic, distance-based, and topological evidence. The topological part is based on persistent homology over token-level transformer embeddings and includes a joint source-target construction that captures how two column representations merge in the same embedding space.

This repository contains the public implementation and the paper-ready results for:

> **MetaMatch: A Topology-Aware Meta-Space for Schema Matching**

The repository is intentionally small. It includes only the code, commands, CSV tables, and figures needed to reproduce or inspect the results reported in the article.

## Main Results

| Question | Main finding | Files |
|---|---|---|
| RQ1 Effectiveness | Random Forest is the best MetaMatch classifier: F1 = 0.81, precision = 0.80, recall = 0.84 over 551 table pairs. MetaMatch is the strongest controlled non-LLM method in the benchmark. | `results/rq1_effectiveness_baselines/` |
| RQ2 Meta-feature selection | Pearson pruning at 0.85 followed by Random Forest importance keeps 10 meta-features while preserving most of the full performance: F1 = 0.79. | `results/rq2_meta_feature_selection/` |
| RQ3 Efficiency | Reduced MetaMatch lowers total runtime from 66568.89 s to 5620.26 s while keeping similar F1. | `results/rq3_efficiency/` |
| RQ4 Family ablation | Syntactic meta-features are strongest alone, but topological meta-features improve combinations, especially with syntactic evidence. | `results/rq4_family_ablation/` |

## Repository Structure

| Path | Content |
|---|---|
| `src/metamatch/` | MetaMatch source code: meta-feature extraction, split construction, modeling, evaluation, and baseline utilities. |
| `scripts/` | Python scripts used to generate the reported tables, figures, and timing results. |
| `run_commands/` | Shell commands for the main experimental blocks. |
| `results/` | Paper-ready CSV results, grouped by research question. |
| `figures/` | Figures used in the article. |
| `pyproject.toml` | Minimal Python project configuration. |

## Experimental Protocol

- Benchmark: Valentine complete-complete table pairs.
- Candidate space: many-to-many schema matching. Every source attribute is paired with every target attribute.
- Splits: six folds built at table-pair level to avoid leakage between train and test table pairs.
- Representation: 60 MetaMatch meta-features:
  - 22 syntactic meta-features;
  - 7 distance-based meta-features;
  - 31 topological meta-features.
- Main classifier: Random Forest with 350 trees and `class_weight="balanced_subsample"`.
- Thresholding: the decision threshold is selected on the training split of each fold and then applied unchanged to the test split.
- Metric: pair-level F1, with precision and recall reported alongside it.

## Environment Setup

```bash
git clone https://github.com/NourKired/MetaMatch.git
cd MetaMatch

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

Or use the setup helper:

```bash
bash run_commands/00_setup.sh
```

## Running MetaMatch

The main train/test pipeline is implemented in:

```bash
scripts/run_rq2_rq3_feature_ablation_rf350.py
```

Run the full and reduced MetaMatch experiments:

```bash
bash run_commands/03_rq2_feature_selection.sh
```

Run the end-to-end timing experiment:

```bash
bash run_commands/05_efficiency_full_vs_reduced_e2e.sh
```

Run the RQ1 report generation from stored scores:

```bash
bash run_commands/01_rq1_effectiveness_baselines.sh
```

## Baselines

The paper compares MetaMatch with:

- LLMATCH
- SMUTF
- MagnetoFTGPT
- MagnetoGPT
- MagnetoFT
- Magneto
- COMA++
- COMA
- Similarity Flooding
- ISResMat
- Cupid
- Distribution Based

Some baselines require their official external packages, pretrained checkpoints, or API access. These external dependencies are not committed here. The repository includes the final paper-ready baseline CSV results, but not private paths, API keys, large model files, or generated intermediate candidate matrices.

Baseline effectiveness results are available in:

```text
results/rq1_effectiveness_baselines/table_1_effectiveness_baselines_summary.csv
```

Pairwise statistical test results are available in:

```text
results/rq1_effectiveness_baselines/rq1_metamatch_vs_baselines_paired_stat_tests_paper.csv
```

## Results by Research Question

### RQ1: Effectiveness and Baselines

Main result files:

```text
results/rq1_effectiveness_baselines/table_1_metamatch_classifiers_all_to_all.csv
results/rq1_effectiveness_baselines/table_1_effectiveness_baselines_summary.csv
results/rq1_effectiveness_baselines/rq1_metamatch_vs_baselines_paired_stat_tests_paper.csv
results/rq1_effectiveness_baselines/figure_2_effectiveness_baselines_metamatch_by_pair.csv
results/rq1_effectiveness_baselines/figure_2_winrate_f1_all_to_all_no_ties_percent_matrix.csv
results/rq1_effectiveness_baselines/figure_rq1_effectiveness_by_group_pair_values.csv
```

Figures:

```text
figures/plot_distribution_baseline.*
figures/plot_winrate_f1_all_to_all_no_ties_matplotlib_percent_heatmap.*
figures/plot_effectiveness_by_relation_distribution.*
figures/plot_effectiveness_by_dataset_distribution.*
```

### RQ2: Meta-feature Selection

The selected reduced representation is obtained with:

1. Pearson correlation pruning with threshold 0.85;
2. Mutual Information to keep the most informative meta-feature among correlated ones;
3. Random Forest importance ranking;
4. top `k = 10` selected meta-features.

The selected meta-features are stored in:

```text
results/rq2_meta_feature_selection/selected_features_pearson085_random_forest_importance_k10.csv
```

Full vs reduced performance is stored in:

```text
results/rq2_meta_feature_selection/rq2_rq3_table_for_paper.csv
```

### RQ3: Efficiency

Runtime results are stored in:

```text
results/rq3_efficiency/table_efficiency_preliminary_application_total_551_pairs_with_reduced.csv
```

Runtime figure:

```text
figures/fig_rq4_total_runtime_bar.*
```

### RQ4: Family Ablation

Ablation results for syntactic, distance-based, and topological meta-feature families are stored in:

```text
results/rq4_family_ablation/family_ablation_complete_vs_reduced_for_paper.csv
results/rq4_family_ablation/rq2_family_ablation_for_paper.csv
```

## Figures Used in the Article

```text
figures/framework_overview.*
figures/topology_vr_h0_h1_h2_metamatch.*
figures/plot_distribution_baseline.*
figures/plot_winrate_f1_all_to_all_no_ties_matplotlib_percent_heatmap.*
figures/plot_effectiveness_by_relation_distribution.*
figures/plot_effectiveness_by_dataset_distribution.*
figures/fig_rq4_total_runtime_bar.*
```

## Notes

Large intermediate files are not included. This includes `.parquet` score matrices, cached embeddings, generated candidate matrices, external pretrained models, and private baseline checkpoints.
