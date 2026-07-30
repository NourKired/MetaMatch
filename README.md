# MetaMatch Companion Repository

GitHub repository: https://github.com/NourKired/MetaMatch

This folder is the public companion artifact for the paper:

**MetaMatch: A Topology-Aware Meta-Space for Schema Matching**

It contains the code, notebooks, commands, paper-ready tables, figures, and preliminary results used to support the experimental section.

## What is included

| Folder | Content |
|---|---|
| `code/src/` | MetaMatch Python package: meta-feature extraction, modeling, splits, evaluation, and baseline runners. |
| `code/scripts/` | Reproduction scripts for effectiveness, baselines, feature selection, efficiency, and ablation. |
| `environment/` | Poetry and pip environments. |
| `notebooks/` | Notebooks used to generate article figures and tables. |
| `results/rq1_effectiveness_baselines/` | RQ1 effectiveness, classifier comparison, win-rate, statistical tests, and runtime tables. |
| `results/rq2_meta_feature_selection/` | RQ2 full vs reduced MetaMatch and selected meta-feature results. |
| `results/rq3_efficiency/` | RQ3 efficiency and full-vs-reduced runtime results. |
| `results/rq4_family_ablation/` | RQ4 family ablation for complete and reduced MetaMatch. |
| `results/preliminary/` | Preliminary correlation pruning and feature-selection benchmark reports. |
| `figures/` | Paper-ready figures in PDF/PNG. |
| `paper/` | Current paper PDF, LaTeX source, and bibliography snapshot. |
| `run_commands/` | Shell commands to reproduce each research question. |

## Main experimental protocol

- Dataset: Valentine complete-complete table pairs.
- Evaluation unit: table pair (`pair_id`), not individual candidate rows.
- Splits: 6 folds built at table-pair level to avoid leakage between train and test candidates from the same table pair.
- MetaMatch representation: 60 meta-features after removing duplicated/non-paper features.
- Main classifier: Random Forest with 350 trees and `class_weight="balanced_subsample"`.
- Inference threshold: learned only on the training part of each fold, then applied to the test part of the same fold.
- Main metric: all-to-all F1 averaged over table pairs, with precision and recall reported alongside F1.

## Research questions

### RQ1: Effectiveness and Baselines

Main files:

- `results/rq1_effectiveness_baselines/table_1_metaspace_classifiers_all_to_all.csv`
- `results/rq1_effectiveness_baselines/table_effectiveness_baselines_metaspace_summary.csv`
- `results/rq1_effectiveness_baselines/rq1_metaspace_vs_baselines_paired_stat_tests.csv`
- `results/rq1_effectiveness_baselines/winrate_f1_all_to_all_no_ties_percent_matrix.csv`
- `figures/plot_distribution_baseline.pdf`
- `figures/plot_winrate_f1_all_to_all_no_ties_matplotlib_percent_heatmap.pdf`

Reproduction command:

```bash
bash run_commands/01_rq1_effectiveness_baselines.sh
```

### RQ2: Meta-feature Selection

Final reduced configuration:

- correlation pruning with Pearson threshold `0.85`;
- Random Forest importance ranking;
- `k=10` selected meta-features.

Main files:

- `results/rq2_meta_feature_selection/selected_features_pearson085_random_forest_importance_k10.csv`
- `results/rq2_meta_feature_selection/best_config_summary_pearson085_random_forest_importance_k10.csv`
- `results/rq2_meta_feature_selection/rq2_rq3_table_for_paper.csv`
- `results/preliminary/feature_selection_first_benchmark_STAGE2_PROTOCOL.html`
- `results/preliminary/feature_correlation_report_REFRESHED_CURRENT.html`

Reproduction command:

```bash
bash run_commands/03_rq2_feature_selection.sh
```

### RQ3: Efficiency

Main files:

- `results/rq1_effectiveness_baselines/table_efficiency_training_runtime_total_551_pairs_with_reduced.csv`
- `results/rq3_efficiency/rq3_full_vs_compact_efficiency_end_to_end_compact_measured.csv`
- `figures/fig_rq4_total_runtime_bar.pdf`

Reproduction command:

```bash
bash run_commands/05_efficiency_full_vs_compact_e2e.sh
```

### RQ4: Meta-feature Family Ablation

Main files:

- `results/rq4_family_ablation/family_ablation_complete_vs_reduced_for_paper.csv`
- `results/rq4_family_ablation/reduced_family_ablation_pair_mean_std_summary.csv`
- `results/rq2_meta_feature_selection/rq2_family_ablation_for_paper.csv`

Reproduction command:

```bash
bash run_commands/03_rq2_feature_selection.sh
```

## Environment

Poetry:

```bash
cd companion_metamatch_vldb_public
poetry install
poetry run python -c "import osirim_occidata; print('ok')"
```

Pip/venv:

```bash
bash run_commands/00_setup.sh
```

## External baselines

Some baselines require external code or pretrained checkpoints:

- Magneto / MagnetoFT / MagnetoGPT / MagnetoFTGPT
- SMUTF
- LLMATCH
- Valentine classical baselines

The scripts in `code/scripts/external_baselines/` and `run_commands/02_baselines_missing_pairs.sh` document how these are called. Paths to external packages and checkpoints should be configured locally and should not be committed with private absolute paths or API keys.

## Large files not included

The full local output directory was about 8.7 GB. This public companion includes paper-ready CSVs, figures, notebooks, and scripts only. Large intermediate `.parquet` score files, cached model outputs, and raw generated candidate matrices are intentionally excluded. The manifest of included files is available in `RESULTS_MANIFEST.csv`.

## Suggested GitHub workflow

```bash
cd companion_metamatch_vldb_public
git init
git add .
git commit -m "Public companion artifact for MetaMatch"
git branch -M main
git remote add origin <your-public-repo-url>
git push -u origin main
```

For a stable paper artifact, create a GitHub release or archive the repository on Zenodo after the final version is fixed.
