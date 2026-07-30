# OSIRIM Occidata - Meta-space + Baselines (Valentine 551)

Pipeline complet prêt à exécuter pour:
- charger les 551 paires Valentine (`source.csv`, `target.csv`, `mapping.json`),
- construire l'espace candidats (produit cartésien colonnes source x colonnes target),
- extraire les meta-features,
- faire 5 splits stratifiés en mode 70/30 (ou 5-fold CV),
- appliquer la règle anti-fuite (alignements présents en test retirés du train),
- entraîner la méthode meta-space,
- évaluer des baselines sur exactement les mêmes jeux de test,
- comparer qualité et temps de calcul.

Meta-features utilisées (alignées avec ton meta-space, sans NLP BLEU/ROUGE):
- `syntax` (texte direct: label + valeurs de colonne),
- `classical` (distances/similarités sur embeddings vectoriels),
- `spectral` (token-level embeddings),
- `topological` (token-level embeddings, TDA; obligatoire).

## Arborescence

- `scripts/prepare_experiment.py`: préparation dataset + folds + tâches.
- `scripts/run_task.py`: exécute 1 tâche (1 CPU).
- `scripts/run_all_local.py`: exécute toutes les tâches localement (jusqu'à 128 jobs).
- `scripts/aggregate_results.py`: agrégation des métriques/temps.
- `scripts/osirim_submit.sh`: soumission SLURM array `%128`, `--cpus-per-task=1`.
- `configs/baselines.example.yaml`: modèle de config pour baselines externes.

## Installation

```bash
python3 -m pip install -e .
```

## 1) Préparer l'expérience

```bash
python3 scripts/prepare_experiment.py \
  --dataset-root valentine \
  --output-root outputs/exp_occidata \
  --split-mode repeated_70_30 \
  --folds 5 \
  --feature-workers 8 \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2
```

Notes:
- `split-mode=repeated_70_30` fait 5 splits stratifiés 70/30 avec objectif de couverture test.
- `split-mode=kfold` fait 5-fold CV classique.
- Pour désactiver les embeddings transformer (fallback hash): `--disable-transformer-embeddings`.
- Les features topologiques sont toujours activées (obligatoires).

## 2) Exécuter les tâches

### Local (max 128 jobs, 1 CPU/job)

```bash
python3 scripts/run_all_local.py \
  --output-root outputs/exp_occidata \
  --max-parallel 128 \
  --model rf
```

### OSIRIM / SLURM

```bash
bash scripts/osirim_submit.sh outputs/exp_occidata rf
```

Le script soumet un job array:
- `--array=0-(N-1)%128`
- `--cpus-per-task=1`

Sur OCCIDATA, tu peux piloter la partition/GPU via variables d'environnement:

```bash
# Array CPU (recommandé pour run_task)
PARTITION=24CPUNodes GPU_COUNT=0 bash scripts/osirim_submit.sh outputs/exp_occidata rf

# Array GPU (si nécessaire)
PARTITION=GPUNodes GPU_COUNT=1 bash scripts/osirim_submit.sh outputs/exp_occidata rf
```

Si `GPU_COUNT>0`, le script ajoute automatiquement:
- `#SBATCH --gres=gpu:<N>`
- `#SBATCH --gres-flags=enforce-binding`

Pour utiliser 128 jobs aussi pendant la préparation des meta-features:

```bash
bash scripts/osirim_submit_pipeline_128.sh outputs/exp_occidata rf
```

Ce script orchestre automatiquement:
1. array `run_prepare_shard.py` (128 shards max en parallèle),
2. merge des shards + génération folds/tasks,
3. soumission de l'array `run_task`.

## 3) Baselines externes

1. Copier le fichier exemple:

```bash
cp configs/baselines.example.yaml configs/baselines.yaml
```

2. Remplacer les `command_template` par tes commandes réelles pour:
- `ISResMat`, `LLMATCH`,
- `Magneto_*`,
- `coma`, `coma_pp`, `coma_inst`,
- `cupid_ext`, `similarity_flooding_ext`, `distribution_based_ext`,
- `SMUTF`.

Par défaut, ce dépôt inclut maintenant un runner local compatible pour:
- `ISResMat`, `LLMATCH`, `Magneto_*`, `SMUTF`

via:
- `scripts/external_baselines/proxy_runner.py`

Important:
- ce sont des implémentations proxy (compatibles protocole d'évaluation), pas les codes officiels de ces méthodes.

Tu reçois automatiquement deux fichiers par méthode/fold dans `results/fold_X/<method>/`:
- `test_manifest.csv`: candidats exacts évalués (`pair_id`, `source_col_norm`, `target_col_norm`, `label`)
- `pair_manifest.jsonl`: chemins exacts des tables/mappings pour ces `pair_id`

Chaque baseline doit produire un fichier (`csv` ou `parquet`) contenant:
- `pair_id`
- `source_col_norm`
- `target_col_norm`
- `score`

Compat interne déjà branchée (même tests): `coma`, `coma_pp`, `coma_inst`, `cupid_ext`, `similarity_flooding_ext`, `distribution_based_ext`.

Pour les baselines LLM/GPT (ex: `LLMATCH`, `Magneto_*_gpt`), ajoute la clé API dans l'environnement avant exécution:

```bash
export OPENAI_API_KEY=\"<ta_cle>\"
```

Sur ton cluster, les scripts chargent automatiquement la clé depuis:
`/projects/sig/nkired/secrets/openai.key`
(modifiable via `OPENAI_KEY_FILE`).

Le script `osirim_quickstart.sh` détecte automatiquement NVIDIA et utilise `--embedding-device cuda` si disponible
(`EMBEDDING_DEVICE=cpu|cuda|auto`, défaut `auto`).
Pour la soumission array en fin de quickstart:
- `TASK_PARTITION` (défaut `24CPUNodes`)
- `TASK_GPU_COUNT` (défaut `0`)
- `PREP_SHARDS` (défaut `128`, via `osirim_submit_pipeline_128.sh`)
- `REUSE_VENV=1` (défaut): réutilise `.venv` existant, évite de retélécharger `torch` à chaque relance.

Tu peux aussi forcer la vérification dans `configs/baselines.yaml` via:

```yaml
required_env: [\"OPENAI_API_KEY\"]
```

Si la variable manque, le run sera marqué `missing_env`.

## 4) Agréger les résultats

```bash
python3 scripts/aggregate_results.py --output-root outputs/exp_occidata
```

Fichiers produits:
- `outputs/exp_occidata/all_task_metrics.csv`
- `outputs/exp_occidata/leaderboard.csv`

`leaderboard.csv` agrège uniquement les runs avec `status=ok` (les méthodes non branchées restent visibles dans `all_task_metrics.csv` avec `missing_backend`).

## Baselines Valentine natives incluses

- `coma_schema`
- `coma_instance`
- `cupid`
- `similarity_flooding`
- `distribution_based`

## Anti-fuite implémentée

Pour chaque fold:
- train/test sont séparés au niveau **pair_id** (paires de tables).
- si un alignement positif `(source_col_norm, target_col_norm)` du test existe aussi en train, il est retiré du train.
