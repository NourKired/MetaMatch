# Data Splits

This directory contains the six train/test folds used in the paper experiments.

The released parquet files expose only the paper representation: 60 MetaMatch meta-features, the binary `label`, and the minimal table-pair context columns needed to reproduce the train/test protocol. Non-paper columns such as spectral descriptors, overlap descriptors, cached runtime fields, and experimental extras are not included.

Each fold is stored as:

```text
data/folds/fold_i/train.parquet
data/folds/fold_i/test.parquet
data/folds/fold_i/split.json
```

The split is built at table-pair level: all candidate source-target attribute pairs from the same table pair are kept in the same fold.
