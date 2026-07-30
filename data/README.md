# Data Splits

This directory contains the six train/test folds used in the paper experiments.

Each fold is stored as:

```text
data/folds/fold_i/train.parquet
data/folds/fold_i/test.parquet
data/folds/fold_i/split.json
```

The split is built at table-pair level: all candidate source-target attribute pairs from the same table pair are kept in the same fold.
