# data/

Not tracked by git. The collector's dataset stays in its own folder (`data.source` and
`data.meta` in `configs/config.yaml`); nothing is copied here.

```
data/processed/
  rows.csv              one row per (change, file): row_key, unit_key, label, file__* facts
  units.csv             one row per change: label, repo, org, date, n_commits, change__* and path__* facts
  splits/<scheme>.csv   unit_key + fold1..foldN columns with train / val / test
```

Keys: `unit_key = full_repo_name|unit_id`, `row_key = full_repo_name|unit_id|file_path`.
Row i of every `artifacts/embeddings/*.npy` array belongs to row i of rows.csv, so re-running
`prepare` on a different dataset means re-running `embed` too (it notices when the row count changed and starts over).
