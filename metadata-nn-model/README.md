# difflabel

Classifies code-change records (bug vs. enhancement, ...) from **CodeBERT code
embeddings** and **BERT metadata embeddings**, including a gated-fusion model
that learns how much to trust each modality. Restructured from a flat set of
scripts into a package; all tunable values now live in JSON files.

## Layout

```
difflabel/
├── configs/                  # every tunable value, as JSON (one file per run type)
│   ├── prepare_csv.json
│   ├── create_embeddings_content_before.json
│   ├── create_embeddings_patch_before.json
│   ├── train_single_code.json
│   ├── train_single_metadata.json
│   └── train_gated_fusion.json
├── scripts/                  # thin CLIs: parse --config, call a pipeline
│   ├── 01_prepare_csv.py
│   ├── 02_create_embeddings.py
│   ├── 03a_train_single_code.py
│   ├── 03b_train_single_metadata.py
│   └── 04_train_gated_fusion.py
├── src/difflabel/
│   ├── config.py             # strict JSON -> dataclass loading, path resolution
│   ├── runtime.py            # seeding, device selection
│   ├── cli.py
│   ├── data/                 # preparation, folds, java_text, datasets, labels, embedding_io, sep_processing
│   ├── embeddings/           # generate.py (CodeBERT -> .pt files), bert.py (metadata embeddings)
│   ├── models/               # single.py, gated_fusion.py, blocks.py, build_model()
│   ├── training/             # loops.py (per-epoch), trainer.py (shared orchestration)
│   ├── evaluation/           # baselines.py (9 classical models), plots.py
│   └── pipelines/            # one module per numbered script, each exposing run(cfg)
├── tests/
├── data/{raw,processed}/     # inputs / prepared CSVs and folds   (git-ignored contents)
├── embeddings/  models/  scalers/  outputs/{metrics,figures}/      (git-ignored contents)
├── pyproject.toml  requirements.txt
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Running

Relative paths in configs are resolved from the project root (override with the
`DIFFLABEL_ROOT` environment variable), so you can run from anywhere.

```bash
# 1. clean + balance raw CSV, write CV folds              -> data/processed/
python scripts/01_prepare_csv.py

# 2. per-row CodeBERT embeddings, once per text column     -> embeddings/
python scripts/02_create_embeddings.py --config configs/create_embeddings_content_before.json
python scripts/02_create_embeddings.py --config configs/create_embeddings_patch_before.json

# 3. single-pipeline DNN + 9 baselines (separate 45993-row dataset)
python scripts/03a_train_single_code.py
python scripts/03b_train_single_metadata.py

# 4. gated-fusion DNN + 9 baselines, for each fold
python scripts/04_train_gated_fusion.py
```

Every script takes `--config path/to/your.json`; copy a shipped config to run
a variation without touching the originals.

## Configuration

* Plain JSON, loaded into dataclasses by `difflabel.config`. **Unknown keys and
  wrong types are rejected** and **all missing keys are reported at once**, so a
  typo can't silently fall back to a default.
* Keys starting with `_` (e.g. `"_comment"`) are ignored; JSON has no comments.
* Output paths in `train_gated_fusion.json` contain `{fold}`, filled per fold.
* `baselines.models` maps model name → constructor kwargs. The random seed is
  injected automatically, and so is the class count for XGBoost/LightGBM.
* `pooling` in the embedding configs is `"pooler"` or `"cls"`.
  `device` anywhere is `"auto"`, `"cpu"`, `"cuda:0"`, ...

### Where the old globals went

| Old module-level constant(s)                                   | Now in                                              |
|----------------------------------------------------------------|-----------------------------------------------------|
| `RAW_CSV`, `OUTPUT_DIR`, `OUTPUT_CSV`, `ROWS_PER_REPO`, `COLUMNS_TO_KEEP` | `prepare_csv.json`                      |
| hard-coded label lists, `columns_to_drop`, `random_state=1`, fold counts/seeds/ratios | `prepare_csv.json` (`labels`, `columns_to_drop`, `balance_seed`, `folds`) |
| `INPUT_CSV`, `COLUMN`, `MODEL_NAME`, `EMBEDDING_DIRECTORY`, `BATCH_SIZE`, `MAX_SEQ_LENGTH`, `DEVICE` | `create_embeddings_*.json` |
| `SEED`, `INPUT_CSV`, `EMBEDDING_DIRECTORY(2)`, `FOLDS`, `FOLDS_DIR` | `train_*.json` top level / `embeddings` |
| `METADATA_MODEL_NAME`, `METADATA_COLUMNS`                      | `train_*.json` → `metadata`                         |
| `HIDDEN_SIZES_*`, `OUTPUT_SIZE`, `DROPOUT_RATE`                | `train_*.json` → `model`                            |
| `BATCH_SIZE`, `NUM_EPOCHS`, `LEARNING_RATE`, scheduler patience/factor, scaler batch size | `train_*.json` → `optim`  |
| `MODEL_PATH`, `SCALER_PATH`, `ML_*`, `DNN_CONFUSION_MATRIX_PNG` | `train_*.json` → `outputs`                         |
| 70/15/15 `train_test_split` ratios                             | `train_single_*.json` → `split`                     |
| classical-model hyperparameters (`n_estimators`, `depth`, ...) | `train_*.json` → `baselines.models`                 |

## Behaviour changes vs. the original scripts

Intentional fixes (the data-prep outputs are otherwise byte-identical to the
original; see Tests):

* `from dump.utils import ...` (03a, 04) was a broken import → package imports.
* `prepare_dataframe(shouldBalance=True)` referenced undefined variables → fixed.
* Embedding pooling was inferred from `"CLS"` appearing in the output directory
  *name* → explicit `pooling` setting.
* `04` overwrote one scaler file for every fold → now `scaler_..._fold{N}.pkl`.
* 03a and 03b both wrote `DNN_confusion_matrix.png` (clobbering each other) →
  distinct names, and all outputs go under `outputs/`.
* A missing embedding file raised `UnboundLocalError` → clear `FileNotFoundError`.
* pandas 3 compatibility (`drop(columns=..., axis=1)` raises there; `groupby.apply`
  no longer passes the group key).
* `tqdm_notebook` → `tqdm.auto`; deprecated `verbose=` on `ReduceLROnPlateau` dropped.
* AUC uses the binary form for 2 classes everywhere, weighted one-vs-rest otherwise.
* The gate-average `print` inside `GatedFusionDiffNN.forward` is now the
  attribute `model.last_gate_mean`.

Refactors that should not change results:

* Model classes are built from helper blocks and no longer take unused constructor
  arguments; layer **names, shapes and construction order are unchanged**, so old
  checkpoints load with `load_state_dict`. Depth now follows the length of the
  `hidden_sizes_*` lists.
* Per-script column *drop* lists became *select* lists, and one-hot labels come from
  the label column alone.
* The duplicated DNN loop, baseline runner and plotting code now exist once.

## Known issues left as-is (worth deciding on)

* **Scaler leakage:** the `StandardScaler` is fit on train + val + **test**.
* **Label leakage:** `train_gated_fusion.json` embeds `commit_message`, which can
  restate the label (03b deliberately excludes it). Remove it from
  `metadata.columns` for a leak-safe run.
* In 04 the classical baselines train on `metadata_tensor` only (as in the notebook).
* `SingleFeatureDataset.__getitem__` does an O(n) DataFrame lookup per item, and
  04 recomputes the metadata embeddings in every fold; both are slow but correct.
* The 45993-row apache dataset used by 03a/03b is not produced by step 1.
* `sep_processing.py` needs an external `sep` package that was not part of the
  upload; it is imported lazily so everything else works without it.

## Tests

```bash
pytest
```

Covers config validation, text helpers, data prep and fold generation (including
a no-leakage check), the CLI, all nine baselines (binary and multi-class), and
model/trainer smoke tests (skipped automatically if `torch` isn't installed).
