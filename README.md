# metadata-nn-model

Classifies code changes in Java projects as **bug**, **enhancement** or **clean**
(Swanson's corrective / adaptive / perfective maintenance; refactoring and performance
count as clean).

- **One sample is one change:** a merged PR (its net diff) or a commit that closes a labelled
  issue.
- **Each file** of the change is represented by frozen CodeBERT vectors of its code.
- **The files** are combined with attention.
- **The result** is fused with leakage-free change metadata.
- **Baselines:** nine classic ML models run on the same features.

The data comes from `repo-data-collector/collector_v2`. Its `data/final/README.md` describes
the dataset.

## Setup

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124   # pick your CUDA build
pip install -e .
```

Every setting lives in `configs/config.yaml`. Check the paths under `data` first.

## Pipeline

```bash
python -m commitclf prepare          # dataset_meta.csv -> data/processed/ (rows, changes, splits); minutes
python -m commitclf embed            # CodeBERT vectors for all files; hours, resumable
python -m commitclf grid --dry-run   # list the experiment grid
python -m commitclf grid             # run it (finished runs are skipped)
```

**`embed`** reads `dataset.csv` once. In that single pass it fills four inputs per file row:

- `content_before`
- `patch`: the full diff
- `patch_before`: context + removed lines
- `patch_after`: context + added lines

Each input is stored twice, as CLS and as mean token pooling (float16, about 1.3 GB per array).
If it stops, running the same command again continues where it left off.
`embed --only message` adds the BERT embedding of the PR title + commit message, needed only for
the leakage block.

## Model

```
file i: [CodeBERT(content_before), CodeBERT(patch), file facts*] --MLP--> h_i
change:  pool(h_1..h_n)  (mean | max | attention: a_i = softmax_i(w·tanh(V h_i)))  -> code side
change facts* --MLP--> metadata side
fusion (concat | gated | attention) -> MLP head -> bug / enhancement / clean
```

\* Metadata, used only by models that use metadata:

- **File facts:** status (added/modified/removed/renamed), log lines added and deleted, test
  file, content omitted, path flags (`src/main`, `src/test`, examples/docs, depth, name patterns
  Test/Exception/Util/Config/Impl/package-info).
- **Change facts:** total churn, number of files and Java files, number of commits, deletion
  share, shares of test/added/removed/renamed files.
- **Not included by default:** the repo name (`--meta change repo`) and the commit message
  (`--meta change message`). The message leaks the label.

**Model types:**
- `code`: code only.
- `metadata`: file facts pooled over files, plus change facts.
- `concat`: both sides concatenated.
- `gated`: `g·tanh(Pc c) + (1−g)·tanh(Pm m)`, with a per-dimension gate `g`.
- `attention`: softmax weights over the two sides, one weight per side and change. The log shows
  the mean code weight.

**Token pooling inside CodeBERT** (`--embed-pool`): `mean` (default), `cls`, or `pooler`, which
is tanh(W·cls+b) and what the paper used.

**Training:** Adam (lr 3e-5, weight decay 1e-4, lr halved after 3 epochs without val macro-F1 improvement), class-weighted cross-entropy, batch of 128 changes. Early stopping
on validation macro-F1 (patience 10, at most 100 epochs). The best checkpoint is evaluated on test.

## Splits

All files of a change always stay in the same part.

| scheme | held out |
|---|---|
| `repo` | whole repositories (StratifiedGroupKFold) |
| `org` | whole organisations; Apache, 17% of the data, lands in one part |
| `random` | random changes; used only to measure the repo shortcut |
| `time` | the newest 15% (1 fold); available, not in the grid |

5 folds of about 70/15/15: the changes are cut into 7 parts; fold i tests on part i and
validates on part i+1.

## Experiment grid (`configs/grid.yaml`)

27 network runs × 5 folds, plus 7 baseline settings.

| block | runs |
|---|---|
| `main` | code, metadata, concat, gated, attention × repo, org |
| `file_pool` | best fusion with mean / max file pooling (repo) |
| `code_input` | code and best fusion with `content_before+patch_before` (paper), `patch` only, `content_before+patch_before+patch_after` (repo) |
| `embed_pool` | code with cls / pooler (repo) |
| `leakage` | metadata and best fusion + commit message (repo) |
| `robustness` | best fusion on single-commit changes (repo, org) |
| `shortcut` | paper-style model with / without repo name, best fusion (random split) |
| `baselines` | code / metadata / both × repo, org; metadata × random |

"Best fusion" is whichever of concat / gated / attention has the highest repo-split macro-F1 in
`main`, so `main` must finish first; the grid does it in order. Useful options:

- `grid --blocks main`: run one block.
- `grid --folds 1`: a quick pass on fold 1.
- `grid --force`: redo finished runs.

## Single runs

Every `experiment` field can be overridden:

```bash
python -m commitclf train --model gated --split org --folds all
python -m commitclf train --model code --code patch --embed-pool cls
python -m commitclf train --meta change repo --split random --name with_repo
python -m commitclf baselines --features metadata --split repo --models XGBoost LogisticRegression
```

## Baselines

- XGBoost, RandomForest, CatBoost, LightGBM, KNN, NaiveBayes, LogisticRegression, AdaBoost,
  LinearSVM. LinearSVM is trained with SGD, because a kernel SVC on about 110k changes would take
  hours per fit.
- Code features are the mean of the change's file vectors. Metadata features are the change facts
  plus path shares.
- Training uses train+val. Classes are weighted where the model allows. Scale-sensitive models
  (LogisticRegression, LinearSVM, KNN) get a StandardScaler. KNN is fit on a 30k subsample.
- Two reference rows: `Majority`, and `RepoMajority` (the repo's most common training class,
  meaningful on the random split).

## Explaining the results (after the grid)

These reuse the saved checkpoints; nothing is retrained except a fast XGBoost for SHAP.

```bash
python -m commitclf explain shap --split repo                 # metadata feature ranking + direction, per class
python -m commitclf explain permutation --run main/metadata_repo   # macro-F1 drop when a feature group is shuffled
python -m commitclf explain permutation --run main/<fusion>_repo
python -m commitclf explain weights --run main/<fusion>_repo  # attention over files, code-vs-metadata weight
python -m commitclf explain tokens --run main/<fusion>_repo --folds 1   # Integrated Gradients on the diff
python -m commitclf explain occlusion --run main/<fusion>_repo --folds 1  # delete one line type, re-predict
```

- **shap:** XGBoost on the change-level metadata, then TreeSHAP. Outputs `shap_ranking.csv`,
  `shap_per_class.csv` (mean |SHAP| plus the direction as a Spearman correlation of feature value
  vs SHAP) and one beeswarm plot per class.
- **permutation:** shuffles a feature group, or a single feature, across the test set and
  measures the drop in macro-F1. The groups are size, status, test files, path/name, number of
  files, commits, and each code input. Output: `permutation.csv`, mean ± std over folds.
- **weights:** per test change, the fusion weight (code side vs change metadata) and the
  attention weight of every file. The summary shows whether test, added or renamed files get
  more or less than an equal share.
- **tokens:** for about 300 test changes (5 files or fewer), Integrated Gradients is pushed
  through CodeBERT down to the tokens of the `patch` input.
  - `token_attribution_summary.csv` gives the share of attribution on added, removed and context
    lines and on hunk headers, compared with their share of tokens (enrichment > 1 = the model
    focuses there).
  - `examples.html` shows highlighted diffs.
  - The run prints a completeness check: attributions should sum to the change in model output.
  - It needs the source `dataset.csv` for the patch text and takes roughly an hour on a GPU.
- **occlusion:** for the same kind of sample, one line type at a time (added, removed, context,
  hunk header) is deleted from every `patch`, the patch is re-embedded with CodeBERT and the
  classifier is run again. As a control, the same number of random lines is deleted.
  - `occlusion_summary.csv`: per variant, the mean drop in the probability of the original
    prediction, how often the prediction flips, and macro-F1 after the deletion. Subsets: all,
    patches that fit in 512 tokens, and each true label.
  - `occlusion_per_change.csv`: one row per change and variant.
  - Reuses the patch texts cached by `tokens` (or reads `dataset.csv` once).

Outputs are written to `artifacts/explain/<analysis>/<run>/`. Don't rerun `prepare` or `embed`
on different data before explaining: the checkpoints are tied to the current rows and embeddings.

## Outputs

```
data/processed/rows.csv, units.csv, splits/<scheme>.csv
artifacts/embeddings/<input>.{cls,mean}.npy, <model>_pooler.pt, <input>.progress.json
artifacts/runs/<block>/<run>/spec.json, summary.csv (per fold + mean/std)
artifacts/runs/<block>/<run>/fold<N>/model.pt, history.csv, test_metrics.json,
                                     test_predictions.csv, confusion_matrix.png
artifacts/baselines/<features>_<split>/metrics.csv, summary.csv, confusion_matrices_fold<N>.png
```

Metrics are per change: accuracy, macro precision/recall/F1, per-class F1, and one-vs-rest
macro ROC-AUC. **Macro-F1 is the main number**, because the full data is 52% bug, 40%
enhancement and 8% clean.

## Known limitations

- CodeBERT reads 512 tokens. Half of the patches and 78% of `content_before` are longer, so
  only their beginning is seen.
- Labels come from the PR or issue. Files inside a change can have different purposes; the
  attention over files softens this but doesn't remove it.
