"""Step 1: clean + balance the raw CSV and generate cross-validation folds."""

from __future__ import annotations

from difflabel.config import PrepareCsvConfig, resolve_path
from difflabel.data.folds import save_stratified_folds, save_stratified_folds_three_way
from difflabel.data.java_text import clean_java_code, extract_code_from_patch
from difflabel.data.preparation import (
    analyze_label_distribution,
    display_repo_counts,
    drop_empty_or_whitespace_rows,
    filter_for_equal_label_count,
    filter_top_repos_equal_representation,
    merge_csv_files,
    prepare_dataframe,
)


def run(cfg: PrepareCsvConfig) -> None:
    label_col, repo_col = cfg.label_column, cfg.repo_column
    output_dir = resolve_path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / cfg.output_csv_name
    raw_csv = resolve_path(cfg.raw_csv)

    # 0. Optionally build the raw CSV from a folder of per-repo CSVs
    if cfg.merge.enabled:
        merge_csv_files(resolve_path(cfg.merge.input_dir), resolve_path(cfg.merge.output_csv))

    # 1. Load + standard label cleanup / column pruning
    df = prepare_dataframe(
        raw_csv,
        should_balance=cfg.balance_labels_early,
        columns_to_keep=cfg.columns_to_keep,
        columns_to_drop=cfg.columns_to_drop,
        all_labels=cfg.labels.all,
        keep_labels=cfg.labels.keep,
        balance_seed=cfg.balance_seed,
        label_col=label_col,
    )
    print(df[label_col].value_counts())

    # 2. Balance rows per repo so no single repo dominates
    display_repo_counts(df, repo_col)
    df_filtered = filter_top_repos_equal_representation(
        df.copy(), labels=cfg.labels.keep, n=cfg.rows_per_repo, repo_col=repo_col, label_col=label_col
    )
    print(df_filtered[label_col].value_counts())
    display_repo_counts(df_filtered, repo_col)

    if cfg.analysis_target_label:
        analyze_label_distribution(df_filtered, cfg.analysis_target_label, repo_col, label_col)

    # 3. Derive `patch_before` from the raw patch text
    df_filtered["patch_before"] = df_filtered["patch"].apply(extract_code_from_patch)

    # 4. Strip license comments / imports / package statements
    for column in ("patch_before", "content_before", "content_after"):
        df_filtered[column] = df_filtered[column].apply(clean_java_code)

    # 5. Drop rows left empty after cleaning
    df = df_filtered
    for column in ("content_before", "patch_before", "content_after"):
        df = drop_empty_or_whitespace_rows(df, column)

    # 6. Re-balance labels once more after all the row drops
    df = filter_for_equal_label_count(df, cfg.labels.keep, cfg.balance_seed, label_col)
    print(df[label_col].value_counts())

    # 7. Persist the cleaned, balanced dataset
    df.to_csv(output_csv, index=False)
    print(f"Saved cleaned dataset to {output_csv} ({len(df)} rows)")

    # 8. Generate both sets of cross-validation folds
    save_stratified_folds(df, output_dir / cfg.folds.kfold_subdir, cfg.folds, repo_col, label_col)
    save_stratified_folds_three_way(df, output_dir / cfg.folds.split_subdir, cfg.folds, repo_col, label_col)
