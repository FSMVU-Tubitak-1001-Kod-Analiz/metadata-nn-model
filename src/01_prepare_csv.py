"""
01_prepare_csv.py
==================
Converted from: CSV_preparation.ipynb

Loads the raw merged PR/commit CSV, cleans it up, balances/filters it by
repository and label, strips license comments and import statements from the
Java code columns, and writes out:

  1. A single balanced CSV (`OUTPUT_CSV`).
  2. Two sets of stratified cross-validation folds derived from that CSV:
       - `folds/foldN_train.csv` / `foldN_test.csv` (train/test only)
       - `folds_70_15_15/foldN_train.csv` / `foldN_val.csv` / `foldN_test.csv`

Run this first. Its output feeds into 02_create_embeddings.py and
04_train_gated_fusion_folds.py.

Expects `merged.csv` (or whatever RAW_CSV points at) to already exist -- e.g.
produced by `utils.merge_csv_files_in_directory()` from a folder of per-repo
CSVs in `inputs/`.
"""

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from utils import (
    prepare_dataframe,
    filterForEqualLabelCount,
    extract_code_from_patch,
    clean_java_code,
)

# ---------------------------------------------------------------------------
# Configuration -- edit these to match your data
# ---------------------------------------------------------------------------
RAW_CSV = "merged.csv"
OUTPUT_DIR = "../new_csv_files_for_rewrite"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "merged_2label_no_license_no_imports_balanced_14768.csv")

# How many rows to keep per repo when balancing label representation.
ROWS_PER_REPO = 440

COLUMNS_TO_KEEP = [
    "pr_title", "commit_message", "file_path", "content_after", "content_before",
    "patch_before", "patch_after", "additions", "deletions",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def drop_empty_or_whitespace_rows(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """Drops rows where `column_name` is empty or whitespace-only."""
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in the DataFrame.")

    df[column_name] = df[column_name].replace(r"^\s*$", np.nan, regex=True)
    return df.dropna(subset=[column_name])


def display_repo_counts(dataframe: pd.DataFrame, repo_col: str = "full_repo_name") -> pd.Series:
    """Prints and returns the number of rows per unique repository."""
    counts = dataframe[repo_col].value_counts()
    print(f"Total unique repos: {len(counts)}")
    print("-" * 30)
    print(counts)
    return counts


def filter_top_repos_equal_representation(
    dataframe: pd.DataFrame, repo_col: str = "full_repo_name", label_col: str = "labels", n: int = 500
) -> pd.DataFrame:
    """
    Keeps exactly `n` rows per repo, with each label ('bug', 'enhancement')
    equally represented (n/2 each). Repos that can't supply enough rows for
    every label are discarded entirely.
    """
    labels = ["bug", "enhancement"]
    num_labels = len(labels)
    target_per_label = n // num_labels
    remainder = n % num_labels

    def balance_group(group: pd.DataFrame):
        available_labels = group[label_col].unique()
        if not all(l in available_labels for l in labels):
            return None

        counts = group[label_col].value_counts()
        for label in labels:
            if counts[label] < target_per_label:
                return None

        sampled_parts = []
        for i, label in enumerate(labels):
            count_to_take = target_per_label + (1 if i < remainder else 0)
            subset = group[group[label_col] == label].head(count_to_take)
            sampled_parts.append(subset)

        return pd.concat(sampled_parts)

    df_final = dataframe.groupby(repo_col, group_keys=False).apply(balance_group)

    if df_final is None or df_final.empty:
        print("Warning: No repositories met the criteria for equal label representation.")
        return pd.DataFrame(columns=dataframe.columns)

    return df_final.reset_index(drop=True)


def analyze_label_distribution(
    dataframe: pd.DataFrame, target_label: str = "clean", repo_col: str = "full_repo_name", label_col: str = "labels"
) -> None:
    """Prints where a given label is concentrated, by org and by repo."""
    df = dataframe.copy()
    df["org"] = df[repo_col].str.split("/").str[0]

    print(f"--- Analysis for label: '{target_label}' ---")

    total_matches = len(df[df[label_col] == target_label])
    print(f"Total '{target_label}' issues in entire dataset: {total_matches}")

    if total_matches == 0:
        print(f"WARNING: The label '{target_label}' was not found at all. Check for typos or capitalization.")
        print(f"Unique labels actually found: {df[label_col].unique()}")
        return

    org_counts = df[df[label_col] == target_label].groupby("org").size().sort_values(ascending=False)
    print(f"\nTop 10 Organizations with '{target_label}':")
    print(org_counts.head(10))

    repo_counts = df[df[label_col] == target_label].groupby(repo_col).size().sort_values(ascending=False)
    print(f"\nTop 10 Repositories with '{target_label}':")
    print(repo_counts.head(10))

    print(f"\n--- Feasibility Check ---")
    print(f"Organizations that have > 10 '{target_label}' issues: {len(org_counts[org_counts > 10])}")
    print(f"Organizations that have > 100 '{target_label}' issues: {len(org_counts[org_counts > 100])}")


def save_stratified_folds(
    dataframe: pd.DataFrame,
    output_dir: str = os.path.join(OUTPUT_DIR, "folds"),
    n_splits: int = 5,
    repo_col: str = "full_repo_name",
    label_col: str = "labels",
) -> None:
    """
    Splits `dataframe` into `n_splits` stratified train/test folds, stratifying
    on the combination of repo + label so both are balanced across the split.
    """
    os.makedirs(output_dir, exist_ok=True)

    df = dataframe.copy()
    df["stratify_col"] = df[repo_col].astype(str) + "_" + df[label_col].astype(str)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    print(f"Starting stratification for {n_splits} folds...")

    for fold_num, (train_index, test_index) in enumerate(skf.split(df, df["stratify_col"]), 1):
        train_df = df.iloc[train_index].drop(columns=["stratify_col"])
        test_df = df.iloc[test_index].drop(columns=["stratify_col"])

        train_filename = os.path.join(output_dir, f"fold{fold_num}_train.csv")
        test_filename = os.path.join(output_dir, f"fold{fold_num}_test.csv")

        train_df.to_csv(train_filename, index=False)
        test_df.to_csv(test_filename, index=False)

        print(f"\nFold {fold_num} Created:")
        print(f"  Train: {len(train_df)} rows | Test: {len(test_df)} rows")

        label_dist = test_df[label_col].value_counts(normalize=True).to_dict()
        print(f"  Label distribution in Test: {label_dist}")

        unique_repos_test = test_df[repo_col].nunique()
        unique_repos_orig = df[repo_col].nunique()
        print(f"  Repo representation: {unique_repos_test}/{unique_repos_orig} repos present in Test set.")

    print(f"\nAll folds saved successfully in '{output_dir}/'")


def save_stratified_folds_70_15_15(
    dataframe: pd.DataFrame,
    output_dir: str = os.path.join(OUTPUT_DIR, "folds_70_15_15"),
    n_splits: int = 5,
    repo_col: str = "full_repo_name",
    label_col: str = "labels",
) -> None:
    """
    Splits `dataframe` into `n_splits` folds with a 70:15:15 train/test/val
    ratio each, stratified on repo + label.
    """
    os.makedirs(output_dir, exist_ok=True)

    df = dataframe.copy()
    df["stratify_col"] = df[repo_col].astype(str) + "_" + df[label_col].astype(str)

    split_1 = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.30, random_state=42)
    split_2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=42)

    print(f"Generating {n_splits} folds with 70:15:15 ratio...")

    for fold_num, (train_index, temp_index) in enumerate(split_1.split(df, df["stratify_col"]), 1):
        train_df = df.iloc[train_index]
        temp_df = df.iloc[temp_index]  # the remaining 30%

        for test_idx, val_idx in split_2.split(temp_df, temp_df["stratify_col"]):
            test_df = temp_df.iloc[test_idx]
            val_df = temp_df.iloc[val_idx]

        train_df = train_df.drop(columns=["stratify_col"])
        test_df = test_df.drop(columns=["stratify_col"])
        val_df = val_df.drop(columns=["stratify_col"])

        fold_prefix = os.path.join(output_dir, f"fold{fold_num}")
        train_df.to_csv(f"{fold_prefix}_train.csv", index=False)
        test_df.to_csv(f"{fold_prefix}_test.csv", index=False)
        val_df.to_csv(f"{fold_prefix}_val.csv", index=False)

        total = len(train_df) + len(test_df) + len(val_df)
        print(f"\nFold {fold_num}:")
        print(f"  Train: {len(train_df)} ({len(train_df)/total:.1%})")
        print(f"  Test:  {len(test_df)} ({len(test_df)/total:.1%})")
        print(f"  Val:   {len(val_df)} ({len(val_df)/total:.1%})")

        val_balance = val_df[label_col].value_counts(normalize=True).to_dict()
        print(f"  Label Balance (Val): {val_balance}")

    print(f"\nSuccessfully saved {n_splits * 3} files ({n_splits} folds x 3 sets) to '{output_dir}/'")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load + do the standard label cleanup/column pruning from utils.py
    df = prepare_dataframe(
        RAW_CSV, shouldBalance=False, should_merge_csv=False, columns_to_keep=COLUMNS_TO_KEEP
    )
    print(df["labels"].value_counts())

    # 2. Balance rows per repo so no single repo dominates the dataset
    display_repo_counts(df)
    df_filtered = filter_top_repos_equal_representation(dataframe=df.copy(), n=ROWS_PER_REPO)
    print(df_filtered["labels"].value_counts())
    display_repo_counts(df_filtered)

    # Optional: inspect where a given label is concentrated before continuing
    analyze_label_distribution(df_filtered, target_label="clean")

    # 3. Derive `patch_before` from the raw patch text
    df_filtered["patch_before"] = df_filtered["patch"].apply(extract_code_from_patch)

    # 4. Strip license comments / imports / package statements from the Java code
    df_filtered["patch_before"] = df_filtered["patch_before"].apply(clean_java_code)
    df_filtered["content_before"] = df_filtered["content_before"].apply(clean_java_code)
    df_filtered["content_after"] = df_filtered["content_after"].apply(clean_java_code)

    # 5. Drop rows left empty after cleaning
    df = df_filtered
    df = drop_empty_or_whitespace_rows(df, "content_before")
    df = drop_empty_or_whitespace_rows(df, "patch_before")
    df = drop_empty_or_whitespace_rows(df, "content_after")

    # 6. Re-balance labels one final time after all the row drops above
    df = filterForEqualLabelCount(df=df)
    print(df["labels"].value_counts())

    # 7. Persist the cleaned, balanced dataset
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved cleaned dataset to {OUTPUT_CSV} ({len(df)} rows)")

    # 8. Generate both sets of cross-validation folds from it
    save_stratified_folds(df)
    save_stratified_folds_70_15_15(df)


if __name__ == "__main__":
    main()
