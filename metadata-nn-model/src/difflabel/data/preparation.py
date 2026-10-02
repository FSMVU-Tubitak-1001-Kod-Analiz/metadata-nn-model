"""CSV loading, label standardisation and balancing (no torch dependency)."""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Merging per-repo CSVs
# ---------------------------------------------------------------------------
def merge_csv_files(input_dir: Path, output_csv: Path) -> pd.DataFrame | None:
    """Recursively merges every ``*.csv`` under ``input_dir`` into ``output_csv``.

    Adds a fresh sequential ``index`` column, which later steps use as the
    per-row id (it also names the per-row embedding files).
    """
    csv_files = glob.glob(os.path.join(str(input_dir), "**", "*.csv"), recursive=True)
    if not csv_files:
        print(f"No CSV files found under {input_dir}!")
        return None

    dataframes = [pd.read_csv(f) for f in tqdm(sorted(csv_files), desc="Reading CSVs")]
    merged = pd.concat(dataframes, ignore_index=True)
    merged["index"] = range(len(merged))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)
    print(f"Merged {len(csv_files)} files into {output_csv}")
    return merged


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
def standardize_labels(dataframe: pd.DataFrame, keywords: Sequence[str], label_col: str = "labels") -> pd.DataFrame:
    """Maps free-text labels onto the canonical ``keywords`` and drops non-matches.

    Matching is case-insensitive and substring-based, e.g. ``"performance-issue"``
    becomes ``"performance"``.
    """
    dataframe = dataframe.copy()
    keyword_map = {k.lower(): k for k in keywords}
    pattern = "|".join(map(re.escape, keyword_map.keys()))

    extracted = dataframe[label_col].str.extract(f"(?i)({pattern})")[0]
    dataframe[label_col] = extracted.str.lower().map(keyword_map)
    return dataframe.dropna(subset=[label_col])


def filter_for_equal_label_count(
    df: pd.DataFrame, labels: Sequence[str], seed: int, label_col: str = "labels"
) -> pd.DataFrame:
    """Down-samples every label in ``labels`` to the size of the smallest one."""
    df = standardize_labels(df, labels, label_col)
    print(f"After standardization, unique labels: {df[label_col].unique()}")

    per_label = [df[df[label_col] == label] for label in labels]
    min_samples = min(len(part) for part in per_label)
    print(min_samples)

    sampled = [part.sample(n=min_samples, random_state=seed) for part in per_label]
    return pd.concat(sampled)


def prepare_dataframe(
    file_path: Path,
    *,
    should_balance: bool,
    columns_to_keep: Sequence[str],
    columns_to_drop: Sequence[str],
    all_labels: Sequence[str],
    keep_labels: Sequence[str],
    balance_seed: int,
    label_col: str = "labels",
) -> pd.DataFrame:
    """Loads the raw CSV, de-duplicates, prunes columns and standardises labels."""
    df = pd.read_csv(file_path)
    df = df.drop_duplicates(keep="first")

    drop = [c for c in columns_to_drop if c not in columns_to_keep]
    df = df.drop(columns=drop)

    df = standardize_labels(df, all_labels, label_col)
    print(df[label_col].value_counts())

    if should_balance:
        df = filter_for_equal_label_count(df, keep_labels, balance_seed, label_col)
    else:
        df = pd.concat([df[df[label_col] == label] for label in keep_labels])

    if "patch" in df.columns:
        df = df.dropna(subset=["patch"])

    return df


# ---------------------------------------------------------------------------
# Row cleanup / repo balancing
# ---------------------------------------------------------------------------
def drop_empty_or_whitespace_rows(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """Drops rows where ``column_name`` is empty or whitespace-only."""
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in the DataFrame.")

    df = df.copy()
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
    dataframe: pd.DataFrame,
    labels: Sequence[str],
    n: int,
    repo_col: str = "full_repo_name",
    label_col: str = "labels",
) -> pd.DataFrame:
    """Keeps exactly ``n`` rows per repo with every label equally represented.

    Repos that can't supply enough rows for every label are discarded entirely.
    If ``n`` isn't divisible by the number of labels, the first labels in
    ``labels`` receive the extra rows.
    """
    target_per_label = n // len(labels)
    remainder = n % len(labels)

    def balance_group(group: pd.DataFrame):
        counts = group[label_col].value_counts()
        if not all(label in counts.index for label in labels):
            return None
        if any(counts[label] < target_per_label for label in labels):
            return None

        parts = []
        for i, label in enumerate(labels):
            take = target_per_label + (1 if i < remainder else 0)
            parts.append(group[group[label_col] == label].head(take))
        return pd.concat(parts)

    # Iterate explicitly rather than groupby().apply(): pandas >= 3 stopped
    # passing the grouping column into apply(), which would drop `repo_col`.
    balanced = [balance_group(group) for _, group in dataframe.groupby(repo_col)]
    balanced = [b for b in balanced if b is not None]

    if not balanced:
        print("Warning: No repositories met the criteria for equal label representation.")
        return pd.DataFrame(columns=dataframe.columns)

    return pd.concat(balanced).reset_index(drop=True)


def analyze_label_distribution(
    dataframe: pd.DataFrame, target_label: str, repo_col: str = "full_repo_name", label_col: str = "labels"
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

    print("\n--- Feasibility Check ---")
    print(f"Organizations that have > 10 '{target_label}' issues: {len(org_counts[org_counts > 10])}")
    print(f"Organizations that have > 100 '{target_label}' issues: {len(org_counts[org_counts > 100])}")


def filter_repo_name(df: pd.DataFrame, name_to_filter: str) -> pd.DataFrame:
    """Keeps rows whose ``full_repo_name`` contains ``name_to_filter`` (case-insensitive)."""
    return df[df["full_repo_name"].str.contains(name_to_filter, case=False, na=False)]
