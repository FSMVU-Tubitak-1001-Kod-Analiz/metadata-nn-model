"""Stratified cross-validation fold generation (no torch dependency)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from difflabel.config import FoldsConfig


def _stratify_key(df: pd.DataFrame, repo_col: str, label_col: str) -> pd.Series:
    return df[repo_col].astype(str) + "_" + df[label_col].astype(str)


def save_stratified_folds(
    dataframe: pd.DataFrame, output_dir: Path, cfg: FoldsConfig, repo_col: str, label_col: str
) -> None:
    """Writes ``foldN_train.csv`` / ``foldN_test.csv`` (stratified on repo + label)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    df = dataframe.copy()
    df["stratify_col"] = _stratify_key(df, repo_col, label_col)

    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.random_state)
    print(f"Starting stratification for {cfg.n_splits} folds...")

    for fold_num, (train_index, test_index) in enumerate(skf.split(df, df["stratify_col"]), 1):
        train_df = df.iloc[train_index].drop(columns=["stratify_col"])
        test_df = df.iloc[test_index].drop(columns=["stratify_col"])

        train_df.to_csv(output_dir / f"fold{fold_num}_train.csv", index=False)
        test_df.to_csv(output_dir / f"fold{fold_num}_test.csv", index=False)

        print(f"\nFold {fold_num} Created:")
        print(f"  Train: {len(train_df)} rows | Test: {len(test_df)} rows")
        print(f"  Label distribution in Test: {test_df[label_col].value_counts(normalize=True).to_dict()}")
        print(f"  Repo representation: {test_df[repo_col].nunique()}/{df[repo_col].nunique()} repos present in Test set.")

    print(f"\nAll folds saved successfully in '{output_dir}/'")


def save_stratified_folds_three_way(
    dataframe: pd.DataFrame, output_dir: Path, cfg: FoldsConfig, repo_col: str, label_col: str
) -> None:
    """Writes ``foldN_{train,test,val}.csv`` (default 70:15:15), stratified on repo + label."""
    output_dir.mkdir(parents=True, exist_ok=True)

    df = dataframe.copy()
    df["stratify_col"] = _stratify_key(df, repo_col, label_col)

    split_1 = StratifiedShuffleSplit(n_splits=cfg.n_splits, test_size=cfg.holdout_fraction, random_state=cfg.random_state)
    split_2 = StratifiedShuffleSplit(n_splits=1, test_size=1 - cfg.test_fraction_of_holdout, random_state=cfg.random_state)

    print(f"Generating {cfg.n_splits} folds (holdout={cfg.holdout_fraction:.0%})...")

    for fold_num, (train_index, temp_index) in enumerate(split_1.split(df, df["stratify_col"]), 1):
        train_df = df.iloc[train_index]
        temp_df = df.iloc[temp_index]

        for test_idx, val_idx in split_2.split(temp_df, temp_df["stratify_col"]):
            test_df = temp_df.iloc[test_idx]
            val_df = temp_df.iloc[val_idx]

        train_df = train_df.drop(columns=["stratify_col"])
        test_df = test_df.drop(columns=["stratify_col"])
        val_df = val_df.drop(columns=["stratify_col"])

        prefix = output_dir / f"fold{fold_num}"
        train_df.to_csv(f"{prefix}_train.csv", index=False)
        test_df.to_csv(f"{prefix}_test.csv", index=False)
        val_df.to_csv(f"{prefix}_val.csv", index=False)

        total = len(train_df) + len(test_df) + len(val_df)
        print(f"\nFold {fold_num}:")
        print(f"  Train: {len(train_df)} ({len(train_df) / total:.1%})")
        print(f"  Test:  {len(test_df)} ({len(test_df) / total:.1%})")
        print(f"  Val:   {len(val_df)} ({len(val_df) / total:.1%})")
        print(f"  Label Balance (Val): {val_df[label_col].value_counts(normalize=True).to_dict()}")

    print(f"\nSuccessfully saved {cfg.n_splits * 3} files ({cfg.n_splits} folds x 3 sets) to '{output_dir}/'")
