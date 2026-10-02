"""Helpers shared by the training pipelines."""

from __future__ import annotations

import pandas as pd

from difflabel.config import EmbeddingFilesConfig, resolve_path
from difflabel.data.embedding_io import load_embedding_features


def code_embedding_dirs(cfg: EmbeddingFilesConfig):
    """Order matters: content_before first, then patch_before (same as the original concat)."""
    return [resolve_path(cfg.content_before_dir), resolve_path(cfg.patch_before_dir)]


def attach_code_embeddings(df: pd.DataFrame, cfg: EmbeddingFilesConfig) -> pd.DataFrame:
    return load_embedding_features(df, code_embedding_dirs(cfg))


def keep_columns(df: pd.DataFrame, columns) -> pd.DataFrame:
    """Selects the columns the models need (replaces the long per-script drop lists)."""
    return df[list(columns)].copy()


def n_classes(df: pd.DataFrame) -> int:
    return len(df["label_tensor"].iloc[0])
