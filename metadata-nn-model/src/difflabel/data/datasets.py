"""PyTorch datasets + dataloader helper."""

from __future__ import annotations

import pandas as pd
from torch.utils.data import DataLoader, Dataset


class SingleFeatureDataset(Dataset):
    """Single-input dataset (03a/03b). Yields ``(feature, label_tensor, index)``.

    ``feature_column`` selects which column is the model input:
    ``"embedding"`` (code pipeline) or ``"metadata_tensor"`` (metadata pipeline).
    """

    def __init__(self, df: pd.DataFrame, feature_column: str):
        self.df = df
        self.feature_column = feature_column
        self.id_list = df["index"].values

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        index = self.id_list[idx]
        row = self.df.loc[self.df["index"] == index]
        return row[self.feature_column].iloc[0], row["label_tensor"].iloc[0], index


class CodeMetadataDataset(Dataset):
    """Dual-input dataset (04). Yields ``(embedding, label_tensor, metadata_tensor, index)``."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.id_list = df["index"].values

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        index = self.id_list[idx]
        row = self.df.loc[self.df["index"] == index]
        return row["embedding"].iloc[0], row["label_tensor"].iloc[0], row["metadata_tensor"].iloc[0], index


def create_dataloader(dataset_cls, df: pd.DataFrame, batch_size: int, shuffle: bool = True, **dataset_kwargs) -> DataLoader:
    return DataLoader(dataset_cls(df, **dataset_kwargs), batch_size=batch_size, shuffle=shuffle)
