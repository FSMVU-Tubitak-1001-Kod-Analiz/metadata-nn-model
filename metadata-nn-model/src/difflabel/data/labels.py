"""One-hot label tensors."""

from __future__ import annotations

import pandas as pd
import torch


def add_label_tensor(df: pd.DataFrame, label_column: str = "labels") -> pd.DataFrame:
    """Adds a ``label_tensor`` column: one float32 one-hot vector per row.

    Class order is the sorted order of the unique labels (``pd.get_dummies``),
    so index 0 is e.g. ``bug`` and index 1 is ``enhancement``.
    """
    one_hot = pd.get_dummies(df[label_column]).astype(int)
    print(f"Label classes (one-hot order): {list(one_hot.columns)}")
    df["label_tensor"] = [torch.tensor(row, dtype=torch.float32) for row in one_hot.values]
    return df
