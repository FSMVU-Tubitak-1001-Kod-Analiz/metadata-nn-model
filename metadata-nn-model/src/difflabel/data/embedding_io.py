"""Reading per-row ``embedding_<index>.pt`` files back into a DataFrame."""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path
from typing import List, Sequence

import pandas as pd
import torch
from tqdm.auto import tqdm


def load_embedding(directory: Path, id_number) -> torch.Tensor:
    """Loads ``embedding_<id_number>.pt`` from ``directory`` onto the CPU."""
    matches = glob.glob(os.path.join(str(directory), f"embedding_{id_number}.pt"))
    if not matches:
        raise FileNotFoundError(f"No embedding file found for ID {id_number} in {directory}")
    return torch.load(matches[0], map_location="cpu")


def load_embedding_features(df: pd.DataFrame, directories: Sequence[Path], index_column: str = "index") -> pd.DataFrame:
    """Adds an ``embedding`` column to ``df``.

    For each row, the embedding with the row's id is loaded from every
    directory in ``directories``, flattened, and concatenated (in the order
    given).  One directory -> a plain embedding; two -> content_before ++
    patch_before.
    """
    embeddings: List[torch.Tensor] = []
    for index in tqdm(df[index_column].values, desc="Loading embeddings"):
        parts = [torch.flatten(load_embedding(d, index)) for d in directories]
        embeddings.append(parts[0] if len(parts) == 1 else torch.cat(parts, dim=0))

    df["embedding"] = embeddings
    return df


def load_all_embeddings(directory: Path) -> List[torch.Tensor]:
    """Loads every ``.pt`` file in ``directory``, ordered by the number in its filename."""
    files = sorted(os.listdir(directory), key=lambda f: int(re.search(r"\d+", f).group()))
    embeddings = []
    for name in tqdm(files, desc="Loading embeddings"):
        path = os.path.join(directory, name)
        if os.path.isfile(path) and path.endswith(".pt"):
            try:
                embeddings.append(torch.load(path))
            except Exception as exc:
                print(f"Error loading file {path}: {exc}")
    return embeddings
