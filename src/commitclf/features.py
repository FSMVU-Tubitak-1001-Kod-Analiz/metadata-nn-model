"""Model inputs: per-file vectors grouped by change, plus change-level metadata."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from commitclf.data.prepare import load_rows, load_units
from commitclf.data.splits import load_split
from commitclf.embeddings import load_input

PARTS = ("train", "val", "test")
SUBSETS = (None, "single_commit")
CHANGE_GROUPS = ("change", "path", "repo", "message")

# Code inputs computed from two stored embeddings instead of read from `embed`:
#   patch_delta = CodeBERT(patch_after) - CodeBERT(patch_before), element-wise per file.
#   The shared context mostly cancels, so it mainly carries the added vs removed lines.
DERIVED_CODE = {"patch_delta": ("patch_after", "patch_before")}


class _Delta:
    """a - b over two [n_rows, hidden] arrays, computed per slice (keeps memory-mapped inputs lazy)."""

    def __init__(self, a, b):
        if a.shape != b.shape:
            raise ValueError(f"Embedding shapes differ: {a.shape} vs {b.shape}")
        self.a, self.b, self.shape = a, b, a.shape

    def __getitem__(self, idx):
        return (np.asarray(self.a[idx], dtype=np.float32) - np.asarray(self.b[idx], dtype=np.float32)).astype(np.float16)


def load_code_input(cfg: dict, name: str, pool: str):
    """A stored embedding input, or a derived one from DERIVED_CODE."""
    if name in DERIVED_CODE:
        a, b = DERIVED_CODE[name]
        return _Delta(load_input(cfg, a, pool), load_input(cfg, b, pool))
    return load_input(cfg, name, pool)


def encode_labels(labels: pd.Series, classes: list[str]) -> torch.Tensor:
    unknown = set(labels) - set(classes)
    if unknown:
        raise ValueError(f"Labels {unknown} not in configured labels {classes}")
    return torch.tensor(labels.map({c: i for i, c in enumerate(classes)}).to_numpy(), dtype=torch.long)


def load_parts(cfg: dict, scheme: str, fold: int, subset: str | None = None) -> dict[str, pd.DataFrame]:
    """The changes in train / val / test for one fold, optionally restricted to a subset."""
    if subset not in SUBSETS:
        raise ValueError(f"Unknown subset '{subset}' (use {SUBSETS})")
    units = load_units(cfg)
    units["part"] = units["unit_key"].map(load_split(cfg, scheme, fold))
    if subset == "single_commit":
        units = units[units["n_commits"] == 1]
    return {p: units[units["part"] == p].reset_index(drop=True) for p in PARTS}


class FileTable:
    """All file rows with their code vectors (+ per-file facts), indexed like rows.csv."""

    def __init__(self, cfg: dict, code: list[str], embed_pool: str, with_facts: bool):
        rows = load_rows(cfg)
        self.unit_rows = pd.Series(np.arange(len(rows))).groupby(rows["unit_key"].to_numpy()).agg(list)
        blocks = [load_code_input(cfg, name, embed_pool) for name in code]
        facts = rows[[c for c in rows.columns if c.startswith("file__")]].to_numpy(dtype=np.float16) \
            if with_facts else np.zeros((len(rows), 0), dtype=np.float16)
        dim = sum(b.shape[1] for b in blocks) + facts.shape[1]
        self.x = torch.empty(len(rows), dim, dtype=torch.float16)
        col = 0
        for block in blocks + [facts]:
            w = block.shape[1]
            for s in range(0, len(rows), 100_000):   # copy from the memory-mapped file in slices
                self.x[s:s + 100_000, col:col + w] = torch.from_numpy(np.array(block[s:s + 100_000]))
            col += w

    @property
    def dim(self) -> int:
        return self.x.shape[1]

    def offsets(self, unit_keys: pd.Series) -> tuple[torch.Tensor, torch.Tensor]:
        """Concatenated row indices of the given changes and the start offset of each change."""
        lists = self.unit_rows.loc[unit_keys.to_numpy()]
        lengths = lists.map(len).to_numpy()
        offsets = np.concatenate([[0], np.cumsum(lengths)])
        return torch.from_numpy(np.concatenate(lists.to_numpy()).astype(np.int64)), torch.from_numpy(offsets)


def change_metadata(cfg: dict, units: pd.DataFrame, groups: list[str], embed_pool: str = "mean") -> torch.Tensor:
    """Change-level metadata for the given changes."""
    parts = []
    for g in groups:
        if g in ("change", "path"):
            cols = [c for c in units.columns if c.startswith(f"{g}__")]
            parts.append(torch.tensor(units[cols].to_numpy(dtype="float32")))
        elif g == "repo":
            repos = sorted(load_units(cfg)["full_repo_name"].unique())
            onehot = pd.get_dummies(pd.Categorical(units["full_repo_name"], categories=repos))
            parts.append(torch.tensor(onehot.to_numpy(dtype="float32")))
        elif g == "message":
            rows = load_rows(cfg)
            first = pd.Series(np.arange(len(rows))).groupby(rows["unit_key"].to_numpy()).first()
            msg = np.asarray(load_input(cfg, "message", "mean" if embed_pool == "pooler" else embed_pool))
            parts.append(torch.tensor(msg[first.loc[units["unit_key"].to_numpy()].to_numpy()], dtype=torch.float32))
        else:
            raise ValueError(f"Unknown metadata group '{g}' (use {CHANGE_GROUPS})")
    return torch.cat(parts, dim=1) if parts else torch.zeros(len(units), 0)


@dataclass
class ChangeSet:
    """One part (train/val/test): changes, their file rows and metadata."""
    units: pd.DataFrame
    y: torch.Tensor
    meta: torch.Tensor
    rows: torch.Tensor      # concatenated file-row indices
    offsets: torch.Tensor   # change i owns rows[offsets[i]:offsets[i+1]]

    def __len__(self) -> int:
        return len(self.units)


def build_sets(cfg: dict, spec: dict, fold: int, table: FileTable) -> dict[str, ChangeSet]:
    parts = load_parts(cfg, spec["split"], fold, spec["subset"])
    uses_meta = spec["model"] != "code"
    out = {}
    for p, units in parts.items():
        rows, offsets = table.offsets(units["unit_key"])
        meta = change_metadata(cfg, units, spec["metadata"] if uses_meta else [], spec["embed_pool"])
        out[p] = ChangeSet(units, encode_labels(units["label"], cfg["labels"]), meta, rows, offsets)
    return out


def collate(table: FileTable, cs: ChangeSet, idx: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Padded file tensor [B, F, D], mask [B, F], metadata [B, M], labels [B]."""
    starts, ends = cs.offsets[idx], cs.offsets[idx + 1]
    counts = ends - starts
    f_max = int(counts.max())
    files = torch.zeros(len(idx), f_max, table.dim, dtype=torch.float16)
    mask = torch.zeros(len(idx), f_max, dtype=torch.bool)
    for b, (s, n) in enumerate(zip(starts.tolist(), counts.tolist())):
        files[b, :n] = table.x[cs.rows[s:s + n]]
        mask[b, :n] = True
    return files, mask, cs.meta[idx], cs.y[idx]


def batches(table: FileTable, cs: ChangeSet, batch_size: int, shuffle: bool, generator=None):
    order = torch.randperm(len(cs), generator=generator) if shuffle else torch.arange(len(cs))
    n = len(order)
    for s in range(0, n, batch_size):
        idx = order[s:s + batch_size]
        if shuffle and len(idx) == 1:   # BatchNorm can't train on a single sample
            continue
        yield collate(table, cs, idx)


def pooled_code(table: FileTable, cs: ChangeSet) -> np.ndarray:
    """Mean of each change's file vectors (fixed-length input for the classic ML baselines)."""
    out = np.zeros((len(cs), table.dim), dtype=np.float32)
    x = table.x
    for i in range(len(cs)):
        s, e = int(cs.offsets[i]), int(cs.offsets[i + 1])
        out[i] = x[cs.rows[s:e]].float().mean(0).numpy()
    return out
