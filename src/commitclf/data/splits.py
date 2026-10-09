"""Train/val/test assignment of changes, written to data/processed/splits/<scheme>.csv.

Each file has one column per fold (fold1..foldN) with values train / val / test.
Splitting happens at the change level, so all files of a change share a part.

  random : changes split at random, stratified by label
  repo   : whole repositories held out (StratifiedGroupKFold, group = repo)
  org    : whole organisations held out (group = org)
  time   : one fold; the newest test_size of changes are test, the next newest val

For random/repo/org the changes are cut into k = round(1 / test_size) parts;
fold i uses part i as test, part i+1 as val and the rest as train.
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from commitclf.data.prepare import load_units, processed_path

SCHEMES = ("random", "repo", "org", "time")
GROUP_COL = {"repo": "full_repo_name", "org": "org"}


def split_path(cfg: dict, scheme: str):
    return processed_path(cfg, "splits") / f"{scheme}.csv"


def _parts(units: pd.DataFrame, scheme: str, k: int, seed: int) -> pd.Series:
    """Assign each change to one of k parts."""
    part = pd.Series(-1, index=units.index)
    if scheme == "random":
        splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        folds = splitter.split(units, units["label"])
    else:
        splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        folds = splitter.split(units, units["label"], groups=units[GROUP_COL[scheme]])
    for i, (_, test_idx) in enumerate(folds):
        part.iloc[test_idx] = i
    return part


def make_scheme(cfg: dict, units: pd.DataFrame, scheme: str) -> pd.DataFrame:
    s = cfg["splits"]
    out = pd.DataFrame({"unit_key": units["unit_key"]})

    if scheme == "time":
        order = units["merged_at"].rank(method="first") / len(units)
        out["fold1"] = "train"
        out.loc[order > 1 - s["test_size"] - s["val_size"], "fold1"] = "val"
        out.loc[order > 1 - s["test_size"], "fold1"] = "test"
        return out

    k = max(3, round(1 / s["test_size"]))
    if s["n_folds"] > k:
        raise ValueError(f"n_folds={s['n_folds']} is more than the {k} parts that test_size allows")
    if scheme in GROUP_COL and units[GROUP_COL[scheme]].nunique() < k:
        raise ValueError(f"Only {units[GROUP_COL[scheme]].nunique()} groups for scheme '{scheme}', need {k}")

    part = _parts(units, scheme, k, cfg["seed"])
    for f in range(s["n_folds"]):
        col = pd.Series("train", index=units.index)
        col[part == (f + 1) % k] = "val"
        col[part == f] = "test"
        out[f"fold{f + 1}"] = col
    return out


def run(cfg: dict, units: pd.DataFrame | None = None) -> None:
    units = load_units(cfg) if units is None else units
    split_path(cfg, "x").parent.mkdir(parents=True, exist_ok=True)
    for scheme in cfg["splits"]["schemes"]:
        if scheme not in SCHEMES:
            raise ValueError(f"Unknown split scheme '{scheme}' (use {SCHEMES})")
        table = make_scheme(cfg, units, scheme)
        table.to_csv(split_path(cfg, scheme), index=False)
        f1 = table["fold1"].value_counts()
        print(f"{scheme:>6}: {table.shape[1] - 1} fold(s); fold1 train={f1.get('train', 0)} "
              f"val={f1.get('val', 0)} test={f1.get('test', 0)}")


def load_split(cfg: dict, scheme: str, fold: int) -> pd.Series:
    """unit_key -> 'train' | 'val' | 'test' for one fold."""
    table = pd.read_csv(split_path(cfg, scheme))
    col = f"fold{fold}"
    if col not in table.columns:
        raise ValueError(f"Split '{scheme}' has no {col} (it has {table.shape[1] - 1} fold(s))")
    return table.set_index("unit_key")[col]


def n_folds(cfg: dict, scheme: str) -> int:
    return len(pd.read_csv(split_path(cfg, scheme), nrows=0).columns) - 1
