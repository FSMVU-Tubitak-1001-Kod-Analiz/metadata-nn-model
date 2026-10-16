"""Train/val/test assignment of changes, written to data/processed/splits/<scheme>.csv.

Each file has one column per fold (fold1..foldN) with values train / val / test.
Splitting happens at the change level, so all files of a change share a part.

  random : changes split at random, stratified by label
  repo   : whole repositories held out (StratifiedGroupKFold, group = repo)
  org    : whole organisations held out (group = org)
  time   : one fold; the newest test_size of changes are test, the next newest val
  given  : test parts read from a file (`splits.given_parts`: unit_key, part), e.g. another
           paper's cross-validation folds; val is a stratified `given_val_size` share of the
           changes, drawn from the remaining parts

For random/repo/org the changes are cut into k = round(1 / test_size) parts;
fold i uses part i as test, part i+1 as val and the rest as train.

repo/org use StratifiedGroupKFold by default (`splits.group_splitter: sklearn`). With many tiny
groups and a few huge ones, scikit-learn <= 1.7 can return very unequal parts (on the CoLaRe set:
581 / 489 / 8 / 244 / 151 changes); `group_splitter: greedy` gives balanced parts that don't depend
on the scikit-learn version.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split

from commitclf.config import resolve
from commitclf.data.prepare import load_units, processed_path

SCHEMES = ("random", "repo", "org", "time", "given")
GROUP_COL = {"repo": "full_repo_name", "org": "org"}


def split_path(cfg: dict, scheme: str):
    return processed_path(cfg, "splits") / f"{scheme}.csv"


def _greedy_group_parts(units: pd.DataFrame, group_col: str, k: int, seed: int) -> pd.Series:
    """Groups in random order, then largest first; each goes to the part whose label counts end up
    closest (squared error) to an equal share of every label."""
    counts = pd.crosstab(units[group_col], units["label"])
    rng = np.random.default_rng(seed)
    counts = counts.iloc[rng.permutation(len(counts))]
    counts = counts.iloc[np.argsort(-counts.sum(axis=1).to_numpy(), kind="stable")]
    target = counts.sum().to_numpy() / k
    filled = np.zeros((k, counts.shape[1]))
    assign = {}
    for group, row in zip(counts.index, counts.to_numpy()):
        cost = ((filled + row - target) ** 2).sum(axis=1) - ((filled - target) ** 2).sum(axis=1)
        p = int(np.argmin(cost))
        filled[p] += row
        assign[group] = p
    return units[group_col].map(assign)


def _parts(units: pd.DataFrame, scheme: str, k: int, seed: int, group_splitter: str = "sklearn") -> pd.Series:
    """Assign each change to one of k parts."""
    part = pd.Series(-1, index=units.index)
    if scheme != "random" and group_splitter == "greedy":
        return _greedy_group_parts(units, GROUP_COL[scheme], k, seed)
    if group_splitter not in ("sklearn", "greedy"):
        raise ValueError(f"Unknown splits.group_splitter '{group_splitter}' (use sklearn or greedy)")
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

    if scheme == "given":
        return given_scheme(cfg, units, out)

    k = max(3, round(1 / s["test_size"]))
    if s["n_folds"] > k:
        raise ValueError(f"n_folds={s['n_folds']} is more than the {k} parts that test_size allows")
    if scheme in GROUP_COL and units[GROUP_COL[scheme]].nunique() < k:
        raise ValueError(f"Only {units[GROUP_COL[scheme]].nunique()} groups for scheme '{scheme}', need {k}")

    part = _parts(units, scheme, k, cfg["seed"], s.get("group_splitter", "sklearn"))
    sizes = part.value_counts().reindex(range(k), fill_value=0)
    if sizes.min() < 0.5 * len(units) / k:
        print(f"WARNING {scheme}: very unequal parts {sizes.tolist()} (expected ~{len(units) // k} each); "
              "consider splits.group_splitter: greedy")
    for f in range(s["n_folds"]):
        col = pd.Series("train", index=units.index)
        col[part == (f + 1) % k] = "val"
        col[part == f] = "test"
        out[f"fold{f + 1}"] = col
    return out


def given_scheme(cfg: dict, units: pd.DataFrame, out: pd.DataFrame) -> pd.DataFrame:
    s = cfg["splits"]
    if not s.get("given_parts"):
        raise ValueError("Scheme 'given' needs splits.given_parts (a CSV with unit_key, part)")
    parts = pd.read_csv(resolve(cfg, s["given_parts"])).set_index("unit_key")["part"]
    part = units["unit_key"].map(parts)
    if part.isna().any():
        raise ValueError(f"{int(part.isna().sum())} changes have no part in {s['given_parts']}")
    ids = sorted(part.unique())
    for f, p in enumerate(ids[:s["n_folds"]]):
        test = (part == p).to_numpy()
        rest = units.index[~test]
        val_share = s.get("given_val_size", s["val_size"]) / (1 - test.mean())
        _, val_idx = train_test_split(rest, test_size=val_share, stratify=units.loc[rest, "label"],
                                      random_state=cfg["seed"] + f)
        col = pd.Series("train", index=units.index)
        col[val_idx] = "val"
        col[test] = "test"
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
