"""Collector dataset -> per-file and per-change tables (no code text).

Writes to data/processed/:
  rows.csv   one row per (change, .java file): keys, label and per-file facts `file__*`
  units.csv  one row per change: label, repo, org, date and change-level facts
             (`change__*`, plus `path__*` = shares of the per-file path flags)

Code columns are never loaded here; `embed` streams them from the source CSV.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from commitclf.config import resolve

KEY_COLS = ["full_repo_name", "unit_id", "file_path"]
META_COLS = KEY_COLS + [
    "org", "unit_type", "label_class", "status", "additions", "deletions", "is_test_file",
    "content_omitted", "n_pr_commits", "n_files_in_pr", "n_java_files_in_pr", "merged_at",
]
STATUSES = ["added", "modified", "removed", "renamed"]


def row_key(df: pd.DataFrame) -> pd.Series:
    return df["full_repo_name"] + "|" + df["unit_id"].astype(str) + "|" + df["file_path"].astype(str)


def unit_key(df: pd.DataFrame) -> pd.Series:
    return df["full_repo_name"] + "|" + df["unit_id"].astype(str)


def processed_path(cfg: dict, name: str):
    return resolve(cfg, cfg["data"]["processed_dir"]) / name


def read_csv_chunks(path, usecols: list[str], chunksize: int, skiprows: int = 0):
    return pd.read_csv(path, usecols=usecols, dtype={"unit_id": str, "file_path": str}, chunksize=chunksize,
                       keep_default_na=False, na_values=[""],
                       skiprows=range(1, skiprows + 1) if skiprows else None)


def read_source(cfg: dict, usecols: list[str], skiprows: int = 0):
    """Iterate over the source CSV (with code) in chunks."""
    return read_csv_chunks(resolve(cfg, cfg["data"]["source"]), usecols, cfg["data"].get("chunksize", 2000), skiprows)


def path_flags(paths: pd.Series) -> pd.DataFrame:
    p = paths.fillna("").str.replace("\\", "/", regex=False)
    name = p.str.rsplit("/", n=1).str[-1]
    low = p.str.lower()
    return pd.DataFrame({
        "path__src_main": low.str.contains("src/main/", regex=False),
        "path__src_test": low.str.contains("src/test/|/test/|/tests/", regex=True),
        "path__examples_docs": low.str.contains(r"(?:^|/)(?:examples?|samples?|demos?|docs?|benchmarks?)/",
                                                regex=True),
        "path__name_test": name.str.contains(r"Tests?\.java$|^Test|IT\.java$", regex=True),
        "path__name_exception": name.str.contains("Exception|Error", regex=True),
        "path__name_util": name.str.contains("Util|Helper", regex=True),
        "path__name_config": name.str.contains("Config|Configuration|Properties|Settings", regex=True),
        "path__name_impl": name.str.contains(r"Impl\.java$", regex=True),
        "path__package_info": name == "package-info.java",
    }).astype("float32").assign(path__depth=np.log1p(p.str.count("/")).astype("float32"))


def file_facts(rows: pd.DataFrame) -> pd.DataFrame:
    """Per-file features, all numeric and roughly unit-scale."""
    facts = pd.DataFrame({
        "file__log_additions": np.log1p(rows["additions"]),
        "file__log_deletions": np.log1p(rows["deletions"]),
        "file__is_test": rows["is_test_file"],
        "file__content_omitted": rows["content_omitted"],
        **{f"file__{s}": (rows["status"] == s).astype(float) for s in STATUSES},
    }, index=rows.index).astype("float32")
    flags = path_flags(rows["file_path"])
    flags.columns = [c.replace("path__", "file__path_") for c in flags.columns]
    return pd.concat([facts, flags.set_axis(rows.index)], axis=1)


def unit_features(rows: pd.DataFrame) -> pd.DataFrame:
    """Change-level facts, aggregated over the change's files."""
    r = rows.assign(churn=rows["additions"] + rows["deletions"])
    r["log_churn"] = np.log1p(r["churn"])
    for s in STATUSES:
        r[f"st_{s}"] = (r["status"] == s).astype(float)
    flags = path_flags(r["file_path"]).set_axis(r.index)
    r = pd.concat([r, flags], axis=1)

    g = r.groupby("unit_key", sort=False)
    out = pd.DataFrame({
        "change__log_additions": np.log1p(g["additions"].sum()),
        "change__log_deletions": np.log1p(g["deletions"].sum()),
        "change__log_java_files": np.log1p(g.size()),
        "change__log_files_in_pr": np.log1p(g["n_files_in_pr"].first()),
        "change__java_file_share": g["n_java_files_in_pr"].first() / g["n_files_in_pr"].first().clip(lower=1),
        "change__log_commits": np.log1p(g["n_pr_commits"].first()),
        "change__mean_log_churn": g["log_churn"].mean(),
        "change__max_log_churn": g["log_churn"].max(),
        "change__deletion_share": g["deletions"].sum() / g["churn"].sum().clip(lower=1),
        "change__test_file_share": g["is_test_file"].mean(),
        "change__content_omitted_share": g["content_omitted"].mean(),
        **{f"change__{s}_share": g[f"st_{s}"].mean() for s in STATUSES},
    })
    return out.join(g[list(flags.columns)].mean())


def load_meta(cfg: dict) -> pd.DataFrame:
    d = cfg["data"]
    path = resolve(cfg, d.get("meta") or d["source"])
    parts = list(tqdm(read_csv_chunks(path, META_COLS, 200_000), desc=f"Reading {path.name}", unit="chunk"))
    rows = pd.concat(parts, ignore_index=True)
    if d.get("meta") and d.get("restrict_to_source"):
        keys = pd.concat([row_key(c) for c in tqdm(read_source(cfg, KEY_COLS), desc="Reading source keys",
                                                     unit="chunk")])
        rows = rows[row_key(rows).isin(set(keys))].reset_index(drop=True)
    return rows


def run(cfg: dict) -> pd.DataFrame:
    rows = load_meta(cfg)

    labels = cfg["labels"]
    unknown = set(rows["label_class"]) - set(labels)
    if unknown:
        print(f"Dropping {rows['label_class'].isin(unknown).sum()} rows with labels {unknown}")
        rows = rows[rows["label_class"].isin(labels)].reset_index(drop=True)

    for col in ["additions", "deletions", "is_test_file", "content_omitted", "n_files_in_pr", "n_java_files_in_pr"]:
        rows[col] = pd.to_numeric(rows[col], errors="coerce").fillna(0)
    # Issue-closing commits are one commit; a PR whose count GitHub couldn't return gets 1.
    commits = pd.to_numeric(rows["n_pr_commits"], errors="coerce")
    rows["n_pr_commits"] = commits.where(rows["unit_type"] != "commit", 1).fillna(1)
    if (rows["n_pr_commits"] > 5000).any():
        print("WARNING: n_pr_commits has values > 5000; the dataset may still hold the uncorrected git counts "
              "(see the collector's DATASET_NOTES.md).")

    rows["row_key"] = row_key(rows)
    rows["unit_key"] = unit_key(rows)
    if rows["row_key"].duplicated().any():
        raise ValueError("Duplicate (repo, unit, file) rows in the source; rebuild the dataset.")

    g = rows.groupby("unit_key", sort=False)
    if (g["label_class"].nunique() > 1).any():
        raise ValueError("Some changes have rows with different labels.")
    units = pd.DataFrame({
        "full_repo_name": g["full_repo_name"].first(),
        "org": g["org"].first(),
        "label": g["label_class"].first(),
        "unit_type": g["unit_type"].first(),
        "merged_at": pd.to_datetime(g["merged_at"].first(), utc=True, format="ISO8601"),
        "n_commits": g["n_pr_commits"].first(),
        "n_rows": g.size(),
    }).join(unit_features(rows))

    out_dir = resolve(cfg, cfg["data"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pd.concat([rows[["row_key", "unit_key", "label_class"]], file_facts(rows)], axis=1)
    table.to_csv(processed_path(cfg, "rows.csv"), index=False)
    units.rename_axis("unit_key").to_csv(processed_path(cfg, "units.csv"))

    print(f"{len(rows)} file rows, {len(units)} changes, {units['full_repo_name'].nunique()} repos, "
          f"{units['org'].nunique()} orgs")
    print(units["label"].value_counts().to_string())
    return units.rename_axis("unit_key").reset_index()


def load_units(cfg: dict) -> pd.DataFrame:
    units = pd.read_csv(processed_path(cfg, "units.csv"))
    units["merged_at"] = pd.to_datetime(units["merged_at"], utc=True, format="ISO8601")
    return units


def load_rows(cfg: dict) -> pd.DataFrame:
    return pd.read_csv(processed_path(cfg, "rows.csv"))
