"""External datasets -> the collector's per-file CSV format, so `prepare` / `embed` / `train` run unchanged.

CoLaRe (Zeng et al. 2024) ships the Levin & Yehudai / Ghadhab Java commit set as
sotas/COLARE-main/dataset/1793/dataset.csv: one row per commit with user, repo, commit,
labels (Corrective / Adaptive / Perfective), msgs (commit message), diffs (`git show`-style
unified diff of the whole commit) and feature (CoLaRe's 29 counts, not used here).

`convert_colare` writes two files to `out_dir`:
  dataset.csv  one row per (commit, changed file with a text diff), collector columns
  parts.csv    unit_key, part: CoLaRe's own 5-fold test assignment, reproduced exactly
               (StratifiedKFold(5, shuffle=True, random_state=42) over their file's row order),
               used by the `given` split scheme

Label mapping (Swanson): Corrective -> bug, Adaptive -> enhancement, Perfective -> clean.
Not available in their data: content_before (only `patch` and its two sides can be embedded),
PR title, dates (merged_at is empty). Every commit is one commit (n_pr_commits = 1).
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from commitclf.data.prepare import path_flags

LABEL_MAP = {"Corrective": "bug", "Adaptive": "enhancement", "Perfective": "clean"}
_FILE_START = re.compile(r"^diff --git ", flags=re.MULTILINE)
_HEADER = re.compile(r"^diff --git a/(.*) b/(.*)$")


def split_files(diff: str) -> list[str]:
    """Whole-commit diff -> one block per file, each starting with 'diff --git'."""
    starts = [m.start() for m in _FILE_START.finditer(diff)]
    return [diff[s:e] for s, e in zip(starts, starts[1:] + [len(diff)])]


def parse_file(block: str) -> dict:
    """One file block -> path, status, patch (from the first @@ on, '' for binary / mode-only changes)."""
    lines = block.splitlines()
    first_hunk = next((i for i, l in enumerate(lines) if l.startswith("@@")), len(lines))
    header = lines[:first_hunk]
    path = None
    for l in header:                       # +++ b/<path>, or --- a/<path> for a deleted file
        if l.startswith("+++ ") and l[4:] != "/dev/null":
            path = l[4:].removeprefix("b/")
        elif l.startswith("rename to "):
            path = path or l[len("rename to "):]
    if path is None:
        for l in header:
            if l.startswith("--- ") and l[4:] != "/dev/null":
                path = l[4:].removeprefix("a/")
    if path is None:
        m = _HEADER.match(lines[0])
        path = m.group(2) if m else lines[0].split()[-1].removeprefix("b/")
    if any(l.startswith("new file mode") for l in header):
        status = "added"
    elif any(l.startswith("deleted file mode") for l in header):
        status = "removed"
    elif any(l.startswith("rename from") for l in header):
        status = "renamed"
    else:
        status = "modified"
    patch = "\n".join(lines[first_hunk:])
    hunk = lines[first_hunk:]
    return {"file_path": path.strip(), "status": status, "patch": patch,
            "additions": sum(l.startswith("+") for l in hunk),
            "deletions": sum(l.startswith("-") for l in hunk)}


def colare_parts(raw: pd.DataFrame) -> pd.Series:
    """CoLaRe's cross_validation_launch.py fold assignment (test part 0..4 per row of their file)."""
    label_map: dict[str, int] = {}
    y = [label_map.setdefault(v, len(label_map)) for v in raw["labels"]]
    part = pd.Series(-1, index=raw.index)
    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    for i, (_, test_idx) in enumerate(skf.split([[i] for i in range(len(raw))], y)):
        part.iloc[test_idx] = i
    return part


def convert_colare(src: str | Path, out_dir: str | Path, files: str = "all") -> None:
    """files='all': every file with a text diff (the data has Vala, Kotlin, XML ... besides Java);
    files='java': only .java files, as in our own dataset (commits without one are dropped)."""
    if files not in ("all", "java"):
        raise ValueError("files must be 'all' or 'java'")
    raw = pd.read_csv(src, keep_default_na=False)
    raw["part"] = colare_parts(raw)
    raw["full_repo_name"] = raw["user"] + "/" + raw["repo"]
    raw["unit_key"] = raw["full_repo_name"] + "|" + raw["commit"]

    dup = raw["unit_key"].duplicated(keep="first")
    conflicts = raw[raw["unit_key"].isin(raw.loc[dup, "unit_key"])].groupby("unit_key")["labels"].nunique()
    print(f"{len(raw)} commits in {src}; {int(dup.sum())} duplicate rows dropped "
          f"({int((conflicts > 1).sum())} with conflicting labels, first kept)")
    raw = raw[~dup].reset_index(drop=True)

    rows, no_files = [], 0
    for r in raw.itertuples(index=False):
        blocks = [parse_file(b) for b in split_files(r.diffs)]
        n_java = sum(b["file_path"].endswith(".java") for b in blocks)
        keep = [b for b in blocks if b["patch"] and (files == "all" or b["file_path"].endswith(".java"))]
        seen = set()
        keep = [b for b in keep if not (b["file_path"] in seen or seen.add(b["file_path"]))]
        if not keep:
            no_files += 1
            continue
        for b in keep:
            rows.append({
                "full_repo_name": r.full_repo_name, "unit_id": r.commit, "org": r.user,
                "unit_type": "commit", "label_class": LABEL_MAP[r.labels], **b,
                "content_omitted": 0, "content_before": "", "n_pr_commits": 1,
                "n_files_in_pr": len(blocks), "n_java_files_in_pr": n_java, "merged_at": "",
                "commit_message": r.msgs, "pr_title": "", "source_label": r.labels,
            })
    df = pd.DataFrame(rows)
    flags = path_flags(df["file_path"])
    df["is_test_file"] = ((flags["path__src_test"] > 0) | (flags["path__name_test"] > 0)).astype(int)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "dataset.csv", index=False)
    units = df.drop_duplicates(["full_repo_name", "unit_id"])
    kept = raw[raw["unit_key"].isin(units["full_repo_name"] + "|" + units["unit_id"])]
    kept[["unit_key", "part"]].to_csv(out_dir / "parts.csv", index=False)

    print(f"files={files}: {no_files} commits without a usable file dropped; "
          f"{len(units)} commits, {len(df)} file rows ({df['file_path'].str.endswith('.java').mean():.0%} .java), "
          f"{units['full_repo_name'].nunique()} repos, {units['org'].nunique()} orgs -> {out_dir}")
    print(units["label_class"].value_counts().to_string())
    print("commits per CoLaRe test part:", kept["part"].value_counts().sort_index().to_dict())
