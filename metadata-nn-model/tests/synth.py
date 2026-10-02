from __future__ import annotations

"""Synthetic raw-data generator shared by tests (and handy for smoke runs)."""


import numpy as np
import pandas as pd

LICENSE = "/*\n * Licensed under the Apache License, Version 2.0\n */\n"


def make_raw_csv(path, n_repos: int = 6, per_label: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    raw_labels = {"bug": ["bug", "Bug", "kind/bug"], "enhancement": ["enhancement", "Enhancement", "type: enhancement"]}
    for r in range(n_repos):
        repo = f"org{r % 3}/repo{r}"
        for label, variants in raw_labels.items():
            for k in range(per_label):
                body = f"int x{k} = {int(rng.integers(0, 1000))};\nreturn x{k};"
                before = f"{LICENSE}package a.b;\nimport java.util.List;\n\nclass C{k} {{\n  void f() {{\n    {body}\n  }}\n}}"
                after = before.replace("return", "return 1 +")
                patch = f"@@ -1,3 +1,3 @@ void f() {{\n-    {body}\n+    int y = {k};\n     return 0;"
                rows.append({
                    "full_repo_name": repo,
                    "labels": variants[k % len(variants)],
                    "pr_title": f"title {r}-{k}",
                    "class_names": "C",
                    "commit_sha": f"sha{r}{label[:1]}{k}",
                    "commit_message": f"{label} change {k}",
                    "file_path": f"src/C{k}.java",
                    "file_name": f"C{k}.java",
                    "additions": int(rng.integers(0, 50)),
                    "deletions": int(rng.integers(0, 50)),
                    "content_before": before,
                    "content_after": after,
                    "patch": patch,
                    "created_at": "2020-01-01",
                    "closed_at": "2020-01-02",
                    "status": "closed",
                })
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df["index"] = range(len(df))
    df.to_csv(path, index=False)
    return df
