"""Run the experiment grid from configs/grid.yaml (see the comments there)."""
from __future__ import annotations

import itertools
from pathlib import Path

import pandas as pd
import yaml

from commitclf.config import resolve
from commitclf.train import SPEC_KEYS, make_spec

LIST_FIELDS = ("code", "metadata")       # fields whose single value is itself a list
FUSION_MODELS = ("concat", "gated", "attention")
SHORT = {"content_before": "cb", "patch_before": "pb", "patch_after": "pa", "patch": "patch", "patch_delta": "pdelta",
         "change": "change", "repo": "repo", "message": "msg", "path": "path"}


def expand(entry: dict) -> list[dict]:
    """One grid entry -> list of concrete settings (cartesian product over list values)."""
    keys, options = [], []
    for k, v in entry.items():
        if k in ("block", "name"):
            continue
        if k in LIST_FIELDS:
            v = v if v and isinstance(v[0], list) else [v]
        elif not isinstance(v, list):
            v = [v]
        keys.append(k)
        options.append(v)
    return [{"block": entry["block"], "name": entry.get("name"), **dict(zip(keys, combo))}
            for combo in itertools.product(*options)]


def run_name(setting: dict) -> str:
    if setting.get("name"):
        return setting["name"]
    if "features" in setting:   # baselines
        return "_".join(p for p in (setting["features"], setting.get("split"), setting.get("subset")) if p)
    parts = [setting.get("model", "default"), setting.get("split", "")]
    tags = {"code": "in", "metadata": "meta", "embed_pool": "emb", "file_pool": "pool", "subset": ""}
    for k, tag in tags.items():
        if k in setting:
            v = setting[k]
            v = "+".join(SHORT.get(x, x) for x in v) if isinstance(v, list) else v
            parts.append(f"{tag}-{v}" if tag else v)
    return "_".join(p for p in parts if p)


def best_fusion(cfg: dict) -> str:
    out_dir = resolve(cfg, cfg["train"]["out_dir"]) / "main"
    scores = {}
    for m in FUSION_MODELS:
        f = out_dir / f"{m}_repo" / "summary.csv"
        if f.exists():
            df = pd.read_csv(f)
            mean = df[df["fold"] == "mean"]
            scores[m] = float((mean if len(mean) else df)["f1"].iloc[0])
    if len(scores) < len(FUSION_MODELS):
        raise RuntimeError("best_fusion needs the `main` block's concat/gated/attention runs on the repo split; "
                           "run `grid --blocks main` first.")
    best = max(scores, key=scores.get)
    print(f"best_fusion = {best} (repo-split macro-F1: " + ", ".join(f"{k} {v:.4f}" for k, v in scores.items()) + ")")
    return best


def is_done(run_dir: Path, n_folds: int) -> bool:
    if not (run_dir / "summary.csv").exists():
        return False
    metrics = run_dir / "metrics.csv"   # baselines: one row per (fold, model)
    if metrics.exists():
        return pd.read_csv(metrics)["fold"].nunique() >= n_folds
    df = pd.read_csv(run_dir / "summary.csv")
    return pd.to_numeric(df["fold"], errors="coerce").notna().sum() >= n_folds


def run(cfg: dict, grid_path: str | Path, blocks: list[str] | None = None, dry_run: bool = False,
        force: bool = False, folds_override: list[str] | None = None) -> None:
    from commitclf import baselines, train
    from commitclf.cli import parse_folds

    grid = yaml.safe_load(open(resolve(cfg, grid_path), encoding="utf-8"))
    jobs = [("train", s) for e in grid.get("runs", []) for s in expand(e)]
    jobs += [("baselines", s) for e in grid.get("baselines", []) for s in expand(e)]
    if blocks:
        jobs = [j for j in jobs if j[1]["block"] in blocks]
    print(f"{len(jobs)} runs" + (f" in blocks {blocks}" if blocks else ""))

    fusion = None
    for i, (kind, s) in enumerate(jobs, 1):
        s = dict(s)
        if s.get("model") == "best_fusion":
            if dry_run:
                s["model"] = "<best_fusion>"
            else:
                fusion = fusion or best_fusion(cfg)
                s["model"] = fusion
        overrides = {k: v for k, v in s.items() if k in SPEC_KEYS}
        spec = make_spec(cfg, **{k: v for k, v in overrides.items() if v != "<best_fusion>"})
        if dry_run:
            spec["model"] = s.get("model", spec["model"])
        name = run_name(s) if kind == "baselines" else f"{s['block']}/{run_name(s)}"
        base = cfg["train"]["out_dir"] if kind == "train" else cfg["baselines"]["out_dir"]
        run_dir = resolve(cfg, base) / name
        folds = parse_folds(folds_override or [str(grid.get("folds", "all"))], cfg, spec["split"])

        status = "done" if is_done(run_dir, len(folds)) and not force else "todo"
        print(f"[{i}/{len(jobs)}] {kind:9} {name:55} {status}"
              + (f"  {spec}" if dry_run else ""))
        if dry_run or status == "done":
            continue
        if kind == "train":
            train.run(cfg, folds, spec, name=name)
        else:
            baselines.run(cfg, folds, spec, features=s.get("features"), name=name)
