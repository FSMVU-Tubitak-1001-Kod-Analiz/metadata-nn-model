"""Train / evaluate the change classifier on one or more folds."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from commitclf.config import resolve
from commitclf.features import FileTable, batches, build_sets
from commitclf.metrics import classification_metrics, plot_confusion_matrices
from commitclf.models import ChangeClassifier
from commitclf.utils import get_device, set_seed

SPEC_KEYS = ("model", "split", "subset", "code", "embed_pool", "file_pool", "metadata", "sim", "facts")


def make_spec(cfg: dict, **overrides) -> dict:
    """Experiment settings: config `experiment` defaults, overridden by non-None arguments."""
    spec = {k: cfg["experiment"].get(k) for k in SPEC_KEYS}
    spec.update({k: v for k, v in overrides.items() if v is not None and k in SPEC_KEYS})
    if spec["model"] == "metadata":
        spec["code"] = []
    if spec["model"] == "code":
        spec["metadata"] = []
    if spec["facts"] is None:   # per-file facts on the file vectors whenever the model uses metadata
        spec["facts"] = True
    return spec


def default_name(spec: dict) -> str:
    name = f"{spec['model']}_{spec['split']}"
    name += f"_{spec['subset']}" if spec["subset"] else ""
    return name + (f"_sim-{spec['sim']}" if spec.get("sim") else "") + ("_nofacts" if spec.get("facts") is False else "")


def make_model(cfg: dict, spec: dict, file_dim: int, meta_dim: int, window_dims: list[int] | None = None) -> ChangeClassifier:
    t = cfg["train"]
    return ChangeClassifier(
        spec["model"], file_dim, meta_dim, len(cfg["labels"]),
        file_hidden=t["file_hidden"], metadata_hidden=t["metadata_hidden"], head_hidden=t["head_hidden"],
        fusion_dim=t["fusion_dim"], file_pool=spec["file_pool"], dropout=t["dropout"], window_dims=window_dims,
    )


def class_weights(y: torch.Tensor, n_classes: int) -> torch.Tensor:
    counts = torch.bincount(y, minlength=n_classes).float()
    return counts.sum() / (n_classes * counts.clamp(min=1))


def to(device, *tensors):
    return [t.to(device, non_blocking=True) for t in tensors]


def unpack(batch, device):
    """(files, mask, meta, y, windows) on the device; windows is None without windowed inputs."""
    files, mask, meta, y = to(device, *batch[:4])
    windows = [(w.to(device, non_blocking=True), m.to(device, non_blocking=True)) for w, m in batch[4]] \
        if len(batch) > 4 else None
    return files, mask, meta, y, windows


def train_epoch(model, table, cs, criterion, optimizer, device, batch_size, generator) -> dict:
    model.train()
    total, correct, seen, fusion_w = 0.0, 0, 0, []
    for batch in batches(table, cs, batch_size, shuffle=True, generator=generator):
        files, mask, meta, y, windows = unpack(batch, device)
        optimizer.zero_grad()
        logits = model(files, mask, meta, windows)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        seen += len(y)
        if model.fusion is not None:
            fusion_w.append(model.fusion.last_weight)
    out = {"loss": total / seen, "accuracy": correct / seen}
    if fusion_w:
        out["code_weight"] = float(np.mean(fusion_w))
    return out


@torch.no_grad()
def predict(model, table, cs, criterion, device, batch_size) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total, probs, ys = 0.0, [], []
    for batch in batches(table, cs, batch_size, shuffle=False):
        files, mask, meta, y, windows = unpack(batch, device)
        logits = model(files, mask, meta, windows)
        total += criterion(logits, y).item() * len(y)
        probs.append(torch.softmax(logits.float(), 1).cpu())
        ys.append(y.cpu())
    return total / len(cs), torch.cat(ys).numpy(), torch.cat(probs).numpy()


def evaluate(model, table, cs, criterion, device, batch_size, labels) -> tuple[dict, np.ndarray, np.ndarray]:
    loss, y_true, proba = predict(model, table, cs, criterion, device, batch_size)
    return {"loss": loss, **classification_metrics(y_true, proba.argmax(1), proba, labels)}, y_true, proba


def train_fold(cfg: dict, spec: dict, fold: int, table: FileTable, run_dir: Path) -> dict:
    t, labels = cfg["train"], cfg["labels"]
    set_seed(cfg["seed"] + fold)
    device = get_device(t.get("device", "cuda"))
    sets = build_sets(cfg, spec, fold, table)
    print(f"Fold {fold}: " + ", ".join(f"{p}={len(s)}" for p, s in sets.items())
          + f" | file dim {table.dim}" + (f" + windows {table.window_dims}" if table.window_dims else "")
          + f", metadata dim {sets['train'].meta.shape[1]}")

    model = make_model(cfg, spec, table.dim, sets["train"].meta.shape[1], table.window_dims).to(device)
    weight = class_weights(sets["train"].y, len(labels)).to(device) if t.get("class_weights", True) else None
    criterion = nn.CrossEntropyLoss(weight=weight)
    plain = nn.CrossEntropyLoss()   # unweighted loss for reporting
    optimizer = torch.optim.Adam(model.parameters(), lr=t["lr"], weight_decay=t.get("weight_decay", 0.0))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=t.get("lr_factor", 0.5), patience=t.get("lr_patience", 3))
    generator = torch.Generator().manual_seed(cfg["seed"] + fold)

    fold_dir = run_dir / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = fold_dir / "model.pt"
    bs = t["batch_size"]

    best_f1, best_epoch, history = -1.0, 0, []
    bar = tqdm(range(1, t["epochs"] + 1), desc=f"{run_dir.name} fold {fold}")
    for epoch in bar:
        tr = train_epoch(model, table, sets["train"], criterion, optimizer, device, bs, generator)
        val, *_ = evaluate(model, table, sets["val"], plain, device, bs, labels)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val["f1"])
        history.append({"epoch": epoch, "lr": lr, **{f"train_{k}": v for k, v in tr.items()},
                        **{f"val_{k}": v for k, v in val.items()}})
        w = f" code_w={tr['code_weight']:.2f}" if "code_weight" in tr else ""
        tqdm.write(f"[{epoch:3d}] train loss={tr['loss']:.4f} acc={tr['accuracy']:.4f} | "
                   f"val loss={val['loss']:.4f} acc={val['accuracy']:.4f} f1={val['f1']:.4f}{w} lr={lr:.1e}")
        if val["f1"] > best_f1:
            best_f1, best_epoch = val["f1"], epoch
            torch.save({"state_dict": model.state_dict(), "spec": spec, "epoch": epoch, "file_dim": table.dim,
                        "meta_dim": sets["train"].meta.shape[1], "labels": labels, "train_cfg": t}, ckpt_path)
        elif epoch - best_epoch >= t.get("patience", 10):
            tqdm.write(f"Early stop: no val F1 improvement for {t['patience']} epochs")
            break
    bar.close()
    pd.DataFrame(history).to_csv(fold_dir / "history.csv", index=False)

    model.load_state_dict(torch.load(ckpt_path, map_location=device)["state_dict"])
    test, y_true, proba = evaluate(model, table, sets["test"], plain, device, bs, labels)
    test["best_epoch"] = best_epoch
    print(f"Fold {fold} test: acc={test['accuracy']:.4f} f1={test['f1']:.4f} auc={test['auc']:.4f} "
          f"(best epoch {best_epoch})")

    (fold_dir / "test_metrics.json").write_text(json.dumps(test, indent=2))
    pred = sets["test"].units[["unit_key", "full_repo_name", "label"]].copy()
    pred["pred"] = [labels[i] for i in proba.argmax(1)]
    for i, c in enumerate(labels):
        pred[f"p_{c}"] = proba[:, i]
    pred.to_csv(fold_dir / "test_predictions.csv", index=False)
    plot_confusion_matrices({f"{run_dir.name} fold {fold}": (y_true, proba.argmax(1))}, labels,
                            fold_dir / "confusion_matrix.png")
    return test


def summarize(rows: list[dict], out_csv: Path) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if len(rows) > 1:
        stats = df.drop(columns="fold").agg(["mean", "std"]).reset_index(names="fold")
        df = pd.concat([df, stats], ignore_index=True)
    df.to_csv(out_csv, index=False)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return df


def run(cfg: dict, folds: list[int], spec: dict, name: str | None = None) -> pd.DataFrame:
    run_dir = resolve(cfg, cfg["train"]["out_dir"]) / (name or default_name(spec))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "spec.json").write_text(json.dumps(spec, indent=2))
    print(f"Run {run_dir.name}: {spec}")

    table = FileTable(cfg, spec["code"], spec["embed_pool"], with_facts=spec["model"] != "code" and spec.get("facts", True),
                      sim=spec.get("sim"), sim_dims=cfg["experiment"].get("sim_dims", 256), seed=cfg["seed"])
    rows = [{"fold": f, **train_fold(cfg, spec, f, table, run_dir)} for f in folds]
    out = summarize(rows, run_dir / "summary.csv")
    from commitclf.grid import rel
    print(f"Results saved in {rel(cfg, run_dir)}")
    return out
