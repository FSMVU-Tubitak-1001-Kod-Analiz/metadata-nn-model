"""Our change classifier with a fine-tuned message encoder (end to end).

Same architecture as a `--meta change message` run: the message vector is appended to the
change-level metadata and goes through the metadata tower. The difference is that the message
encoder (BERT by default) is trained together with the classifier instead of being frozen.
The code side still uses the frozen CodeBERT file vectors from `embed`.

Settings come from `finetune` in the config (defaults below), the rest from `train`:
    model: bert-base-uncased   max_length: 128   pool: cls | mean
    encoder_lr: 2.0e-5         epochs: 30        patience: 6       fp16: true
    batch_size: (train.batch_size)                head_lr: (train.lr)
The message text is built like the `message` embedding input (embeddings.inputs.message:
columns + separator) and cached in <processed_dir>/messages.csv.

The best model of each fold is kept in memory only (a BERT checkpoint per fold is ~0.5 GB);
predictions and metrics are written like `train` does.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from commitclf.config import resolve
from commitclf.data.prepare import KEY_COLS, processed_path, read_source, unit_key
from commitclf.embeddings import build_texts
from commitclf.features import ChangeSet, FileTable, build_sets, collate
from commitclf.metrics import classification_metrics, plot_confusion_matrices
from commitclf.train import class_weights, make_model, summarize
from commitclf.utils import get_device, set_seed

DEFAULTS = {"model": "bert-base-uncased", "max_length": 128, "pool": "cls", "encoder_lr": 2.0e-5,
            "epochs": 30, "patience": 6, "fp16": True}


def settings(cfg: dict) -> dict:
    ft = {**DEFAULTS, **(cfg.get("finetune") or {})}
    ft.setdefault("batch_size", cfg["train"]["batch_size"])
    ft.setdefault("head_lr", cfg["train"]["lr"])
    return ft


def load_messages(cfg: dict) -> pd.Series:
    """unit_key -> message text (first file row of each change), cached next to units.csv."""
    path = processed_path(cfg, "messages.csv")
    if path.exists():
        return pd.read_csv(path, keep_default_na=False).set_index("unit_key")["text"]
    inp = cfg["embeddings"]["inputs"]["message"]
    cols, sep = inp["columns"], inp.get("separator", " | ")
    parts = []
    for chunk in tqdm(read_source(cfg, sorted(set(KEY_COLS) | set(cols))), desc="Reading messages", unit="chunk"):
        chunk = chunk.assign(unit_key=unit_key(chunk)).drop_duplicates("unit_key")
        parts.append(pd.DataFrame({"unit_key": chunk["unit_key"].to_numpy(),
                                   "text": build_texts(chunk, cols, sep, clean=False)}))
    msgs = pd.concat(parts).drop_duplicates("unit_key")
    msgs.to_csv(path, index=False)
    return msgs.set_index("unit_key")["text"]


class MessageModel(nn.Module):
    """ChangeClassifier whose metadata input is [change facts, encoder(message)]."""

    def __init__(self, base: nn.Module, encoder: nn.Module, pool: str, fp16: bool = False):
        super().__init__()
        self.base, self.encoder, self.pool, self.fp16 = base, encoder, pool, fp16

    def encode(self, ids: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        # Mixed precision only for the transformer; the classifier (BatchNorm, index_put) stays float32.
        with torch.autocast(device_type=ids.device.type, dtype=torch.float16, enabled=self.fp16):
            h = self.encoder(input_ids=ids, attention_mask=attn).last_hidden_state
        h = h.float()
        if self.pool == "cls":
            return h[:, 0]
        m = attn.unsqueeze(-1).float()
        return (h * m).sum(1) / m.sum(1).clamp(min=1)

    def forward(self, files, mask, meta, ids, attn):
        return self.base(files, mask, torch.cat((meta, self.encode(ids, attn)), dim=1))


def batches(table: FileTable, cs: ChangeSet, tokens: dict, batch_size: int, shuffle: bool, generator=None):
    order = torch.randperm(len(cs), generator=generator) if shuffle else torch.arange(len(cs))
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        if shuffle and len(idx) == 1:   # BatchNorm can't train on a single sample
            continue
        files, mask, meta, y = collate(table, cs, idx)
        yield files, mask, meta, tokens["input_ids"][idx], tokens["attention_mask"][idx], y


@torch.no_grad()
def predict(model, table, cs, tokens, device, batch_size) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, ys = [], []
    for *x, y in batches(table, cs, tokens, batch_size, shuffle=False):
        x = [t.to(device, non_blocking=True) for t in x]
        logits = model(*x)
        probs.append(torch.softmax(logits.float(), 1).cpu())
        ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(probs).numpy()


def train_fold(cfg: dict, spec: dict, fold: int, table: FileTable, messages: pd.Series, run_dir: Path) -> dict:
    from transformers import AutoModel, AutoTokenizer
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()

    t, ft, labels = cfg["train"], settings(cfg), cfg["labels"]
    set_seed(cfg["seed"] + fold)
    device = get_device(t.get("device", "cuda"))
    fp16 = bool(ft["fp16"]) and device.type == "cuda"
    sets = build_sets(cfg, spec, fold, table)

    tokenizer = AutoTokenizer.from_pretrained(ft["model"])
    tokens = {}
    for p, cs in sets.items():
        texts = messages.reindex(cs.units["unit_key"]).fillna("").astype(str).tolist()
        tokens[p] = tokenizer(texts, truncation=True, padding="max_length", max_length=ft["max_length"],
                              return_tensors="pt")
    encoder = AutoModel.from_pretrained(ft["model"])
    meta_dim = sets["train"].meta.shape[1] + encoder.config.hidden_size
    model = MessageModel(make_model(cfg, spec, table.dim, meta_dim), encoder, ft["pool"], fp16).to(device)
    print(f"Fold {fold}: " + ", ".join(f"{p}={len(s)}" for p, s in sets.items())
          + f" | file dim {table.dim}, metadata dim {meta_dim} (incl. {ft['model']} message)")

    weight = class_weights(sets["train"].y, len(labels)).to(device) if t.get("class_weights", True) else None
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": ft["encoder_lr"], "weight_decay": 0.01},
        {"params": model.base.parameters(), "lr": ft["head_lr"], "weight_decay": t.get("weight_decay", 0.0)},
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=t.get("lr_factor", 0.5), patience=t.get("lr_patience", 3))
    if hasattr(torch.amp, "GradScaler"):          # torch >= 2.3
        scaler = torch.amp.GradScaler("cuda", enabled=fp16)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=fp16)
    generator = torch.Generator().manual_seed(cfg["seed"] + fold)
    bs = ft["batch_size"]

    fold_dir = run_dir / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    best_f1, best_epoch, best_state, history = -1.0, 0, None, []
    bar = tqdm(range(1, ft["epochs"] + 1), desc=f"{run_dir.name} fold {fold}")
    for epoch in bar:
        model.train()
        total, seen = 0.0, 0
        for *x, y in batches(table, sets["train"], tokens["train"], bs, shuffle=True, generator=generator):
            x = [v.to(device, non_blocking=True) for v in x]
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(*x)
            loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total += loss.item() * len(y)
            seen += len(y)
        y_val, p_val = predict(model, table, sets["val"], tokens["val"], device, bs)
        val = classification_metrics(y_val, p_val.argmax(1), p_val, labels)
        lrs = [g["lr"] for g in optimizer.param_groups]
        scheduler.step(val["f1"])
        history.append({"epoch": epoch, "encoder_lr": lrs[0], "head_lr": lrs[1], "train_loss": total / seen,
                        **{f"val_{k}": v for k, v in val.items()}})
        tqdm.write(f"[{epoch:3d}] train loss={total / seen:.4f} | val acc={val['accuracy']:.4f} "
                   f"f1={val['f1']:.4f} lr={lrs[0]:.1e}/{lrs[1]:.1e}")
        if val["f1"] > best_f1:
            best_f1, best_epoch = val["f1"], epoch
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= ft["patience"]:
            tqdm.write(f"Early stop: no val F1 improvement for {ft['patience']} epochs")
            break
    bar.close()
    pd.DataFrame(history).to_csv(fold_dir / "history.csv", index=False)

    model.load_state_dict(best_state)
    y_true, proba = predict(model, table, sets["test"], tokens["test"], device, bs)
    test = {**classification_metrics(y_true, proba.argmax(1), proba, labels), "best_epoch": best_epoch}
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
    del model, encoder, best_state, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return test


def run(cfg: dict, folds: list[int], spec: dict, name: str | None = None) -> pd.DataFrame:
    if spec["model"] == "code":
        raise ValueError("The message joins the metadata side; use metadata, concat, gated or attention")
    spec = {**spec, "metadata": [g for g in spec["metadata"] if g != "message"]}
    ft = settings(cfg)
    default = f"{spec['model']}_{spec['split']}" + (f"_{spec['subset']}" if spec["subset"] else "") + "_msg-ft"
    run_dir = resolve(cfg, cfg["train"]["out_dir"]) / (name or f"finetune/{default}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "spec.json").write_text(json.dumps({**spec, "finetune": ft}, indent=2))
    print(f"Run {run_dir.name}: {spec} | finetune {ft}")

    messages = load_messages(cfg)
    table = FileTable(cfg, spec["code"], spec["embed_pool"], with_facts=True)
    rows = [{"fold": f, **train_fold(cfg, spec, f, table, messages, run_dir)} for f in folds]
    return summarize(rows, run_dir / "summary.csv")
