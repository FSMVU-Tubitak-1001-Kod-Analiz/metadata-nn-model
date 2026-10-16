"""Our change classifier with fine-tuned encoders (end to end).

Two encoders can be trained together with the classifier instead of being frozen:

  message (default on)  BERT on the commit message. Its vector is appended to the change-level
                        metadata, as in a `--meta change message` run.
  code (--code-ft)      CodeBERT on each file's code input(s) (`--code`, e.g. patch). The file
                        vector is [CodeBERT(input) for each input, per-file facts], as in `train`,
                        but computed by the trained CodeBERT instead of read from `embed`.

Without --code-ft the code side uses the frozen file vectors from `embed`.

Code fine-tuning is meant for small sets (it keeps every file's text in memory):
  - Each change keeps at most `max_files` files, the ones with the largest churn. Frozen runs
    use all files.
  - Changes are processed `micro_batch` at a time, with gradient accumulation up to `batch_size`.
  - Gradient checkpointing is on, to fit max_files x code_max_length tokens per change.

Settings come from `finetune` in the config (defaults below); the rest come from `train`:
    model: bert-base-uncased   max_length: 128     pool: cls | mean    encoder_lr: 2.0e-5
    code_model: microsoft/codebert-base           code_max_length: 512 code_lr: 2.0e-5
    max_files: 8               micro_batch: 8      grad_checkpointing: true
    epochs: 30                 patience: 6         fp16: true
    batch_size: (train.batch_size)                 head_lr: (train.lr)
CodeBERT pooling follows --embed-pool (mean | cls | pooler).

The message text is built like the `message` embedding input (embeddings.inputs.message:
columns + separator) and cached in <processed_dir>/messages.csv.

The best model of each fold is kept in memory only (a BERT checkpoint per fold is ~0.5 GB).
Predictions and metrics are written the same way `train` writes them.
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
from commitclf.data.prepare import KEY_COLS, load_rows, processed_path, read_source, row_key, unit_key
from commitclf.embeddings import DERIVED, build_texts
from commitclf.features import ChangeSet, FileTable, build_sets, collate
from commitclf.metrics import classification_metrics, plot_confusion_matrices
from commitclf.train import class_weights, make_model, summarize
from commitclf.utils import get_device, set_seed

DEFAULTS = {"model": "bert-base-uncased", "max_length": 128, "pool": "cls", "encoder_lr": 2.0e-5,
            "code_model": "microsoft/codebert-base", "code_max_length": 512, "code_lr": 2.0e-5,
            "max_files": 8, "micro_batch": 8, "grad_checkpointing": True,
            "epochs": 30, "patience": 6, "fp16": True}
MAX_CODE_ROWS = 200_000


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


def load_code_texts(cfg: dict, inputs: list[str]) -> dict[str, list[str]]:
    """input name -> one text per row of rows.csv, preprocessed as `embed` does."""
    rows = load_rows(cfg)
    if len(rows) > MAX_CODE_ROWS:
        raise ValueError(f"{len(rows)} file rows: code fine-tuning keeps all texts in memory and is meant "
                         f"for small sets (<= {MAX_CODE_ROWS} rows)")
    index = pd.Series(np.arange(len(rows)), index=rows["row_key"])
    e = cfg["embeddings"]
    specs = {name: e["inputs"][name] for name in inputs}
    src_cols = sorted({DERIVED.get(c, (c, None))[0] for s in specs.values() for c in s["columns"]} | set(KEY_COLS))
    out = {name: [""] * len(rows) for name in inputs}
    for chunk in tqdm(read_source(cfg, src_cols), desc="Reading code", unit="chunk"):
        chunk = chunk.reset_index(drop=True)
        idx = row_key(chunk).map(index)
        keep = idx.notna().to_numpy()
        if not keep.any():
            continue
        sel, pos = chunk[keep], idx[keep].astype(int).to_numpy()
        for name, s in specs.items():
            texts = build_texts(sel, s["columns"], s.get("separator", " | "), e.get("clean_code", True))
            for p, t in zip(pos, texts):
                out[name][p] = t
    return out


def file_churn(cfg: dict) -> np.ndarray:
    rows = load_rows(cfg)
    return (np.expm1(rows["file__log_additions"]) + np.expm1(rows["file__log_deletions"])).to_numpy()


def pool_hidden(out, attn: torch.Tensor, pool: str) -> torch.Tensor:
    if pool == "pooler":
        return out.pooler_output.float()
    h = out.last_hidden_state.float()
    if pool == "cls":
        return h[:, 0]
    m = attn.unsqueeze(-1).float()
    return (h * m).sum(1) / m.sum(1).clamp(min=1)


class FinetuneModel(nn.Module):
    """ChangeClassifier with optional trained encoders for the message and the files' code."""

    def __init__(self, base: nn.Module, msg_encoder: nn.Module | None, msg_pool: str,
                 code_encoder: nn.Module | None, code_pool: str, n_code_inputs: int, fp16: bool):
        super().__init__()
        self.base, self.msg_encoder, self.code_encoder = base, msg_encoder, code_encoder
        self.msg_pool, self.code_pool, self.n_code_inputs, self.fp16 = msg_pool, code_pool, n_code_inputs, fp16

    def _encode(self, encoder, ids, attn, pool):
        # Mixed precision only inside the transformers; the classifier (BatchNorm, index_put) stays float32.
        with torch.autocast(device_type=ids.device.type, dtype=torch.float16, enabled=self.fp16):
            out = encoder(input_ids=ids, attention_mask=attn)
        return pool_hidden(out, attn, pool)

    def forward(self, files, mask, meta, msg=None, code=None):
        if self.code_encoder is not None:
            # code: list (one per input) of (ids, attn) for the real files, in mask order
            vecs = [self._encode(self.code_encoder, ids, attn, self.code_pool) for ids, attn in code]
            real = torch.cat(vecs, dim=1)
            code_part = torch.zeros(*mask.shape, real.shape[1], device=real.device, dtype=torch.float32)
            code_part[mask] = real
            files = torch.cat((code_part, files.float()), dim=2)
        if self.msg_encoder is not None:
            meta = torch.cat((meta, self._encode(self.msg_encoder, *msg, self.msg_pool)), dim=1)
        return self.base(files, mask, meta)


class Batcher:
    """Batches of one part: file facts / frozen vectors, metadata, labels, plus encoder inputs."""

    def __init__(self, table: FileTable, cs: ChangeSet, msg_tokens: dict | None, code_texts: dict | None,
                 code_tokenizer, code_max_length: int, churn: np.ndarray | None, max_files: int):
        self.table, self.cs, self.msg_tokens = table, cs, msg_tokens
        self.code_texts, self.code_tokenizer, self.code_max_length = code_texts, code_tokenizer, code_max_length
        if code_texts is not None:   # keep the max_files files with the largest churn per change
            rows, offsets = [], [0]
            for i in range(len(cs)):
                r = cs.rows[int(cs.offsets[i]):int(cs.offsets[i + 1])].numpy()
                if len(r) > max_files:
                    r = r[np.argsort(-churn[r], kind="stable")[:max_files]]
                rows.append(r)
                offsets.append(offsets[-1] + len(r))
            self.cs = ChangeSet(cs.units, cs.y, cs.meta, torch.from_numpy(np.concatenate(rows).astype(np.int64)),
                                torch.tensor(offsets))

    def __len__(self) -> int:
        return len(self.cs)

    def batches(self, batch_size: int, shuffle: bool, generator=None):
        order = torch.randperm(len(self.cs), generator=generator) if shuffle else torch.arange(len(self.cs))
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            if shuffle and len(idx) == 1:   # BatchNorm can't train on a single sample
                continue
            files, mask, meta, y = collate(self.table, self.cs, idx)
            extra = {}
            if self.msg_tokens is not None:
                extra["msg"] = (self.msg_tokens["input_ids"][idx], self.msg_tokens["attention_mask"][idx])
            if self.code_texts is not None:
                starts, ends = self.cs.offsets[idx], self.cs.offsets[idx + 1]
                rows = torch.cat([self.cs.rows[a:b] for a, b in zip(starts.tolist(), ends.tolist())]).tolist()
                extra["code"] = []
                for texts in self.code_texts.values():
                    tok = self.code_tokenizer([texts[r] for r in rows], truncation=True, padding=True,
                                              max_length=self.code_max_length, return_tensors="pt")
                    extra["code"].append((tok["input_ids"], tok["attention_mask"]))
            yield files, mask, meta, y, extra


def to_device(extra: dict, device) -> dict:
    out = {}
    for k, v in extra.items():
        if k == "msg":
            out[k] = tuple(t.to(device, non_blocking=True) for t in v)
        else:
            out[k] = [tuple(t.to(device, non_blocking=True) for t in pair) for pair in v]
    return out


@torch.no_grad()
def predict(model, batcher: Batcher, device, batch_size) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, ys = [], []
    for files, mask, meta, y, extra in batcher.batches(batch_size, shuffle=False):
        logits = model(files.to(device), mask.to(device), meta.to(device), **to_device(extra, device))
        probs.append(torch.softmax(logits.float(), 1).cpu())
        ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(probs).numpy()


def train_fold(cfg: dict, spec: dict, fold: int, table: FileTable, messages: pd.Series | None,
               code_texts: dict | None, churn: np.ndarray | None, run_dir: Path) -> dict:
    from transformers import AutoModel, AutoTokenizer
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()

    t, ft, labels = cfg["train"], settings(cfg), cfg["labels"]
    set_seed(cfg["seed"] + fold)
    device = get_device(t.get("device", "cuda"))
    fp16 = bool(ft["fp16"]) and device.type == "cuda"
    sets = build_sets(cfg, {**spec, "model": spec["model"]}, fold, table)

    msg_encoder, msg_tokens = None, {p: None for p in sets}
    if messages is not None:
        tokenizer = AutoTokenizer.from_pretrained(ft["model"])
        for p, cs in sets.items():
            texts = messages.reindex(cs.units["unit_key"]).fillna("").astype(str).tolist()
            msg_tokens[p] = tokenizer(texts, truncation=True, padding="max_length", max_length=ft["max_length"],
                                      return_tensors="pt")
        msg_encoder = AutoModel.from_pretrained(ft["model"])

    code_encoder, code_tokenizer = None, None
    if code_texts is not None:
        code_tokenizer = AutoTokenizer.from_pretrained(ft["code_model"])
        code_encoder = AutoModel.from_pretrained(ft["code_model"])
        if ft["grad_checkpointing"]:
            code_encoder.gradient_checkpointing_enable()
    batchers = {p: Batcher(table, cs, msg_tokens[p], code_texts, code_tokenizer, ft["code_max_length"], churn,
                           ft["max_files"]) for p, cs in sets.items()}

    file_dim = table.dim + (len(code_texts) * code_encoder.config.hidden_size if code_encoder is not None else 0)
    meta_dim = sets["train"].meta.shape[1] + (msg_encoder.config.hidden_size if msg_encoder is not None else 0)
    model = FinetuneModel(make_model(cfg, spec, file_dim, meta_dim), msg_encoder, ft["pool"], code_encoder,
                          spec["embed_pool"], len(code_texts or {}), fp16).to(device)
    print(f"Fold {fold}: " + ", ".join(f"{p}={len(s)}" for p, s in sets.items())
          + f" | file dim {file_dim}, metadata dim {meta_dim}"
          + (f" | code: {ft['code_model']} on {list(code_texts)}, <= {ft['max_files']} files" if code_texts else "")
          + (f" | message: {ft['model']}" if messages is not None else ""))

    weight = class_weights(sets["train"].y, len(labels)).to(device) if t.get("class_weights", True) else None
    criterion = nn.CrossEntropyLoss(weight=weight)
    groups = [{"params": model.base.parameters(), "lr": ft["head_lr"], "weight_decay": t.get("weight_decay", 0.0)}]
    if msg_encoder is not None:
        groups.append({"params": model.msg_encoder.parameters(), "lr": ft["encoder_lr"], "weight_decay": 0.01})
    if code_encoder is not None:
        groups.append({"params": model.code_encoder.parameters(), "lr": ft["code_lr"], "weight_decay": 0.01})
    optimizer = torch.optim.AdamW(groups)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=t.get("lr_factor", 0.5), patience=t.get("lr_patience", 3))
    if hasattr(torch.amp, "GradScaler"):          # torch >= 2.3
        scaler = torch.amp.GradScaler("cuda", enabled=fp16)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=fp16)
    generator = torch.Generator().manual_seed(cfg["seed"] + fold)
    bs = ft["batch_size"]
    micro = min(ft["micro_batch"], bs) if code_encoder is not None else bs
    accum = max(1, round(bs / micro))

    fold_dir = run_dir / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    best_f1, best_epoch, best_state, history = -1.0, 0, None, []
    bar = tqdm(range(1, ft["epochs"] + 1), desc=f"{run_dir.name} fold {fold}")
    for epoch in bar:
        model.train()
        total, seen, step = 0.0, 0, 0
        optimizer.zero_grad()
        for files, mask, meta, y, extra in batchers["train"].batches(micro, shuffle=True, generator=generator):
            y = y.to(device)
            logits = model(files.to(device), mask.to(device), meta.to(device), **to_device(extra, device))
            loss = criterion(logits, y)
            scaler.scale(loss / accum).backward()
            step += 1
            if step % accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            total += loss.item() * len(y)
            seen += len(y)
        if step % accum:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        y_val, p_val = predict(model, batchers["val"], device, micro)
        val = classification_metrics(y_val, p_val.argmax(1), p_val, labels)
        lrs = [g["lr"] for g in optimizer.param_groups]
        scheduler.step(val["f1"])
        history.append({"epoch": epoch, **{f"lr{i}": lr for i, lr in enumerate(lrs)}, "train_loss": total / seen,
                        **{f"val_{k}": v for k, v in val.items()}})
        tqdm.write(f"[{epoch:3d}] train loss={total / seen:.4f} | val acc={val['accuracy']:.4f} "
                   f"f1={val['f1']:.4f} lr=" + "/".join(f"{lr:.1e}" for lr in lrs))
        if val["f1"] > best_f1:
            best_f1, best_epoch = val["f1"], epoch
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= ft["patience"]:
            tqdm.write(f"Early stop: no val F1 improvement for {ft['patience']} epochs")
            break
    bar.close()
    pd.DataFrame(history).to_csv(fold_dir / "history.csv", index=False)

    model.load_state_dict(best_state)
    y_true, proba = predict(model, batchers["test"], device, micro)
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
    del model, best_state, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return test


def run(cfg: dict, folds: list[int], spec: dict, name: str | None = None, code: bool = False,
        message: bool = True) -> pd.DataFrame:
    if not (code or message):
        raise ValueError("Nothing to fine-tune: use the message (default) and/or --code-ft")
    if message and spec["model"] == "code":
        raise ValueError("The message joins the metadata side; use metadata, concat, gated or attention, "
                         "or --no-message")
    if code and not spec["code"]:
        raise ValueError("--code-ft needs code inputs (--code patch ...)")
    spec = {**spec, "metadata": [g for g in spec["metadata"] if g != "message"]}
    ft = settings(cfg)
    tags = (["code-ft"] if code else []) + (["msg-ft"] if message else [])
    default = f"{spec['model']}_{spec['split']}" + (f"_{spec['subset']}" if spec["subset"] else "") + "_" + "_".join(tags)
    run_dir = resolve(cfg, cfg["train"]["out_dir"]) / (name or f"finetune/{default}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "spec.json").write_text(json.dumps({**spec, "finetune": {**ft, "code": code, "message": message}},
                                                  indent=2))
    print(f"Run {run_dir.name}: {spec} | finetune code={code} message={message} {ft}")

    messages = load_messages(cfg) if message else None
    code_texts, churn = (load_code_texts(cfg, spec["code"]), file_churn(cfg)) if code else (None, None)
    # Frozen file vectors only when the code isn't fine-tuned; per-file facts whenever metadata is used.
    table = FileTable(cfg, [] if code else spec["code"], spec["embed_pool"], with_facts=spec["model"] != "code")
    rows = [{"fold": f, **train_fold(cfg, spec, f, table, messages, code_texts, churn, run_dir)} for f in folds]
    return summarize(rows, run_dir / "summary.csv")
