"""Windowed embeddings: every window of a file's text gets its own vector, so long files are not cut at 512 tokens.

A file's patch is split at hunk headers ('@@'). Consecutive hunks are packed into windows of at
most `max_length` tokens; a hunk longer than that is cut into several windows. At most
`max_windows` windows per file are kept (the first ones). An empty text gives one empty window.

With `side: before` / `side: after`, each hunk is first turned into that side (context + removed /
context + added lines, cleaned like `embed` cleans patch_before / patch_after), and the packing is
decided on the longer side of each hunk, so the before and after windows cover the same hunks.

Config (`embeddings.windows`), e.g.
    windows:
      patch_win: {model: codebert, column: patch, max_length: 512, max_windows: 8}
      patch_before_win: {model: codebert, column: patch, side: before, max_length: 512, max_windows: 8}
      patch_after_win: {model: codebert, column: patch, side: after, max_length: 512, max_windows: 8}

`python -m commitclf embed-windows patch_win` writes to the embeddings dir:
    patch_win.cls.npy / patch_win.mean.npy   [n_windows, hidden] float16, windows of row 0 first, then row 1 ...
    patch_win.offsets.npy                    [n_rows + 1] int64: row i owns windows offsets[i]:offsets[i+1]
    patch_win.progress.json                  window statistics; "complete": true when done
It resumes from per-chunk part files (patch_win.parts/) if interrupted.

Use as a code input:
    --code patch_win       windows pooled by learned attention inside the model (per file)
    --code patch_winmean   plain mean of the windows, precomputed (no model change)
"""
from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from commitclf.data.prepare import KEY_COLS, load_rows, read_source, row_key
from commitclf.embeddings import emb_dir
from commitclf.utils import get_device


def window_specs(cfg: dict) -> dict:
    return cfg["embeddings"].get("windows") or {}


def split_hunks(text: str) -> list[str]:
    """A unified diff -> one text per hunk, each starting with its '@@' line."""
    hunks, cur = [], []
    for line in str(text).splitlines():
        if line.startswith("@@") and cur:
            hunks.append("\n".join(cur))
            cur = []
        cur.append(line)
    if cur:
        hunks.append("\n".join(cur))
    return hunks


def _plan(lengths: list[int], body: int) -> list[list[tuple[int, int, int]]]:
    """Group hunks into windows: each window is a list of (hunk, part, n_parts).
    Consecutive hunks are packed while they fit (+1 token for the newline between them);
    a hunk longer than `body` gets n_parts = ceil(length / body) windows of its own."""
    windows, cur, used = [], [], 0
    for h, n in enumerate(lengths):
        if n > body:
            if cur:
                windows.append(cur)
                cur, used = [], 0
            parts = -(-n // body)
            windows += [[(h, p, parts)] for p in range(parts)]
            continue
        need = n + (1 if cur else 0)
        if cur and used + need > body:
            windows.append(cur)
            cur, used, need = [], 0, n
        cur.append((h, 0, 1))
        used += need
    if cur or not windows:
        windows.append(cur)
    return windows


def _build(plan: list[list[tuple[int, int, int]]], ids: list[list[int]], newline: list[int]) -> list[list[int]]:
    out = []
    for window in plan:
        w = []
        for h, part, parts in window:
            size = -(-len(ids[h]) // parts) if parts > 1 else len(ids[h])
            piece = ids[h][part * size:(part + 1) * size]
            w += (newline if w else []) + piece
        out.append(w)
    return out


def make_windows(text: str, tokenizer, max_length: int, max_windows: int, side: str | None = None,
                 clean: bool = False) -> tuple[list[list[int]], int]:
    """Token-id windows (without special tokens) of one patch, and the number before capping.

    side=None: windows of the patch text itself. side='before' / 'after': windows of that side of
    each hunk (context + removed / context + added lines, as `patch_before` / `patch_after`), with
    the hunk grouping decided on the longer of the two sides, so window k of the before side and
    window k of the after side always cover the same hunks."""
    from commitclf.data.java import clean_java, patch_after, patch_before
    body = max_length - 2
    hunks = split_hunks(text)
    if not hunks:
        return [[]], 1
    newline = tokenizer("\n", add_special_tokens=False)["input_ids"]
    if side is None:
        ids = tokenizer(hunks, add_special_tokens=False)["input_ids"]
        lengths = [len(i) for i in ids]
    else:
        prep = (lambda t: clean_java(t)) if clean else (lambda t: t)
        sides = {"before": [prep(patch_before(h)) for h in hunks], "after": [prep(patch_after(h)) for h in hunks]}
        if side not in sides:
            raise ValueError(f"side must be before or after, got '{side}'")
        ids_by_side = {k: tokenizer(v, add_special_tokens=False)["input_ids"] for k, v in sides.items()}
        lengths = [max(len(a), len(b)) for a, b in zip(ids_by_side["before"], ids_by_side["after"])]
        ids = ids_by_side[side]
    plan = _plan(lengths, body)
    return _build(plan, ids, newline)[:max_windows], len(plan)


class WindowEncoder:
    def __init__(self, model_name: str, batch_size: int, device, fp16: bool):
        from transformers import AutoModel, AutoTokenizer
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_error()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.batch_size, self.device = batch_size, device
        self.fp16 = fp16 and device.type == "cuda"

    @torch.no_grad()
    def __call__(self, windows: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
        """(cls, mean) float16 for each window (token ids without special tokens)."""
        tok = self.tokenizer
        hidden = self.model.config.hidden_size
        cls_out = np.zeros((len(windows), hidden), dtype=np.float16)
        mean_out = np.zeros((len(windows), hidden), dtype=np.float16)
        order = np.argsort([len(w) for w in windows])              # similar lengths per batch
        for s in range(0, len(order), self.batch_size):
            idx = order[s:s + self.batch_size]
            seqs = [[tok.cls_token_id] + windows[i] + [tok.sep_token_id] for i in idx]
            width = max(len(q) for q in seqs)
            ids = torch.full((len(seqs), width), tok.pad_token_id, dtype=torch.long)
            att = torch.zeros((len(seqs), width), dtype=torch.long)
            for j, q in enumerate(seqs):
                ids[j, :len(q)] = torch.tensor(q)
                att[j, :len(q)] = 1
            ids, att = ids.to(self.device), att.to(self.device)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.fp16):
                h = self.model(input_ids=ids, attention_mask=att).last_hidden_state.float()
            m = att.unsqueeze(-1).float()
            cls_out[idx] = h[:, 0].cpu().numpy().astype(np.float16)
            mean_out[idx] = ((h * m).sum(1) / m.sum(1).clamp(min=1)).cpu().numpy().astype(np.float16)
        return cls_out, mean_out


def _paths(cfg: dict, name: str) -> dict:
    d = emb_dir(cfg)
    return {"cls": d / f"{name}.cls.npy", "mean": d / f"{name}.mean.npy", "offsets": d / f"{name}.offsets.npy",
            "progress": d / f"{name}.progress.json", "parts": d / f"{name}.parts"}


def run(cfg: dict, name: str, overwrite: bool = False) -> None:
    specs = window_specs(cfg)
    if name not in specs:
        raise KeyError(f"'{name}' is not in embeddings.windows ({list(specs)})")
    spec = specs[name]
    model_spec = cfg["embeddings"]["models"][spec["model"]]
    paths = _paths(cfg, name)
    if overwrite and paths["parts"].exists():
        shutil.rmtree(paths["parts"])
    paths["parts"].mkdir(parents=True, exist_ok=True)

    rows = load_rows(cfg)
    index = pd.Series(np.arange(len(rows)), index=rows["row_key"])
    device = get_device(cfg["train"].get("device", "cuda"))
    enc = WindowEncoder(model_spec["name"], cfg["embeddings"].get("batch_size", 32), device,
                        cfg["embeddings"].get("fp16", True))
    max_len = min(spec.get("max_length", 512), model_spec.get("max_length", 512))
    max_win = spec.get("max_windows", 8)
    side = spec.get("side")
    clean = bool(side) and cfg["embeddings"].get("clean_code", True)   # as `embed` does for patch_before/after
    print(f"Windows '{name}' over column '{spec['column']}'" + (f", {side} side" if side else "")
          + f": {max_len} tokens, at most {max_win} per file")

    for i, chunk in enumerate(tqdm(read_source(cfg, sorted(set(KEY_COLS) | {spec["column"]})), desc="chunks")):
        part = paths["parts"] / f"part_{i:05d}.npz"
        if part.exists():
            continue
        chunk = chunk.reset_index(drop=True)
        idx = row_key(chunk).map(index)
        keep = idx.notna().to_numpy()
        texts = chunk.loc[keep, spec["column"]].fillna("").astype(str).tolist()
        made = [make_windows(t, enc.tokenizer, max_len, max_win, side=side, clean=clean) for t in texts]
        flat = [w for ws, _ in made for w in ws]
        empty = np.zeros((0, enc.model.config.hidden_size), np.float16)
        cls, mean = enc(flat) if flat else (empty, empty)
        np.savez(part, rows=idx[keep].astype(np.int64).to_numpy(), counts=np.array([len(ws) for ws, _ in made]),
                 full=np.array([n for _, n in made]), cls=cls, mean=mean)

    # Assemble in row order.
    parts = sorted(paths["parts"].glob("part_*.npz"))
    data = [np.load(p) for p in parts]
    row_ids = np.concatenate([d["rows"] for d in data])
    counts = np.concatenate([d["counts"] for d in data])
    full = np.concatenate([d["full"] for d in data])
    if len(np.unique(row_ids)) != len(rows):
        raise ValueError(f"{len(np.unique(row_ids))} of {len(rows)} rows found in the source")
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    order = np.argsort(row_ids, kind="stable")
    offsets = np.concatenate([[0], np.cumsum(counts[order])]).astype(np.int64)
    for pool in ("cls", "mean"):
        src = np.concatenate([d[pool] for d in data])
        out = np.lib.format.open_memmap(paths[pool], mode="w+", dtype=np.float16, shape=(int(offsets[-1]), src.shape[1]))
        for k, r in enumerate(order):
            out[offsets[k]:offsets[k + 1]] = src[starts[r]:starts[r] + counts[r]]
        out.flush()
        del out
    np.save(paths["offsets"], offsets)
    stats = {"complete": True, "n_rows": len(rows), "n_windows": int(offsets[-1]),
             "mean_windows": float(counts.mean()), "files_capped": int((full > max_win).sum()),
             "files_multi_window": int((counts > 1).sum()), "max_length": max_len, "max_windows": max_win}
    paths["progress"].write_text(json.dumps(stats, indent=2))
    print(f"Done: {stats}")


def load_windows(cfg: dict, name: str, pool: str) -> tuple[np.ndarray, np.ndarray]:
    """(vectors [n_windows, hidden] float16, offsets [n_rows + 1]) of a windowed input."""
    paths = _paths(cfg, name)
    prog = json.loads(paths["progress"].read_text()) if paths["progress"].exists() else {}
    if not prog.get("complete"):
        raise FileNotFoundError(f"Windowed embedding '{name}' is missing or unfinished; run `embed-windows {name}`.")
    offsets = np.load(paths["offsets"])
    if pool in ("cls", "mean"):
        return np.load(paths[pool], mmap_mode="r"), offsets
    if pool != "pooler":
        raise ValueError(f"Unknown pooling '{pool}' (use cls, mean or pooler)")
    w = torch.load(emb_dir(cfg) / f"{window_specs(cfg)[name]['model']}_pooler.pt")
    cls = np.load(paths["cls"], mmap_mode="r")
    out = np.empty(cls.shape, dtype=np.float16)
    for s in range(0, len(cls), 100_000):
        x = torch.from_numpy(np.asarray(cls[s:s + 100_000], dtype=np.float32))
        out[s:s + 100_000] = torch.tanh(x @ w["weight"].T + w["bias"]).numpy()
    return out, offsets


def window_mean(cfg: dict, name: str, pool: str) -> np.ndarray:
    """[n_rows, hidden] float16: the plain mean of each row's windows."""
    vecs, offsets = load_windows(cfg, name, pool)
    counts = np.diff(offsets)
    out = np.zeros((len(counts), vecs.shape[1]), dtype=np.float32)
    for s in range(0, len(counts), 50_000):
        e = min(s + 50_000, len(counts))
        block = np.asarray(vecs[offsets[s]:offsets[e]], dtype=np.float32)
        sums = np.add.reduceat(block, offsets[s:e] - offsets[s], axis=0) if len(block) else np.zeros((e - s, vecs.shape[1]))
        out[s:e] = sums / np.maximum(counts[s:e], 1)[:, None]
    return out.astype(np.float16)
