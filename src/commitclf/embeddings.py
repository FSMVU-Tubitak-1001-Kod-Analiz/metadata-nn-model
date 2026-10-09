"""Frozen transformer embeddings, one vector per file row of data/processed/rows.csv.

For every input in `embeddings.inputs` two arrays are written to artifacts/embeddings/:
    <input>.cls.npy    [n_rows, hidden] float16, last-layer vector of the first token
    <input>.mean.npy   [n_rows, hidden] float16, mean of the last-layer token vectors (padding excluded)
Row i of each array belongs to row i of rows.csv. The "pooler" pooling (what the paper used)
is tanh(W·cls + b) with the model's pooler weights, recomputed from cls when loading.

All selected inputs are filled in one pass over the source CSV. Progress is saved every
`save_every` chunks, so running the same command again continues where it stopped.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from commitclf.config import resolve
from commitclf.data.java import clean_java, patch_after, patch_before
from commitclf.data.prepare import KEY_COLS, load_rows, read_source, row_key
from commitclf.utils import get_device

DERIVED = {"patch_before": ("patch", patch_before), "patch_after": ("patch", patch_after)}
CLEANED = {"content_before", "content_after", "patch_before", "patch_after"}
POOLS = ("cls", "mean")


def emb_dir(cfg: dict) -> Path:
    return resolve(cfg, cfg["embeddings"]["dir"])


def array_path(cfg: dict, name: str, pool: str) -> Path:
    return emb_dir(cfg) / f"{name}.{pool}.npy"


def progress_path(cfg: dict, name: str) -> Path:
    return emb_dir(cfg) / f"{name}.progress.json"


def read_progress(cfg: dict, name: str) -> dict:
    p = progress_path(cfg, name)
    return json.loads(p.read_text()) if p.exists() else {"source_rows": 0, "written": 0, "complete": False}


def build_texts(chunk: pd.DataFrame, columns: list[str], separator: str, clean: bool) -> list[str]:
    parts = []
    for col in columns:
        src, fn = DERIVED.get(col, (col, None))
        s = chunk[src].fillna("").astype(str)
        if fn is not None:
            s = s.map(fn)
        if clean and col in CLEANED:
            s = s.map(clean_java)
        parts.append(s)
    if len(parts) == 1:
        return parts[0].tolist()
    return pd.concat(parts, axis=1).agg(separator.join, axis=1).tolist()


class Encoder:
    def __init__(self, model_name: str, max_length: int, batch_size: int, device, fp16: bool):
        from transformers import AutoModel, AutoTokenizer
        from transformers import logging as hf_logging

        hf_logging.set_verbosity_error()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.max_length, self.batch_size, self.device = max_length, batch_size, device
        self.fp16 = fp16 and device.type == "cuda"
        self.hidden = self.model.config.hidden_size

    def pooler_weights(self) -> dict | None:
        pooler = getattr(self.model, "pooler", None)
        if pooler is None:
            return None
        return {"weight": pooler.dense.weight.detach().cpu().float(), "bias": pooler.dense.bias.detach().cpu().float()}

    @torch.no_grad()
    def __call__(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Returns (cls, mean), each [len(texts), hidden] float16."""
        unique = list(dict.fromkeys(texts))   # e.g. one commit message shared by all files of a change
        cls_out, mean_out = [], []
        for start in range(0, len(unique), self.batch_size):
            tokens = self.tokenizer(unique[start:start + self.batch_size], truncation=True, padding=True,
                                    max_length=self.max_length, return_tensors="pt").to(self.device)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.fp16):
                hidden = self.model(**tokens).last_hidden_state.float()
            mask = tokens["attention_mask"].unsqueeze(-1).float()
            cls_out.append(hidden[:, 0].cpu())
            mean_out.append(((hidden * mask).sum(1) / mask.sum(1).clamp(min=1)).cpu())
        lookup = {t: i for i, t in enumerate(unique)}
        position = np.array([lookup[t] for t in texts])
        cls = torch.cat(cls_out).numpy()[position]
        mean = torch.cat(mean_out).numpy()[position]
        return cls.astype(np.float16), mean.astype(np.float16)


def run(cfg: dict, only: list[str] | None = None, overwrite: bool = False) -> None:
    e = cfg["embeddings"]
    names = only or [n for n in e["inputs"] if n != "message"]
    unknown = [n for n in names if n not in e["inputs"]]
    if unknown:
        raise KeyError(f"Unknown embedding inputs {unknown}; defined: {list(e['inputs'])}")

    rows = load_rows(cfg)
    n_rows = len(rows)
    index = pd.Series(np.arange(n_rows), index=rows["row_key"])
    emb_dir(cfg).mkdir(parents=True, exist_ok=True)
    device = get_device(cfg["train"].get("device", "cuda"))

    # Which inputs still need work, and from which source row.
    progress, active = {}, []
    for name in names:
        prog = {"source_rows": 0, "written": 0, "complete": False} if overwrite else read_progress(cfg, name)
        if prog.get("complete") and prog.get("n_rows") == n_rows:
            print(f"{name}: already complete, skipping (use --overwrite to redo)")
            continue
        if prog.get("n_rows") not in (None, n_rows):
            print(f"{name}: rows.csv changed since the last run, starting over")
            prog = {"source_rows": 0, "written": 0, "complete": False}
        prog["n_rows"] = n_rows
        progress[name] = prog
        active.append(name)
    if not active:
        return

    encoders, arrays = {}, {}
    for name in active:
        spec = e["models"][e["inputs"][name]["model"]]
        key = e["inputs"][name]["model"]
        if key not in encoders:
            encoders[key] = Encoder(spec["name"], spec.get("max_length", 512), e.get("batch_size", 32),
                                    device, e.get("fp16", True))
            weights = encoders[key].pooler_weights()
            if weights is not None:
                torch.save(weights, emb_dir(cfg) / f"{key}_pooler.pt")
        hidden = encoders[key].hidden
        for pool in POOLS:
            path = array_path(cfg, name, pool)
            fresh = progress[name]["source_rows"] == 0 or not path.exists()
            arrays[name, pool] = np.lib.format.open_memmap(path, mode="w+" if fresh else "r+",
                                                           dtype=np.float16, shape=(n_rows, hidden))
            if fresh:
                progress[name].update(source_rows=0, written=0)

    start = min(progress[n]["source_rows"] for n in active)
    source_cols = sorted({DERIVED.get(c, (c, None))[0] for n in active for c in e["inputs"][n]["columns"]}
                         | set(KEY_COLS))
    print(f"Embedding {active} from source row {start} (columns {source_cols})")

    def save_progress():
        for arr in arrays.values():
            arr.flush()
        for n in active:
            progress_path(cfg, n).write_text(json.dumps(progress[n]))

    pos = start
    bar = tqdm(desc="rows", unit="row", initial=start)
    for i, chunk in enumerate(read_source(cfg, source_cols, skiprows=start)):
        chunk = chunk.reset_index(drop=True)
        idx = row_key(chunk).map(index)
        for name in active:
            inp = e["inputs"][name]
            # rows this input still needs: in rows.csv and past its own progress
            need = idx.notna() & (np.arange(pos, pos + len(chunk)) >= progress[name]["source_rows"])
            if need.any():
                sel = chunk[need.to_numpy()]
                texts = build_texts(sel, inp["columns"], inp.get("separator", " | "), e.get("clean_code", True))
                cls, mean = encoders[inp["model"]](texts)
                target = idx[need].astype(int).to_numpy()
                arrays[name, "cls"][target] = cls
                arrays[name, "mean"][target] = mean
                progress[name]["written"] += int(need.sum())
            progress[name]["source_rows"] = max(progress[name]["source_rows"], pos + len(chunk))
        pos += len(chunk)
        bar.update(len(chunk))
        if (i + 1) % e.get("save_every", 20) == 0:
            save_progress()
    bar.close()

    for name in active:
        progress[name]["complete"] = True
        if progress[name]["written"] != n_rows:
            print(f"WARNING {name}: wrote {progress[name]['written']} of {n_rows} rows. Some rows of rows.csv "
                  "were not found in the source (wrong source file?).")
    save_progress()
    print(f"Done: {active} -> {emb_dir(cfg)}")


def load_input(cfg: dict, name: str, pool: str) -> np.ndarray:
    """[n_rows, hidden] float16 array for one embedding input and pooling (cls | mean | pooler)."""
    if not read_progress(cfg, name).get("complete"):
        raise FileNotFoundError(f"Embedding '{name}' is missing or unfinished; run `embed --only {name}`.")
    if pool in POOLS:
        return np.load(array_path(cfg, name, pool), mmap_mode="r")
    if pool != "pooler":
        raise ValueError(f"Unknown pooling '{pool}' (use cls, mean or pooler)")
    w = torch.load(emb_dir(cfg) / f"{cfg['embeddings']['inputs'][name]['model']}_pooler.pt")
    cls = np.load(array_path(cfg, name, "cls"), mmap_mode="r")
    out = np.empty(cls.shape, dtype=np.float16)
    for s in range(0, len(cls), 100_000):
        x = torch.from_numpy(np.asarray(cls[s:s + 100_000], dtype=np.float32))
        out[s:s + 100_000] = torch.tanh(x @ w["weight"].T + w["bias"]).numpy()
    return out
