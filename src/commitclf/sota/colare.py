"""CoLaRe (Zeng et al., SANER 2024) run unchanged on our folds and scored with our metrics.

What is theirs (loaded from sotas/COLARE-main/commit_classifier, not copied):
  - the model classes in model/models.py: CCModel (message + code hunks + 29 line counts),
    CodeFeatModel (code + counts, no message), MessageCodeModel, MessageFeatModel
  - the data pipeline in dataset/ccdataset.py: their own diff parser, 5 files x 2 hunks
    for the Java set, 256 tokens per text, their tokenizer, their collate function
  - the inputs: their dataset/1793/dataset.csv, with their precomputed 29 features
  - the training recipe of their Trainer:
      - AdamW (eps 1e-6, no weight decay), CodeBERT embeddings + layers 0-2 at 1e-5,
        layers 3-6 at 2e-5, layers 7-11 at 4e-5, everything else at 1e-4, pooler not trained
      - unweighted cross-entropy, batch 3 with 2-step gradient accumulation, 10 epochs,
        no scheduler, no clipping, fp32
What is ours:
  - which commits are train / val / test: our split files (`given` = CoLaRe's own 5 folds,
    `repo` = held-out repositories). Only the 1,775 commits of our converted set are used
    (their 3 duplicate rows and 3 commits without a text diff are left out)
  - the tested epoch: best validation macro-F1 (their CV mode has no validation set and tests
    the last epoch)
  - metrics, predictions and the output layout: the same as `train` / `finetune`
  - one guard: an optimizer step whose gradients are not finite is skipped (and counted in
    history.csv). Their gate divides by ||H|| with H = g * W n + h_bias and h_bias starts at
    zero, so a commit whose 29 counts are all zero (12 in this set) gives NaN gradients if it is
    in the very first optimizer step; without the guard the whole fold collapses to NaN

Their package `multimodal_transformers` (pinned to torch 1.13 / transformers 4.27) is replaced by
a minimal stand-in for the only part their models use: TabularConfig and TabularFeatCombiner with
combine_feat_method="gating_on_cat_and_num_feats_then_sum". It is transcribed from
multimodal-transformers 0.2a0 (MIT licence), numerical features only (their models pass no
categorical ones).
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from commitclf.config import resolve
from commitclf.data.splits import load_split
from commitclf.metrics import classification_metrics, plot_confusion_matrices
from commitclf.train import summarize
from commitclf.utils import get_device, set_seed

VARIANTS = ("CCModel", "CodeFeatModel", "MessageCodeModel", "MessageFeatModel")
THEIR_LABELS = {0: "Perfective", 1: "Adaptive", 2: "Corrective"}           # ccdataset.CommitReader
TO_OURS = {"Perfective": "clean", "Adaptive": "enhancement", "Corrective": "bug"}
CODEBERT = "microsoft/codebert-base"


# ----------------------------------------------------------------------------------------------
# Stand-in for multimodal_transformers (gating_on_cat_and_num_feats_then_sum, numerical only)
# ----------------------------------------------------------------------------------------------
class TabularConfig:
    def __init__(self, num_labels, mlp_division=4, combine_feat_method="text_only", mlp_dropout=0.1,
                 numerical_bn=True, use_simple_classifier=True, mlp_act="relu", gating_beta=0.2,
                 numerical_feat_dim=0, cat_feat_dim=0, **kwargs):
        self.num_labels, self.mlp_division, self.combine_feat_method = num_labels, mlp_division, combine_feat_method
        self.mlp_dropout, self.numerical_bn, self.use_simple_classifier = mlp_dropout, numerical_bn, use_simple_classifier
        self.mlp_act, self.gating_beta = mlp_act, gating_beta
        self.numerical_feat_dim, self.cat_feat_dim = numerical_feat_dim, cat_feat_dim


class TabularFeatCombiner(nn.Module):
    def __init__(self, tabular_config: TabularConfig):
        super().__init__()
        c = tabular_config
        if c.combine_feat_method != "gating_on_cat_and_num_feats_then_sum":
            raise NotImplementedError(c.combine_feat_method)
        if c.cat_feat_dim:
            raise NotImplementedError("categorical features")
        self.tabular_config = c
        self.text_out_dim, self.numerical_feat_dim = c.text_feat_dim, c.numerical_feat_dim
        if self.numerical_feat_dim > self.text_out_dim:
            raise NotImplementedError("numerical_feat_dim > text_feat_dim (needs their MLP)")
        self.num_bn = nn.BatchNorm1d(self.numerical_feat_dim) if c.numerical_bn and self.numerical_feat_dim else None
        self.act_func = {"relu": nn.ReLU(), "tanh": nn.Tanh(), "sigmoid": nn.Sigmoid()}[c.mlp_act]
        if self.numerical_feat_dim > 0:
            self.g_num_layer = nn.Linear(self.numerical_feat_dim + self.text_out_dim, self.text_out_dim)
            self.dropout_num = nn.Dropout(c.mlp_dropout)
            self.h_num_layer = nn.Linear(self.numerical_feat_dim, self.text_out_dim, bias=False)
        self.h_bias = nn.Parameter(torch.zeros(self.text_out_dim))
        self.layer_norm = nn.LayerNorm(self.text_out_dim)   # created but unused in their forward
        self.final_out_dim = self.text_out_dim

    def forward(self, text_feats, cat_feats=None, numerical_feats=None):
        if numerical_feats is None:
            numerical_feats = torch.zeros((text_feats.shape[0], 0), device=text_feats.device)
        if self.num_bn is not None:
            numerical_feats = self.num_bn(numerical_feats)
        if numerical_feats.shape[1] != 0:
            g_num = self.dropout_num(self.act_func(self.g_num_layer(torch.cat([text_feats, numerical_feats], dim=1))))
            g_mult_num = g_num * self.h_num_layer(numerical_feats)
        else:
            g_mult_num = 0
        h = g_mult_num + self.h_bias
        norm = torch.norm(text_feats, dim=1) / torch.norm(h, dim=1)
        alpha = torch.clamp(norm * self.tabular_config.gating_beta, min=0, max=1)
        return text_feats + alpha[:, None] * h


def _install_mmt_shim() -> None:
    pkg = types.ModuleType("multimodal_transformers")
    sub = types.ModuleType("multimodal_transformers.model")
    cfg_mod = types.ModuleType("multimodal_transformers.model.tabular_config")
    comb_mod = types.ModuleType("multimodal_transformers.model.tabular_combiner")
    cfg_mod.TabularConfig, comb_mod.TabularFeatCombiner = TabularConfig, TabularFeatCombiner
    pkg.model, sub.tabular_config, sub.tabular_combiner = sub, cfg_mod, comb_mod
    for name, mod in [("multimodal_transformers", pkg), ("multimodal_transformers.model", sub),
                      ("multimodal_transformers.model.tabular_config", cfg_mod),
                      ("multimodal_transformers.model.tabular_combiner", comb_mod)]:
        sys.modules.setdefault(name, mod)


# ----------------------------------------------------------------------------------------------
# Loading their code
# ----------------------------------------------------------------------------------------------
def _load_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_colare(colare_dir: Path, model_path: str | None = None):
    """Their models module, dataset module and model registry.

    model_path: load CodeBERT from here instead of the hub name hard-coded in their code
    (a local copy, or a tiny model for testing)."""
    code_dir = colare_dir / "commit_classifier"
    if not (code_dir / "model" / "models.py").exists():
        raise FileNotFoundError(f"CoLaRe code not found under {code_dir}")
    _install_mmt_shim()
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))       # their modules do `from registry import ...`
    import registry                              # noqa: E402  (their package)
    models = _load_file("colare_models", code_dir / "model" / "models.py")
    data = _load_file("colare_ccdataset", code_dir / "dataset" / "ccdataset.py")
    if model_path:
        for mod, attr in [(models, "AutoModel"), (data, "AutoTokenizer")]:
            orig = getattr(mod, attr)

            class Redirect:
                @staticmethod
                def from_pretrained(name, *a, _orig=orig, **k):
                    return _orig.from_pretrained(model_path if name == CODEBERT else name, *a, **k)
            setattr(mod, attr, Redirect)
    return models, data, registry.MODEL_REGISTRY


# ----------------------------------------------------------------------------------------------
# Training (their Trainer's recipe) and evaluation (ours)
# ----------------------------------------------------------------------------------------------
def param_groups(model: nn.Module) -> list[dict]:
    """Their Trainer._build_optimizer, written generically: the same groups for every variant."""
    enc = model.code_change_encoder.hunk_encoder if hasattr(model, "code_change_encoder") else model.message_encoder
    layers = enc.encoder.layer
    groups = [{"params": list(enc.embeddings.parameters()) + [p for l in layers[0:3] for p in l.parameters()], "lr": 1e-5},
              {"params": [p for l in layers[3:7] for p in l.parameters()], "lr": 2e-5},
              {"params": [p for l in layers[7:12] for p in l.parameters()], "lr": 4e-5}]
    in_encoder = {id(p) for p in enc.parameters()}            # the pooler is in no group: never updated
    groups.append({"params": [p for p in model.parameters() if id(p) not in in_encoder], "lr": 1e-4})
    return groups


def fold_csvs(raw: pd.DataFrame, parts: pd.Series, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for p in ("train", "val", "test"):
        paths[p] = out_dir / f"{p}.csv"
        raw[raw["unit_key"].map(parts) == p].drop(columns=["unit_key"]).to_csv(paths[p], index=False)
    return paths


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, ys = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        probs.append(torch.softmax(model(**batch).float(), 1).cpu())
        ys.append(batch["labels"].cpu())
    return torch.cat(ys).numpy(), torch.cat(probs).numpy()


def to_our_order(y: np.ndarray, proba: np.ndarray, labels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Their class ids (0 Perfective, 1 Adaptive, 2 Corrective) -> our label order."""
    ours = [labels.index(TO_OURS[THEIR_LABELS[i]]) for i in range(3)]
    out = np.zeros_like(proba)
    out[:, ours] = proba
    return np.array(ours)[y], out


def train_fold(cfg: dict, variant: str, split: str, fold: int, raw: pd.DataFrame, mods, opts: dict,
               run_dir: Path) -> dict:
    models, data, registry = mods
    labels = cfg["labels"]
    set_seed(cfg["seed"] + fold)
    device = get_device(cfg["train"].get("device", "cuda"))

    fold_dir = run_dir / f"fold{fold}"
    paths = fold_csvs(raw, load_split(cfg, split, fold), fold_dir / "data")
    conf = types.SimpleNamespace(file_num_limit=opts["file_limit"], hunk_num_limit=opts["hunk_limit"])
    sets = {p: data.CommitDataset(str(path), conf, use_roberta=False) for p, path in paths.items()}
    collate = partial(data.collate_fn, msg_collator=sets["train"].message_collator)
    gen = torch.Generator().manual_seed(cfg["seed"] + fold)
    train_loader = torch.utils.data.DataLoader(sets["train"], batch_size=opts["batch_size"], shuffle=True,
                                               collate_fn=collate, drop_last=True, generator=gen)
    eval_loaders = {p: torch.utils.data.DataLoader(sets[p], batch_size=opts["eval_batch_size"], shuffle=False,
                                                   collate_fn=collate) for p in ("val", "test")}
    print(f"Fold {fold}: " + ", ".join(f"{p}={len(s)}" for p, s in sets.items())
          + f" | CoLaRe {variant}, {opts['file_limit']} files x {opts['hunk_limit']} hunks")

    model = registry.get_obj(variant)(None).to(device)
    optimizer = torch.optim.AdamW(param_groups(model), eps=1e-6, weight_decay=0.0)   # = transformers.AdamW defaults
    criterion = nn.CrossEntropyLoss()

    best_f1, best_epoch, best_state, history, global_step = -1.0, 0, None, [], 0
    for epoch in range(1, opts["epochs"] + 1):
        model.train()
        optimizer.zero_grad()
        total, seen, skipped = 0.0, 0, 0
        for batch in tqdm(train_loader, desc=f"{run_dir.name} fold {fold} epoch {epoch}", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = criterion(model(**batch), batch["labels"])
            total += loss.item() * len(batch["labels"])
            seen += len(batch["labels"])
            (loss / opts["grad_acc"]).backward()
            global_step += 1
            if global_step % opts["grad_acc"] == 0:
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                if all(torch.isfinite(g).all() for g in grads):
                    optimizer.step()
                else:
                    skipped += 1
                optimizer.zero_grad()
        y_val, p_val = to_our_order(*predict(model, eval_loaders["val"], device), labels)
        val = classification_metrics(y_val, p_val.argmax(1), p_val, labels)
        history.append({"epoch": epoch, "train_loss": total / seen, "skipped_steps": skipped,
                        **{f"val_{k}": v for k, v in val.items()}})
        tqdm.write(f"[{epoch:3d}] train loss={total / seen:.4f} | val acc={val['accuracy']:.4f} f1={val['f1']:.4f}"
                   + (f" | skipped {skipped} non-finite steps" if skipped else ""))
        if val["f1"] > best_f1:
            best_f1, best_epoch = val["f1"], epoch
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
    pd.DataFrame(history).to_csv(fold_dir / "history.csv", index=False)

    model.load_state_dict(best_state)
    y_true, proba = to_our_order(*predict(model, eval_loaders["test"], device), labels)
    test = {**classification_metrics(y_true, proba.argmax(1), proba, labels), "best_epoch": best_epoch}
    print(f"Fold {fold} test: acc={test['accuracy']:.4f} f1={test['f1']:.4f} auc={test['auc']:.4f} "
          f"(best epoch {best_epoch})")
    (fold_dir / "test_metrics.json").write_text(json.dumps(test, indent=2))
    test_rows = raw[raw["unit_key"].map(load_split(cfg, split, fold)) == "test"]
    pred = pd.DataFrame({"unit_key": test_rows["unit_key"].to_numpy(),
                         "label": [TO_OURS[l] for l in test_rows["labels"]]})
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


def run(cfg: dict, folds: list[int], variant: str, split: str, colare_dir: str, src: str,
        model_path: str | None = None, name: str | None = None, **opts) -> pd.DataFrame:
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    opts = {"file_limit": 5, "hunk_limit": 2, "batch_size": 3, "grad_acc": 2, "epochs": 10, "eval_batch_size": 6,
            **{k: v for k, v in opts.items() if v is not None}}
    mods = load_colare(resolve(cfg, colare_dir), model_path)

    raw = pd.read_csv(resolve(cfg, src))
    raw["unit_key"] = raw["user"] + "/" + raw["repo"] + "|" + raw["commit"]
    raw = raw.drop_duplicates("unit_key").reset_index(drop=True)
    in_split = raw["unit_key"].isin(load_split(cfg, split, folds[0]).index)
    print(f"{len(raw)} unique commits in {src}; {int(in_split.sum())} in our split files "
          f"(left out: {int((~in_split).sum())})")
    raw = raw[in_split].reset_index(drop=True)

    run_dir = resolve(cfg, cfg["train"]["out_dir"]) / (name or f"colare/{variant}_{split}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "spec.json").write_text(json.dumps({"method": "CoLaRe", "variant": variant, "split": split,
                                                   **opts, "selection": "best val macro-F1"}, indent=2))
    print(f"Run {run_dir.name}: {opts}")
    for f in folds:
        train_fold(cfg, variant, split, f, raw, mods, opts, run_dir)
    # summary over every finished fold in the run folder, so a single fold can be rerun
    rows = [{"fold": int(d.name[4:]), **json.loads((d / "test_metrics.json").read_text())}
            for d in sorted(run_dir.glob("fold*"), key=lambda d: int(d.name[4:]))
            if (d / "test_metrics.json").exists()]
    return summarize(rows, run_dir / "summary.csv")
