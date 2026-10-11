"""Explaining finished runs. Nothing here retrains a network; it reuses the saved checkpoints.

    shap         XGBoost on the change-level metadata + SHAP: feature ranking and direction per class
    permutation  shuffle one feature (group) of a trained network's test inputs, measure the macro-F1 drop
    weights      per test change: attention over its files and the code-vs-metadata fusion weight
    tokens       Integrated Gradients through CodeBERT on the `patch` input: which diff lines
                 (added / removed / context / hunk header) the prediction rests on, plus an HTML
                 page with highlighted examples
    occlusion    remove one kind of diff line (added / removed / context / hunk header) from every
                 file of a change, re-embed with CodeBERT and measure how much the prediction changes;
                 compared with removing the same number of random lines

Outputs go to artifacts/explain/<analysis>/<run or split>/.
"""
from __future__ import annotations

import html
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from commitclf.config import resolve
from commitclf.data.prepare import KEY_COLS, load_rows, read_source, row_key
from commitclf.features import ChangeSet, FileTable, batches, build_sets, encode_labels, load_parts
from commitclf.utils import get_device, set_seed

# Readable names for the paper
NAMES = {
    "change__log_additions": "lines added (log)", "change__log_deletions": "lines deleted (log)",
    "change__log_java_files": "Java files changed (log)", "change__log_files_in_pr": "files changed, all types (log)",
    "change__java_file_share": "share of Java files", "change__log_commits": "commits (log)",
    "change__mean_log_churn": "mean churn per file (log)", "change__max_log_churn": "max churn per file (log)",
    "change__deletion_share": "deletion share of churn", "change__test_file_share": "share of test files",
    "change__content_omitted_share": "share of oversized files", "change__added_share": "share of added files",
    "change__modified_share": "share of modified files", "change__removed_share": "share of removed files",
    "change__renamed_share": "share of renamed files", "path__src_main": "share under src/main",
    "path__src_test": "share under src/test", "path__examples_docs": "share in examples/docs",
    "path__name_test": "share with test-like names", "path__name_exception": "share named *Exception/*Error",
    "path__name_util": "share named *Util/*Helper", "path__name_config": "share named *Config/*Properties",
    "path__name_impl": "share named *Impl", "path__package_info": "share of package-info.java",
    "path__depth": "mean path depth (log)",
}

FILE_GROUPS = {
    "file: size": ["file__log_additions", "file__log_deletions"],
    "file: status": ["file__added", "file__modified", "file__removed", "file__renamed"],
    "file: test file": ["file__is_test", "file__path_src_test", "file__path_name_test"],
    "file: path / name": ["file__path_src_main", "file__path_examples_docs", "file__path_name_exception",
                          "file__path_name_util", "file__path_name_config", "file__path_name_impl",
                          "file__path_package_info", "file__path_depth"],
    "file: oversized": ["file__content_omitted"],
}
CHANGE_GROUPS = {
    "change: size": ["change__log_additions", "change__log_deletions", "change__mean_log_churn",
                     "change__max_log_churn", "change__deletion_share"],
    "change: number of files": ["change__log_java_files", "change__log_files_in_pr", "change__java_file_share"],
    "change: commits": ["change__log_commits"],
    "change: status shares": ["change__added_share", "change__modified_share", "change__removed_share",
                              "change__renamed_share"],
    "change: test share": ["change__test_file_share"],
    "change: oversized share": ["change__content_omitted_share"],
}


def out_dir(cfg: dict, analysis: str, name: str) -> Path:
    d = resolve(cfg, "artifacts/explain") / analysis / name.replace("/", "__")
    d.mkdir(parents=True, exist_ok=True)
    return d


def summarize_folds(df: pd.DataFrame, by: list[str], value_cols: list[str]) -> pd.DataFrame:
    agg = df.groupby(by, sort=False)[value_cols].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return agg.reset_index()


# ---------------------------------------------------------------------------- loading a finished run

def run_dir(cfg: dict, run: str) -> Path:
    d = resolve(cfg, cfg["train"]["out_dir"]) / run
    if not (d / "spec.json").exists():
        raise FileNotFoundError(f"No finished run at {d} (expected spec.json). Use e.g. --run main/attention_repo")
    return d


def load_model(cfg: dict, rdir: Path, fold: int, device):
    from commitclf.train import make_model
    ckpt = torch.load(rdir / f"fold{fold}" / "model.pt", map_location=device)
    model = make_model({**cfg, "train": ckpt["train_cfg"]}, ckpt["spec"], ckpt["file_dim"], ckpt["meta_dim"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


def run_folds(rdir: Path) -> list[int]:
    return sorted(int(p.name[4:]) for p in rdir.glob("fold*") if (p / "model.pt").exists())


def local_test(table: FileTable, cs: ChangeSet) -> tuple[SimpleNamespace, ChangeSet]:
    """Copy the test files into a small table so features can be permuted without touching the big one."""
    x = table.x[cs.rows].clone()
    return (SimpleNamespace(x=x, dim=x.shape[1]),
            ChangeSet(cs.units, cs.y, cs.meta.clone(), torch.arange(len(x)), cs.offsets))


@torch.no_grad()
def predict(model, table, cs, device, batch_size=256, collect=False):
    """Class probabilities for every change (+ file attention and fusion weights if collect)."""
    probs, file_w, code_w = [], [], []
    for files, mask, meta, _ in batches(table, cs, batch_size, shuffle=False):
        logits = model(files.to(device), mask.to(device), meta.to(device))
        probs.append(torch.softmax(logits.float(), 1).cpu())
        if collect:
            w = model.files.pool.last_weights
            file_w.extend([w[i, :int(mask[i].sum())].cpu().numpy() for i in range(len(w))] if w is not None
                          else [None] * len(mask))
            code_w.append(model.fusion.last_code_weights.cpu() if model.fusion is not None
                          else torch.full((len(mask),), float("nan")))
    out = torch.cat(probs).numpy()
    return (out, file_w, torch.cat(code_w).numpy()) if collect else out


def macro_f1(y, proba) -> float:
    from sklearn.metrics import f1_score
    return f1_score(y, proba.argmax(1), average="macro", labels=range(proba.shape[1]), zero_division=0)


# ---------------------------------------------------------------------------- shap

def tree_shap(model, X: np.ndarray, n_classes: int) -> np.ndarray:
    """Exact TreeSHAP values from XGBoost itself (no `shap` package): [n, features, classes]."""
    import xgboost as xgb
    contrib = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
    contrib = np.asarray(contrib)
    if contrib.ndim == 2:                     # binary: [n, features + 1]
        contrib = contrib[:, None, :]
    return contrib[:, :, :-1].transpose(0, 2, 1)   # drop the bias column -> [n, features, classes]


def beeswarm(plt, sv: np.ndarray, X: np.ndarray, names: list[str], title: str, path: Path,
             top: int = 15, max_points: int = 3000, seed: int = 0) -> None:
    """SHAP summary plot: one row per feature, dots = changes, colour = feature value (low blue, high red)."""
    order = np.argsort(np.abs(sv).mean(0))[::-1][:top][::-1]
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(sv), min(max_points, len(sv)), replace=False)
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(order) + 1.5))
    for row, j in enumerate(order):
        x, v = sv[pick, j], X[pick, j]
        rank = pd.Series(v).rank(pct=True).to_numpy() if np.ptp(v) > 0 else np.full(len(v), 0.5)
        ax.scatter(x, row + rng.uniform(-0.3, 0.3, len(x)), c=rank, cmap="coolwarm", s=6, alpha=0.6,
                   vmin=0, vmax=1, linewidths=0)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_yticks(range(len(order)), [names[j] for j in order])
    ax.set_xlabel("SHAP value (impact on the class log-odds)")
    ax.set_title(title)
    sm = plt.cm.ScalarMappable(cmap="coolwarm")
    cb = fig.colorbar(sm, ax=ax, ticks=[0, 1], pad=0.01)
    cb.ax.set_yticklabels(["low", "high"])
    cb.set_label("feature value")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def run_shap(cfg: dict, split: str, folds: list[int], subset=None) -> pd.DataFrame:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from xgboost import XGBClassifier

    from commitclf.baselines import balanced_weights

    labels, d = cfg["labels"], out_dir(cfg, "shap", split + (f"_{subset}" if subset else ""))
    rows = []
    for fold in folds:
        parts = load_parts(cfg, split, fold, subset)
        train = pd.concat([parts["train"], parts["val"]], ignore_index=True)
        test = parts["test"]
        cols = [c for c in train.columns if c.startswith(("change__", "path__"))]
        y_train = encode_labels(train["label"], labels).numpy()
        model = XGBClassifier(tree_method="hist", eval_metric="mlogloss", random_state=cfg["seed"], n_jobs=-1)
        model.fit(train[cols].to_numpy(np.float32), y_train, sample_weight=balanced_weights(y_train, len(labels)))

        X_test = test[cols].to_numpy(np.float32)
        y_test = encode_labels(test["label"], labels).numpy()
        f1 = macro_f1(y_test, model.predict_proba(X_test))
        sv = tree_shap(model, X_test, len(labels))   # [n, features, classes]
        print(f"Fold {fold}: XGBoost macro-F1 {f1:.4f}, SHAP on {len(X_test)} test changes")

        for k, cls in enumerate(labels):
            for j, c in enumerate(cols):
                v = sv[:, j, k]
                corr = pd.Series(X_test[:, j]).corr(pd.Series(v), method="spearman") if X_test[:, j].std() > 0 else 0.0
                rows.append({"fold": fold, "class": cls, "feature": c, "name": NAMES.get(c, c),
                             "mean_abs_shap": float(np.abs(v).mean()), "direction": float(corr)})
        if fold == folds[0]:
            for k, cls in enumerate(labels):
                beeswarm(plt, sv[:, :, k], X_test, [NAMES.get(c, c) for c in cols],
                         f"SHAP for class '{cls}' (fold {fold}, {split} split)", d / f"beeswarm_{cls}.png",
                         seed=cfg["seed"])

    df = pd.DataFrame(rows)
    df.to_csv(d / "shap_per_fold.csv", index=False)
    per_class = summarize_folds(df, ["class", "feature", "name"], ["mean_abs_shap", "direction"])
    per_class.to_csv(d / "shap_per_class.csv", index=False)
    overall = (df.groupby(["fold", "feature", "name"])["mean_abs_shap"].sum().reset_index()
               .pipe(summarize_folds, ["feature", "name"], ["mean_abs_shap"])
               .sort_values("mean_abs_shap_mean", ascending=False))
    overall.to_csv(d / "shap_ranking.csv", index=False)
    print(overall.head(15).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"-> {d}")
    return overall


# ---------------------------------------------------------------------------- permutation

def feature_groups(spec: dict, rows_cols: list[str], units_cols: list[str], file_dim: int, per_feature: bool):
    """{group name: (file-vector columns, metadata columns)}."""
    fact_names = [c for c in rows_cols if c.startswith("file__")] if spec["model"] != "code" else []
    n_code = file_dim - len(fact_names)
    hidden = n_code // max(1, len(spec["code"])) if spec["code"] else 0
    groups = {}
    for i, name in enumerate(spec["code"]):
        groups[f"code: {name}"] = (list(range(i * hidden, (i + 1) * hidden)), [])
    fpos = {c: n_code + i for i, c in enumerate(fact_names)}
    meta_cols = []
    for g in (spec["metadata"] if spec["model"] != "code" else []):
        if g == "change":
            meta_cols += [c for c in units_cols if c.startswith("change__")]
        elif g == "repo":
            meta_cols += ["<repo>"]
        elif g == "message":
            meta_cols += ["<message>"]
    if fact_names:
        for gname, cols in FILE_GROUPS.items():
            groups[gname] = ([fpos[c] for c in cols if c in fpos], [])
        if per_feature:
            for c in fact_names:
                groups[f"  {c}"] = ([fpos[c]], [])
    change_cols = [c for c in meta_cols if c.startswith("change__")]
    mpos = {c: i for i, c in enumerate(change_cols)}
    if change_cols:
        for gname, cols in CHANGE_GROUPS.items():
            groups[gname] = ([], [mpos[c] for c in cols if c in mpos])
        if per_feature:
            for c in change_cols:
                groups[f"  {c}"] = ([], [mpos[c]])
    return groups, len(change_cols)


def run_permutation(cfg: dict, run: str, folds: list[int] | None, repeats: int = 3, per_feature: bool = True):
    device = get_device(cfg["train"].get("device", "cuda"))
    rdir = run_dir(cfg, run)
    spec = json.loads((rdir / "spec.json").read_text())
    folds = folds or run_folds(rdir)
    d = out_dir(cfg, "permutation", run)
    table = FileTable(cfg, spec["code"], spec["embed_pool"], with_facts=spec["model"] != "code")
    rows_cols = list(load_rows(cfg).columns)

    out = []
    for fold in folds:
        set_seed(cfg["seed"] + fold)
        model = load_model(cfg, rdir, fold, device)
        cs = build_sets(cfg, spec, fold, table)["test"]
        t, c = local_test(table, cs)
        y = c.y.numpy()
        base = macro_f1(y, predict(model, t, c, device))
        groups, n_change = feature_groups(spec, rows_cols, list(cs.units.columns), table.dim, per_feature)
        # repo / message columns (experiments only) as one group each
        extra = c.meta.shape[1] - n_change
        if extra > 0:
            groups["change: repo / message (extra)"] = ([], list(range(n_change, c.meta.shape[1])))
        print(f"Fold {fold}: base macro-F1 {base:.4f}, {len(groups)} groups")
        g = torch.Generator().manual_seed(cfg["seed"] + fold)
        x0, m0 = t.x.clone(), c.meta.clone()
        for name, (fcols, mcols) in tqdm(groups.items(), desc=f"fold {fold}"):
            if not fcols and not mcols:
                continue
            drops = []
            for _ in range(repeats):
                if fcols:   # shuffle across all test files (keeps the group's columns together)
                    p = torch.randperm(len(x0), generator=g)
                    t.x[:, fcols] = x0[p][:, fcols]
                if mcols:   # shuffle across test changes
                    p = torch.randperm(len(m0), generator=g)
                    c.meta[:, mcols] = m0[p][:, mcols]
                drops.append(base - macro_f1(y, predict(model, t, c, device)))
                t.x.copy_(x0)
                c.meta.copy_(m0)
            out.append({"fold": fold, "feature": name.strip(), "group": not name.startswith("  "),
                        "name": NAMES.get(name.strip(), name.strip()),
                        "f1_drop": float(np.mean(drops)), "base_f1": base})
    df = pd.DataFrame(out)
    df.to_csv(d / "permutation_per_fold.csv", index=False)
    summ = summarize_folds(df, ["feature", "name", "group"], ["f1_drop"]).sort_values(["group", "f1_drop_mean"],
                                                                                ascending=[False, False])
    summ.to_csv(d / "permutation.csv", index=False)
    print(summ.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"-> {d}")
    return summ


# ---------------------------------------------------------------------------- weights

def run_weights(cfg: dict, run: str, folds: list[int] | None):
    device = get_device(cfg["train"].get("device", "cuda"))
    rdir = run_dir(cfg, run)
    spec = json.loads((rdir / "spec.json").read_text())
    folds = folds or run_folds(rdir)
    d = out_dir(cfg, "weights", run)
    table = FileTable(cfg, spec["code"], spec["embed_pool"], with_facts=spec["model"] != "code")
    rows = load_rows(cfg)
    labels = cfg["labels"]

    changes, files = [], []
    for fold in folds:
        model = load_model(cfg, rdir, fold, device)
        cs = build_sets(cfg, spec, fold, table)["test"]
        proba, file_w, code_w = predict(model, table, cs, device, collect=True)
        for i in range(len(cs)):
            s, e = int(cs.offsets[i]), int(cs.offsets[i + 1])
            r = rows.iloc[cs.rows[s:e].numpy()]
            changes.append({"fold": fold, "unit_key": cs.units["unit_key"].iat[i], "label": cs.units["label"].iat[i],
                            "pred": labels[int(proba[i].argmax())], "n_files": e - s, "code_weight": code_w[i]})
            if file_w[i] is not None:
                for k in range(e - s):
                    files.append({"fold": fold, "unit_key": changes[-1]["unit_key"], "label": changes[-1]["label"],
                                  "file_path": r["row_key"].iat[k].split("|", 2)[2], "weight": float(file_w[i][k]),
                                  "uniform": 1.0 / (e - s), "is_test": int(r["file__is_test"].iat[k]),
                                  "status": next((st for st in ("added", "modified", "removed", "renamed")
                                                  if r[f"file__{st}"].iat[k] == 1), "?")})
    ch = pd.DataFrame(changes)
    ch.to_csv(d / "change_weights.csv", index=False)
    print("Mean code-side fusion weight by true class:")
    print(ch.groupby("label")["code_weight"].agg(["mean", "std"]).to_string(float_format=lambda v: f"{v:.3f}"))
    if files:
        fw = pd.DataFrame(files)
        fw.to_csv(d / "file_weights.csv", index=False)
        multi = fw[fw.groupby(["fold", "unit_key"])["weight"].transform("size") > 1]
        # attention relative to uniform (1.0 = average share, >1 = favoured)
        multi = multi.assign(ratio=multi["weight"] / multi["uniform"])
        summary = pd.concat([
            multi.groupby("is_test")["ratio"].mean().rename(lambda v: f"test file = {v}"),
            multi.groupby("status")["ratio"].mean().rename(lambda v: f"status = {v}"),
        ]).rename("attention / uniform share").to_frame()
        summary.to_csv(d / "file_attention_summary.csv")
        print("\nAttention over files in multi-file changes (1.0 = equal share, >1 = favoured):")
        print(summary.to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"-> {d}")


# ---------------------------------------------------------------------------- tokens (Integrated Gradients)

def fetch_texts(cfg: dict, keys: set[str], column: str, cache: Path) -> dict[str, str]:
    if cache.exists():
        texts = pickle.loads(cache.read_bytes())
        if keys <= set(texts):
            return texts
    texts = {}
    for chunk in tqdm(read_source(cfg, KEY_COLS + [column]), desc=f"Reading {column} from source", unit="chunk"):
        k = row_key(chunk)
        hit = k.isin(keys).to_numpy()
        texts.update(zip(k[hit], chunk.loc[hit, column].fillna("").astype(str)))
        if len(texts) == len(keys):
            break
    cache.write_bytes(pickle.dumps(texts))
    return texts


def line_types(text: str) -> tuple[np.ndarray, list[str]]:
    """Start offset of every line and its type: added, removed, context, header."""
    lines = text.split("\n")
    starts = np.cumsum([0] + [len(l) + 1 for l in lines[:-1]])
    types = ["header" if l.startswith("@@") else "added" if l.startswith("+") else
             "removed" if l.startswith("-") else "context" for l in lines]
    return starts, types


def sample_changes(cs: ChangeSet, n_classes: int, n: int, max_files: int, seed: int) -> list[int]:
    """Test changes with <= max_files files, n // n_classes per true class (same sample for tokens and occlusion)."""
    n_files = (cs.offsets[1:] - cs.offsets[:-1]).numpy()
    rng = np.random.default_rng(seed)
    pick = []
    for k in range(n_classes):
        idx = np.where((cs.y.numpy() == k) & (n_files <= max_files))[0]
        pick += list(rng.choice(idx, min(len(idx), n // n_classes), replace=False))
    return sorted(int(i) for i in pick)


def run_tokens(cfg: dict, run: str, fold: int, n: int = 300, steps: int = 32, max_files: int = 5,
               target: str = "pred", examples: int = 4, ig_batch: int = 32):
    from transformers import AutoModel, AutoTokenizer

    device = get_device(cfg["train"].get("device", "cuda"))
    rdir = run_dir(cfg, run)
    spec = json.loads((rdir / "spec.json").read_text())
    if "patch" not in spec["code"]:
        raise ValueError(f"Run {run} does not use the `patch` input (code = {spec['code']})")
    d = out_dir(cfg, "tokens", f"{run}_fold{fold}")
    labels = cfg["labels"]
    e = cfg["embeddings"]
    mspec = e["models"][e["inputs"]["patch"]["model"]]

    table = FileTable(cfg, spec["code"], spec["embed_pool"], with_facts=spec["model"] != "code")
    rows = load_rows(cfg)
    clf = load_model(cfg, rdir, fold, device)
    for p in clf.parameters():
        p.requires_grad_(False)
    cs = build_sets(cfg, spec, fold, table)["test"]
    proba = predict(clf, table, cs, device)

    pick = sample_changes(cs, len(labels), n, max_files, cfg["seed"])
    keys = {rows["row_key"].iat[int(r)] for i in pick for r in cs.rows[cs.offsets[i]:cs.offsets[i + 1]]}
    texts = fetch_texts(cfg, keys, "patch", d / "patch_texts.pkl")

    tok = AutoTokenizer.from_pretrained(mspec["name"])
    enc = AutoModel.from_pretrained(mspec["name"]).to(device).eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    pooler = enc.pooler if spec["embed_pool"] == "pooler" else None
    hidden = enc.config.hidden_size
    slot = spec["code"].index("patch")
    sl = slice(slot * hidden, (slot + 1) * hidden)
    word_emb = enc.get_input_embeddings()

    def embed_from(inputs_embeds, mask):
        h = enc(inputs_embeds=inputs_embeds, attention_mask=mask).last_hidden_state
        if spec["embed_pool"] == "mean":
            m = mask.unsqueeze(-1).float()
            return (h * m).sum(1) / m.sum(1).clamp(min=1)
        cls = h[:, 0]
        return torch.tanh(pooler.dense(cls)) if pooler is not None else cls

    records, shown = [], {c: 0 for c in labels}
    page = ["<html><head><meta charset='utf-8'><style>body{font-family:sans-serif} pre{font-size:12px;"
            "line-height:1.5;white-space:pre-wrap} .f{margin:12px 0;padding:8px;border:1px solid #ccc}</style>"
            "</head><body><h2>Token attributions (Integrated Gradients) on the diff</h2><p>Green pushes towards "
            f"the explained class, red away from it. Run {html.escape(run)}, fold {fold}.</p>"]

    for i in tqdm(pick, desc="Integrated Gradients"):
        s, e_ = int(cs.offsets[i]), int(cs.offsets[i + 1])
        files = table.x[cs.rows[s:e_]].float().unsqueeze(0).to(device)              # [1, F, D]
        fmask = torch.ones(1, e_ - s, dtype=torch.bool, device=device)
        meta = cs.meta[i:i + 1].to(device)
        cls_idx = int(proba[i].argmax()) if target == "pred" else int(cs.y[i])
        label, pred = labels[int(cs.y[i])], labels[int(proba[i].argmax())]
        html_files = []
        for j in range(e_ - s):
            rk = rows["row_key"].iat[int(cs.rows[s + j])]
            text = texts.get(rk, "")
            t = tok(text, truncation=True, max_length=mspec.get("max_length", 512), return_offsets_mapping=True,
                    return_tensors="pt")
            offsets = t.pop("offset_mapping")[0].numpy()
            ids, mask = t["input_ids"].to(device), t["attention_mask"].to(device)
            x = word_emb(ids)
            special = torch.tensor([o[0] == o[1] for o in offsets], device=device)
            base_ids = torch.where(special, ids, torch.full_like(ids, tok.pad_token_id))
            base = word_emb(base_ids)
            grads = torch.zeros_like(x)
            alphas = (torch.arange(steps, device=device) + 0.5) / steps
            for a in alphas.split(ig_batch):   # interpolation steps per GPU batch
                emb = (base + a.view(-1, 1, 1) * (x - base)).requires_grad_(True)
                v = embed_from(emb, mask.expand(len(a), -1))
                f = files.expand(len(a), -1, -1).clone()
                f[:, j, sl] = v
                logits = clf(f, fmask.expand(len(a), -1), meta.expand(len(a), -1))
                g, = torch.autograd.grad(torch.log_softmax(logits, 1)[:, cls_idx].sum(), emb)
                grads += g.sum(0, keepdim=True)
            attr = ((x - base) * grads / steps).sum(-1)[0].detach().cpu().numpy()
            with torch.no_grad():   # completeness check: attributions should sum to f(x) - f(baseline)
                ends = torch.cat([base, x])
                f = files.expand(2, -1, -1).clone()
                f[:, j, sl] = embed_from(ends, mask.expand(2, -1))
                lp = torch.log_softmax(clf(f, fmask.expand(2, -1), meta.expand(2, -1)), 1)[:, cls_idx]
                gap = float(lp[1] - lp[0])

            starts, types = line_types(text)
            tok_type = [None if o[0] == o[1] else types[int(np.searchsorted(starts, o[0], side="right") - 1)]
                        for o in offsets]
            total = np.abs(attr[[k for k, tt in enumerate(tok_type) if tt]]).sum() or 1.0
            rec = {"fold": fold, "unit_key": cs.units["unit_key"].iat[i], "file": rk.split("|", 2)[2],
                   "label": label, "pred": pred, "explained": labels[cls_idx], "n_tokens": len(ids[0]),
                   "truncated": len(ids[0]) >= mspec.get("max_length", 512),
                   "ig_sum": float(attr.sum()), "ig_target_gap": gap}
            for tt in ("added", "removed", "context", "header"):
                ks = [k for k, x_ in enumerate(tok_type) if x_ == tt]
                rec[f"attr_{tt}"] = float(np.abs(attr[ks]).sum() / total) if ks else 0.0
                rec[f"tokens_{tt}"] = len(ks) / max(1, sum(x_ is not None for x_ in tok_type))
            records.append(rec)

            if shown[label] < examples and label == pred:
                scale = np.abs(attr).max() or 1.0
                spans, last = [], 0
                for k, (a0, a1) in enumerate(offsets):
                    if a0 == a1:
                        continue
                    spans.append(html.escape(text[last:a0]))
                    w = attr[k] / scale
                    color = f"rgba(0,160,0,{w:.2f})" if w > 0 else f"rgba(220,0,0,{-w:.2f})"
                    spans.append(f"<span style='background:{color}'>{html.escape(text[a0:a1])}</span>")
                    last = a1
                spans.append(html.escape(text[last:last + 2000]))
                html_files.append(f"<div class='f'><b>{html.escape(rec['file'])}</b><pre>{''.join(spans)}</pre></div>")
        if html_files:
            shown[label] += 1
            page.append(f"<h3>{html.escape(cs.units['unit_key'].iat[i])} - true {label}, predicted {pred} "
                        f"(p={proba[i].max():.2f})</h3>" + "".join(html_files))

    df = pd.DataFrame(records)
    df.to_csv(d / "token_attribution_per_file.csv", index=False)
    cols = [f"attr_{t}" for t in ("added", "removed", "context", "header")] + \
           [f"tokens_{t}" for t in ("added", "removed", "context", "header")]
    summary = df.groupby("label")[cols].mean()
    summary.loc["all"] = df[cols].mean()
    for t in ("added", "removed", "context", "header"):   # >1: more attribution than the share of tokens
        summary[f"enrichment_{t}"] = summary[f"attr_{t}"] / summary[f"tokens_{t}"].replace(0, np.nan)
    summary.to_csv(d / "token_attribution_summary.csv")
    (d / "examples.html").write_text("\n".join(page + ["</body></html>"]), encoding="utf-8")
    print(summary.to_string(float_format=lambda v: f"{v:.3f}"))
    err = ((df["ig_sum"] - df["ig_target_gap"]).abs() / df["ig_target_gap"].abs().clip(lower=1e-3)).median()
    print(f"Integrated Gradients completeness: median relative error {err:.1%} (raise --steps if this is large)")
    print(f"{len(df)} files from {len(pick)} changes; {df['truncated'].mean():.0%} of patches were cut at "
          f"{mspec.get('max_length', 512)} tokens\n-> {d}")


# ---------------------------------------------------------------------------- occlusion (line removal)

LINE_TYPES = ("added", "removed", "context", "header")


def drop_lines(text: str, kind: str, rng: np.random.Generator | None = None, k: int = 0) -> str:
    """Remove all lines of one kind from a diff, or (kind='random') k random lines of any kind."""
    lines = text.split("\n")
    types = line_types(text)[1]
    if kind == "random":
        k = min(k, len(lines))
        gone = set(rng.choice(len(lines), k, replace=False).tolist()) if k else set()
        return "\n".join(l for i, l in enumerate(lines) if i not in gone)
    return "\n".join(l for l, t in zip(lines, types) if t != kind)


def run_occlusion(cfg: dict, run: str, fold: int, n: int = 300, max_files: int = 5, batch_size: int = 64):
    from commitclf.embeddings import Encoder, emb_dir

    device = get_device(cfg["train"].get("device", "cuda"))
    rdir = run_dir(cfg, run)
    spec = json.loads((rdir / "spec.json").read_text())
    if "patch" not in spec["code"]:
        raise ValueError(f"Run {run} does not use the `patch` input (code = {spec['code']})")
    d = out_dir(cfg, "occlusion", f"{run}_fold{fold}")
    labels = cfg["labels"]
    e = cfg["embeddings"]
    model_key = e["inputs"]["patch"]["model"]
    mspec = e["models"][model_key]
    max_len = mspec.get("max_length", 512)

    table = FileTable(cfg, spec["code"], spec["embed_pool"], with_facts=spec["model"] != "code")
    rows = load_rows(cfg)
    clf = load_model(cfg, rdir, fold, device)
    cs = build_sets(cfg, spec, fold, table)["test"]
    pick = sample_changes(cs, len(labels), n, max_files, cfg["seed"])
    keys = {rows["row_key"].iat[int(r)] for i in pick for r in cs.rows[cs.offsets[i]:cs.offsets[i + 1]]}
    cache = out_dir(cfg, "tokens", f"{run}_fold{fold}") / "patch_texts.pkl"   # shared with `tokens`
    texts = fetch_texts(cfg, keys, "patch", cache)

    enc = Encoder(mspec["name"], max_len, batch_size, device, e.get("fp16", True))
    pooler = torch.load(emb_dir(cfg) / f"{model_key}_pooler.pt") if spec["embed_pool"] == "pooler" else None
    hidden = enc.hidden
    sl = slice(spec["code"].index("patch") * hidden, (spec["code"].index("patch") + 1) * hidden)

    def vectors(batch: list[str]) -> torch.Tensor:
        cls, mean = enc(batch)
        if spec["embed_pool"] == "mean":
            return torch.from_numpy(mean).float()
        x = torch.from_numpy(cls).float()
        return torch.tanh(x @ pooler["weight"].T + pooler["bias"]) if pooler is not None else x

    variants = ["full"] + [f"no_{t}" for t in LINE_TYPES] + [f"random_like_{t}" for t in ("added", "removed", "context")]
    rng = np.random.default_rng(cfg["seed"])
    out = []
    for i in tqdm(pick, desc="Occlusion"):
        s, e_ = int(cs.offsets[i]), int(cs.offsets[i + 1])
        keys_i = [rows["row_key"].iat[int(cs.rows[s + j])] for j in range(e_ - s)]
        originals = [texts.get(k, "") for k in keys_i]
        counts = [{t: line_types(x)[1].count(t) for t in LINE_TYPES} for x in originals]
        edited = []
        for v in variants:
            for x, c in zip(originals, counts):
                if v == "full":
                    edited.append(x)
                elif v.startswith("no_"):
                    edited.append(drop_lines(x, v[3:]))
                else:
                    edited.append(drop_lines(x, "random", rng, c[v.split("_like_")[1]]))
        vec = vectors(edited).view(len(variants), e_ - s, hidden)

        files = table.x[cs.rows[s:e_]].float().unsqueeze(0).repeat(len(variants), 1, 1)
        files[:, :, sl] = vec
        with torch.no_grad():
            p = torch.softmax(clf(files.to(device), torch.ones(len(variants), e_ - s, dtype=torch.bool, device=device),
                                  cs.meta[i:i + 1].expand(len(variants), -1).to(device)).float(), 1).cpu().numpy()
        pred_full = int(p[0].argmax())
        truncated = any(len(enc.tokenizer(x, truncation=False)["input_ids"]) > max_len for x in originals)
        for v_i, v in enumerate(variants):
            kind = v[3:] if v.startswith("no_") else v.split("_like_")[1] if "_like_" in v else None
            out.append({"fold": fold, "unit_key": cs.units["unit_key"].iat[i], "label": labels[int(cs.y[i])],
                        "variant": v, "has_lines": bool(kind is None or sum(c[kind] for c in counts) > 0),
                        "lines_removed": 0 if kind is None else sum(c[kind] for c in counts),
                        "truncated": truncated, "pred": labels[int(p[v_i].argmax())],
                        "p_pred_full": float(p[v_i, pred_full]), "p_true": float(p[v_i, int(cs.y[i])])})

    df = pd.DataFrame(out)
    full = df[df["variant"] == "full"].set_index("unit_key")
    df["drop_p_pred"] = df["unit_key"].map(full["p_pred_full"]) - df["p_pred_full"]
    df["flipped"] = df["pred"] != df["unit_key"].map(full["pred"])
    df.to_csv(d / "occlusion_per_change.csv", index=False)

    def summarize(sub: pd.DataFrame) -> pd.DataFrame:
        from sklearn.metrics import f1_score
        rows_ = []
        for v in variants:
            x = sub[(sub["variant"] == v) & sub["has_lines"]]
            if not len(x):
                continue
            rows_.append({"variant": v, "changes": len(x), "mean_lines_removed": x["lines_removed"].mean(),
                          "drop_in_p(pred)": x["drop_p_pred"].mean(), "flip_rate": x["flipped"].mean(),
                          "macro_f1": f1_score(x["label"], x["pred"], average="macro", labels=labels, zero_division=0)})
        return pd.DataFrame(rows_).set_index("variant") if rows_ else pd.DataFrame()

    parts = {"all": df, "not truncated": df[~df["truncated"]]}
    parts.update({f"label = {c}": df[df["label"] == c] for c in labels})
    tables = []
    for name, sub in parts.items():
        t = summarize(sub)
        if t.empty:
            print(f"(no changes for subset '{name}')")
            continue
        t.insert(0, "subset", name)
        tables.append(t)
    summary = pd.concat(tables)
    summary.to_csv(d / "occlusion_summary.csv")
    with pd.option_context("display.width", 200):
        for name in ("all", "not truncated"):
            if not (summary["subset"] == name).any():
                continue
            print(f"\n{name}:")
            print(summary[summary["subset"] == name].drop(columns="subset").to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\n{len(pick)} changes; per-class tables in {d / 'occlusion_summary.csv'}\n-> {d}")
