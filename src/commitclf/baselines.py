"""Classic ML baselines on fixed-length per-change features.

code     = mean of the change's file vectors (same embeddings as the network)
metadata = the run's change-level groups + `path` shares (these models can't see single files)
Trained on train + val (no model selection), evaluated on test. Classes are weighted by
inverse frequency where the model supports it. Scale-sensitive models (logistic regression,
linear SVM, KNN) get a StandardScaler inside their pipeline.

Two reference rows are added: `Majority` (always the most common training class) and
`RepoMajority` (the most common class of the change's repo in training; meaningful only
on the random split, where test repos also appear in training).
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from commitclf.config import resolve
from commitclf.features import FileTable, build_sets, change_metadata, load_parts, pooled_code
from commitclf.metrics import classification_metrics, plot_confusion_matrices
from commitclf.train import default_name

SCALED = {"LogisticRegression", "LinearSVM", "KNN"}


def make_models(seed: int, n_classes: int) -> dict:
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier

    models = {
        "XGBoost": XGBClassifier(tree_method="hist", eval_metric="mlogloss", random_state=seed, n_jobs=-1),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1),
        "CatBoost": CatBoostClassifier(iterations=100, learning_rate=0.1, depth=6, verbose=0,
                                       random_seed=seed, allow_writing_files=False),
        "LightGBM": LGBMClassifier(random_state=seed, verbose=-1),
        "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "NaiveBayes": GaussianNB(),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=seed),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=seed),
        # linear SVM trained with SGD; modified_huber gives probabilities (needed for AUC)
        "LinearSVM": SGDClassifier(loss="modified_huber", alpha=1e-4, max_iter=50, random_state=seed),
    }
    return {k: make_pipeline(StandardScaler(), m) if k in SCALED else m for k, m in models.items()}


def fit(model, X, y, sample_weight):
    if hasattr(model, "steps"):   # pipeline: route the weight to the last step
        step = model.steps[-1][0]
        if step == "kneighborsclassifier":
            return model.fit(X, y)
        return model.fit(X, y, **{f"{step}__sample_weight": sample_weight})
    return model.fit(X, y, sample_weight=sample_weight)


def balanced_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.bincount(y, minlength=n_classes).astype(float)
    return (len(y) / (n_classes * np.maximum(counts, 1)))[y]


def reference_rows(train_units, test_units, y_train, y_test, n_classes):
    majority = np.bincount(y_train, minlength=n_classes).argmax()
    out = {"Majority": np.full(len(y_test), majority)}
    repo_major = (pd.DataFrame({"repo": train_units["full_repo_name"].to_numpy(), "y": y_train})
                  .groupby("repo")["y"].agg(lambda s: s.value_counts().idxmax()))
    out["RepoMajority"] = test_units["full_repo_name"].map(repo_major).fillna(majority).astype(int).to_numpy()
    return {k: (pred, np.eye(n_classes)[pred]) for k, pred in out.items()}


def run(cfg: dict, folds: list[int], spec: dict, features: str | None = None,
        only: list[str] | None = None, name: str | None = None) -> pd.DataFrame:
    features = features or cfg["baselines"]["features"]
    labels, n_classes = cfg["labels"], len(cfg["labels"])
    use_code, use_meta = features in ("code", "both"), features in ("metadata", "both")
    meta_groups = [g for g in spec["metadata"] if g != "message"] or ["change"]
    meta_groups += [g for g in ["path"] if g not in meta_groups]
    meta_groups += ["message"] if "message" in spec["metadata"] else []

    out_dir = resolve(cfg, cfg["baselines"]["out_dir"]) / (name or f"{features}_{default_name(spec).split('_', 1)[1]}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "spec.json").write_text(json.dumps({**spec, "features": features,
                                                   "metadata_groups": meta_groups if use_meta else []}, indent=2))
    print(f"Baselines {out_dir.name}: features={features}, code={spec['code'] if use_code else []}, "
          f"metadata={meta_groups if use_meta else []}")

    code_spec = spec["code"] if use_code else []
    table = FileTable(cfg, code_spec, spec["embed_pool"], with_facts=False) if code_spec else None
    rows = []
    for fold in folds:
        sets = build_sets(cfg, {**spec, "model": "code", "metadata": []}, fold, table) if table else None
        parts = {p: s.units for p, s in sets.items()} if sets else load_parts(cfg, spec["split"], fold, spec["subset"])
        train_units = pd.concat([parts["train"], parts["val"]], ignore_index=True)
        test_units = parts["test"]

        def features_for(units, part_names):
            blocks = []
            if use_code:
                blocks.append(np.concatenate([pooled_code(table, sets[p]) for p in part_names]))
            if use_meta:
                blocks.append(change_metadata(cfg, units, meta_groups, spec["embed_pool"]).numpy())
            return np.concatenate(blocks, axis=1)

        X_train, X_test = features_for(train_units, ["train", "val"]), features_for(test_units, ["test"])
        y_train = train_units["label"].map({c: i for i, c in enumerate(labels)}).to_numpy()
        y_test = test_units["label"].map({c: i for i, c in enumerate(labels)}).to_numpy()
        w_train = balanced_weights(y_train, n_classes)
        print(f"Fold {fold}: X_train={X_train.shape}, X_test={X_test.shape}")

        cms = {}
        for ref, (pred, proba) in reference_rows(train_units, test_units, y_train, y_test, n_classes).items():
            m = classification_metrics(y_test, pred, proba, labels)
            rows.append({"fold": fold, "model": ref, **m, "fit_s": 0.0, "predict_s": 0.0})
            print(f"  {ref:<20} acc={m['accuracy']:.4f} f1={m['f1']:.4f}")

        rng = np.random.default_rng(cfg["seed"] + fold)
        skip = set(cfg["baselines"].get("skip", {}).get(features, []))
        for model_name, model in make_models(cfg["seed"], n_classes).items():
            if (only and model_name not in only) or (not only and model_name in skip):
                continue
            Xf, yf, wf = X_train, y_train, w_train
            if model_name == "KNN" and len(X_train) > cfg["baselines"].get("knn_max_train", 30000):
                keep = rng.choice(len(X_train), cfg["baselines"]["knn_max_train"], replace=False)
                Xf, yf, wf = X_train[keep], y_train[keep], w_train[keep]
            t0 = time.time()
            fit(model, Xf, yf, wf)
            fit_time = time.time() - t0
            t0 = time.time()
            proba = np.asarray(model.predict_proba(X_test))
            predict_time = time.time() - t0
            pred = proba.argmax(1)

            m = classification_metrics(y_test, pred, proba, labels)
            rows.append({"fold": fold, "model": model_name, **m, "fit_s": fit_time, "predict_s": predict_time})
            cms[model_name] = (y_test, pred)
            print(f"  {model_name:<20} acc={m['accuracy']:.4f} f1={m['f1']:.4f} auc={m['auc']:.4f} "
                  f"({fit_time:.0f}s)")
            pd.DataFrame(rows).to_csv(out_dir / "metrics.csv", index=False)  # save as we go

        pd.DataFrame(rows).to_csv(out_dir / "metrics.csv", index=False)
        plot_confusion_matrices(cms, labels, out_dir / f"confusion_matrices_fold{fold}.png")

    df = pd.DataFrame(rows)
    agg = df.drop(columns="fold").groupby("model", sort=False).agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg.to_csv(out_dir / "summary.csv")
    print(agg[["accuracy_mean", "f1_mean", "auc_mean"]].to_string(float_format=lambda v: f"{v:.4f}"))
    return df
