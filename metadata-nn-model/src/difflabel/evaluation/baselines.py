"""Nine classical-ML baselines, configured entirely from JSON.

Constructor arguments for each model come from ``baselines.models`` in the
config.  Two things are injected here because they're derived rather than
chosen: the random seed and (where the library wants it) the class count.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

from difflabel.config import BaselinesConfig, ConfigError

METRIC_HEADER = ["Model Name", "Accuracy", "Precision", "Recall", "F1 Score", "AUC", "Prediction Time (s)"]


def _xgboost(params, seed, n_classes):
    import xgboost as xgb
    return xgb.XGBClassifier(**{"num_class": n_classes, "random_state": seed, **params})


def _lightgbm(params, seed, n_classes):
    from lightgbm import LGBMClassifier
    extra = {"num_class": n_classes} if n_classes > 2 else {}  # binary LightGBM rejects num_class
    return LGBMClassifier(**{**extra, "random_state": seed, **params})


def _catboost(params, seed, n_classes):
    from catboost import CatBoostClassifier
    # allow_writing_files=False stops CatBoost littering the cwd with catboost_info/
    return CatBoostClassifier(**{"random_seed": seed, "allow_writing_files": False, **params})


def _sklearn(cls_path: str, takes_seed: bool = True) -> Callable:
    def factory(params, seed, n_classes):
        module, name = cls_path.rsplit(".", 1)
        cls = getattr(__import__(module, fromlist=[name]), name)
        seed_kw = {"random_state": seed} if takes_seed else {}
        return cls(**{**seed_kw, **params})
    return factory


# Names usable as keys in the config's `baselines.models`.
MODEL_FACTORIES: Dict[str, Callable] = {
    "XGBoost": _xgboost,
    "RandomForest": _sklearn("sklearn.ensemble.RandomForestClassifier"),
    "CatBoost": _catboost,
    "LightGBM": _lightgbm,
    "KNN": _sklearn("sklearn.neighbors.KNeighborsClassifier", takes_seed=False),
    "NaiveBayes": _sklearn("sklearn.naive_bayes.GaussianNB", takes_seed=False),
    "LogisticRegression": _sklearn("sklearn.linear_model.LogisticRegression"),
    "AdaBoost": _sklearn("sklearn.ensemble.AdaBoostClassifier"),
    "SVM": _sklearn("sklearn.svm.SVC"),
}


def build_models(cfg: BaselinesConfig, seed: int, n_classes: int) -> List[Tuple[str, Any]]:
    unknown = sorted(set(cfg.models) - set(MODEL_FACTORIES))
    if unknown:
        raise ConfigError(f"baselines.models: unknown model(s) {unknown}. Available: {sorted(MODEL_FACTORIES)}")
    return [(name, MODEL_FACTORIES[name](params, seed, n_classes)) for name, params in cfg.models.items()]


def _class_id(label) -> int:
    """One-hot torch tensor / array -> class id; scalars pass through unchanged."""
    if hasattr(label, "numpy"):
        return label.numpy().argmax()
    if np.ndim(label) > 0:
        return np.argmax(label)
    return label


def _to_class_ids(df: pd.DataFrame) -> np.ndarray:
    return df["label_tensor"].apply(_class_id).values


def run_baselines(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    cfg: BaselinesConfig,
    seed: int,
    n_classes: int,
    metrics_csv: Path,
    confusion_png: Path,
) -> None:
    """Fits every configured baseline, appends its metrics to ``metrics_csv``, plots confusion matrices."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    from .plots import plot_ml_confusion_matrices

    train_df = pd.concat([df_train, df_val], axis=0, ignore_index=True) if cfg.train_on_train_and_val else df_train

    X_train = np.vstack(train_df[cfg.feature_column])
    y_train = _to_class_ids(train_df)
    X_test = np.vstack(df_test[cfg.feature_column])
    y_test = _to_class_ids(df_test)

    models = build_models(cfg, seed, n_classes)

    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_csv, mode="w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(METRIC_HEADER)

    metrics: Dict[str, Dict[str, float]] = {}
    for name, model in models:
        model.fit(X_train, y_train)

        start = time.time()
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        prediction_time = time.time() - start

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="macro")
        recall = recall_score(y_test, y_pred, average="macro")
        f1 = f1_score(y_test, y_pred, average="macro")
        if len(np.unique(y_test)) == 2:
            auc = roc_auc_score(y_test, y_proba[:, 1])
        else:
            auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")

        metrics[name] = {"Accuracy": accuracy, "Precision": precision, "Recall": recall,
                         "F1 Score": f1, "AUC": auc, "Prediction Time (s)": prediction_time}

        print(f"Metrics for {name}:")
        for key, value in metrics[name].items():
            print(f"  {key}: {value:.4f}")

        with open(metrics_csv, mode="a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow([name, accuracy, precision, recall, f1, auc, prediction_time])

    print(f"\nMetrics saved incrementally to {metrics_csv}")
    plot_ml_confusion_matrices(models, X_test, y_test, metrics, confusion_png)
