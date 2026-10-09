from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray,
                           labels: list[str] | None = None) -> dict:
    """Accuracy, macro P/R/F1, AUC (binary or one-vs-rest macro) and per-class F1."""
    y_proba = np.asarray(y_proba, dtype="float64")
    y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)  # some models' float32 rows don't sum to exactly 1
    classes = np.arange(y_proba.shape[1])
    try:
        if len(classes) == 2:
            auc = roc_auc_score(y_true, y_proba[:, 1])
        else:
            auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro", labels=classes)
    except ValueError:   # a class missing from y_true (tiny test sets)
        auc = float("nan")
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", labels=classes, zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", labels=classes, zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", labels=classes, zero_division=0),
        "auc": auc,
    }
    per_class = f1_score(y_true, y_pred, average=None, labels=classes, zero_division=0)
    names = labels or [str(c) for c in classes]
    out.update({f"f1_{n}": v for n, v in zip(names, per_class)})
    return out


def plot_confusion_matrices(results: dict[str, tuple[np.ndarray, np.ndarray]],
                            labels: list[str], path: Path, ncols: int = 3) -> None:
    """results: {title: (y_true, y_pred)} -> one PNG with a grid of matrices."""
    n = len(results)
    if n == 0:
        return
    ncols = min(ncols, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)
    for ax, (title, (y_true, y_pred)) in zip(axes.flat, results.items()):
        cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
        ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{title}\nacc={accuracy_score(y_true, y_pred):.3f} "
                     f"F1={f1_score(y_true, y_pred, average='macro', labels=range(len(labels))):.3f}")
    for ax in list(axes.flat)[n:]:
        ax.remove()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
