"""Confusion-matrix plots (torch is imported lazily so baselines work without it)."""

from __future__ import annotations

from pathlib import Path


def plot_dnn_confusion_matrix(test_outputs, y_test, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import torch
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    _, predicted = torch.max(test_outputs.detach().cpu(), 1)
    true_labels = torch.argmax(y_test, dim=1).cpu()

    cm = confusion_matrix(true_labels.numpy(), predicted.numpy())
    ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    plt.title("Confusion Matrix for Label Prediction")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, format="png", dpi=300)
    plt.close()
    print(f"Saved {out_path}")


def plot_ml_confusion_matrices(models, X_test, y_test, metrics, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    confusion_matrices = {name: confusion_matrix(y_test, model.predict(X_test)) for name, model in models}

    num_models = len(confusion_matrices)
    rows = (num_models + 2) // 3
    fig, axes = plt.subplots(nrows=rows, ncols=3, figsize=(18, 6 * rows))
    fig.suptitle("Confusion Matrices for Trained Models", fontsize=20)
    flat_axes = axes.flatten()

    for ax, (name, cm) in zip(flat_axes, confusion_matrices.items()):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_title(f"{name} Accuracy:{metrics[name]['Accuracy']}")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

    for ax in flat_axes[num_models:]:
        fig.delaxes(ax)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f"All confusion matrices saved to {out_path}")
