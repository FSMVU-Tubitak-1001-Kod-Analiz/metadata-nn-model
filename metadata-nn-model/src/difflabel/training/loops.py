"""Per-epoch train / evaluate / test loops.

``*_single`` handle ``(X, y, idx)`` batches (03a/03b); ``*_dual`` handle
``(X, y, X_metadata, idx)`` batches (04).  Behaviour is unchanged from the
original notebooks, including two details worth knowing about:

* ``evaluate_*`` steps the LR scheduler on the validation loss.
* AUC is binary (positive class = column 1) for 2 classes and weighted
  one-vs-rest otherwise -- see ``_auc``.
"""

from __future__ import annotations

import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def calculate_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """Fraction of ``outputs`` whose argmax matches the (1-D) ``labels``."""
    _, predicted = torch.max(outputs, 1)
    return (predicted == labels).sum().item() / labels.size(0)


def _scale(scaler, batch_X: torch.Tensor, device: str) -> torch.Tensor:
    return torch.tensor(scaler.transform(batch_X), dtype=torch.float32).to(device)


def _auc(all_labels, all_probabilities) -> float:
    """Binary (2 classes): AUC of column 1. Otherwise weighted one-vs-rest multi-class AUC.

    Identical to the original notebooks for their configurations (3-class single
    pipelines, 2-class gated fusion); it additionally avoids sklearn's error when
    2-class scores are passed to the multi-class code path.
    """
    probs = np.vstack(all_probabilities)
    if probs.shape[1] == 2:
        return roc_auc_score(all_labels, probs[:, 1])
    return roc_auc_score(all_labels, probs, average="weighted", multi_class="ovr")


# ---------------------------------------------------------------------------
# Single-input pipeline
# ---------------------------------------------------------------------------
def train_single(model, loader, scaler, optimizer, criterion, device) -> Tuple[float, float]:
    """One training epoch. Returns ``(avg_loss, avg_accuracy)``."""
    model.train()
    running_loss = running_accuracy = 0.0

    for batch_X, batch_y, _idx in loader:
        batch_X = _scale(scaler, batch_X, device)
        batch_y = torch.argmax(batch_y.to(device), dim=1)

        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_accuracy += calculate_accuracy(outputs, batch_y)

    return running_loss / len(loader), running_accuracy / len(loader)


def evaluate_single(model, loader, scaler, criterion, scheduler, device="cpu"):
    """Validation/test evaluation. Returns ``(loss, acc, precision, recall, f1, inference_time, auc)``."""
    running_loss = running_accuracy = 0.0
    all_labels: List[int] = []
    all_predictions: List[int] = []
    all_probabilities: List[np.ndarray] = []

    start_time = time.time()

    for batch_X, batch_y, _idx in loader:
        batch_X = _scale(scaler, batch_X, device)
        batch_y = torch.argmax(batch_y.to(device), dim=1)

        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        probabilities = F.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        all_labels.extend(batch_y.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())
        all_probabilities.extend(probabilities.cpu().numpy())

        running_loss += loss.item()
        running_accuracy += calculate_accuracy(outputs, batch_y)

    inference_time = time.time() - start_time
    val_loss = running_loss / len(loader)
    val_accuracy = running_accuracy / len(loader)

    precision = precision_score(all_labels, all_predictions, average="weighted")
    recall = recall_score(all_labels, all_predictions, average="weighted")
    f1 = f1_score(all_labels, all_predictions, average="weighted")
    auc = _auc(all_labels, all_probabilities)

    if scheduler is not None:
        scheduler.step(val_loss)

    return val_loss, val_accuracy, precision, recall, f1, inference_time, auc


def test_single(model, loader, scaler, criterion, device):
    """Returns ``(test_loss, test_accuracy, predicted_outputs, true_labels)``."""
    running_loss = running_accuracy = 0.0
    predicted_y, y = [], []

    for batch_X, batch_y, _idx in loader:
        batch_X = _scale(scaler, batch_X, device)
        batch_y = batch_y.to(device)
        batch_y_1d = torch.argmax(batch_y, dim=1)

        outputs = model(batch_X)
        loss = criterion(outputs, batch_y_1d)

        running_loss += loss.item()
        running_accuracy += calculate_accuracy(outputs, batch_y_1d)

        predicted_y.append(outputs)
        y.append(batch_y)

    return (
        running_loss / len(loader),
        running_accuracy / len(loader),
        torch.cat(predicted_y, dim=0),
        torch.cat(y, dim=0),
    )


# ---------------------------------------------------------------------------
# Dual-input (gated fusion) pipeline
# ---------------------------------------------------------------------------
def train_dual(model, loader, scaler, optimizer, criterion, device) -> Tuple[float, float]:
    """One training epoch. Returns ``(avg_loss, avg_accuracy)``."""
    model.train()
    running_loss = running_accuracy = 0.0

    for batch_X, batch_y, batch_X_metadata, _idx in loader:
        batch_X = _scale(scaler, batch_X, device)
        batch_X_metadata = batch_X_metadata.to(device)
        batch_y = torch.argmax(batch_y.to(device), dim=1)

        outputs = model(batch_X, batch_X_metadata)
        loss = criterion(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_accuracy += calculate_accuracy(outputs, batch_y)

    return running_loss / len(loader), running_accuracy / len(loader)


def evaluate_dual(model, loader, scaler, criterion, scheduler, device="cpu"):
    """Validation/test evaluation. Returns ``(loss, acc, precision, recall, f1, inference_time, auc)``."""
    running_loss = running_accuracy = 0.0
    all_labels: List[int] = []
    all_predictions: List[int] = []
    all_probabilities: List[np.ndarray] = []

    start_time = time.time()

    for batch_X, batch_y, batch_X_metadata, _idx in loader:
        batch_X = _scale(scaler, batch_X, device)
        batch_X_metadata = batch_X_metadata.to(device)
        batch_y = torch.argmax(batch_y.to(device), dim=1)

        outputs = model(batch_X, batch_X_metadata)
        loss = criterion(outputs, batch_y)

        probabilities = F.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        all_labels.extend(batch_y.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())
        all_probabilities.extend(probabilities.cpu().numpy())

        running_loss += loss.item()
        running_accuracy += calculate_accuracy(outputs, batch_y)

    inference_time = time.time() - start_time
    val_loss = running_loss / len(loader)
    val_accuracy = running_accuracy / len(loader)

    precision = precision_score(all_labels, all_predictions, average="weighted")
    recall = recall_score(all_labels, all_predictions, average="weighted")
    f1 = f1_score(all_labels, all_predictions, average="weighted")

    auc = _auc(all_labels, all_probabilities)

    if scheduler is not None:
        scheduler.step(val_loss)

    return val_loss, val_accuracy, precision, recall, f1, inference_time, auc


def test_dual(model, loader, scaler, criterion, device):
    """Returns ``(test_loss, test_accuracy, predicted_outputs, true_labels)``."""
    running_loss = running_accuracy = 0.0
    predicted_y, y = [], []

    for batch_X, batch_y, batch_X_metadata, _idx in loader:
        batch_X = _scale(scaler, batch_X, device)
        batch_X_metadata = batch_X_metadata.to(device)
        batch_y = batch_y.to(device)
        batch_y_1d = torch.argmax(batch_y, dim=1)

        outputs = model(batch_X, batch_X_metadata)
        loss = criterion(outputs, batch_y_1d)

        running_loss += loss.item()
        running_accuracy += calculate_accuracy(outputs, batch_y_1d)

        predicted_y.append(outputs)
        y.append(batch_y)

    return (
        running_loss / len(loader),
        running_accuracy / len(loader),
        torch.cat(predicted_y, dim=0),
        torch.cat(y, dim=0),
    )
