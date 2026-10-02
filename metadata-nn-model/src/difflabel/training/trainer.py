"""Shared DNN training orchestration (scaler fit, epoch loop, best-checkpoint, test report).

The three original scripts repeated this flow verbatim and differed only in the
model class and which loop functions they called; that difference is captured
by :class:`LoopFns`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Tuple

import joblib
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

from difflabel.config import OptimConfig
from difflabel.data.datasets import create_dataloader

from . import loops


@dataclass(frozen=True)
class LoopFns:
    train: Callable
    evaluate: Callable
    test: Callable


SINGLE_LOOPS = LoopFns(loops.train_single, loops.evaluate_single, loops.test_single)
DUAL_LOOPS = LoopFns(loops.train_dual, loops.evaluate_dual, loops.test_dual)


def fit_scaler(full_loader, scaler_path: Path) -> StandardScaler:
    """Fits a StandardScaler on the first element of every batch and saves it.

    NOTE: as in the original notebooks, the loader passed in covers the *whole*
    dataset (train + val + test), so the scaler statistics see the test rows.
    """
    scaler = StandardScaler()
    for batch in full_loader:
        scaler.partial_fit(batch[0])
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    return scaler


def _print_eval(prefix: str, loss, acc, precision, recall, f1, inference_time, auc) -> None:
    print(f"{prefix}Inference Time: {inference_time:.4f} seconds")
    print(f"{prefix}Loss: {loss:.4f}, Accuracy: {acc:.4f}")
    print(f"{prefix}Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    print(f"{prefix}AUC: {auc}")


def fit_dnn(
    *,
    build_model: Callable[[], nn.Module],
    loop_fns: LoopFns,
    dataset_cls,
    dataset_kwargs: dict,
    df: pd.DataFrame,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    optim_cfg: OptimConfig,
    device: str,
    model_path: Path,
    scaler_path: Path,
    log_prefix: str = "",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Trains a DNN, keeps the best-validation-accuracy checkpoint, then tests it.

    Returns ``(test_outputs, y_test)`` for confusion-matrix plotting.
    """
    bs = optim_cfg.batch_size
    train_loader = create_dataloader(dataset_cls, df_train, batch_size=bs, shuffle=True, **dataset_kwargs)
    val_loader = create_dataloader(dataset_cls, df_val, batch_size=bs, shuffle=True, **dataset_kwargs)
    test_loader = create_dataloader(dataset_cls, df_test, batch_size=bs, shuffle=True, **dataset_kwargs)
    scaler_loader = create_dataloader(
        dataset_cls, df, batch_size=optim_cfg.scaler_batch_size, shuffle=True, **dataset_kwargs
    )

    scaler = fit_scaler(scaler_loader, scaler_path)

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=optim_cfg.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=optim_cfg.scheduler_mode, patience=optim_cfg.scheduler_patience, factor=optim_cfg.scheduler_factor
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    best_accuracy = 0.0

    for epoch in range(optim_cfg.num_epochs):
        model.train()
        model.to(device)
        avg_loss, avg_accuracy = loop_fns.train(model, train_loader, scaler, optimizer, criterion, device)
        print(
            f"{log_prefix}Epoch [{epoch + 1}/{optim_cfg.num_epochs}], "
            f"Train Loss: {avg_loss:.4f}, Train Accuracy: {avg_accuracy:.4f}"
        )

        # Validation runs on the CPU, as in the original notebooks.
        model.eval()
        model.to("cpu")
        with torch.no_grad():
            val_metrics = loop_fns.evaluate(model, val_loader, scaler, criterion, scheduler, device="cpu")
        _print_eval(f"{log_prefix}Val ", *val_metrics)

        val_accuracy = val_metrics[1]
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), model_path)

    # Reload the best checkpoint and report final test-set metrics.
    model = build_model().to(device)
    model.load_state_dict(torch.load(model_path))

    with torch.no_grad():
        model.eval()
        model.to("cpu")
        test_metrics = loop_fns.evaluate(model, test_loader, scaler, criterion, scheduler, device="cpu")
    _print_eval(f"{log_prefix}Test ", *test_metrics)

    with torch.no_grad():
        model.to(device)
        model.eval()
        _, test_accuracy, test_outputs, y_test = loop_fns.test(model, test_loader, scaler, criterion, device)
    print(f"{log_prefix}Accuracy on test data: {test_accuracy:.4f}")
    print(test_outputs.shape)

    return test_outputs, y_test
