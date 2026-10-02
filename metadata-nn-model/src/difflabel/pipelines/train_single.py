"""Step 3a / 3b: single-pipeline DNN (+ classical baselines).

``pipeline = "code"``     -> concatenated content_before + patch_before CodeBERT embeddings
``pipeline = "metadata"`` -> BERT embedding of the configured metadata columns
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from difflabel.config import SingleTrainConfig, resolve_path
from difflabel.data.datasets import SingleFeatureDataset
from difflabel.data.labels import add_label_tensor
from difflabel.embeddings.bert import add_metadata_embeddings
from difflabel.evaluation.baselines import run_baselines
from difflabel.evaluation.plots import plot_dnn_confusion_matrix
from difflabel.models import build_model
from difflabel.runtime import empty_cuda_cache, resolve_device, set_seed
from difflabel.training.trainer import SINGLE_LOOPS, fit_dnn

from ._common import attach_code_embeddings, keep_columns, n_classes

FEATURE_COLUMN = {"code": "embedding", "metadata": "metadata_tensor"}


def load_data(cfg: SingleTrainConfig, device: str) -> pd.DataFrame:
    df = pd.read_csv(resolve_path(cfg.input_csv))
    df = add_label_tensor(df, cfg.label_column)

    if cfg.pipeline == "code":
        df = attach_code_embeddings(df, cfg.embeddings)
    else:
        df = add_metadata_embeddings(df, cfg.metadata, device)

    return keep_columns(df, ["index", "label_tensor", FEATURE_COLUMN[cfg.pipeline]])


def split_by_index(df: pd.DataFrame, cfg: SingleTrainConfig):
    X_index = df[["index"]]
    idx_train, idx_holdout = train_test_split(X_index, test_size=cfg.split.test_size, random_state=cfg.seed)
    idx_test, idx_val = train_test_split(idx_holdout, test_size=cfg.split.val_fraction_of_holdout, random_state=cfg.seed)

    def pick(idx):
        return df[df["index"].isin(idx["index"].values)].reset_index(drop=True)

    return pick(idx_train), pick(idx_val), pick(idx_test)


def run(cfg: SingleTrainConfig) -> None:
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    feature_column = FEATURE_COLUMN[cfg.pipeline]

    df = load_data(cfg, device)
    df_train, df_val, df_test = split_by_index(df, cfg)
    empty_cuda_cache()

    feature_dim = len(df_train[feature_column].iloc[0])
    kind = cfg.pipeline

    test_outputs, y_test = fit_dnn(
        build_model=lambda: build_model(
            kind, cfg.model,
            code_dim=feature_dim if kind == "code" else None,
            meta_dim=feature_dim if kind == "metadata" else None,
        ),
        loop_fns=SINGLE_LOOPS,
        dataset_cls=SingleFeatureDataset,
        dataset_kwargs={"feature_column": feature_column},
        df=df, df_train=df_train, df_val=df_val, df_test=df_test,
        optim_cfg=cfg.optim,
        device=device,
        model_path=resolve_path(cfg.outputs.model_path),
        scaler_path=resolve_path(cfg.outputs.scaler_path),
    )
    plot_dnn_confusion_matrix(test_outputs, y_test, resolve_path(cfg.outputs.dnn_confusion_png))

    if cfg.baselines.enabled:
        run_baselines(
            df_train, df_val, df_test, cfg.baselines,
            seed=cfg.seed, n_classes=n_classes(df),
            metrics_csv=resolve_path(cfg.outputs.ml_metrics_csv),
            confusion_png=resolve_path(cfg.outputs.ml_confusion_png),
        )
