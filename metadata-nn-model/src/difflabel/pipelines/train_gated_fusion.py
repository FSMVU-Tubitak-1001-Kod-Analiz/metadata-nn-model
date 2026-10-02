"""Step 4: gated-fusion DNN + baselines, once per cross-validation fold."""

from __future__ import annotations

import pandas as pd

from difflabel.config import GatedFusionTrainConfig, format_path, resolve_path
from difflabel.data.datasets import CodeMetadataDataset
from difflabel.data.labels import add_label_tensor
from difflabel.embeddings.bert import add_metadata_embeddings
from difflabel.evaluation.baselines import run_baselines
from difflabel.evaluation.plots import plot_dnn_confusion_matrix
from difflabel.models import build_model
from difflabel.runtime import empty_cuda_cache, resolve_device, set_seed
from difflabel.training.trainer import DUAL_LOOPS, fit_dnn

from ._common import attach_code_embeddings, keep_columns, n_classes


def load_fold_data(cfg: GatedFusionTrainConfig, fold: int, device: str):
    folds_dir = resolve_path(cfg.folds_dir)
    raw = {split: pd.read_csv(folds_dir / f"fold{fold}_{split}.csv") for split in ("train", "val", "test")}

    df = pd.concat([raw["train"], raw["val"], raw["test"]], axis=0).reset_index(drop=True)

    df = add_label_tensor(df, cfg.label_column)
    df = add_metadata_embeddings(df, cfg.metadata, device)
    df = keep_columns(df, ["index", "label_tensor", "metadata_tensor"])
    df = attach_code_embeddings(df, cfg.embeddings)

    sizes = {split: len(frame) for split, frame in raw.items()}
    print(f"Loaded {sizes['train']} train, {sizes['val']} val, and {sizes['test']} test samples.")

    def pick(split):
        return df[df["index"].isin(raw[split]["index"].values)].reset_index(drop=True)

    empty_cuda_cache()
    return df, pick("train"), pick("val"), pick("test")


def run(cfg: GatedFusionTrainConfig) -> None:
    device = resolve_device(cfg.device)

    for fold in cfg.folds:
        set_seed(cfg.seed)
        print(f"\n{'=' * 60}\nFold {fold}\n{'=' * 60}")

        df, df_train, df_val, df_test = load_fold_data(cfg, fold, device)

        code_dim = len(df_train["embedding"].iloc[0])
        meta_dim = len(df_train["metadata_tensor"].iloc[0])

        test_outputs, y_test = fit_dnn(
            build_model=lambda: build_model("gated_fusion", cfg.model, code_dim=code_dim, meta_dim=meta_dim),
            loop_fns=DUAL_LOOPS,
            dataset_cls=CodeMetadataDataset,
            dataset_kwargs={},
            df=df, df_train=df_train, df_val=df_val, df_test=df_test,
            optim_cfg=cfg.optim,
            device=device,
            model_path=format_path(cfg.outputs.model_path, fold=fold),
            scaler_path=format_path(cfg.outputs.scaler_path, fold=fold),
            log_prefix=f"[Fold {fold}] ",
        )
        plot_dnn_confusion_matrix(test_outputs, y_test, format_path(cfg.outputs.dnn_confusion_png, fold=fold))

        if cfg.baselines.enabled:
            run_baselines(
                df_train, df_val, df_test, cfg.baselines,
                seed=cfg.seed, n_classes=n_classes(df),
                metrics_csv=format_path(cfg.outputs.ml_metrics_csv, fold=fold),
                confusion_png=format_path(cfg.outputs.ml_confusion_png, fold=fold),
            )
