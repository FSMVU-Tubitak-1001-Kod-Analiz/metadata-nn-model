"""
common.py
=========
Shared building blocks used by the three training scripts:

    03a_train_single_pipeline_code_embeddings.py
    03b_train_single_pipeline_metadata_embeddings.py
    04_train_gated_fusion_folds.py

This module intentionally contains no experiment-specific configuration
(paths, hyperparameters, etc.) -- those live at the top of each numbered
script so you can tell at a glance what a given run actually did.

Everything here was lifted directly out of the original notebooks; the only
changes are cosmetic (docstrings, type hints, removing dead/commented-out
code) and de-duplication (the same seed/accuracy/dataset code was copy-pasted
across all three original notebooks).
"""

from __future__ import annotations

import os
import random
import time
from typing import Iterable, List, Sequence, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Seeds python/numpy/torch (+ CUDA) for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Metadata text -> BERT embedding helpers
# (used by the "metadata" pipeline and the gated-fusion pipeline)
# ---------------------------------------------------------------------------
def get_bert_embedding_tensor(text, tokenizer, model, max_length: int = 128) -> torch.Tensor:
    """Embeds a single string with a HF encoder, returning its [CLS] vector (on CPU)."""
    if pd.isna(text) or text is None:
        return torch.zeros(model.config.hidden_size, dtype=torch.float32)
    inputs = tokenizer(
        str(text),
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    if torch.cuda.is_available():
        inputs = {key: val.cuda() for key, val in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().cpu()


def embed_columns_with_bert(
    df: pd.DataFrame,
    columns_to_embed: Union[str, Sequence[str]],
    tokenizer,
    model,
    output_column_name: str = "metadata_tensor",
    separator: str = " | ",
) -> pd.DataFrame:
    """
    Generates BERT embeddings for one or more columns and adds them to the DataFrame.

    Args:
        df: The input DataFrame.
        columns_to_embed: A single column name, or a list of column names whose
            values get concatenated (joined by `separator`) before being embedded
            together as one piece of text.
        tokenizer: The BERT tokenizer.
        model: The pre-trained BERT model.
        output_column_name: Name of the new column to store embeddings in.
        separator: String used to join multiple columns' values.

    Returns:
        The DataFrame with the new embedding column added.
    """
    if isinstance(columns_to_embed, str):
        columns_to_embed = [columns_to_embed]

    for col in columns_to_embed:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the DataFrame.")

    print(f"Generating embeddings from column(s): {', '.join(columns_to_embed)}")

    def create_combined_text(row) -> str:
        values = [str(row[col]) for col in columns_to_embed]
        return separator.join(values)

    df[output_column_name] = df.apply(
        lambda row: get_bert_embedding_tensor(create_combined_text(row), tokenizer, model),
        axis=1,
    )
    return df


# ---------------------------------------------------------------------------
# Accuracy metric shared by every training loop
# ---------------------------------------------------------------------------
def calculate_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """Fraction of `outputs` whose argmax matches the (already 1D) `labels`."""
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    return correct / labels.size(0)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
class SingleFeatureDataset(Dataset):
    """
    Dataset for the single-input-pipeline scripts (03a / 03b).

    Both original notebooks used an identical class (called `CodeDataset`)
    that differed only in *which* dataframe column it read as the feature
    vector ('embedding' for the code pipeline, 'metadata_tensor' for the
    metadata pipeline). That's parameterized here via `feature_column`.

    Returns (feature_tensor, label_tensor, index) per item, matching the
    original `for batch_X, batch_y, idx in loader` unpacking.
    """

    def __init__(self, df: pd.DataFrame, feature_column: str):
        self.df = df
        self.feature_column = feature_column
        self.id_list = df["index"].values

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        index = self.id_list[idx]
        feature_tensor = self.df.loc[self.df["index"] == index, self.feature_column].iloc[0]
        label_tensor = self.df.loc[self.df["index"] == index, "label_tensor"].iloc[0]
        return feature_tensor, label_tensor, index


class CodeMetadataDataset(Dataset):
    """
    Dataset for the dual-input gated-fusion script (04).

    Returns (embedding, label_tensor, metadata_tensor, index) per item,
    matching `for batch_X, batch_y, batch_X_metadata, idx in loader`.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.id_list = df["index"].values

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        index = self.id_list[idx]
        embedding = self.df.loc[self.df["index"] == index, "embedding"].iloc[0]
        metadata_tensor = self.df.loc[self.df["index"] == index, "metadata_tensor"].iloc[0]
        label_tensor = self.df.loc[self.df["index"] == index, "label_tensor"].iloc[0]
        return embedding, label_tensor, metadata_tensor, index


def create_dataloader(dataset_cls, df: pd.DataFrame, batch_size: int = 32, shuffle: bool = True, **dataset_kwargs) -> DataLoader:
    """Small convenience wrapper mirroring the notebooks' `create_dataloader`."""
    dataset = dataset_cls(df, **dataset_kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ---------------------------------------------------------------------------
# Model: single-input pipelines (used by 03a and 03b)
# ---------------------------------------------------------------------------
#
# IMPORTANT: these are two *different* architectures, not one class reused
# with different inputs. In the original notebooks both scripts start from
# the same `CodeDiffNN` class, but:
#   - the "code" notebook (03a) has the 3-layer *code* branch active
#     (code_fc1/2/3, widths from hidden_sizes_embedding_pipeline) and the
#     metadata branch fully commented out.
#   - the "metadata" notebook (03b) has the 2-layer *metadata* branch active
#     (metadata_fc1/2, widths from hidden_sizes_metadata_pipeline) and the
#     code branch fully commented out.
# Both are kept as separate classes here (rather than unified into one
# parametric class) so that layer names/shapes -- and therefore any
# checkpoints already trained from the original notebooks -- stay
# load-compatible via `model.load_state_dict(...)`.


class CodeOnlyDiffNN(nn.Module):
    """Single-pipeline ANN with the *code* branch active. Used by 03a."""

    def __init__(
        self,
        code_input_size: int,
        metadata_input_size: int,  # unused -- kept for call-signature parity with the original notebook
        hidden_sizes_embedding_pipeline: Sequence[int],
        hidden_sizes_metadata_pipeline: Sequence[int],  # unused -- see note above
        hidden_sizes_merged_pipeline: Sequence[int],
        output_size: int,
        dropout_rate: float,
    ):
        super().__init__()

        self.code_fc1 = nn.Linear(code_input_size, hidden_sizes_embedding_pipeline[0])
        self.code_bn1 = nn.BatchNorm1d(hidden_sizes_embedding_pipeline[0])
        self.code_relu1 = nn.ReLU()
        self.code_dropout1 = nn.Dropout(dropout_rate)

        self.code_fc2 = nn.Linear(hidden_sizes_embedding_pipeline[0], hidden_sizes_embedding_pipeline[1])
        self.code_bn2 = nn.BatchNorm1d(hidden_sizes_embedding_pipeline[1])
        self.code_relu2 = nn.ReLU()
        self.code_dropout2 = nn.Dropout(dropout_rate)

        self.code_fc3 = nn.Linear(hidden_sizes_embedding_pipeline[1], hidden_sizes_embedding_pipeline[2])
        self.code_bn3 = nn.BatchNorm1d(hidden_sizes_embedding_pipeline[2])
        self.code_relu3 = nn.ReLU()
        self.code_dropout3 = nn.Dropout(dropout_rate)

        combined_input_size = hidden_sizes_embedding_pipeline[2]
        self.combined_fc1 = nn.Linear(combined_input_size, hidden_sizes_merged_pipeline[0])
        self.combined_bn1 = nn.BatchNorm1d(hidden_sizes_merged_pipeline[0])
        self.combined_relu1 = nn.ReLU()
        self.combined_dropout1 = nn.Dropout(dropout_rate)

        self.combined_fc2 = nn.Linear(hidden_sizes_merged_pipeline[0], hidden_sizes_merged_pipeline[1])
        self.combined_bn2 = nn.BatchNorm1d(hidden_sizes_merged_pipeline[1])
        self.combined_relu2 = nn.ReLU()
        self.combined_dropout2 = nn.Dropout(dropout_rate)

        self.combined_fc3 = nn.Linear(hidden_sizes_merged_pipeline[1], hidden_sizes_merged_pipeline[2])
        self.combined_bn3 = nn.BatchNorm1d(hidden_sizes_merged_pipeline[2])
        self.combined_relu3 = nn.ReLU()
        self.combined_dropout3 = nn.Dropout(dropout_rate)

        self.combined_fc4 = nn.Linear(hidden_sizes_merged_pipeline[2], hidden_sizes_merged_pipeline[3])
        self.combined_bn4 = nn.BatchNorm1d(hidden_sizes_merged_pipeline[3])
        self.combined_relu4 = nn.ReLU()
        self.combined_dropout4 = nn.Dropout(dropout_rate)

        self.fc_out = nn.Linear(hidden_sizes_merged_pipeline[3], output_size)

    def forward(self, code_x: torch.Tensor) -> torch.Tensor:
        code_x = self.code_fc1(code_x)
        code_x = self.code_bn1(code_x)
        code_x = self.code_relu1(code_x)
        code_x = self.code_dropout1(code_x)

        code_x = self.code_fc2(code_x)
        code_x = self.code_bn2(code_x)
        code_x = self.code_relu2(code_x)
        code_x = self.code_dropout2(code_x)

        code_x = self.code_fc3(code_x)
        code_x = self.code_bn3(code_x)
        code_x = self.code_relu3(code_x)
        code_x = self.code_dropout3(code_x)

        combined_x = code_x

        combined_x = self.combined_fc1(combined_x)
        combined_x = self.combined_bn1(combined_x)
        combined_x = self.combined_relu1(combined_x)
        combined_x = self.combined_dropout1(combined_x)

        combined_x = self.combined_fc2(combined_x)
        combined_x = self.combined_bn2(combined_x)
        combined_x = self.combined_relu2(combined_x)
        combined_x = self.combined_dropout2(combined_x)

        combined_x = self.combined_fc3(combined_x)
        combined_x = self.combined_bn3(combined_x)
        combined_x = self.combined_relu3(combined_x)
        combined_x = self.combined_dropout3(combined_x)

        combined_x = self.combined_fc4(combined_x)
        combined_x = self.combined_bn4(combined_x)
        combined_x = self.combined_relu4(combined_x)
        combined_x = self.combined_dropout4(combined_x)

        return self.fc_out(combined_x)


class MetadataOnlyDiffNN(nn.Module):
    """Single-pipeline ANN with the *metadata* branch active. Used by 03b."""

    def __init__(
        self,
        code_input_size: int,  # unused -- kept for call-signature parity with the original notebook
        metadata_input_size: int,
        hidden_sizes_embedding_pipeline: Sequence[int],  # unused -- see note above
        hidden_sizes_metadata_pipeline: Sequence[int],
        hidden_sizes_merged_pipeline: Sequence[int],
        output_size: int,
        dropout_rate: float,
    ):
        super().__init__()

        self.metadata_fc1 = nn.Linear(metadata_input_size, hidden_sizes_metadata_pipeline[0])
        self.metadata_bn1 = nn.BatchNorm1d(hidden_sizes_metadata_pipeline[0])
        self.metadata_relu1 = nn.ReLU()
        self.metadata_dropout1 = nn.Dropout(dropout_rate)

        self.metadata_fc2 = nn.Linear(hidden_sizes_metadata_pipeline[0], hidden_sizes_metadata_pipeline[1])
        self.metadata_bn2 = nn.BatchNorm1d(hidden_sizes_metadata_pipeline[1])
        self.metadata_relu2 = nn.ReLU()
        self.metadata_dropout2 = nn.Dropout(dropout_rate)

        combined_input_size = hidden_sizes_metadata_pipeline[1]
        self.combined_fc1 = nn.Linear(combined_input_size, hidden_sizes_merged_pipeline[0])
        self.combined_bn1 = nn.BatchNorm1d(hidden_sizes_merged_pipeline[0])
        self.combined_relu1 = nn.ReLU()
        self.combined_dropout1 = nn.Dropout(dropout_rate)

        self.combined_fc2 = nn.Linear(hidden_sizes_merged_pipeline[0], hidden_sizes_merged_pipeline[1])
        self.combined_bn2 = nn.BatchNorm1d(hidden_sizes_merged_pipeline[1])
        self.combined_relu2 = nn.ReLU()
        self.combined_dropout2 = nn.Dropout(dropout_rate)

        self.combined_fc3 = nn.Linear(hidden_sizes_merged_pipeline[1], hidden_sizes_merged_pipeline[2])
        self.combined_bn3 = nn.BatchNorm1d(hidden_sizes_merged_pipeline[2])
        self.combined_relu3 = nn.ReLU()
        self.combined_dropout3 = nn.Dropout(dropout_rate)

        self.combined_fc4 = nn.Linear(hidden_sizes_merged_pipeline[2], hidden_sizes_merged_pipeline[3])
        self.combined_bn4 = nn.BatchNorm1d(hidden_sizes_merged_pipeline[3])
        self.combined_relu4 = nn.ReLU()
        self.combined_dropout4 = nn.Dropout(dropout_rate)

        self.fc_out = nn.Linear(hidden_sizes_merged_pipeline[3], output_size)

    def forward(self, metadata_x: torch.Tensor) -> torch.Tensor:
        metadata_x = self.metadata_fc1(metadata_x)
        metadata_x = self.metadata_bn1(metadata_x)
        metadata_x = self.metadata_relu1(metadata_x)
        metadata_x = self.metadata_dropout1(metadata_x)

        metadata_x = self.metadata_fc2(metadata_x)
        metadata_x = self.metadata_bn2(metadata_x)
        metadata_x = self.metadata_relu2(metadata_x)
        metadata_x = self.metadata_dropout2(metadata_x)

        combined_x = metadata_x

        combined_x = self.combined_fc1(combined_x)
        combined_x = self.combined_bn1(combined_x)
        combined_x = self.combined_relu1(combined_x)
        combined_x = self.combined_dropout1(combined_x)

        combined_x = self.combined_fc2(combined_x)
        combined_x = self.combined_bn2(combined_x)
        combined_x = self.combined_relu2(combined_x)
        combined_x = self.combined_dropout2(combined_x)

        combined_x = self.combined_fc3(combined_x)
        combined_x = self.combined_bn3(combined_x)
        combined_x = self.combined_relu3(combined_x)
        combined_x = self.combined_dropout3(combined_x)

        combined_x = self.combined_fc4(combined_x)
        combined_x = self.combined_bn4(combined_x)
        combined_x = self.combined_relu4(combined_x)
        combined_x = self.combined_dropout4(combined_x)

        return self.fc_out(combined_x)


# ---------------------------------------------------------------------------
# Model: dual-input gated-fusion pipeline (used by 04)
# ---------------------------------------------------------------------------
class GatedFusionDiffNN(nn.Module):
    """The gated-fusion ANN architecture from the "Rewrite ... exp_architecture" notebook."""

    def __init__(
        self,
        code_input_size: int,
        metadata_input_size: int,
        hidden_sizes_embedding_pipeline: Sequence[int],
        hidden_sizes_metadata_pipeline: Sequence[int],
        hidden_sizes_merged_pipeline: Sequence[int],
        output_size: int,
        dropout_rate: float,
        modality_dropout_p: float = 0.1,
        fusion_dim: int = None,
    ):
        super().__init__()

        self.modality_dropout_p = modality_dropout_p

        # Pipeline for code embeddings
        self.code_fc1 = nn.Linear(code_input_size, hidden_sizes_embedding_pipeline[0])
        self.code_bn1 = nn.BatchNorm1d(hidden_sizes_embedding_pipeline[0])
        self.code_relu1 = nn.ReLU()
        self.code_dropout1 = nn.Dropout(dropout_rate)

        self.code_fc2 = nn.Linear(hidden_sizes_embedding_pipeline[0], hidden_sizes_embedding_pipeline[1])
        self.code_bn2 = nn.BatchNorm1d(hidden_sizes_embedding_pipeline[1])
        self.code_relu2 = nn.ReLU()
        self.code_dropout2 = nn.Dropout(dropout_rate)

        self.code_fc3 = nn.Linear(hidden_sizes_embedding_pipeline[1], hidden_sizes_embedding_pipeline[2])
        self.code_bn3 = nn.BatchNorm1d(hidden_sizes_embedding_pipeline[2])
        self.code_relu3 = nn.ReLU()
        self.code_dropout3 = nn.Dropout(dropout_rate)

        # Pipeline for metadata
        self.metadata_fc1 = nn.Linear(metadata_input_size, hidden_sizes_metadata_pipeline[0])
        self.metadata_bn1 = nn.BatchNorm1d(hidden_sizes_metadata_pipeline[0])
        self.metadata_relu1 = nn.ReLU()
        self.metadata_dropout1 = nn.Dropout(dropout_rate)

        self.metadata_fc2 = nn.Linear(hidden_sizes_metadata_pipeline[0], hidden_sizes_metadata_pipeline[1])
        self.metadata_bn2 = nn.BatchNorm1d(hidden_sizes_metadata_pipeline[1])
        self.metadata_relu2 = nn.ReLU()
        self.metadata_dropout2 = nn.Dropout(dropout_rate)

        # --- Gated fusion ---
        code_dim = hidden_sizes_embedding_pipeline[2]
        meta_dim = hidden_sizes_metadata_pipeline[1]

        self.fusion_dim = fusion_dim if fusion_dim is not None else code_dim

        self.code_proj = nn.Linear(code_dim, self.fusion_dim)
        self.meta_proj = nn.Linear(meta_dim, self.fusion_dim)
        self.gate_fc = nn.Linear(code_dim + meta_dim, self.fusion_dim)

        # Combining stage
        self.combined_fc1 = nn.Linear(self.fusion_dim, hidden_sizes_merged_pipeline[0])
        self.combined_bn1 = nn.BatchNorm1d(hidden_sizes_merged_pipeline[0])
        self.combined_relu1 = nn.ReLU()
        self.combined_dropout1 = nn.Dropout(dropout_rate)

        self.combined_fc2 = nn.Linear(hidden_sizes_merged_pipeline[0], hidden_sizes_merged_pipeline[1])
        self.combined_bn2 = nn.BatchNorm1d(hidden_sizes_merged_pipeline[1])
        self.combined_relu2 = nn.ReLU()
        self.combined_dropout2 = nn.Dropout(dropout_rate)

        self.combined_fc3 = nn.Linear(hidden_sizes_merged_pipeline[1], hidden_sizes_merged_pipeline[2])
        self.combined_bn3 = nn.BatchNorm1d(hidden_sizes_merged_pipeline[2])
        self.combined_relu3 = nn.ReLU()
        self.combined_dropout3 = nn.Dropout(dropout_rate)

        self.combined_fc4 = nn.Linear(hidden_sizes_merged_pipeline[2], hidden_sizes_merged_pipeline[3])
        self.combined_bn4 = nn.BatchNorm1d(hidden_sizes_merged_pipeline[3])
        self.combined_relu4 = nn.ReLU()
        self.combined_dropout4 = nn.Dropout(dropout_rate)

        self.fc_out = nn.Linear(hidden_sizes_merged_pipeline[3], output_size)

    def forward(self, code_x: torch.Tensor, metadata_x: torch.Tensor) -> torch.Tensor:
        code_x = self.code_fc1(code_x)
        code_x = self.code_bn1(code_x)
        code_x = self.code_relu1(code_x)
        code_x = self.code_dropout1(code_x)

        code_x = self.code_fc2(code_x)
        code_x = self.code_bn2(code_x)
        code_x = self.code_relu2(code_x)
        code_x = self.code_dropout2(code_x)

        code_x = self.code_fc3(code_x)
        code_x = self.code_bn3(code_x)
        code_x = self.code_relu3(code_x)
        code_x = self.code_dropout3(code_x)

        metadata_x = self.metadata_fc1(metadata_x)
        metadata_x = self.metadata_bn1(metadata_x)
        metadata_x = self.metadata_relu1(metadata_x)
        metadata_x = self.metadata_dropout1(metadata_x)

        metadata_x = self.metadata_fc2(metadata_x)
        metadata_x = self.metadata_bn2(metadata_x)
        metadata_x = self.metadata_relu2(metadata_x)
        metadata_x = self.metadata_dropout2(metadata_x)

        # Gated fusion: project both modalities into a shared space and
        # let a learned gate decide how much to trust each one.
        h_code = torch.tanh(self.code_proj(code_x))
        h_meta = torch.tanh(self.meta_proj(metadata_x))

        gate = torch.sigmoid(self.gate_fc(torch.cat((code_x, metadata_x), dim=1)))
        print("Average Gate (Code reliance weight):", gate.mean().item())

        fused_x = gate * h_code + (1 - gate) * h_meta

        combined_x = self.combined_fc1(fused_x)
        combined_x = self.combined_bn1(combined_x)
        combined_x = self.combined_relu1(combined_x)
        combined_x = self.combined_dropout1(combined_x)

        combined_x = self.combined_fc2(combined_x)
        combined_x = self.combined_bn2(combined_x)
        combined_x = self.combined_relu2(combined_x)
        combined_x = self.combined_dropout2(combined_x)

        combined_x = self.combined_fc3(combined_x)
        combined_x = self.combined_bn3(combined_x)
        combined_x = self.combined_relu3(combined_x)
        combined_x = self.combined_dropout3(combined_x)

        combined_x = self.combined_fc4(combined_x)
        combined_x = self.combined_bn4(combined_x)
        combined_x = self.combined_relu4(combined_x)
        combined_x = self.combined_dropout4(combined_x)

        return self.fc_out(combined_x)


# ---------------------------------------------------------------------------
# Train / evaluate / test loops -- single-input pipeline (03a / 03b)
# ---------------------------------------------------------------------------
def train_single(model, loader, scaler, optimizer, criterion, device) -> tuple[float, float]:
    """One training epoch for the single-input pipeline. Returns (avg_loss, avg_accuracy)."""
    model.train()
    running_loss = 0.0
    running_accuracy = 0.0

    for batch_X, batch_y, idx in loader:
        batch_X = scaler.transform(batch_X)
        batch_X = torch.tensor(batch_X, dtype=torch.float32)

        batch_y = batch_y.to(device)
        batch_X = batch_X.to(device)
        batch_y = torch.argmax(batch_y, dim=1)

        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accuracy = calculate_accuracy(outputs, batch_y)
        running_loss += loss.item()
        running_accuracy += accuracy

    return running_loss / len(loader), running_accuracy / len(loader)


def evaluate_single(model, loader, scaler, criterion, scheduler, device="cpu"):
    """
    Runs validation/test-time evaluation (no gradient updates) for the
    single-input pipeline. Also steps `scheduler` on the resulting val loss,
    exactly as the original notebooks did.

    Returns (val_loss, val_accuracy, precision, recall, f1, inference_time, auc).
    """
    running_loss = 0.0
    running_accuracy = 0.0

    all_labels: List[int] = []
    all_predictions: List[int] = []
    all_probabilities: List[np.ndarray] = []

    start_time = time.time()

    for batch_X, batch_y, idx in loader:
        batch_X = scaler.transform(batch_X)
        batch_X = torch.tensor(batch_X, dtype=torch.float32).to(device)

        batch_y = batch_y.to(device)
        batch_y = torch.argmax(batch_y, dim=1)

        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        accuracy = calculate_accuracy(outputs, batch_y)

        probabilities = F.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        all_labels.extend(batch_y.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())
        all_probabilities.extend(probabilities.cpu().numpy())

        running_loss += loss.item()
        running_accuracy += accuracy

    inference_time = time.time() - start_time

    val_loss = running_loss / len(loader)
    val_accuracy = running_accuracy / len(loader)

    precision = precision_score(all_labels, all_predictions, average="weighted")
    recall = recall_score(all_labels, all_predictions, average="weighted")
    f1 = f1_score(all_labels, all_predictions, average="weighted")
    auc = roc_auc_score(all_labels, all_probabilities, average="weighted", multi_class="ovr")

    if scheduler is not None:
        scheduler.step(val_loss)

    return val_loss, val_accuracy, precision, recall, f1, inference_time, auc


def test_single(model, loader, scaler, criterion, device):
    """
    Runs the model over `loader` and returns raw outputs/labels for later
    analysis (e.g. a confusion matrix), plus loss/accuracy.

    Returns (test_loss, test_accuracy, predicted_outputs, true_labels).
    """
    running_loss = 0.0
    running_accuracy = 0.0
    predicted_y = []
    y = []

    for batch_X, batch_y, idx in loader:
        batch_X = scaler.transform(batch_X)
        batch_X = torch.tensor(batch_X, dtype=torch.float32)

        batch_y = batch_y.to(device)
        batch_X = batch_X.to(device)
        batch_y_1d = torch.argmax(batch_y, dim=1)

        outputs = model(batch_X)
        loss = criterion(outputs, batch_y_1d)

        accuracy = calculate_accuracy(outputs, batch_y_1d)
        running_loss += loss.item()
        running_accuracy += accuracy

        predicted_y.append(outputs)
        y.append(batch_y)

    test_loss = running_loss / len(loader)
    test_accuracy = running_accuracy / len(loader)

    return test_loss, test_accuracy, torch.cat(predicted_y, dim=0), torch.cat(y, dim=0)


# ---------------------------------------------------------------------------
# Train / evaluate / test loops -- dual-input gated-fusion pipeline (04)
# ---------------------------------------------------------------------------
def train_dual(model, loader, scaler, optimizer, criterion, device) -> tuple[float, float]:
    """One training epoch for the gated-fusion pipeline. Returns (avg_loss, avg_accuracy)."""
    model.train()
    running_loss = 0.0
    running_accuracy = 0.0

    for batch_X, batch_y, batch_X_metadata, idx in loader:
        batch_X = scaler.transform(batch_X)
        batch_X = torch.tensor(batch_X, dtype=torch.float32)

        batch_y = batch_y.to(device)
        batch_X = batch_X.to(device)
        batch_X_metadata = batch_X_metadata.to(device)
        batch_y = torch.argmax(batch_y, dim=1)

        outputs = model(batch_X, batch_X_metadata)
        loss = criterion(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accuracy = calculate_accuracy(outputs, batch_y)
        running_loss += loss.item()
        running_accuracy += accuracy

    return running_loss / len(loader), running_accuracy / len(loader)


def evaluate_dual(model, loader, scaler, criterion, scheduler, device="cpu"):
    """Validation/test-time evaluation for the gated-fusion pipeline (no gradient updates)."""
    running_loss = 0.0
    running_accuracy = 0.0

    all_labels: List[int] = []
    all_predictions: List[int] = []
    all_probabilities: List[np.ndarray] = []

    start_time = time.time()

    for batch_X, batch_y, batch_X_metadata, idx in loader:
        batch_X = scaler.transform(batch_X)
        batch_X = torch.tensor(batch_X, dtype=torch.float32).to(device)

        batch_y = batch_y.to(device)
        batch_X_metadata = batch_X_metadata.to(device)
        batch_y = torch.argmax(batch_y, dim=1)

        outputs = model(batch_X, batch_X_metadata)
        loss = criterion(outputs, batch_y)

        accuracy = calculate_accuracy(outputs, batch_y)

        probabilities = F.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        all_labels.extend(batch_y.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())
        all_probabilities.extend(probabilities.cpu().numpy())

        running_loss += loss.item()
        running_accuracy += accuracy

    inference_time = time.time() - start_time

    val_loss = running_loss / len(loader)
    val_accuracy = running_accuracy / len(loader)

    precision = precision_score(all_labels, all_predictions, average="weighted")
    recall = recall_score(all_labels, all_predictions, average="weighted")
    f1 = f1_score(all_labels, all_predictions, average="weighted")

    # Binary AUC (positive class = column 1), matching the original notebook.
    all_probabilities_np = np.vstack(all_probabilities)
    auc = roc_auc_score(all_labels, all_probabilities_np[:, 1])

    if scheduler is not None:
        scheduler.step(val_loss)

    return val_loss, val_accuracy, precision, recall, f1, inference_time, auc


def test_dual(model, loader, scaler, criterion, device):
    """
    Runs the gated-fusion model over `loader`, returning raw outputs/labels
    plus loss/accuracy: (test_loss, test_accuracy, predicted_outputs, true_labels).
    """
    running_loss = 0.0
    running_accuracy = 0.0
    predicted_y = []
    y = []

    for batch_X, batch_y, batch_X_metadata, idx in loader:
        batch_X = scaler.transform(batch_X)
        batch_X = torch.tensor(batch_X, dtype=torch.float32)

        batch_y = batch_y.to(device)
        batch_X = batch_X.to(device)
        batch_X_metadata = batch_X_metadata.to(device)
        batch_y_1d = torch.argmax(batch_y, dim=1)

        outputs = model(batch_X, batch_X_metadata)
        loss = criterion(outputs, batch_y_1d)

        accuracy = calculate_accuracy(outputs, batch_y_1d)
        running_loss += loss.item()
        running_accuracy += accuracy

        predicted_y.append(outputs)
        y.append(batch_y)

    test_loss = running_loss / len(loader)
    test_accuracy = running_accuracy / len(loader)

    return test_loss, test_accuracy, torch.cat(predicted_y, dim=0), torch.cat(y, dim=0)
