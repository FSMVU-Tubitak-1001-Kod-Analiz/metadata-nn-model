"""
03b_train_single_pipeline_metadata_embeddings.py
=================================================
Converted from: final_9ML_1DNN_new_evaluation_content_before_concat_patch_before_1pipeline_metadata.ipynb

Trains + evaluates 10 classifiers (1 custom ANN + 9 classical ML models) on a
single feature pipeline built entirely from *metadata* -- no code embeddings
at all. The metadata pipeline embeds `full_repo_name`, `file_path`,
`additions`, and `deletions` (joined into one string) with `bert-base-uncased`
and uses that as the model's only input.

`pr_title` and `commit_message` are deliberately excluded: the original
notebook flags them as a data-leakage risk (they can literally restate the
label, e.g. "fix bug in X").

Outputs:
  - MODEL_PATH               (best DNN checkpoint, by validation accuracy)
  - model_metrics_1pipeline_noleak_metadata.csv (metrics for the 9 ML baselines)
  - all_confusion_matrices_1pipeline_noleak_metadata.png
  - DNN_confusion_matrix.png

NOTE on data: same INPUT_CSV caveat as 03a -- see that script's docstring.
Since this script doesn't need pre-computed code embeddings, it only needs
the CSV itself (no dependency on 02_create_embeddings.py).
"""

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from transformers import AutoModel, AutoTokenizer
from transformers import logging as hf_logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from common import (
    set_seed,
    embed_columns_with_bert,
    SingleFeatureDataset,
    create_dataloader,
    MetadataOnlyDiffNN,
    train_single,
    evaluate_single,
    test_single,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42

INPUT_CSV = "new_csv_files_for_cb/merged_no_license_no_empty_only_apache_balanced_45993.csv"
METADATA_MODEL_NAME = "bert-base-uncased"
METADATA_COLUMNS = ["full_repo_name", "file_path", "additions", "deletions"]

MODEL_PATH = "models/CB_pooler_embeddings_content_before_concat_patch_before_1pipeline_noleak_metadata_only_apache_balanced_45993.pth"
SCALER_PATH = "scalers/scaler_metadata_45993.pkl"

ML_METRICS_CSV = "model_metrics_1pipeline_noleak_metadata.csv"
ML_CONFUSION_MATRIX_PNG = "all_confusion_matrices_1pipeline_noleak_metadata.png"
DNN_CONFUSION_MATRIX_PNG = "DNN_confusion_matrix.png"

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
NUM_EPOCHS = 100
LEARNING_RATE = 1e-5

HIDDEN_SIZES_EMBEDDING_PIPELINE = [1024, 2048, 4096]  # unused by MetadataOnlyDiffNN; kept for parity
HIDDEN_SIZES_METADATA_PIPELINE = [512, 1024]
HIDDEN_SIZES_MERGED_PIPELINE = [2048, 1024, 512, 256]
OUTPUT_SIZE = 3
DROPOUT_RATE = 0.2


def load_data() -> pd.DataFrame:
    hf_logging.set_verbosity_error()

    df = pd.read_csv(INPUT_CSV)

    # One-hot encode labels -> a single float label_tensor per row
    df_encoded_labels = pd.get_dummies(df, columns=["labels"])
    y = df_encoded_labels.drop(columns=[
        "full_repo_name", "pr_title", "commit_message", "file_path",
        "additions", "deletions", "content_before", "content_after",
        "index", "patch_before", "patch_after",
    ])
    y = y.astype(int)
    df["label_tensor"] = [torch.tensor(row, dtype=torch.float32) for row in y.values]

    tokenizer = AutoTokenizer.from_pretrained(METADATA_MODEL_NAME)
    bert_model = AutoModel.from_pretrained(METADATA_MODEL_NAME)
    if torch.cuda.is_available():
        bert_model = bert_model.cuda()

    # pr_title / commit_message are excluded: they can leak the label directly.
    df = embed_columns_with_bert(df, METADATA_COLUMNS, tokenizer, bert_model)

    df = df.drop(columns=[
        "full_repo_name", "pr_title", "commit_message", "file_path",
        "additions", "deletions", "content_before", "content_after",
        "patch_before", "patch_after", "labels",
    ])
    return df


def train_dnn(df: pd.DataFrame):
    X_index = df[["index"]]
    X_index_train, X_index_test = train_test_split(X_index, test_size=0.3, random_state=SEED)
    X_index_test, X_index_val = train_test_split(X_index_test, test_size=0.5, random_state=SEED)

    df_train = df[df["index"].isin(X_index_train["index"].values)].reset_index(drop=True)
    df_val = df[df["index"].isin(X_index_val["index"].values)].reset_index(drop=True)
    df_test = df[df["index"].isin(X_index_test["index"].values)].reset_index(drop=True)
    torch.cuda.empty_cache()

    train_loader = create_dataloader(SingleFeatureDataset, df_train, batch_size=BATCH_SIZE, shuffle=True, feature_column="metadata_tensor")
    val_loader = create_dataloader(SingleFeatureDataset, df_val, batch_size=BATCH_SIZE, shuffle=True, feature_column="metadata_tensor")
    test_loader = create_dataloader(SingleFeatureDataset, df_test, batch_size=BATCH_SIZE, shuffle=True, feature_column="metadata_tensor")
    full_loader_for_scaler = create_dataloader(SingleFeatureDataset, df, batch_size=512, shuffle=True, feature_column="metadata_tensor")

    scaler = StandardScaler()
    for batch_X, batch_y, idx in full_loader_for_scaler:
        scaler.partial_fit(batch_X)
    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)

    code_embedding_dim = 1  # unused by MetadataOnlyDiffNN
    metadata_input_size = len(df_train["metadata_tensor"][0])

    model = MetadataOnlyDiffNN(
        code_embedding_dim, metadata_input_size,
        HIDDEN_SIZES_EMBEDDING_PIPELINE, HIDDEN_SIZES_METADATA_PIPELINE, HIDDEN_SIZES_MERGED_PIPELINE,
        OUTPUT_SIZE, DROPOUT_RATE,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=400, factor=0.1, verbose=True)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    best_accuracy = 0.0
    for epoch in range(NUM_EPOCHS):
        model.train()
        model.to(DEVICE)
        avg_loss, avg_accuracy = train_single(model, train_loader, scaler, optimizer, criterion, DEVICE)
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Train Loss: {avg_loss:.4f}, Train Accuracy: {avg_accuracy:.4f}")

        model.eval()
        model.to("cpu")
        with torch.no_grad():
            val_loss, val_accuracy, precision, recall, f1, inference_time, auc = evaluate_single(
                model, val_loader, scaler, criterion, scheduler, device="cpu"
            )
            print(f"Inference Time: {inference_time:.4f} seconds")
            print(f"Val Loss:{val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
            print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
            print(f"AUC: {auc}")

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), MODEL_PATH)

    # Reload the best checkpoint and report final test-set metrics
    model = MetadataOnlyDiffNN(
        code_embedding_dim, metadata_input_size,
        HIDDEN_SIZES_EMBEDDING_PIPELINE, HIDDEN_SIZES_METADATA_PIPELINE, HIDDEN_SIZES_MERGED_PIPELINE,
        OUTPUT_SIZE, DROPOUT_RATE,
    ).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))

    with torch.no_grad():
        model.eval()
        model.to("cpu")
        val_loss, val_accuracy, precision, recall, f1, inference_time, auc = evaluate_single(
            model, test_loader, scaler, criterion, scheduler, device="cpu"
        )
        print(f"Accuracy on test data: {val_accuracy}")
        print(f"Inference Time: {inference_time} seconds")
        print(f"Loss:{val_loss}")
        print(f"Precision: {precision}, Recall: {recall}, F1-Score: {f1}")
        print(f"AUC: {auc}")

    with torch.no_grad():
        model.to(DEVICE)
        model.eval()
        test_loss, test_accuracy, test_outputs, y_test = test_single(model, test_loader, scaler, criterion, DEVICE)
        print(f"Accuracy on test data: {test_accuracy:.4f}")
        print(test_outputs.shape)

    plot_dnn_confusion_matrix(test_outputs, y_test)

    return df_train, df_val, df_test


def plot_dnn_confusion_matrix(test_outputs, y_test):
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    _, predicted = torch.max(test_outputs.detach().cpu(), 1)
    labels_1d = torch.argmax(y_test, dim=1).cpu()

    y_test_np = labels_1d.numpy()
    predicted_np = predicted.numpy()

    cm = confusion_matrix(y_test_np, predicted_np)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix for Label Prediction")
    plt.savefig(DNN_CONFUSION_MATRIX_PNG, format="png", dpi=300)
    plt.close()
    print(f"Saved {DNN_CONFUSION_MATRIX_PNG}")


def train_ml_baselines(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame):
    import time
    import csv
    import xgboost as xgb
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import LogisticRegression

    X_train = np.vstack(df_train["metadata_tensor"])
    y_train = df_train["label_tensor"].apply(
        lambda x: x.numpy().argmax() if isinstance(x, torch.Tensor) else x
    ).values

    X_test = np.vstack(df_test["metadata_tensor"])
    y_test = df_test["label_tensor"].apply(
        lambda x: x.numpy().argmax() if isinstance(x, torch.Tensor) else x
    ).values

    models = [
        ("XGBoost", xgb.XGBClassifier(objective="multi:softmax", num_class=3, eval_metric="mlogloss", random_state=SEED)),
        ("RandomForest", RandomForestClassifier(n_estimators=100, random_state=SEED)),
        ("CatBoost", CatBoostClassifier(iterations=100, learning_rate=0.1, depth=6, verbose=0, random_seed=SEED)),
        ("LightGBM", LGBMClassifier(num_class=3, random_state=SEED)),
        ("KNN", KNeighborsClassifier(n_neighbors=5)),
        ("NaiveBayes", GaussianNB()),
        ("LogisticRegression", LogisticRegression(max_iter=1000, random_state=SEED)),
        ("AdaBoost", AdaBoostClassifier(n_estimators=100, random_state=SEED)),
        ("SVM", SVC(probability=True, kernel="linear", random_state=SEED)),
    ]

    with open(ML_METRICS_CSV, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Model Name", "Accuracy", "Precision", "Recall", "F1 Score", "AUC", "Prediction Time (s)"])

    metrics = {}
    for model_name, model in models:
        model.fit(X_train, y_train)

        start_time = time.time()
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        prediction_time = time.time() - start_time

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="macro")
        recall = recall_score(y_test, y_pred, average="macro")
        f1 = f1_score(y_test, y_pred, average="macro")
        auc = roc_auc_score(y_test, y_pred_proba, multi_class="ovr", average="macro")

        metrics[model_name] = {
            "Accuracy": accuracy, "Precision": precision, "Recall": recall,
            "F1 Score": f1, "AUC": auc, "Prediction Time (s)": prediction_time,
        }

        print(f"Metrics for {model_name}:")
        print(f"  Accuracy: {accuracy}")
        print(f"  Precision: {precision}")
        print(f"  Recall: {recall}")
        print(f"  F1 Score: {f1}")
        print(f"  AUC: {auc:.4f}")
        print(f"  Prediction Time (s): {prediction_time:.4f}")

        with open(ML_METRICS_CSV, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([model_name, accuracy, precision, recall, f1, auc, prediction_time])

    print(f"\nMetrics saved incrementally to {ML_METRICS_CSV}")

    plot_ml_confusion_matrices(models, X_test, y_test, metrics)


def plot_ml_confusion_matrices(models, X_test, y_test, metrics):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    confusion_matrices = {}
    for model_name, model in models:
        y_pred = model.predict(X_test)
        confusion_matrices[model_name] = confusion_matrix(y_test, y_pred)

    num_models = len(confusion_matrices)
    rows = (num_models + 2) // 3
    fig, axes = plt.subplots(nrows=rows, ncols=3, figsize=(18, 6 * rows))
    fig.suptitle("Confusion Matrices for Trained Models", fontsize=20)

    for ax, (model_name, cm) in zip(axes.flatten(), confusion_matrices.items()):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_title(f"{model_name} Accuracy:{metrics[model_name]['Accuracy']}")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

    for i in range(num_models, len(axes.flatten())):
        fig.delaxes(axes.flatten()[i])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(ML_CONFUSION_MATRIX_PNG)
    plt.close()
    print(f"All confusion matrices saved to {ML_CONFUSION_MATRIX_PNG}")


def main():
    set_seed(SEED)
    df = load_data()
    df_train, df_val, df_test = train_dnn(df)
    train_ml_baselines(df_train, df_val, df_test)


if __name__ == "__main__":
    main()
