"""
04_train_gated_fusion_folds.py
================================
Converted from: Rewrite_content_before_concat_patch_before_metadata_embedding_exp_architecture.ipynb

Trains + evaluates the gated-fusion architecture (code embeddings + metadata
embeddings, combined via a learned gate) plus 9 classical ML baselines, for
each of the 5 stratified 70:15:15 folds produced by 01_prepare_csv.py.

For each fold N this script:
  1. Loads new_csv_files_for_rewrite/folds_70_15_15/foldN_{train,val,test}.csv
  2. Builds label tensors, embeds metadata (`commit_message`, `file_path`,
     `additions`, `deletions`) with bert-base-uncased
     (NOTE: unlike 03b, `commit_message` IS included here even though the
     original notebook's own comment elsewhere warns pr_title/commit_message
     can leak the label -- that warning was not carried through to this
     script's actual column list. Kept as-is for fidelity; drop
     "commit_message" from METADATA_COLUMNS below if you want the same
     leak-safety guarantee as 03b.)
  3. Loads + concatenates the content_before/patch_before code embeddings
     (from 02_create_embeddings.py) into an `embedding` column
  4. Trains the gated-fusion DNN (GatedFusionDiffNN) for NUM_EPOCHS, saving
     the best-validation-accuracy checkpoint to `models/..._foldN.pth`
  5. Reloads that checkpoint and reports test-set metrics
  6. Trains 9 classical ML baselines on the metadata_tensor features and
     saves their metrics + confusion matrices

Outputs per fold (file names carry the `_fold{N}` suffix):
  - models/CB_pooler_embeddings_..._gatedfusion_14768_Adam_BN_w_commsg_fold{N}.pth
  - model_metrics_adam_bn_w_commsg_fold{N}.csv
  - DNN_confusion_matrix_adam_bn_w_commsg_fold{N}.png
  - all_confusion_matrices_adam_bn_w_commsg_fold{N}.png
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
from sklearn.preprocessing import StandardScaler

from dump.utils import load_and_concat_embeddings_from_directories
from common import (
    set_seed,
    embed_columns_with_bert,
    CodeMetadataDataset,
    create_dataloader,
    GatedFusionDiffNN,
    train_dual,
    evaluate_dual,
    test_dual,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42

FOLDS = [1, 2, 3, 4, 5]
FOLDS_DIR = "../new_csv_files_for_rewrite/folds_70_15_15"

EMBEDDING_DIRECTORY = "embeddings/CB_pooler_embeddings_content_before_2label_no_license_no_imports_balanced_14768"
EMBEDDING_DIRECTORY2 = "embeddings/CB_pooler_embeddings_patch_before_2label_no_license_no_imports_balanced_14768"

METADATA_MODEL_NAME = "bert-base-uncased"
METADATA_COLUMNS = ["commit_message", "file_path", "additions", "deletions"]

SCALER_PATH = "scalers/scaler_cb_pooler_14768.pkl"

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
NUM_EPOCHS = 100
LEARNING_RATE = 1e-5

HIDDEN_SIZES_EMBEDDING_PIPELINE = [1024, 2048, 4096]
HIDDEN_SIZES_METADATA_PIPELINE = [1024, 2048]
HIDDEN_SIZES_MERGED_PIPELINE = [4096, 2048, 1024, 512]
OUTPUT_SIZE = 2
DROPOUT_RATE = 0.2


def load_fold_data(fold_num: int):
    train_csv_path = os.path.join(FOLDS_DIR, f"fold{fold_num}_train.csv")
    val_csv_path = os.path.join(FOLDS_DIR, f"fold{fold_num}_val.csv")
    test_csv_path = os.path.join(FOLDS_DIR, f"fold{fold_num}_test.csv")

    df_train_raw = pd.read_csv(train_csv_path)
    df_val_raw = pd.read_csv(val_csv_path)
    df_test_raw = pd.read_csv(test_csv_path)

    df = pd.concat([df_train_raw, df_val_raw, df_test_raw], axis=0).reset_index(drop=True)

    hf_logging.set_verbosity_error()

    df_encoded_labels = pd.get_dummies(df, columns=["labels"])
    y = df_encoded_labels.drop(columns=[
        "full_repo_name", "pr_title", "patch", "commit_message", "file_path",
        "additions", "deletions", "content_before", "content_after",
        "index", "patch_before",
    ])
    y = y.astype(int)
    df["label_tensor"] = [torch.tensor(row, dtype=torch.float32) for row in y.values]

    tokenizer = AutoTokenizer.from_pretrained(METADATA_MODEL_NAME)
    bert_model = AutoModel.from_pretrained(METADATA_MODEL_NAME)
    if torch.cuda.is_available():
        bert_model = bert_model.cuda()

    df = embed_columns_with_bert(df, METADATA_COLUMNS, tokenizer, bert_model)

    df = df.drop(columns=[
        "full_repo_name", "pr_title", "file_path", "commit_message",
        "additions", "deletions", "content_before", "content_after",
        "patch_before", "patch", "labels",
    ])

    df = load_and_concat_embeddings_from_directories(
        directory_path_1=EMBEDDING_DIRECTORY, directory_path_2=EMBEDDING_DIRECTORY2, df=df
    )

    train_indices = df_train_raw["index"].values
    val_indices = df_val_raw["index"].values
    test_indices = df_test_raw["index"].values
    print(f"Loaded {len(train_indices)} train, {len(val_indices)} val, and {len(test_indices)} test samples.")

    df_train = df[df["index"].isin(train_indices)].reset_index(drop=True)
    df_val = df[df["index"].isin(val_indices)].reset_index(drop=True)
    df_test = df[df["index"].isin(test_indices)].reset_index(drop=True)
    torch.cuda.empty_cache()

    return df, df_train, df_val, df_test


def train_dnn_for_fold(fold_num: int, df, df_train, df_val, df_test, model_path: str):
    train_loader = create_dataloader(CodeMetadataDataset, df_train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = create_dataloader(CodeMetadataDataset, df_val, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = create_dataloader(CodeMetadataDataset, df_test, batch_size=BATCH_SIZE, shuffle=True)
    full_loader_for_scaler = create_dataloader(CodeMetadataDataset, df, batch_size=512, shuffle=True)

    scaler = StandardScaler()
    for batch_X, batch_y, batch_X_metadata, idx in full_loader_for_scaler:
        scaler.partial_fit(batch_X)
    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)

    code_embedding_dim = df_train["embedding"][5].shape[0]
    metadata_input_size = len(df_train["metadata_tensor"][0])

    model = GatedFusionDiffNN(
        code_input_size=code_embedding_dim,
        metadata_input_size=metadata_input_size,
        hidden_sizes_embedding_pipeline=HIDDEN_SIZES_EMBEDDING_PIPELINE,
        hidden_sizes_metadata_pipeline=HIDDEN_SIZES_METADATA_PIPELINE,
        hidden_sizes_merged_pipeline=HIDDEN_SIZES_MERGED_PIPELINE,
        output_size=OUTPUT_SIZE,
        dropout_rate=DROPOUT_RATE,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=400, factor=0.1, verbose=True)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    best_accuracy = 0.0
    for epoch in range(NUM_EPOCHS):
        model.train()
        model.to(DEVICE)
        avg_loss, avg_accuracy = train_dual(model, train_loader, scaler, optimizer, criterion, DEVICE)
        print(f"[Fold {fold_num}] Epoch [{epoch + 1}/{NUM_EPOCHS}], Train Loss: {avg_loss:.4f}, Train Accuracy: {avg_accuracy:.4f}")

        model.eval()
        model.to("cpu")
        with torch.no_grad():
            val_loss, val_accuracy, precision, recall, f1, inference_time, auc = evaluate_dual(
                model, val_loader, scaler, criterion, scheduler, device="cpu"
            )
            print(f"Inference Time: {inference_time:.4f} seconds")
            print(f"Val Loss:{val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
            print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
            print(f"AUC: {auc}")

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), model_path)

    # Reload the best checkpoint and report final test-set metrics
    model = GatedFusionDiffNN(
        code_input_size=code_embedding_dim,
        metadata_input_size=metadata_input_size,
        hidden_sizes_embedding_pipeline=HIDDEN_SIZES_EMBEDDING_PIPELINE,
        hidden_sizes_metadata_pipeline=HIDDEN_SIZES_METADATA_PIPELINE,
        hidden_sizes_merged_pipeline=HIDDEN_SIZES_MERGED_PIPELINE,
        output_size=OUTPUT_SIZE,
        dropout_rate=DROPOUT_RATE,
    ).to(DEVICE)
    model.load_state_dict(torch.load(model_path))

    with torch.no_grad():
        model.eval()
        model.to("cpu")
        val_loss, val_accuracy, precision, recall, f1, inference_time, auc = evaluate_dual(
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
        test_loss, test_accuracy, test_outputs, y_test = test_dual(model, test_loader, scaler, criterion, DEVICE)
        print(f"Accuracy on test data: {test_accuracy:.4f}")
        print(test_outputs.shape)

    return test_outputs, y_test


def plot_dnn_confusion_matrix(test_outputs, y_test, out_path: str):
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
    plt.savefig(out_path, format="png", dpi=300)
    plt.close()
    print(f"Saved {out_path}")


def train_ml_baselines_for_fold(df_train, df_val, df_test, metrics_csv: str, confusion_png: str):
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

    df_ml_train = pd.concat([df_train, df_val], axis=0, ignore_index=True)

    # NOTE: matches the original notebook -- the ML baselines here train on
    # `metadata_tensor` only (the `np.concatenate([embedding, metadata_tensor])`
    # variant is left commented out in the source, same as below).
    X_train = np.vstack(df_ml_train["metadata_tensor"])
    y_train = df_ml_train["label_tensor"].apply(
        lambda x: x.numpy().argmax() if isinstance(x, torch.Tensor) else x
    ).values

    X_test = np.vstack(df_test["metadata_tensor"])
    y_test = df_test["label_tensor"].apply(
        lambda x: x.numpy().argmax() if isinstance(x, torch.Tensor) else x
    ).values

    models = [
        ("XGBoost", xgb.XGBClassifier(objective="multi:softmax", num_class=2, eval_metric="mlogloss", random_state=SEED)),
        ("RandomForest", RandomForestClassifier(n_estimators=100, random_state=SEED)),
        ("CatBoost", CatBoostClassifier(iterations=100, learning_rate=0.1, depth=6, verbose=0, random_seed=SEED)),
        ("LightGBM", LGBMClassifier(random_state=SEED)),
        ("KNN", KNeighborsClassifier(n_neighbors=5)),
        ("NaiveBayes", GaussianNB()),
        ("LogisticRegression", LogisticRegression(max_iter=1000, random_state=SEED)),
        ("AdaBoost", AdaBoostClassifier(n_estimators=100, random_state=SEED)),
        ("SVM", SVC(probability=True, kernel="linear", random_state=SEED)),
    ]

    with open(metrics_csv, mode="w", newline="", encoding="utf-8") as file:
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

        n_classes = len(np.unique(y_test))
        if n_classes == 2:
            auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        else:
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

        with open(metrics_csv, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([model_name, accuracy, precision, recall, f1, auc, prediction_time])

    print(f"\nMetrics saved incrementally to {metrics_csv}")

    plot_ml_confusion_matrices(models, X_test, y_test, metrics, confusion_png)


def plot_ml_confusion_matrices(models, X_test, y_test, metrics, out_path: str):
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
    plt.savefig(out_path)
    plt.close()
    print(f"All confusion matrices saved to {out_path}")


def main():
    for fold_num in FOLDS:
        set_seed(SEED)
        print(f"\n{'=' * 60}\nFold {fold_num}\n{'=' * 60}")

        suffix = f"_adam_bn_w_commsg_fold{fold_num}"
        model_path = f"models/CB_pooler_embeddings_content_before_concat_patch_before_exp_architecture_gatedfusion_14768_Adam_BN_w_commsg_fold{fold_num}.pth"
        dnn_confusion_png = f"DNN_confusion_matrix{suffix}.png"
        ml_metrics_csv = f"model_metrics{suffix}.csv"
        ml_confusion_png = f"all_confusion_matrices{suffix}.png"

        df, df_train, df_val, df_test = load_fold_data(fold_num)

        test_outputs, y_test = train_dnn_for_fold(fold_num, df, df_train, df_val, df_test, model_path)
        plot_dnn_confusion_matrix(test_outputs, y_test, dnn_confusion_png)

        train_ml_baselines_for_fold(df_train, df_val, df_test, ml_metrics_csv, ml_confusion_png)


if __name__ == "__main__":
    main()
