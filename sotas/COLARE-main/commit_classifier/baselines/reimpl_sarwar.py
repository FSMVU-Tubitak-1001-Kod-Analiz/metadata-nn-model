import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict
from simpletransformers.classification import ClassificationModel, ClassificationArgs
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

"""
Experiments results:

acc = 0.75968
macro_precision = 0.70546
macro_recall = 0.68735
macro_f1 = 0.69374
corrective_precision = 0.81983
corrective_recall = 0.84532
corrective_f1 = 0.83203
adaptive_precision = 0.61857
adaptive_recall = 0.56498
adaptive_f1 = 0.58588
perfective_precision = 0.67798
perfective_recall = 0.65175
perfective_f1 = 0.66331
"""


def compute_metrics(tgts, preds):
    acc = accuracy_score(tgts, preds)
    macro_precision = precision_score(tgts, preds, average="macro")
    macro_recall = recall_score(tgts, preds, average="macro")
    macro_f1 = f1_score(tgts, preds, average="macro")

    corrective_precision = precision_score(
        y_true=tgts, y_pred=preds, labels=[0], average=None
    )
    corrective_recall = recall_score(
        y_true=tgts, y_pred=preds, labels=[0], average=None
    )
    corrective_f1 = f1_score(tgts, preds, labels=[0], average=None)

    adaptive_precision = precision_score(
        y_true=tgts, y_pred=preds, labels=[1], average=None
    )
    adaptive_recall = recall_score(y_true=tgts, y_pred=preds, labels=[1], average=None)
    adaptive_f1 = f1_score(tgts, preds, labels=[1], average=None)

    perfective_precision = precision_score(
        y_true=tgts, y_pred=preds, labels=[2], average=None
    )
    perfective_recall = recall_score(
        y_true=tgts, y_pred=preds, labels=[2], average=None
    )
    perfective_f1 = f1_score(tgts, preds, labels=[2], average=None)
    metrics = {
        "acc": acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "corrective_precision": corrective_precision,
        "corrective_recall": corrective_recall,
        "corrective_f1": corrective_f1,
        "adaptive_precision": adaptive_precision,
        "adaptive_recall": adaptive_recall,
        "adaptive_f1": adaptive_f1,
        "perfective_precision": perfective_precision,
        "perfective_recall": perfective_recall,
        "perfective_f1": perfective_f1,
    }
    return metrics


def average_metrics(metrics: list):
    average_metrics = defaultdict(list)
    for metric in metrics:
        if isinstance(metric, dict):
            for key, value in metric.items():
                average_metrics[key].append(value)
        else:
            average_metrics["default_metric"].append(value)

    for key in average_metrics:
        average_metrics[key] = np.mean(average_metrics[key])
    return average_metrics


if __name__ == "__main__":
    data_path = "data/multi-lang-v4/dataset.csv"
    dataset = pd.read_csv(data_path)
    dataset = dataset[["msgs", "maintenance_type"]]
    dataset.rename(columns={"msgs": "text", "maintenance_type": "labels"}, inplace=True)

    # the same stratified 5 fold for evaluating model's performance
    dataset_x = [[i] for i in range(len(dataset))]
    dataset_y = [row["labels"] for _, row in dataset.iterrows()]
    label_map = {}
    for index, val in enumerate(dataset_y):
        if val not in label_map:
            label_map[val] = len(label_map)
        dataset_y[index] = label_map[val]
    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

    def convert_label(label):
        mp = {
            "P": 0,
            "A": 1,
            "C": 2,
        }
        return mp[label]

    dataset["labels"] = dataset["labels"].map(convert_label)

    cross_validation_metrics = []
    for fold_index, (train_index, test_index) in enumerate(
        skf.split(dataset_x, dataset_y)
    ):
        train_dataset = dataset.iloc[train_index]
        test_dataset = dataset.iloc[test_index]

        # The same argument following Sarwar et al.'s model
        # https://github.com/usmansarwar23/-Multi-label-Classification-of-Commit-Messages-using-Transfer-Learning
        model_args = ClassificationArgs(
            train_batch_size=8,
            num_train_epochs=4,
            overwrite_output_dir=True,
            use_multiprocessing=False,
            use_multiprocessing_for_evaluation=False,
        )

        model = ClassificationModel(
            "distilbert",
            "distilbert-base-uncased",
            num_labels=3,
            use_cuda=True,
            args=model_args,
        )
        model.train_model(train_dataset)

        text_to_predict = test_dataset["text"].to_list()
        predictions, raw_outputs = model.predict(text_to_predict)

        fold_metrics = compute_metrics(test_dataset["labels"].to_list(), predictions)
        cross_validation_metrics.append(fold_metrics)

    for key, value in average_metrics(cross_validation_metrics).items():
        print(f"{key} = {value:.5f}")
