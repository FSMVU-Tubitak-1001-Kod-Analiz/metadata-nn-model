import pandas
import numpy as np
from collections import defaultdict
from pathlib import Path
from sklearn.model_selection import KFold, StratifiedKFold

import torch
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


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


def create_model():
    model = Sequential()
    model.add(Dense(400, activation="relu", input_dim=769, kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.2, noise_shape=None, seed=None))

    model.add(Dense(400, activation="relu", kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.2, noise_shape=None, seed=None))

    model.add(Dense(3, activation="softmax"))
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    model.summary()
    return model


def train_and_test_for_one_fold(train_dataset, test_dataset):
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    tokenized_text_train = train_dataset["msgs"].apply(
        (
            lambda x: tokenizer.encode(
                x, max_length=80, add_special_tokens=True, truncation=True
            )
        )
    )
    tokenized_text_test = test_dataset["msgs"].apply(
        (
            lambda x: tokenizer.encode(
                x, max_length=80, add_special_tokens=True, truncation=True
            )
        )
    )

    max_len = 0
    for i in tokenized_text_train.values:
        if len(i) > max_len:
            max_len = len(i)
    for i in tokenized_text_test.values:
        if len(i) > max_len:
            max_len = len(i)
    padded_train = np.array(
        [i + [0] * (max_len - len(i)) for i in tokenized_text_train.values]
    )
    padded_test = np.array(
        [i + [0] * (max_len - len(i)) for i in tokenized_text_test.values]
    )
    attention_mask_train = np.where(padded_train != 0, 1, 0)
    attention_mask_test = np.where(padded_test != 0, 1, 0)

    model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    input_ids_train = torch.tensor(padded_train)
    input_ids_test = torch.tensor(padded_test)
    attention_mask_train = torch.tensor(attention_mask_train)
    attention_mask_test = torch.tensor(attention_mask_test)

    with torch.no_grad():
        last_hidden_states_train = model(
            input_ids_train, attention_mask=attention_mask_train
        )
        last_hidden_states_test = model(
            input_ids_test, attention_mask=attention_mask_test
        )
    bert_features_train = last_hidden_states_train[0][:, 0, :].numpy()
    bert_features_test = last_hidden_states_test[0][:, 0, :].numpy()

    cc_train = train_dataset.drop(
        train_dataset.columns.difference(["Contains Bug Fix?"]), axis=1
    ).replace(np.nan, 0)
    cc_test = test_dataset.drop(
        test_dataset.columns.difference(["Contains Bug Fix?"]), axis=1
    ).replace(np.nan, 0)
    merged_cc = pandas.concat((cc_train, cc_test))
    sc = StandardScaler()
    sc.fit(merged_cc)
    cc_train = sc.transform(cc_train)
    cc_test = sc.transform(cc_test)

    labels_train = train_dataset["labels"]
    labels_test = test_dataset["labels"]
    merged_label = pandas.concat((labels_train, labels_test))
    encoder = LabelBinarizer()
    encoder.fit(merged_label)
    labels_train = encoder.transform(labels_train)
    labels_test = encoder.transform(labels_test)
    labels_test = np.argmax(labels_test, axis=1)

    all_input_features_train = np.concatenate((bert_features_train, cc_train), axis=1)
    all_input_features_test = np.concatenate((bert_features_test, cc_test), axis=1)

    model = create_model()
    history = model.fit(
        all_input_features_train, labels_train, epochs=100, batch_size=32, verbose=1
    )
    predictions = np.argmax(model.predict(all_input_features_test), axis=1)
    return compute_metrics(labels_test, predictions)


def cross_validation_launch():
    dataset = pandas.read_csv("dataset.csv")
    label = "labels"

    dataset_x = [[i] for i in range(len(dataset))]
    dataset_y = [row[label] for _, row in dataset.iterrows()]
    label_map = {}
    for index, val in enumerate(dataset_y):
        if val not in label_map:
            label_map[val] = len(label_map)
        dataset_y[index] = label_map[val]

    # split dataset
    kfold_metrics_list = []
    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    cross_validation_metrics = []
    for fold_index, (train_index, test_index) in enumerate(
        skf.split(dataset_x, dataset_y)
    ):
        train_dataset = dataset.iloc[train_index]
        test_dataset = dataset.iloc[test_index]

        metrics = train_and_test_for_one_fold(train_dataset, test_dataset)
        print(metrics)
        kfold_metrics_list.append(metrics)

    print(average_metrics(kfold_metrics_list))


if __name__ == "__main__":
    cross_validation_launch()
