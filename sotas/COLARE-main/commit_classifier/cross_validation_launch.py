import os
import sys
import pandas
import numpy as np
from collections import defaultdict
from pathlib import Path
from sklearn.model_selection import KFold, StratifiedKFold

from trainer import Trainer, Tester
from trainer import train_logger as logger


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


def _save_launch_args(config):
    config.to_json(os.path.join(config.save_dir, "config.json"))

    flag_args = {"enable_checkpoint", "do_train", "do_test", "enable_cv"}
    lauch_command = " ".join(
        f"--{k} {v}" for k, v in config.namespace.items() if k not in flag_args
    )

    for arg in flag_args:
        if config.namespace[arg]:
            lauch_command += f" --{arg}"

    lauch_command = f"python {sys.argv[0]} " + lauch_command

    with open(os.path.join(config.save_dir, "train_command.txt"), "w") as f:
        f.write(lauch_command)


import random
import string


def generate_random_string(length):
    characters = string.ascii_letters + string.digits
    random_string = "".join(random.choice(characters) for _ in range(length))
    return random_string


def cross_validation_launch(config):
    # if not config.dataset:
    #     raise RuntimeError(f"--dataset should given when enbale cross validation")

    # make directory to store cross validation splits
    data_dir = Path(config.data_dir).parent
    cv_dir = data_dir.joinpath(
        f"{Path(config.data_dir).name}-5flod-{generate_random_string(5)}"
    )
    cv_dir.mkdir(exist_ok=True, parents=True)

    # save launch args
    _save_launch_args(config)

    # update config.data_dir
    dataset_path = os.path.join(config.data_dir, "dataset.csv")
    config.data_dir = str(cv_dir)

    # load dataset
    dataset = pandas.read_csv(dataset_path)
    if "labels" in dataset:
        label = "labels"
    else:
        label = "maintenance_type"
    assert label in dataset, f"{label} not in dataset ({dataset_path})"

    # extract sample index and label to split
    dataset_x = [[i] for i in range(len(dataset))]
    dataset_y = [row[label] for _, row in dataset.iterrows()]
    label_map = {}
    for index, val in enumerate(dataset_y):
        if val not in label_map:
            label_map[val] = len(label_map)
        dataset_y[index] = label_map[val]

    # split dataset
    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    cross_validation_metrics = []
    for fold_index, (train_index, test_index) in enumerate(
        skf.split(dataset_x, dataset_y)
    ):
        train_dataset = dataset.iloc[train_index]
        test_dataset = dataset.iloc[test_index]

        train_dataset.to_csv(os.path.join(cv_dir, "train.csv"), index=False)
        test_dataset.to_csv(os.path.join(cv_dir, "test.csv"), index=False)

        assert config.do_train, "--do_train if required if using cross validation"
        assert config.do_test, "--do_test if required if using cross validation"

        if os.path.exists(os.path.join(config.save_dir, "checkpoint.pt")):
            os.remove(os.path.join(config.save_dir, "checkpoint.pt"))
        if os.path.exists(os.path.join(config.save_dir, "best_model.pt")):
            os.remove(os.path.join(config.save_dir, "best_model.pt"))

        logger.info(f"start training for fold {fold_index}")
        trainer = Trainer(config, save_train_args=False)
        trainer.train()
        tester = Tester(config)
        metrics = tester.test()
        cross_validation_metrics.append(metrics)
    for key, value in average_metrics(cross_validation_metrics).items():
        logger.info(f"{key} = {value:.5f}")
