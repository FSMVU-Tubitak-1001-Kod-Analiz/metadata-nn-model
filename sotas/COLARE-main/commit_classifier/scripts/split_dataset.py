import sys
import os
from pathlib import Path
from datasets import load_dataset

TRAIN_SIZE = 0.7
TEST_SIZE = 0.2


def split(file: str, delimiter=","):
    file = Path(file).absolute()
    dataset = load_dataset("csv", data_files=str(file), delimiter=delimiter)["train"]

    train_size = TRAIN_SIZE
    test_size = TEST_SIZE * len(dataset) / ((1 - TRAIN_SIZE) * len(dataset))

    dataset = dataset.train_test_split(train_size=train_size, seed=123)
    train, test_val = dataset["train"], dataset["test"]
    dataset = test_val.train_test_split(train_size=test_size, seed=123)
    test, val = dataset["train"], dataset["test"]

    train.set_format("pandas")
    val.set_format("pandas")
    test.set_format("pandas")

    train_df = train[:]
    val_df = val[:]
    test_df = test[:]

    train_df.to_csv(file.parent.joinpath("train.csv"), index=False)
    val_df.to_csv(file.parent.joinpath("eval.csv"), index=False)
    test_df.to_csv(file.parent.joinpath("test.csv"), index=False)


if __name__ == "__main__":
    split(sys.argv[1])
