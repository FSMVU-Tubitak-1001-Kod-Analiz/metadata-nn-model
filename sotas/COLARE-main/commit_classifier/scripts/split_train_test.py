import sys
import os
from pathlib import Path
from datasets import load_dataset

TRAIN_SIZE = 0.8
TEST_SIZE = 0.2


def split(file: str, delimiter=","):
    file = Path(file).absolute()
    dataset = load_dataset("csv", data_files=str(file), delimiter=delimiter)["train"]

    train_size = TRAIN_SIZE
    dataset = dataset.train_test_split(train_size=train_size, seed=42)
    train, test = dataset["train"], dataset["test"]

    train.set_format("pandas")
    test.set_format("pandas")

    train_df = train[:]
    test_df = test[:]

    train_df.to_csv(file.parent.joinpath("train.csv"), index=False)
    test_df.to_csv(file.parent.joinpath("test.csv"), index=False)


if __name__ == "__main__":
    split(sys.argv[1])
