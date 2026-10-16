import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter


def str2int(label):
    if label == "P":
        return 0
    if label == "A":
        return 1
    if label == "C":
        return 2
    raise NotImplementedError


if __name__ == "__main__":
    df = pd.read_csv("dataset.csv")
    x = [i for i in range(len(df))]
    y = [str2int(row["maintenance_type"]) for _, row in df.iterrows()]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.125, random_state=42, stratify=y_train
    )

    print(f"train num: {len(x_train)}")
    print(f"eval num: {len(x_val)}")
    print(f"test num: {len(x_test)}")

    print(f"train label: {Counter(y_train)}")
    print(f"eval label: {Counter(y_val)}")
    print(f"test label: {Counter(y_test)}")

    print(set(x_train).intersection(set(x_test)))
    print(set(x_train).intersection(set(x_val)))
    print(set(x_test).intersection(set(x_val)))

    train_df = df[df.index.isin(x_train)]
    print(train_df["language"].unique())
    print(train_df)
    train_df.to_csv("train.csv", index=False)

    test_df = df[df.index.isin(x_test)]
    print(test_df["language"].unique())
    print(test_df)
    test_df.to_csv("test.csv", index=False)

    eval_df = df[df.index.isin(x_val)]
    print(eval_df["language"].unique())
    print(eval_df)
    eval_df.to_csv("eval.csv", index=False)
