import openai
import pandas as pd
import tiktoken
from time import sleep
from tqdm import tqdm
import numpy as np
from argparse import ArgumentParser
from collections import defaultdict

"""
RESULT = {
    "acc": 0.5034680349798346,
    "macro_precision": 0.5357732075805598,
    "macro_recall": 0.6262106025140279,
    "macro_f1": 0.500644570846388,
    "corrective_precision": 0.4968461645904359,
    "corrective_recall": 0.9008450704225351,
    "corrective_f1": 0.6400489388706199,
    "adaptive_precision": 0.2858077981925908,
    "adaptive_recall": 0.6522163120567376,
    "adaptive_f1": 0.396422595142354,
    "perfective_precision": 0.8246656599586526,
    "perfective_recall": 0.32557042506281086,
    "perfective_f1": 0.46546217852619004,
}
"""

model_engine = "gpt-3.5-turbo"


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo-0301":  # note: future models may deviate from this
        num_tokens = 0
        for message in messages:
            num_tokens += (
                4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            )
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not presently implemented for model {model}.
  See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )


def generate_promot(row):
    base_promot = """
    Please act as a commit classifier,
    categorize git commit into three categories: Adaptive, Perfective, and Corrective.
    The Adaptive category corresponds to commits that address the modifications to the project in order to adapt it to the new environment such as the feature addition.
    The Perfective category corresponds to commits that address enhancement of the project, such as enhancement of performance and refactoring of source code.
    The Corrective category corresponds to commits that address the fix of bugs and faults in the project.
    I will provide you with the commit message and code diff for a commit,
    and you need to give me the category label for this commit.
    Please avoid any explanations and only provide the category label.
    """

    commit_message_prompt = f"commit message: {row['msgs']}\n"
    commit_codediff_prompt = f"code diff: {row['diffs']}\n"

    prompt = base_promot + commit_message_prompt + commit_codediff_prompt

    messages = [{"role": "user", "content": prompt}]

    l, r = 0, len(commit_codediff_prompt) - 1
    while l <= r:
        mid = (l + r) // 2
        prompt = base_promot + commit_message_prompt + commit_codediff_prompt[:mid]
        messages = [{"role": "user", "content": prompt}]

        ok = num_tokens_from_messages(messages) < 4096
        if ok:
            l = mid + 1
        else:
            r = mid - 1
    prompt = base_promot + commit_message_prompt + commit_codediff_prompt[:r]

    return prompt


def get_chatgpt_labels(fold):
    filename = f"folds/fold{fold}.csv"
    df = pd.read_csv(filename)
    if "gpt_label" not in df:
        df["gpt_label"] = "None"
    for index, row in tqdm(df.iterrows(), total=len(df)):
        if row["gpt_label"] != "None":
            continue
        ok = False
        failed_cnt = 0
        skip = False
        while not ok:
            try:
                prompt = generate_promot(row)

                res = openai.ChatCompletion.create(
                    model=model_engine,
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                )
                ok = True
            except Exception as e:
                print(e)
                failed_cnt += 1
                sleep(10)

            if failed_cnt > 10:
                skip = True
                break

        if skip:
            continue

        generated_text = res["choices"][0]["message"]["content"].strip()
        df.at[index, "gpt_label"] = generated_text
        df.to_csv(filename, index=False)
        sleep(20)


from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


def convert_to_number1(x):
    mp = {
        "Corrective": 0,
        "Adaptive": 1,
        "Perfective": 2,
    }
    return mp[x]


def convert_to_number2(x):
    mp = {
        "C": 0,
        "A": 1,
        "P": 2,
    }
    return mp[x]


def postproces(fold):
    filename = f"folds/fold{fold}.csv"
    df = pd.read_csv(filename)

    print(len(df))
    print(df["gpt_label"].unique())

    def recover_label(x):
        mp = {
            "Corrective": "Corrective",
            "Adaptive": "Adaptive",
            "Perfective": "Perfective",
            "Category: Perfective": "Perfective",
            "Category: Adaptive": "Adaptive",
            "Category: Corrective": "Corrective",
            "Category label: Perfective": "Perfective",
            "Category label: Adaptive": "Adaptive",
            "Category label: Corrective": "Corrective",
            "Categorization: Perfective": "Perfective",
            "Categorization: Adaptive": "Adaptive",
            "Categorization: Corrective": "Corrective",
        }
        if x in mp:
            return mp[x]
        return "None"

    df["gpt_label"] = df["gpt_label"].apply(recover_label)
    print(df["gpt_label"].unique())

    gpt_label = df["gpt_label"].apply(convert_to_number1)
    preds = gpt_label.to_list()
    tgts = df["maintenance_type"].apply(convert_to_number2).to_list()

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


def average_5folds():
    results = []
    for i in range(1, 6):
        results.append(postproces(i))
    print(average_metrics(results))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--task", type=str)
    parser.add_argument("--fold", type=str)
    parser.add_argument("--api", type=str)
    args = parser.parse_args()

    openai.api_key = args.api

    if args.task == "complete":
        get_chatgpt_labels(args.fold)
    if args.task == "postprocess":
        postproces(args.fold)
    if args.task == "average":
        average_5folds()
