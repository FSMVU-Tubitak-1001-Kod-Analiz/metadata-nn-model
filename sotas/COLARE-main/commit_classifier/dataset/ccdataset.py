import json
import copy
import os
import functools
import torch
from collections import defaultdict
from itertools import chain
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding
from tqdm import tqdm
from registry import DATASET_REGISTRY
import pandas


NUM_FILE_HUNK = []
NUM_COMMIT_FILE = []


class Hunk:
    def __init__(self, code_lines: str = None) -> None:
        self.codes = code_lines

    def __str__(self) -> str:
        return str(self.codes)


class FileChangeInstance:
    def __init__(
        self,
        filename: str,
        hunks_add=None,
        hunks_delete=None,
    ) -> None:
        self.filename = filename
        self.hunks_add = []
        self.hunks_delete = []
        if hunks_add is not None:
            if not isinstance(hunks_add, list):
                hunks_add = [hunks_add]
            self.hunks_add.extend(hunks_add)
        if hunks_delete is not None:
            if not isinstance(hunks_delete, list):
                hunks_delete = [hunks_delete]
            self.hunks_delete.extend(hunks_delete)

    def append_hunk_added(self, hunk: Hunk):
        self.hunks_add.append(hunk)

    def append_hunk_deleted(self, hunk: Hunk):
        self.hunks_delete.append(hunk)

    def __str__(self) -> str:
        res = f"<\n  file_change: {self.filename}\n"
        res += f"  hunks_added = [{','.join(str(hunk) for hunk in self.hunks_add)}]\n"
        res += f"  hunks_deleted = [{','.join(str(hunk) for hunk in self.hunks_delete)}]\n>"
        return res


class FileChangeList:
    def __init__(self, changes) -> None:
        self.changes = changes
        self.file_attention_mask = None
        self.hunk_attention_mask = None


class CodeDiffParser:
    def __init__(
        self,
        file_num_limit: int = None,
        hunk_num_limit: int = None,
        code_num_limit: int = None,
    ) -> None:
        self.file_num_limit = file_num_limit or int(1e9)
        self.hunk_num_limit = hunk_num_limit or int(1e9)
        self.code_num_limit = code_num_limit or int(1e9)

    def parse_one_hunk(self, diff: str):
        code_added, code_deleted = [], []
        for line in diff.splitlines():
            if line.startswith("+"):
                code_added.append(line[1:].strip())
            elif line.startswith("-"):
                code_deleted.append(line[1:].strip())
            else:
                code_added.append(line.strip())
                code_deleted.append(line.strip())
        code_added = code_added[: self.code_num_limit]
        code_deleted = code_deleted[: self.code_num_limit]
        return Hunk("\n".join(code_added)), Hunk("\n".join(code_deleted))

    def parse_one_file_diff(self, diff: str) -> FileChangeInstance:
        diff_lines = diff.splitlines()
        hunk_index = [
            index for index, line in enumerate(diff_lines) if line.startswith("@@")
        ]
        NUM_FILE_HUNK.append(len(hunk_index))
        hunk_diffs = []
        for i in range(len(hunk_index)):
            if i + 1 == len(hunk_index):
                hunk_diffs.append(diff_lines[hunk_index[i] :])
            else:
                hunk_diffs.append(diff_lines[hunk_index[i] : hunk_index[i + 1]])
        hunks_added, hunks_deleted = [], []
        for hunk in hunk_diffs:
            hunk_added, hunk_deleted = self.parse_one_hunk("\n".join(hunk))
            hunks_added.append(hunk_added)
            hunks_deleted.append(hunk_deleted)
        filename = diff_lines[0].split()[-1]
        filename = filename[filename.find("/") + 1 :]
        return FileChangeInstance(
            filename,
            hunks_added[: self.hunk_num_limit],
            hunks_deleted[: self.hunk_num_limit],
        )

    def parse(self, diff: str):
        file_count = diff.count("diff --git")
        NUM_COMMIT_FILE.append(file_count)
        block_index = []
        for _ in range(file_count):
            if not block_index:
                start_index = 0
            else:
                start_index = block_index[-1] + 1
            block_index.append(diff.find("diff --git", start_index))

        file_diffs = []
        for i in range(len(block_index)):
            if i + 1 == len(block_index):
                file_diffs.append(diff[block_index[i] :])
            else:
                file_diffs.append(diff[block_index[i] : block_index[i + 1]])

        file_change_instances = []
        for file_diff in file_diffs:
            file_change_instances.append(self.parse_one_file_diff(file_diff))
        return FileChangeList(file_change_instances[: self.file_num_limit])


class CommitReader:
    def __init__(
        self,
        file_num_limit: int = 10,
        hunk_num_limit: int = 10,
        code_num_limit: int = 256,
    ) -> None:
        self.file_num_limit = file_num_limit
        self.hunk_num_limit = hunk_num_limit
        self.code_num_limit = code_num_limit
        self.diff_parser = CodeDiffParser(
            file_num_limit, hunk_num_limit, code_num_limit
        )
        self.total_files = []
        self.total_hunks = []
        self.total_codes = []

    def series_to_dict(self, series: pandas.Series) -> dict:
        if "labels" in series:
            label_map = {
                "Perfective": 0,
                "Adaptive": 1,
                "Corrective": 2,
            }
            label = label_map[series["labels"]]
        else:
            label_map = {
                "P": 0,
                "A": 1,
                "C": 2,
            }
            label = label_map[series["maintenance_type"]]
        commit_message = series["msgs"]
        file_change_instances = self.diff_parser.parse(series["diffs"])

        return {
            "commit_message": commit_message,
            "file_changes": file_change_instances,
            "label": label,
            "commit_sha": series["commit"],
            "diff": series["diffs"],
            "numerical_features": json.loads(series["feature"]),
        }

    def read(self, data_file):
        df = pandas.read_csv(data_file)
        for _, series in df.iterrows():
            yield self.series_to_dict(series)


@DATASET_REGISTRY.register("CommitDataset")
class CommitDataset(Dataset):
    def __init__(self, data_path: str, config, use_roberta=False) -> None:
        super().__init__()
        self.data_path = data_path
        self.config = config
        self.data_reader = CommitReader(config.file_num_limit, config.hunk_num_limit)
        self.commit_messages = []
        self.commit_filechanges = []
        self.commit_labels = []
        self.commit_sha = []
        self.commit_features = []
        self.commit_diffs = []
        self.use_roberta = use_roberta
        for item in self.data_reader.read(data_path):
            self.commit_messages.append(item["commit_message"])
            self.commit_filechanges.append(item["file_changes"])
            self.commit_labels.append(item["label"])
            self.commit_sha.append(item["commit_sha"])
            self.commit_features.append(item["numerical_features"])
            self.commit_diffs.append(item["diff"])
        self.message_tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/codebert-base"
        )
        self.code_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        if use_roberta:
            self.roberta_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        else:
            self.roberta_tokenizer = None
        self.message_collator = DataCollatorWithPadding(self.message_tokenizer)
        self.code_collator = DataCollatorWithPadding(self.code_tokenizer)
        self.total_tokens = []
        self.preprocess()

    def preprocess(self):
        # tokenize commit message
        msg_tokenizer = (
            self.message_tokenizer if not self.use_roberta else self.roberta_tokenizer
        )
        self.commit_messages_tokenized = [
            msg_tokenizer(
                message,
                truncation=True,
                padding="max_length",
                max_length=256,
            )
            for message in self.commit_messages
        ]

        if self.use_roberta:
            self.commit_diffs_tokenized = [
                self.message_tokenizer(
                    diff,
                    truncation=True,
                    padding="max_length",
                )
                for diff in self.commit_diffs
            ]

        # tokenize codes
        self.commit_filechanges_tokenized = copy.deepcopy(self.commit_filechanges)
        for file_changes in tqdm(self.commit_filechanges_tokenized, desc="tokenizing"):
            for file_change in file_changes.changes:
                for hunk in chain(file_change.hunks_add, file_change.hunks_delete):
                    hunk.codes = self.code_tokenizer(
                        hunk.codes,
                        truncation=True,
                        padding="max_length",
                        max_length=256,
                    )
        # padding codes
        for file_changes in tqdm(self.commit_filechanges_tokenized, desc="padding"):
            self.padding_files(file_changes)

        for index, file_changes in tqdm(
            enumerate(self.commit_filechanges_tokenized), desc="padding"
        ):
            assert file_changes.file_attention_mask[0] != 1, self.commit_sha[index]

        # reformat codes
        self.codes_add_input_ids = []
        self.codes_add_attention_mask = []
        self.codes_delete_input_ids = []
        self.codes_delete_attention_mask = []
        self.file_attention_mask = []
        self.hunk_attention_mask = []
        self.reformat_codes()

    def reformat_codes(self):
        for commit_id in tqdm(
            range(len(self.commit_filechanges_tokenized)), desc="reformating"
        ):
            commit_add_input_ids = []
            commit_add_attention_mask = []
            commit_delete_input_ids = []
            commit_delete_attention_mask = []
            self.file_attention_mask.append(
                self.commit_filechanges_tokenized[commit_id].file_attention_mask
            )
            self.hunk_attention_mask.append(
                self.commit_filechanges_tokenized[commit_id].hunk_attention_mask
            )
            for file_id in range(
                len(self.commit_filechanges_tokenized[commit_id].changes)
            ):
                file_add_input_ids = []
                file_add_attention_mask = []
                file_delete_input_ids = []
                file_delete_attention_mask = []

                for hunk_id in range(
                    len(
                        self.commit_filechanges_tokenized[commit_id]
                        .changes[file_id]
                        .hunks_add
                    )
                ):
                    file_add_input_ids.append(
                        self.commit_filechanges_tokenized[commit_id]
                        .changes[file_id]
                        .hunks_add[hunk_id]
                        .codes["input_ids"]
                    )
                    file_delete_input_ids.append(
                        self.commit_filechanges_tokenized[commit_id]
                        .changes[file_id]
                        .hunks_delete[hunk_id]
                        .codes["input_ids"]
                    )
                    file_add_attention_mask.append(
                        self.commit_filechanges_tokenized[commit_id]
                        .changes[file_id]
                        .hunks_add[hunk_id]
                        .codes["attention_mask"]
                    )
                    file_delete_attention_mask.append(
                        self.commit_filechanges_tokenized[commit_id]
                        .changes[file_id]
                        .hunks_delete[hunk_id]
                        .codes["attention_mask"]
                    )
                commit_add_input_ids.append(file_add_input_ids)
                commit_add_attention_mask.append(file_add_attention_mask)
                commit_delete_input_ids.append(file_delete_input_ids)
                commit_delete_attention_mask.append(file_delete_attention_mask)
                assert len(file_add_input_ids) == len(commit_add_input_ids[-1])
                assert len(file_delete_input_ids) == len(commit_delete_input_ids[-1])
            self.codes_add_input_ids.append(commit_add_input_ids)
            self.codes_add_attention_mask.append(commit_add_attention_mask)
            self.codes_delete_input_ids.append(commit_delete_input_ids)
            self.codes_delete_attention_mask.append(commit_delete_attention_mask)
            assert len(commit_add_input_ids) == len(self.codes_add_input_ids[-1])
            assert len(commit_delete_input_ids) == len(self.codes_delete_input_ids[-1])

    def padding_hunk(self, hunk):
        if hunk.codes is None:
            hunk.codes = self.code_tokenizer(
                "",
                truncation=True,
                padding="max_length",
                max_length=256,
            )

    def padding_file(self, file):
        assert len(file.hunks_add) == len(file.hunks_delete)
        file.hunk_attention_mask = [0] * len(file.hunks_add) + [1] * (
            self.config.hunk_num_limit - len(file.hunks_add)
        )
        if file.hunk_attention_mask[0] == 1:
            file.hunk_attention_mask[0] = 0
        while len(file.hunks_add) < self.config.hunk_num_limit:
            file.hunks_add.append(Hunk())

        for hunk in file.hunks_add:
            self.padding_hunk(hunk)

        while len(file.hunks_delete) < self.config.hunk_num_limit:
            file.hunks_delete.append(Hunk())

        for hunk in file.hunks_delete:
            self.padding_hunk(hunk)

    def padding_files(self, files):
        files.file_attention_mask = [0] * len(files.changes) + [1] * (
            self.config.file_num_limit - len(files.changes)
        )
        while len(files.changes) < self.config.file_num_limit:
            files.changes.append(FileChangeInstance("pad_file"))
        for file in files.changes:
            self.padding_file(file)
        files.hunk_attention_mask = []
        for file in files.changes:
            files.hunk_attention_mask.append(file.hunk_attention_mask)

    def __len__(self) -> int:
        return len(self.commit_messages)

    def __getitem__(self, index) -> dict:
        item = {
            "label": self.commit_labels[index],
            "msg_input_ids": self.commit_messages_tokenized[index]["input_ids"],
            "msg_attention_mask": self.commit_messages_tokenized[index][
                "attention_mask"
            ],
            "codes_add_input_ids": self.codes_add_input_ids[index],
            "codes_add_attention_mask": self.codes_add_attention_mask[index],
            "codes_delete_input_ids": self.codes_delete_input_ids[index],
            "codes_delete_attention_mask": self.codes_delete_attention_mask[index],
            "file_attention_mask": self.file_attention_mask[index],
            "hunk_attention_mask": self.hunk_attention_mask[index],
            "commit_sha": self.commit_sha[index],
            "numerical_features": self.commit_features[index],
        }
        if self.use_roberta:
            item.update(
                {
                    "diff_input_ids": self.commit_diffs_tokenized[index]["input_ids"],
                    "diff_attention_mask": self.commit_diffs_tokenized[index][
                        "attention_mask"
                    ],
                }
            )
        return item


def collate_fn(
    batch,
    msg_collator: DataCollatorWithPadding,
):
    msg_features = defaultdict(list)
    code_add_features = defaultdict(list)
    code_delete_features = defaultdict(list)
    diff_features = defaultdict(list)
    file_attention_mask = []
    hunk_attention_mask = []
    commit_sha = []
    numerical_features = []
    for item in batch:
        for key, value in item.items():
            if key.startswith("msg"):
                msg_features[key[4:]].append(value)
            elif key == "label":
                msg_features[key].append(value)
            elif key.startswith("codes_add"):
                code_add_features[key[10:]].append(value)
            elif key.startswith("codes_delete"):
                code_delete_features[key[13:]].append(value)
            elif key == "file_attention_mask":
                file_attention_mask.append(value)
            elif key == "hunk_attention_mask":
                hunk_attention_mask.append(value)
            elif key == "commit_sha":
                commit_sha.append(value)
            elif key == "numerical_features":
                numerical_features.append(value)
            elif key.startswith("diff"):
                diff_features[key[5:]].append(value)

    msg_features = msg_collator(msg_features)
    msg_features["codes_add_input_ids"] = torch.tensor(code_add_features["input_ids"])
    msg_features["codes_add_attention_mask"] = torch.tensor(
        code_add_features["attention_mask"]
    )
    msg_features["codes_delete_input_ids"] = torch.tensor(
        code_delete_features["input_ids"]
    )
    msg_features["codes_delete_attention_mask"] = torch.tensor(
        code_delete_features["attention_mask"]
    )
    msg_features["file_attention_mask"] = torch.tensor(
        file_attention_mask, dtype=torch.bool
    )
    msg_features["hunk_attention_mask"] = torch.tensor(
        hunk_attention_mask, dtype=torch.bool
    )
    msg_features["numerical_features"] = torch.tensor(
        numerical_features, dtype=torch.float32
    )

    if diff_features:
        msg_features["diff_input_ids"] = torch.tensor(diff_features["input_ids"])
        msg_features["diff_attention_mask"] = torch.tensor(
            diff_features["attention_mask"]
        )
    return msg_features


if __name__ == "__main__":

    class Config:
        def __init__(self) -> None:
            self.file_num_limit = 4
            self.hunk_num_limit = 2
            self.code_num_limit = 300

    # dataset = CommitDataset("data/1793-7t1e2t/1793_final_refined.csv", Config())

    # instance_id = 413

    # codebert_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    # print(dataset.message_tokenizer.decode(dataset[instance_id]["msg_input_ids"]))
    # print(codebert_tokenizer.decode(dataset[instance_id]["codes_add_input_ids"][1][0]))
    # print(
    #     codebert_tokenizer.decode(dataset[instance_id]["codes_delete_input_ids"][1][0])
    # )
    # print(dataset[instance_id]["file_attention_mask"])
    # print(dataset[instance_id]["hunk_attention_mask"])

    # import torch

    # t = torch.tensor(dataset[1024]["codes_add_input_ids"])
    # print(t.shape)

    # train_loader = DataLoader(
    #     dataset,
    #     batch_size=3,
    #     shuffle=True,
    #     collate_fn=functools.partial(
    #         collate_fn,
    #         msg_collator=dataset.message_collator,
    #     ),
    #     drop_last=True,
    # )

    # item = next(iter(train_loader))

    # print(item)
    # x = item["codes_add_input_ids"].view(-1, 300).numpy().tolist()
    # for xx in x:
    #     print(codebert_tokenizer.decode(xx))

    # from model_codebert import CCModel

    # model = CCModel(None)

    # model(**item)

    filename = "data/multi-lang-v4/dataset.csv"

    dataset = CommitDataset(filename, Config(), use_roberta=True)
    NUM_FILE_HUNK = sorted(NUM_FILE_HUNK)
    NUM_COMMIT_FILE = sorted(NUM_COMMIT_FILE)

    def check_num(arr, val):
        index = 0
        while index < len(arr):
            if arr[index] <= val:
                index += 1
            else:
                break
        return index / len(arr)

    while True:
        hunk_num = int(input("hunk num: "))
        print(check_num(NUM_FILE_HUNK, hunk_num))
        file_num = int(input("file num: "))
        print(check_num(NUM_COMMIT_FILE, file_num))
