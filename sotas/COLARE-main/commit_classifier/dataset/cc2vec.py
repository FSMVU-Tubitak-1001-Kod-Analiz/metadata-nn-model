from typing import Union
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

from nltk.tokenize import word_tokenize


NUM_FILE_HUNK = []
NUM_HUNK_LINE = []
NUM_LINE_WORD = []
NUM_COMMIT_FILE = []


class CodeDiffParser:
    def __init__(
        self,
        file_num_limit: int = None,
        hunk_num_limit: int = None,
        line_num_limit: int = None,
        word_num_limit: int = None,
    ) -> None:
        self.file_num_limit = file_num_limit or int(1e9)
        self.hunk_num_limit = hunk_num_limit or int(1e9)
        self.line_num_limit = line_num_limit or int(1e9)
        self.word_num_limit = word_num_limit or int(1e9)
        self.word_set = set()

    def parse_one_line(self, diff: str):
        tokens = word_tokenize(diff)
        NUM_LINE_WORD.append(len(tokens))
        for token in tokens:
            self.word_set.add(token)
        return tokens[: self.word_num_limit]

    def parse_one_hunk(self, diff: str):
        code_line_added = []
        code_line_deleted = []
        for line in diff.splitlines():
            if line.startswith("+"):
                code_line_added.append(line[1:].strip())
            elif line.startswith("-"):
                code_line_deleted.append(line[1:].strip())

        NUM_HUNK_LINE.append(len(code_line_added))
        NUM_HUNK_LINE.append(len(code_line_deleted))

        code_line_added = code_line_added[: self.line_num_limit]
        code_line_deleted = code_line_deleted[: self.line_num_limit]

        for index, line in enumerate(code_line_added):
            code_line_added[index] = self.parse_one_line(line)
        for index, line in enumerate(code_line_deleted):
            code_line_deleted[index] = self.parse_one_line(line)

        # code_line_added = [line_num_limit][word_num_limit]
        return code_line_added, code_line_deleted

    def parse_one_file_diff(self, diff: str):
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

        hunks_added = hunks_added[: self.hunk_num_limit]
        hunks_deleted = hunks_deleted[: self.hunk_num_limit]

        # [hunk_num][line_num][word_num]
        return hunks_added, hunks_deleted

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

        files_added, files_deleted = [], []
        for file_diff in file_diffs:
            file_added, file_deleted = self.parse_one_file_diff(file_diff)
            files_added.append(file_added)
            files_deleted.append(file_deleted)

        files_added = files_added[: self.file_num_limit]
        files_deleted = files_deleted[: self.file_num_limit]

        return files_added, files_deleted


class CommitReader:
    def __init__(
        self,
        file_num_limit: int = 10,
        hunk_num_limit: int = 10,
        line_num_limit: int = 10,
        word_num_limit: int = 10,
    ) -> None:
        self.file_num_limit = file_num_limit
        self.hunk_num_limit = hunk_num_limit
        self.line_num_limit = line_num_limit
        self.word_num_limit = word_num_limit

        self.diff_parser = CodeDiffParser(
            file_num_limit=file_num_limit,
            hunk_num_limit=hunk_num_limit,
            line_num_limit=line_num_limit,
            word_num_limit=word_num_limit,
        )

    def get_word_dictionary(self):
        word_dict = {}
        for word in self.diff_parser.word_set:
            word_dict[word] = len(word_dict) + 1
        assert "<PAD>" not in word_dict
        word_dict["<PAD>"] = len(word_dict) + 1
        return word_dict

    def series_to_dict(self, series: pandas.Series) -> dict:
        label_map = {
            "P": 0,
            "A": 1,
            "C": 2,
        }
        label = label_map[series["maintenance_type"]]
        commit_message = series["msgs"]
        added_code, delted_code = self.diff_parser.parse(series["diffs"])

        return {
            "commit_message": commit_message,
            "added_code": added_code,
            "deleted_code": delted_code,
            "label": label,
            "commit_sha": series["commit"],
            "numerical_features": json.loads(series["feature"]),
        }

    def read(self, data_file):
        df = pandas.read_csv(data_file)
        for _, series in df.iterrows():
            yield self.series_to_dict(series)


@DATASET_REGISTRY.register("CC2Vec")
class CC2VecDataset(Dataset):
    def __init__(self, data_path: str, config, use_roberta=False) -> None:
        super().__init__()
        self.data_path = data_path
        self.config = config

        # the same num limit with CC2Vec
        self.file_num_limit = 2
        self.hunk_num_limit = 5
        self.line_num_limit = 8
        self.word_num_limit = 32

        self.data_reader = CommitReader(
            file_num_limit=self.file_num_limit,
            hunk_num_limit=self.hunk_num_limit,
            line_num_limit=self.line_num_limit,
            word_num_limit=self.word_num_limit,
        )

        self.commit_messages = []
        self.commit_added_code = []
        self.commit_deleted_code = []
        self.commit_labels = []
        self.commit_sha = []
        self.commit_features = []
        for item in self.data_reader.read(data_path):
            self.commit_messages.append(item["commit_message"])
            self.commit_added_code.append(item["added_code"])
            self.commit_deleted_code.append(item["deleted_code"])
            self.commit_labels.append(item["label"])
            self.commit_sha.append(item["commit_sha"])
            self.commit_features.append(item["numerical_features"])
        self.word_dictionary = self.data_reader.get_word_dictionary()
        self.id2word = {v: k for k, v in self.word_dictionary.items()}
        self.message_tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/codebert-base"
        )
        self.message_collator = DataCollatorWithPadding(self.message_tokenizer)
        self.total_tokens = []
        self.preprocess()

    def preprocess(self):
        # tokenize commit message
        self.commit_messages_tokenized = [
            self.message_tokenizer(
                message,
                truncation=True,
                padding="max_length",
                max_length=256,
            )
            for message in self.commit_messages
        ]

        # padding codes
        for index, files_change in enumerate(self.commit_added_code):
            self.commit_added_code[index] = self.padding_files(files_change)
        for index, files_change in enumerate(self.commit_deleted_code):
            self.commit_deleted_code[index] = self.padding_files(files_change)

    def get_padded_word(self):
        return "<PAD>"

    def get_padded_line(self):
        return [self.get_padded_word() for _ in range(self.word_num_limit)]

    def get_padded_hunk(self):
        return [self.get_padded_line() for _ in range(self.line_num_limit)]

    def get_padded_file(self):
        return [self.get_padded_hunk() for _ in range(self.hunk_num_limit)]

    def padding_files(self, files: list):
        # files: [file][hunk][line][word]

        for _ in range(self.file_num_limit - len(files)):
            files.append(self.get_padded_file())

        for index, file in enumerate(files):
            files[index] = self.padding_file(file)

        return files

    def padding_file(self, file: list):
        # file: [hunk][line][word]

        for _ in range(self.hunk_num_limit - len(file)):
            file.append(self.get_padded_hunk())

        for index, hunk in enumerate(file):
            file[index] = self.padding_hunk(hunk)

        return file

    def padding_hunk(self, hunk: list):
        # hunk: [line][word]

        for _ in range(self.line_num_limit - len(hunk)):
            hunk.append(self.get_padded_line())

        for index, line in enumerate(hunk):
            hunk[index] = self.padding_line(line)

        return hunk

    def padding_line(self, line: list):
        # line: [word]

        for _ in range(self.word_num_limit - len(line)):
            line.append(self.get_padded_word())

        return self.encode_line(line)

    def encode_line(self, line: list):
        return [self.word_dictionary[word] for word in line]

    def decode_line(self, line: list):
        return [self.id2word[word] for word in line]

    def decode_hunk(self, hunk: list):
        return [self.decode_line(line) for line in hunk]

    def decode_file(self, file: list):
        return [self.decode_hunk(hunk) for hunk in file]

    def decode_files(self, files: list):
        return [self.decode_file(file) for file in files]

    def __len__(self) -> int:
        return len(self.commit_messages)

    def __getitem__(self, index) -> dict:
        return {
            "label": self.commit_labels[index],
            "msg_input_ids": self.commit_messages_tokenized[index]["input_ids"],
            "msg_attention_mask": self.commit_messages_tokenized[index][
                "attention_mask"
            ],
            "commit_added_code": self.commit_added_code[index],
            "commit_deleted_code": self.commit_deleted_code[index],
            "commit_sha": self.commit_sha[index],
            "numerical_features": self.commit_features[index],
        }


def collate_fn(batch, msg_collator: DataCollatorWithPadding):
    msg_features = defaultdict(list)
    added_code = []
    deleted_code = []
    commit_sha = []
    numerical_features = []

    for item in batch:
        for key, value in item.items():
            if key.startswith("msg"):
                msg_features[key[4:]].append(value)
            elif key == "label":
                msg_features[key].append(value)
            elif key == "commit_added_code":
                added_code.append(value)
            elif key == "commit_deleted_code":
                deleted_code.append(value)
            elif key == "commit_sha":
                commit_sha.append(value)
            elif key == "numerical_features":
                numerical_features.append(value)

    msg_features = msg_collator(msg_features)
    msg_features["added_code"] = torch.tensor(added_code, dtype=torch.long)
    msg_features["deleted_code"] = torch.tensor(deleted_code, dtype=torch.long)
    msg_features["numerical_features"] = torch.tensor(
        numerical_features, dtype=torch.float32
    )
    return msg_features
