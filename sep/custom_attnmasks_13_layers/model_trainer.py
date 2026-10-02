import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import datasets
import tqdm
from datasets import Dataset, concatenate_datasets
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from transformers.tokenization_utils_base import BatchEncoding
import torch
import os
import json
from roberta_for_mlm_13_layers import RobertaForMLM13Layers


class ModelTrainer:
    def __init__(self, model, tokenizer, data_collator, save_path):
        self.trainer = None
        self.model = model
        self.data_collator = data_collator
        self.tokenizer = tokenizer
        self.save_path = save_path
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    def add_to_vocabulary(self, keyword_tokens):
        tokens_added = self.tokenizer.add_tokens(keyword_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))
        print("Added", tokens_added, "tokens")

    def tokenize_loop(self, codes):
        code_tqdm = tqdm.tqdm(enumerate(codes), total=len(codes))
        tokenized = []
        for i, sample in code_tqdm:
            tokenized.append(self.tokenizer(sample, return_tensors='pt', truncation=True, padding=True, max_length=510))
            codes[i] = None  # scuffed memory management
        return datasets.Dataset.from_list(tokenized)

    @staticmethod
    def _pool(file_name, tokenizer, samples, position_embeddings):
        with open(file_name, "w") as file:
            samples = [sample["sample"] for sample in samples]
            json.dump(tokenizer(samples, truncation=True, padding=True, max_length=position_embeddings).data, file)

    def tokenize_to_files(self, codes, output_folder, keep_tokenize_results, position_embeddings):
        """
        Split codes to tokenize into batches of 1000 samples. We need to do this because sadly HuggingFace tokenizer
        get stuck when it is initiated with a very large data file. If the files are in memory though, it's all cool.
        """
        file_size = 10000

        print("Writing tokenize results to files at:", output_folder)

        code_tqdm = tqdm.tqdm(range(0, len(codes), file_size), position=0, total=(len(codes) // file_size) + 1)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder, exist_ok=True)

        with ProcessPoolExecutor(16) as executor:
            futures = []
            for i in range(0, len(codes), file_size):
                file_name = output_folder + str(i // file_size) + ".json"
                samples = codes[i: i + file_size]
                futures.append(executor.submit(ModelTrainer._pool, file_name, self.tokenizer, samples, position_embeddings))

            for future in as_completed(futures):
                if future.exception():
                    print(future.exception())
                code_tqdm.update(1)

        print("Combining results...")
        files = os.listdir(output_folder)
        file_tqdm = tqdm.tqdm(files, position=1, leave=True)

        dataset_list = []
        for i in file_tqdm:
            i = os.path.join(output_folder, i)
            with open(i, "r") as f:
                be = BatchEncoding(json.load(f))
                be.convert_to_tensors("pt")  # this operation is inplace
                dataset_list.append(Dataset.from_dict(be))

        file_tqdm.close()

        if not keep_tokenize_results:
            print("Removing temporary files")
            shutil.rmtree(output_folder)

        return concatenate_datasets(dataset_list)

    def train(self, train_codes, test_codes, train_output, test_output, keep_tokenize_results=False,
              position_embeddings=510, **kwargs):
        print("Tokenizing train dataset...")
        train_dataset = self.tokenize_to_files(train_codes, train_output, keep_tokenize_results, position_embeddings)
        del train_codes

        print("Tokenizing validation dataset...")
        val_output = test_output
        val_dataset = self.tokenize_to_files(test_codes, val_output, keep_tokenize_results, position_embeddings)
        del test_codes

        # if self.tokenizer.pad_token is None:
        #     print("Tokenizer pad token is none, adding new special pad token")
        #     self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        training_args = TrainingArguments(
            output_dir=self.save_path,
            report_to=["tensorboard"],
            logging_strategy="step",
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            save_total_limit=2,
            learning_rate=2e-5,
            num_train_epochs=kwargs.pop("num_train_epochs", 100),
            weight_decay=0.01,
            per_device_train_batch_size=32, #24
            per_device_eval_batch_size=32            
        )
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=self.data_collator
        )

        start = datetime.now()
        try:
            self.trainer.train()
        except KeyboardInterrupt as e:
            pass

        end = datetime.now()
        diff = end - start
        metadata = {
            "tokenizer_name": self.tokenizer.name_or_path,
            "training_time": diff
        }
        metadata = metadata | kwargs
        metadata = {key: str(value) for key, value in metadata.items()}
        metadata["training_args"] = training_args.to_dict()

        try:
            with open(os.path.join(self.save_path, "metadata.json"), "w") as metadata_file:
                json.dump(metadata, metadata_file, indent=4)
        except Exception as e:
            print(
                "Exception when writing metadata, I didn't want it to crash the program, not at this point, so we will just keep going.")
            print(metadata)
            print(e)

        self.trainer.save_model(os.path.join(self.save_path, "saved_model"))
        return self.trainer
