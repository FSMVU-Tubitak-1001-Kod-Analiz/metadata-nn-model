import os
import sys
import shutil
import torch
import random
import numpy as np
import logging

from transformers import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm


from model import build_model
from dataset import build_data_loaders
from config import Config
from args import parse_args
from utils import *

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, config: Config, save_train_args: bool = True):
        self.config = config
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.eval_loader = None
        self.device = None
        self.epoch = 0
        self.n_no_improve = 0
        self.best_metric = 0
        self.save_train_args = save_train_args

    def save_launch_args(self):
        config = self.config
        config.to_json(os.path.join(config.save_dir, "config.json"))

        flag_args = {
            "enable_checkpoint",
            "do_train",
            "do_test",
            "enable_cv",
            "use_roberta",
        }
        lauch_command = " ".join(
            f"--{k} {v}" for k, v in config.namespace.items() if k not in flag_args
        )

        for arg in flag_args:
            if config.namespace[arg]:
                lauch_command += f" --{arg}"

        lauch_command = f"python {sys.argv[0]} " + lauch_command

        with open(os.path.join(config.save_dir, "train_command.txt"), "w") as f:
            f.write(lauch_command)

    def _build_criterion(self, config: Config):
        return torch.nn.CrossEntropyLoss()

    def _build_optimizer(self, model: torch.nn.Module, config: Config):
        parameters_list = [
            {
                "params": model.msg_encoder.embeddings.parameters(),
                "lr": 1e-5,
            },
            {
                "params": model.msg_encoder.encoder.layer[0].parameters(),
                "lr": 1e-5,
            },
            {
                "params": model.msg_encoder.encoder.layer[1].parameters(),
                "lr": 1e-5,
            },
            {
                "params": model.msg_encoder.encoder.layer[2].parameters(),
                "lr": 1e-5,
            },
            {
                "params": model.msg_encoder.encoder.layer[3].parameters(),
                "lr": 2e-5,
            },
            {
                "params": model.msg_encoder.encoder.layer[4].parameters(),
                "lr": 2e-5,
            },
            {
                "params": model.msg_encoder.encoder.layer[5].parameters(),
                "lr": 2e-5,
            },
            {
                "params": model.msg_encoder.encoder.layer[6].parameters(),
                "lr": 2e-5,
            },
            {
                "params": model.msg_encoder.encoder.layer[7].parameters(),
                "lr": 4e-5,
            },
            {
                "params": model.msg_encoder.encoder.layer[8].parameters(),
                "lr": 4e-5,
            },
            {
                "params": model.msg_encoder.encoder.layer[9].parameters(),
                "lr": 4e-5,
            },
            {
                "params": model.msg_encoder.encoder.layer[10].parameters(),
                "lr": 4e-5,
            },
            {
                "params": model.msg_encoder.encoder.layer[11].parameters(),
                "lr": 4e-5,
            },
            {
                "params": model.code_change_encoder.parameters(),
                "lr": 1e-4,
            },
            {"params": model.feature_combiner.parameters(), "lr": 1e-4},
            {"params": model.classifier.parameters(), "lr": 1e-4},
            {"params": model.text_code_combiner.parameters(), "lr": 1e-4},
        ]
        return AdamW(parameters_list)

    def prepare(self):
        if self.config.device == "gpu":
            device = "cuda:0"
        elif self.config.device == "gpu2":
            device = "cuda:1"

        if torch.cuda.is_available():
            self.device = torch.device(device)
            logger.warn(self.device)
        else:
            self.device = torch.device("cpu")

        if self.config.enable_cv:
            self.train_loader, self.test_loader = build_data_loaders(self.config)
        else:
            self.train_loader, self.test_loader, self.eval_loader = build_data_loaders(
                self.config
            )
        self.model = build_model(self.config).to(self.device)
        self.criterion = self._build_criterion(self.config)
        self.optimizer = self._build_optimizer(self.model, self.config)
        if hasattr(self.model, "bert_config"):
            logger.info(f"bert config: {self.model.bert_config}")

    def resume_from_checkpoint(self):
        if os.path.exists(os.path.join(self.config.save_dir, "checkpoint.pt")):
            checkpoint = torch.load(os.path.join(self.config.save_dir, "checkpoint.pt"))
            self.model.load_state_dict(checkpoint["state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            # self.scheduler.load_state_dict(checkpoint["scheduler"])
            self.epoch = checkpoint["epoch"]
            self.n_no_improve = checkpoint["n_no_imporve"]
            self.best_metric = checkpoint["best_metric"]

    def save_checkpoint(self, is_best: bool = False):
        filename = os.path.join(self.config.save_dir, "checkpoint.pt")
        if os.path.exists(filename):
            os.remove(filename)
        torch.save(
            {
                "epoch": self.epoch,
                "n_no_imporve": self.n_no_improve,
                "best_metric": self.best_metric,
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                # "scheduler": self.scheduler.state_dict(),
            },
            filename,
        )
        if is_best:
            best_model = os.path.join(self.config.save_dir, "best_model.pt")
            if os.path.exists(best_model):
                os.remove(best_model)
            shutil.copyfile(filename, best_model)

    def model_forward(self, batch, state_word, state_sent, state_hunk):
        assert self.model is not None, "trainer should be prepared"
        batch = move_to_device(batch, self.device)
        layer_out = self.model(
            hid_state_word=state_word,
            hid_state_sent=state_sent,
            hid_state_hunk=state_hunk,
            **batch,
        )
        loss = self.criterion(layer_out, batch["labels"])
        return layer_out, batch["labels"], loss

    def model_evaluate(self):
        self.model.eval()
        self.model.set_eval()
        with torch.no_grad():
            losses, preds, tgts = [], [], []
            state_word = self.model.init_hidden_word().to(self.device)
            state_sent = self.model.init_hidden_sent().to(self.device)
            state_hunk = self.model.init_hidden_hunk().to(self.device)
            for batch in tqdm(self.eval_loader, total=len(self.eval_loader)):
                out, tgt, loss = self.model_forward(
                    batch,
                    state_word=state_word,
                    state_hunk=state_hunk,
                    state_sent=state_sent,
                )
                losses.append(loss.item())
                pred = (
                    torch.nn.functional.softmax(out, dim=1)
                    .argmax(dim=1)
                    .cpu()
                    .detach()
                    .numpy()
                )
                preds.append(pred)
                tgt = tgt.cpu().detach().numpy()
                tgts.append(tgt)
        tgts = [x for xx in tgts for x in xx]
        preds = [x for xx in preds for x in xx]
        metrics = {
            "loss": np.mean(losses),
            "accuracy": accuracy_score(tgts, preds),
            "macro_f1": f1_score(tgts, preds, average="macro"),
        }
        self.model.train()
        self.model.set_train()
        return metrics

    def model_test(self):
        self.model.eval()
        self.model.set_eval()
        with torch.no_grad():
            losses, preds, tgts = [], [], []
            state_word = self.model.init_hidden_word().to(self.device)
            state_sent = self.model.init_hidden_sent().to(self.device)
            state_hunk = self.model.init_hidden_hunk().to(self.device)
            for batch in tqdm(self.test_loader, total=len(self.test_loader)):
                out, tgt, loss = self.model_forward(
                    batch,
                    state_word=state_word,
                    state_hunk=state_hunk,
                    state_sent=state_sent,
                )
                losses.append(loss.item())
                pred = (
                    torch.nn.functional.softmax(out, dim=1)
                    .argmax(dim=1)
                    .cpu()
                    .detach()
                    .numpy()
                )
                preds.append(pred)
                tgt = tgt.cpu().detach().numpy()
                tgts.append(tgt)
        tgts = [x for xx in tgts for x in xx]
        preds = [x for xx in preds for x in xx]
        metrics = {
            "loss": np.mean(losses),
            "acc": accuracy_score(tgts, preds),
            "weighted_f1": f1_score(tgts, preds, average="weighted"),
        }
        logger.info(
            f"[epoch {self.epoch}] test accuracy: {accuracy_score(tgts, preds)} macro_f1: {f1_score(tgts, preds, average='macro')}"
        )
        self.model.train()
        self.model.set_train()

    def train(self):
        config = self.config
        self.prepare()
        if self.save_train_args:
            self.save_launch_args()
        if config.enable_checkpoint:
            self.resume_from_checkpoint()

        logger.info("********** Running training **********")
        logger.info(f"  Num Examples = {config.train_loader_len}")
        logger.info(f"  Num Epochs = {config.max_epochs}")
        logger.info(f"  Train Batch Size = {config.batch_size}")
        logger.info(f"  Accumulate Gradient Step = {config.grad_acc}")
        logger.info(f"  Train Arguments = {config}")
        logger.info(f"  Model Structure = {self.model}")

        # metrics = self.model_evaluate()
        # logger.info(f"start training, acc = {metrics['acc']:.5f}")

        global_step = 0
        while self.epoch <= config.max_epochs:
            self.model.train()
            self.model.set_train()
            self.optimizer.zero_grad()

            # train one epoch
            preds, tgts = [], []
            for batch in tqdm(self.train_loader, total=len(self.train_loader)):
                state_word = self.model.init_hidden_word().to(self.device)
                state_sent = self.model.init_hidden_sent().to(self.device)
                state_hunk = self.model.init_hidden_hunk().to(self.device)
                out, tgt, loss = self.model_forward(
                    batch=batch,
                    state_word=state_word,
                    state_hunk=state_hunk,
                    state_sent=state_sent,
                )
                pred = (
                    torch.nn.functional.softmax(out, dim=1)
                    .argmax(dim=1)
                    .cpu()
                    .detach()
                    .numpy()
                )
                preds.append(pred)
                tgt = tgt.cpu().detach().numpy()
                tgts.append(tgt)
                if config.grad_acc > 1:
                    loss = loss / config.grad_acc

                loss.backward()
                global_step += 1

                if global_step % config.grad_acc == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    # self.scheduler.step()

            tgts = [x for xx in tgts for x in xx]
            preds = [x for xx in preds for x in xx]
            logger.info(f"train accuracy: {accuracy_score(tgts, preds)}")
            # evaluate after training one epoch
            metrics = self.model_evaluate()
            self.model_test()
            if metrics["accuracy"] > self.best_metric:
                self.n_no_improve = 0
                self.best_metric = metrics["accuracy"]
                is_improved = True
            else:
                self.n_no_improve += 1
                is_improved = False

            # save checkpoint every epoch
            if config.enable_checkpoint:
                self.save_checkpoint(is_improved)

            # log epoch info
            logger.info(f"[epoch {self.epoch}]  train loss = {loss:.5f}")
            logger.info(
                f"[epoch {self.epoch}]  evaluate loss = {metrics['loss']:.5f}, accuracy = {metrics['accuracy']:.5f}, "
                f"macro_f1 = {metrics['macro_f1']:.5f}, best_metric = {self.best_metric:.5f}, epoch_no_improve = {self.n_no_improve}"
            )

            # check whther n_no_improve is not less than patience
            if self.n_no_improve >= config.patience:
                logger.info(
                    f"no improve with {config.patience} epoch, finish learning."
                )
                break

            # increse epch
            self.epoch += 1


if __name__ == "__main__":
    # some transformers environment variable
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # parse argument namespace to Config
    config = Config(parse_args())
    config.save_dir = os.path.join(config.save_dir, config.name)
    os.makedirs(config.save_dir, exist_ok=True)

    set_global_seed(config)
    set_global_logger(logger, config)

    # start training
    trainer = Trainer(config)
    trainer.train()

    logging.info("finished")
