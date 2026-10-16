import os
import torch
import pandas
import logging

from model import build_model
from dataset import build_data_loaders
from args import parse_args
from config import Config
from utils import *
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

logger = logging.getLogger(__file__)

BSET_MODEL_NAME = "best_model.pt"


class Tester:
    def __init__(self, config: Config):
        self.config = config
        self.prepare()

    def _build_criterion(self, config: Config):
        return torch.nn.CrossEntropyLoss()

    def model_forward(self, batch):
        assert self.model is not None, "trainer should be prepared"
        batch = move_to_device(batch, self.device)
        layer_out = self.model(**batch)
        loss = self.criterion(layer_out, batch["labels"])
        return layer_out, batch["labels"], loss

    def prepare(self):
        if self.config.device == "gpu" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # create necessary components
        self.test_loader, self.test_dataset = build_data_loaders(
            self.config, testing=True
        )
        self.model = build_model(self.config).to(self.device)
        self.criterion = self._build_criterion(self.config)

        if not os.path.exists(os.path.join(self.config.save_dir, BSET_MODEL_NAME)):
            raise RuntimeError(
                f"there is no best_model checkpoint in {self.config.save_dir}"
            )

        # load components weight from check point
        checkpoint = torch.load(os.path.join(self.config.save_dir, BSET_MODEL_NAME))
        print(os.path.join(self.config.save_dir, BSET_MODEL_NAME))
        self.model.load_state_dict(checkpoint["state_dict"])

    def export_failed_cases(self, preds, tgts):
        assert len(preds) == len(tgts)

        convert_map = {0: "Corrective", 1: "Adaptive", 2: "Perfective"}
        miss_cases = 0
        commit, user, repo, target, predict = [], [], [], [], []
        for index, (pred, tgt) in enumerate(zip(preds, tgts)):
            if pred != tgt:
                commit.append(self.test_dataset.get_commit_by_index(index))
                user.append(self.test_dataset.get_user_by_index(index))
                repo.append(self.test_dataset.get_repo_by_index(index))
                target.append(convert_map[tgt])
                predict.append(convert_map[pred])
                miss_cases += 1

        df = pandas.DataFrame(
            {
                "commit": commit,
                "user": user,
                "repo": repo,
                "target": target,
                "predict": predict,
            }
        )
        df.to_csv(os.path.join(self.config.save_dir, "failed_cases.csv"), index=False)
        logger.info(
            f"failed cases exported to {os.path.join(self.config.save_dir, 'failed_cases.csv')} "
            f"with accuracy {(len(tgts) - miss_cases) / len(tgts):.5f}"
        )

    def test(self):
        self.model.eval()
        with torch.no_grad():
            losses, preds, tgts = [], [], []
            for batch in self.test_loader:
                out, tgt, loss = self.model_forward(batch)
                losses.append(loss)
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
        adaptive_recall = recall_score(
            y_true=tgts, y_pred=preds, labels=[1], average=None
        )
        adaptive_f1 = f1_score(tgts, preds, labels=[1], average=None)

        perfective_precision = precision_score(
            y_true=tgts, y_pred=preds, labels=[2], average=None
        )
        perfective_recall = recall_score(
            y_true=tgts, y_pred=preds, labels=[2], average=None
        )
        perfective_f1 = f1_score(tgts, preds, labels=[2], average=None)
        cm = confusion_matrix(tgts, preds)

        logger.info(f"test finished!")
        logger.info(f"test accuracy = {acc:.5f}")

        # self.export_failed_cases(preds, tgts)
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
            "cm": cm,
        }
        logger.info(metrics)
        return metrics


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

    tester = Tester(config)
    tester.test()

    logging.info("finished")
