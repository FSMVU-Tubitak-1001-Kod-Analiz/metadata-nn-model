from .trainer_cc2vec import Trainer as CC2VecTrainer
from .trainer_cc2vec import logger as CC2VecLogger

from .trainer import Trainer
from .trainer import logger as train_logger
from .test_model import Tester
from .test_model import logger as test_logger

__all__ = [
    "CC2VecTrainer",
    "CC2VecLogger",
    "Trainer",
    "train_logger",
    "Tester",
    "test_logger",
]
