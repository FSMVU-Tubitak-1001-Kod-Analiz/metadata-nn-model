import os
import logging
from args import parse_args
from trainer import Trainer, train_logger, Tester, test_logger
from config import Config
from utils import *
from cross_validation_launch import cross_validation_launch


def normal_lauch(conifg: Config):
    if config.dataset == "CC2Vec":
        from trainer import CC2VecTrainer, CC2VecLogger

        set_global_logger(CC2VecLogger, config)
        if config.do_train:
            trainer = CC2VecTrainer(config)
            trainer.train()
        if config.do_test:
            tester = Tester(config)
            tester.test()
    else:
        if conifg.do_train:
            trainer = Trainer(conifg)
            trainer.train()
        if conifg.do_test:
            tester = Tester(config)
            tester.test()


if __name__ == "__main__":
    # some transformers environment variable
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # parse argument namespace to Config
    config = Config(parse_args())
    config.save_dir = os.path.join(config.save_dir, config.name)
    os.makedirs(config.save_dir, exist_ok=True)

    set_global_seed(config)
    set_global_logger(train_logger, config)
    set_global_logger(test_logger, config)

    if config.enable_cv:
        cross_validation_launch(config)
    else:
        normal_lauch(config)

    logging.info("finished")
