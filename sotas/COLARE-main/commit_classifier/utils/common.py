import os
import sys
import logging
import coloredlogs
import random
import torch
import numpy as np
from config import Config


def set_global_seed(config: Config):
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if config.device in ("gpu", "gpu2"):
        torch.cuda.manual_seed_all(config.seed)


def set_global_logger(logger, config: Config):
    logger.setLevel(level=logging.INFO)
    color_formatter = coloredlogs.ColoredFormatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    uncolor_formatter = logging.Formatter("[%(asctime)s] - [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(
        os.path.join(config.save_dir, "runlong.log"),
        mode="a",
        delay=False,
    )
    console_handler = logging.StreamHandler(sys.stdout)
    file_handler.setFormatter(uncolor_formatter)
    console_handler.setFormatter(color_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def move_to_device(batch, device):
    if isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [move_to_device(v, device) for v in batch]
    else:
        return batch.to(device=device)
