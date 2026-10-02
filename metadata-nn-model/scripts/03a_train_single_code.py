#!/usr/bin/env python
"""Train DNN + baselines on code embeddings only.

Usage:  python scripts/03a_train_single_code.py [--config configs/train_single_code.json]
"""
from difflabel.cli import run_cli
from difflabel.config import SingleTrainConfig
from difflabel.pipelines import train_single

if __name__ == "__main__":
    run_cli(SingleTrainConfig, train_single.run, "configs/train_single_code.json", "Train DNN + baselines on code embeddings only.")
