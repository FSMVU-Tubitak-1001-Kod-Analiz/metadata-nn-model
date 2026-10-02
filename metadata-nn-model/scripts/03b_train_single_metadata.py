#!/usr/bin/env python
"""Train DNN + baselines on metadata embeddings only.

Usage:  python scripts/03b_train_single_metadata.py [--config configs/train_single_metadata.json]
"""
from difflabel.cli import run_cli
from difflabel.config import SingleTrainConfig
from difflabel.pipelines import train_single

if __name__ == "__main__":
    run_cli(SingleTrainConfig, train_single.run, "configs/train_single_metadata.json", "Train DNN + baselines on metadata embeddings only.")
