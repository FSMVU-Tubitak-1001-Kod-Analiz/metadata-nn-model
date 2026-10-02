#!/usr/bin/env python
"""Train the gated-fusion DNN + baselines for every fold.

Usage:  python scripts/04_train_gated_fusion.py [--config configs/train_gated_fusion.json]
"""
from difflabel.cli import run_cli
from difflabel.config import GatedFusionTrainConfig
from difflabel.pipelines import train_gated_fusion

if __name__ == "__main__":
    run_cli(GatedFusionTrainConfig, train_gated_fusion.run, "configs/train_gated_fusion.json", "Train the gated-fusion DNN + baselines for every fold.")
