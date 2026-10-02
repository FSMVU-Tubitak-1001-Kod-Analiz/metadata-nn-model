#!/usr/bin/env python
"""Clean + balance the raw CSV and generate CV folds.

Usage:  python scripts/01_prepare_csv.py [--config configs/prepare_csv.json]
"""
from difflabel.cli import run_cli
from difflabel.config import PrepareCsvConfig
from difflabel.pipelines import prepare_csv

if __name__ == "__main__":
    run_cli(PrepareCsvConfig, prepare_csv.run, "configs/prepare_csv.json", "Clean + balance the raw CSV and generate CV folds.")
