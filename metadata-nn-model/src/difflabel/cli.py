"""Tiny shared CLI used by every script in ``scripts/``."""

from __future__ import annotations

import argparse
from typing import Callable, Type

from difflabel.config import load_config


def run_cli(config_cls: Type, run_fn: Callable, default_config: str, description: str) -> None:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config", default=default_config,
        help=f"Path to a JSON config (relative paths resolve from the project root). Default: {default_config}",
    )
    args = parser.parse_args()
    run_fn(load_config(config_cls, args.config))
