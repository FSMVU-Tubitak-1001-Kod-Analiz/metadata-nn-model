"""Config loading. The config is a plain nested dict read from YAML."""
from __future__ import annotations

from pathlib import Path

import yaml

DEFAULT_CONFIG = "configs/config.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    path = Path(path).resolve()
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Relative paths in the config are relative to the project root
    # (the folder that contains configs/).
    cfg["root"] = path.parent.parent
    return cfg


def resolve(cfg: dict, rel: str | Path) -> Path:
    """Turn a path from the config into an absolute path under the project root."""
    p = Path(rel)
    return p if p.is_absolute() else cfg["root"] / p
