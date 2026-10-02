"""JSON configuration loading.

Every tunable value (paths, hyperparameters, column lists, label sets, ...)
lives in a JSON file under ``configs/``.  This module turns those files into
strictly-validated dataclasses:

* unknown keys are rejected (catches typos instead of silently ignoring them)
* missing required keys are reported together, with their full dotted path
* keys starting with ``_`` (e.g. ``"_comment"``) are ignored, since JSON has
  no native comment syntax

Relative paths inside configs are resolved against the project root by
:func:`resolve_path`.
"""

from __future__ import annotations

import dataclasses
import json
import os
import typing
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar

T = TypeVar("T")


class ConfigError(ValueError):
    """Raised when a config file is malformed or fails validation."""


# ---------------------------------------------------------------------------
# Project root / path handling
# ---------------------------------------------------------------------------
def _find_project_root() -> Path:
    env = os.environ.get("DIFFLABEL_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "pyproject.toml").exists():
        return candidate
    return Path.cwd()


PROJECT_ROOT: Path = _find_project_root()


def resolve_path(path: str | os.PathLike) -> Path:
    """Absolute paths pass through; relative ones are anchored at PROJECT_ROOT."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else PROJECT_ROOT / p


def format_path(template: str, **fields: Any) -> Path:
    """Fills ``{fold}``-style placeholders in a path template, then resolves it."""
    return resolve_path(template.format(**fields))


# ---------------------------------------------------------------------------
# Generic strict JSON -> dataclass conversion
# ---------------------------------------------------------------------------
def _strip_comments(data: Any) -> Any:
    if isinstance(data, dict):
        return {k: _strip_comments(v) for k, v in data.items() if not str(k).startswith("_")}
    if isinstance(data, list):
        return [_strip_comments(v) for v in data]
    return data


def _convert(tp: Any, value: Any, path: str) -> Any:
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)

    if tp is Any:
        return value

    if origin is typing.Union:  # Optional[X]
        if value is None and type(None) in args:
            return None
        non_none = [a for a in args if a is not type(None)]
        return _convert(non_none[0], value, path)

    if dataclasses.is_dataclass(tp):
        return _build(tp, value, path)

    if origin in (list, List):
        if not isinstance(value, list):
            raise ConfigError(f"{path}: expected a list, got {type(value).__name__}")
        return [_convert(args[0], v, f"{path}[{i}]") for i, v in enumerate(value)]

    if origin in (dict, Dict):
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected an object, got {type(value).__name__}")
        return {k: _convert(args[1], v, f"{path}.{k}") for k, v in value.items()}

    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path}: expected true/false, got {value!r}")
        return value
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{path}: expected an integer, got {value!r}")
        return value
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path}: expected a number, got {value!r}")
        return float(value)
    if tp is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path}: expected a string, got {value!r}")
        return value

    return value


def _build(cls: Type[T], data: Any, path: str) -> T:
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected an object, got {type(data).__name__}")

    hints = typing.get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls)}

    unknown = sorted(set(data) - set(fields))
    if unknown:
        valid = ", ".join(sorted(fields))
        raise ConfigError(f"{path}: unknown key(s) {unknown}. Valid keys: {valid}")

    kwargs: Dict[str, Any] = {}
    missing: List[str] = []
    for name, f in fields.items():
        if name in data:
            kwargs[name] = _convert(hints[name], data[name], f"{path}.{name}")
        elif f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:  # type: ignore[misc]
            missing.append(f"{path}.{name}")
    if missing:
        raise ConfigError(f"Missing required config key(s): {', '.join(missing)}")

    return cls(**kwargs)


def load_config(cls: Type[T], path: str | os.PathLike) -> T:
    """Reads a JSON file and builds an instance of dataclass ``cls`` from it."""
    p = resolve_path(path)
    if not p.is_file():
        raise ConfigError(f"Config file not found: {p}")
    try:
        with open(p, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{p} is not valid JSON: {exc}") from exc
    return _build(cls, _strip_comments(raw), p.name)


# ---------------------------------------------------------------------------
# Config schemas
# ---------------------------------------------------------------------------
# -- 01_prepare_csv ---------------------------------------------------------
@dataclasses.dataclass
class MergeConfig:
    enabled: bool
    input_dir: str
    output_csv: str


@dataclasses.dataclass
class FoldsConfig:
    n_splits: int
    random_state: int
    holdout_fraction: float            # share of rows leaving the train split (70:15:15 -> 0.30)
    test_fraction_of_holdout: float    # share of the holdout that becomes test (rest is val)
    kfold_subdir: str
    split_subdir: str


@dataclasses.dataclass
class LabelsConfig:
    all: List[str]    # keywords recognised when standardising raw label text
    keep: List[str]   # labels that survive into the final dataset


@dataclasses.dataclass
class PrepareCsvConfig:
    raw_csv: str
    output_dir: str
    output_csv_name: str
    rows_per_repo: int
    columns_to_keep: List[str]
    columns_to_drop: List[str]
    labels: LabelsConfig
    balance_labels_early: bool
    balance_seed: int
    merge: MergeConfig
    folds: FoldsConfig
    repo_column: str = "full_repo_name"
    label_column: str = "labels"
    analysis_target_label: Optional[str] = None


# -- 02_create_embeddings ---------------------------------------------------
@dataclasses.dataclass
class CreateEmbeddingsConfig:
    input_csv: str
    column: str
    model_name: str
    embedding_dir: str
    pooling: str            # "pooler" or "cls"
    device: str
    batch_size: int
    max_seq_length: int
    index_column: str = "index"


# -- training (shared pieces) -----------------------------------------------
@dataclasses.dataclass
class ModelConfig:
    hidden_sizes_embedding_pipeline: List[int]
    hidden_sizes_metadata_pipeline: List[int]
    hidden_sizes_merged_pipeline: List[int]
    output_size: int
    dropout_rate: float
    modality_dropout_p: float = 0.1
    fusion_dim: Optional[int] = None


@dataclasses.dataclass
class OptimConfig:
    batch_size: int
    num_epochs: int
    learning_rate: float
    scaler_batch_size: int
    scheduler_mode: str
    scheduler_patience: int
    scheduler_factor: float


@dataclasses.dataclass
class SplitConfig:
    test_size: float                 # fraction held out from train
    val_fraction_of_holdout: float   # fraction of the held-out part that becomes val


@dataclasses.dataclass
class MetadataEmbeddingConfig:
    model_name: str
    columns: List[str]
    max_length: int
    separator: str


@dataclasses.dataclass
class EmbeddingFilesConfig:
    content_before_dir: str
    patch_before_dir: str


@dataclasses.dataclass
class BaselinesConfig:
    enabled: bool
    feature_column: str                       # "embedding" or "metadata_tensor"
    train_on_train_and_val: bool
    models: Dict[str, Dict[str, Any]]         # name -> constructor kwargs (seed/num_class injected)


@dataclasses.dataclass
class OutputsConfig:
    model_path: str
    scaler_path: str
    dnn_confusion_png: str
    ml_metrics_csv: str
    ml_confusion_png: str


# -- 03a / 03b: single-pipeline training ------------------------------------
@dataclasses.dataclass
class SingleTrainConfig:
    pipeline: str                    # "code" or "metadata"
    seed: int
    device: str
    input_csv: str
    model: ModelConfig
    optim: OptimConfig
    split: SplitConfig
    baselines: BaselinesConfig
    outputs: OutputsConfig
    embeddings: Optional[EmbeddingFilesConfig] = None     # required for pipeline == "code"
    metadata: Optional[MetadataEmbeddingConfig] = None    # required for pipeline == "metadata"
    label_column: str = "labels"

    def __post_init__(self) -> None:
        if self.pipeline not in ("code", "metadata"):
            raise ConfigError(f"pipeline must be 'code' or 'metadata', got {self.pipeline!r}")
        if self.pipeline == "code" and self.embeddings is None:
            raise ConfigError("pipeline 'code' requires an 'embeddings' section")
        if self.pipeline == "metadata" and self.metadata is None:
            raise ConfigError("pipeline 'metadata' requires a 'metadata' section")


# -- 04: gated-fusion, k-fold -----------------------------------------------
@dataclasses.dataclass
class GatedFusionTrainConfig:
    seed: int
    device: str
    folds: List[int]
    folds_dir: str
    embeddings: EmbeddingFilesConfig
    metadata: MetadataEmbeddingConfig
    model: ModelConfig
    optim: OptimConfig
    baselines: BaselinesConfig
    outputs: OutputsConfig           # path templates may contain {fold}
    label_column: str = "labels"
