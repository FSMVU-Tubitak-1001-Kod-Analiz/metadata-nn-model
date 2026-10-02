import json
from pathlib import Path

import pytest

from difflabel.config import (
    ConfigError, CreateEmbeddingsConfig, GatedFusionTrainConfig, PrepareCsvConfig,
    SingleTrainConfig, load_config, resolve_path,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"

SHIPPED = [
    ("prepare_csv.json", PrepareCsvConfig),
    ("create_embeddings_content_before.json", CreateEmbeddingsConfig),
    ("create_embeddings_patch_before.json", CreateEmbeddingsConfig),
    ("train_single_code.json", SingleTrainConfig),
    ("train_single_metadata.json", SingleTrainConfig),
    ("train_gated_fusion.json", GatedFusionTrainConfig),
]


@pytest.mark.parametrize("name,cls", SHIPPED)
def test_shipped_configs_load(name, cls):
    cfg = load_config(cls, CONFIGS / name)
    assert isinstance(cfg, cls)


def _write(tmp_path, data):
    p = tmp_path / "c.json"
    p.write_text(json.dumps(data))
    return p


def test_unknown_key_rejected(tmp_path):
    data = json.loads((CONFIGS / "create_embeddings_content_before.json").read_text())
    data["batchsize"] = 8  # typo
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(CreateEmbeddingsConfig, _write(tmp_path, data))


def test_missing_keys_all_reported(tmp_path):
    data = json.loads((CONFIGS / "create_embeddings_content_before.json").read_text())
    del data["column"], data["batch_size"]
    with pytest.raises(ConfigError) as exc:
        load_config(CreateEmbeddingsConfig, _write(tmp_path, data))
    assert "column" in str(exc.value) and "batch_size" in str(exc.value)


def test_wrong_type_rejected(tmp_path):
    data = json.loads((CONFIGS / "create_embeddings_content_before.json").read_text())
    data["batch_size"] = "32"
    with pytest.raises(ConfigError, match="integer"):
        load_config(CreateEmbeddingsConfig, _write(tmp_path, data))


def test_pipeline_requires_matching_section(tmp_path):
    data = json.loads((CONFIGS / "train_single_code.json").read_text())
    del data["embeddings"]
    with pytest.raises(ConfigError, match="embeddings"):
        load_config(SingleTrainConfig, _write(tmp_path, data))


def test_comment_keys_ignored_and_paths_resolve():
    cfg = load_config(PrepareCsvConfig, CONFIGS / "prepare_csv.json")
    assert resolve_path(cfg.output_dir).is_absolute()
    assert resolve_path("/abs/path") == Path("/abs/path")
