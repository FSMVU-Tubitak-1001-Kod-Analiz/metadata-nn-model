"""Torch-dependent tests (skipped automatically when torch isn't installed)."""
import pytest

torch = pytest.importorskip("torch")

import numpy as np
import pandas as pd

from difflabel.config import ModelConfig, OptimConfig
from difflabel.data.datasets import CodeMetadataDataset, SingleFeatureDataset
from difflabel.models import CodeOnlyDiffNN, GatedFusionDiffNN, MetadataOnlyDiffNN, build_model
from difflabel.training.trainer import DUAL_LOOPS, SINGLE_LOOPS, fit_dnn

MODEL_CFG = ModelConfig(
    hidden_sizes_embedding_pipeline=[16, 12, 8], hidden_sizes_metadata_pipeline=[10, 6],
    hidden_sizes_merged_pipeline=[12, 10, 8, 6], output_size=2, dropout_rate=0.1,
)
OPTIM = OptimConfig(batch_size=16, num_epochs=2, learning_rate=1e-3, scaler_batch_size=32,
                    scheduler_mode="min", scheduler_patience=400, scheduler_factor=0.1)


def test_state_dict_keys_match_original_checkpoint_layout():
    keys = set(build_model("code", MODEL_CFG, code_dim=5).state_dict())
    expected = {f"code_{k}{i}.{p}" for i in (1, 2, 3) for k in ("fc", "bn") for p in ("weight", "bias")}
    expected |= {f"combined_{k}{i}.{p}" for i in (1, 2, 3, 4) for k in ("fc", "bn") for p in ("weight", "bias")}
    expected |= {"fc_out.weight", "fc_out.bias"}
    assert expected <= keys
    assert all(k in expected or k.endswith(("running_mean", "running_var", "num_batches_tracked")) for k in keys)

    gated = set(build_model("gated_fusion", MODEL_CFG, code_dim=5, meta_dim=4).state_dict())
    for name in ("code_proj", "meta_proj", "gate_fc"):
        assert f"{name}.weight" in gated


def test_forward_shapes():
    x, m = torch.randn(4, 5), torch.randn(4, 4)
    assert CodeOnlyDiffNN(5, [16, 12, 8], [12, 10, 8, 6], 2, 0.1).eval()(x).shape == (4, 2)
    assert MetadataOnlyDiffNN(4, [10, 6], [12, 10, 8, 6], 3, 0.1).eval()(m).shape == (4, 3)
    g = GatedFusionDiffNN(5, 4, [16, 12, 8], [10, 6], [12, 10, 8, 6], 2, 0.1).eval()
    assert g(x, m).shape == (4, 2) and 0 <= g.last_gate_mean <= 1


def _frames(n=96, code_dim=5, meta_dim=4):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, n)
    df = pd.DataFrame({
        "index": range(n),
        "embedding": [torch.tensor(rng.normal(size=code_dim) + c, dtype=torch.float32) for c in y],
        "metadata_tensor": [torch.tensor(rng.normal(size=meta_dim) + c, dtype=torch.float32) for c in y],
        "label_tensor": [torch.tensor(np.eye(2)[c], dtype=torch.float32) for c in y],
    })
    return df, df.iloc[:60].reset_index(drop=True), df.iloc[60:78].reset_index(drop=True), df.iloc[78:].reset_index(drop=True)


@pytest.mark.parametrize("kind", ["single", "dual"])
def test_fit_dnn_smoke(tmp_path, kind):
    df, tr, va, te = _frames()
    if kind == "single":
        build = lambda: build_model("code", MODEL_CFG, code_dim=5)
        loops, ds, kw = SINGLE_LOOPS, SingleFeatureDataset, {"feature_column": "embedding"}
    else:
        build = lambda: build_model("gated_fusion", MODEL_CFG, code_dim=5, meta_dim=4)
        loops, ds, kw = DUAL_LOOPS, CodeMetadataDataset, {}
    out, y = fit_dnn(build_model=build, loop_fns=loops, dataset_cls=ds, dataset_kwargs=kw, df=df, df_train=tr,
                     df_val=va, df_test=te, optim_cfg=OPTIM, device="cpu",
                     model_path=tmp_path / "m.pth", scaler_path=tmp_path / "s.pkl")
    assert out.shape == (len(te), 2) and y.shape == (len(te), 2)
    assert (tmp_path / "m.pth").exists() and (tmp_path / "s.pkl").exists()
