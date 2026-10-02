import numpy as np
import pandas as pd

from difflabel.config import BaselinesConfig
from difflabel.evaluation.baselines import build_models, run_baselines

FAST = {
    "XGBoost": {"objective": "multi:softmax", "eval_metric": "mlogloss", "n_estimators": 5},
    "RandomForest": {"n_estimators": 5},
    "CatBoost": {"iterations": 5, "learning_rate": 0.1, "depth": 3, "verbose": 0},
    "LightGBM": {"n_estimators": 5, "verbose": -1},
    "KNN": {"n_neighbors": 3},
    "NaiveBayes": {},
    "LogisticRegression": {"max_iter": 200},
    "AdaBoost": {"n_estimators": 5},
    "SVM": {"probability": True, "kernel": "linear"},
}


def _frame(n, n_classes, seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_classes, n)
    X = rng.normal(size=(n, 6)) + y[:, None]
    labels = [np.eye(n_classes)[i] for i in y]
    return pd.DataFrame({"feat": list(X), "label_tensor": labels})


def test_all_baselines_run_binary_and_multiclass(tmp_path):
    for n_classes in (2, 3):
        cfg = BaselinesConfig(enabled=True, feature_column="feat", train_on_train_and_val=True, models=FAST)
        run_baselines(_frame(120, n_classes, 1), _frame(30, n_classes, 2), _frame(60, n_classes, 3), cfg,
                      seed=0, n_classes=n_classes,
                      metrics_csv=tmp_path / f"m{n_classes}.csv", confusion_png=tmp_path / f"c{n_classes}.png")
        metrics = pd.read_csv(tmp_path / f"m{n_classes}.csv")
        assert list(metrics["Model Name"]) == list(FAST)
        assert (metrics["Accuracy"] > 0.5).all()
        assert (tmp_path / f"c{n_classes}.png").stat().st_size > 0


def test_unknown_baseline_rejected():
    import pytest
    from difflabel.config import ConfigError
    cfg = BaselinesConfig(enabled=True, feature_column="f", train_on_train_and_val=False, models={"Nope": {}})
    with pytest.raises(ConfigError, match="Nope"):
        build_models(cfg, 0, 2)
