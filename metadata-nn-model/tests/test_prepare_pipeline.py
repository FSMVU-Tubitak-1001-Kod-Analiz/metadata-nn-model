import dataclasses
from pathlib import Path

import pandas as pd

from difflabel.config import PrepareCsvConfig, load_config
from difflabel.pipelines import prepare_csv

from synth import make_raw_csv

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_prepare_csv_end_to_end(tmp_path):
    raw = tmp_path / "raw.csv"
    make_raw_csv(raw, n_repos=4, per_label=60)

    cfg = load_config(PrepareCsvConfig, CONFIGS / "prepare_csv.json")
    cfg = dataclasses.replace(cfg, raw_csv=str(raw), output_dir=str(tmp_path / "out"), rows_per_repo=100)
    prepare_csv.run(cfg)

    out = tmp_path / "out"
    final = pd.read_csv(out / cfg.output_csv_name)
    assert set(final["labels"]) == {"bug", "enhancement"}
    assert final["labels"].value_counts().nunique() == 1                    # balanced
    assert "patch_before" in final.columns
    assert not final["content_before"].str.contains("import ").any()        # imports stripped
    assert not final["content_before"].str.contains("Licensed").any()       # license stripped

    for n in range(1, 6):
        for split in ("train", "test"):
            assert (out / "folds" / f"fold{n}_{split}.csv").exists()
        sizes = {s: len(pd.read_csv(out / "folds_70_15_15" / f"fold{n}_{s}.csv")) for s in ("train", "val", "test")}
        assert sum(sizes.values()) == len(final)
        assert abs(sizes["train"] / len(final) - 0.70) < 0.02
        # no leakage between splits
        ids = [set(pd.read_csv(out / "folds_70_15_15" / f"fold{n}_{s}.csv")["index"]) for s in ("train", "val", "test")]
        assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])


def test_cli_script_runs_from_json_config(tmp_path):
    """scripts/01_prepare_csv.py --config <json>, exactly as a user would run it."""
    import json, os, subprocess, sys

    root = Path(__file__).resolve().parents[1]
    raw = tmp_path / "raw.csv"
    make_raw_csv(raw, n_repos=4, per_label=60)

    data = json.loads((CONFIGS / "prepare_csv.json").read_text())
    data.update(raw_csv=str(raw), output_dir=str(tmp_path / "out"), rows_per_repo=100)
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(data))

    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    proc = subprocess.run([sys.executable, str(root / "scripts" / "01_prepare_csv.py"), "--config", str(cfg_path)],
                          capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert (tmp_path / "out" / data["output_csv_name"]).exists()

    bad = subprocess.run([sys.executable, str(root / "scripts" / "01_prepare_csv.py"), "--config", str(tmp_path / "nope.json")],
                         capture_output=True, text=True, env=env)
    assert bad.returncode != 0 and "Config file not found" in bad.stderr
