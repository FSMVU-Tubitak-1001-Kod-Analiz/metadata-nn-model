"""Command line entry point:  python -m commitclf <command> [options]

    prepare     collector dataset -> data/processed/{rows,units}.csv + split files
    splits      (re)build the split files from units.csv
    embed       CodeBERT / BERT vectors per file row (resumable)
    train       train/evaluate the change classifier
    baselines   train/evaluate the classic ML models
    grid        run the experiment grid in configs/grid.yaml

Experiment options (train, baselines) override `experiment` in the config:
    --model code|metadata|concat|gated|attention   --split repo|org|random|time
    --code content_before patch ...   --embed-pool mean|cls|pooler   --file-pool mean|max|attention
    --meta change [repo] [message]    --subset single_commit
"""
from __future__ import annotations

import argparse

from commitclf.config import DEFAULT_CONFIG, load_config
from commitclf.data.splits import SCHEMES
from commitclf.models import FILE_POOLS, MODEL_TYPES


def parse_folds(values: list[str], cfg: dict, scheme: str) -> list[int]:
    if values == ["all"]:
        from commitclf.data.splits import n_folds
        return list(range(1, n_folds(cfg, scheme) + 1))
    return [int(v) for v in values]


def add_experiment_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", choices=MODEL_TYPES)
    p.add_argument("--split", choices=SCHEMES)
    p.add_argument("--subset", choices=["single_commit"])
    p.add_argument("--code", nargs="*", help="embedding inputs per file")
    p.add_argument("--embed-pool", choices=["mean", "cls", "pooler"])
    p.add_argument("--file-pool", choices=FILE_POOLS)
    p.add_argument("--meta", nargs="*", help="change-level metadata groups: change repo message")
    p.add_argument("--folds", nargs="+", default=["1"], help="fold numbers or 'all' (default: 1)")
    p.add_argument("--name", help="output folder name (default: derived from the settings)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="commitclf", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-c", "--config", default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("prepare", help="build rows/units tables and splits")
    sub.add_parser("splits", help="rebuild the split files")

    p = sub.add_parser("embed", help="compute embeddings")
    p.add_argument("--only", nargs="+", help="inputs from embeddings.inputs (default: all except message)")
    p.add_argument("--overwrite", action="store_true")

    add_experiment_args(sub.add_parser("train", help="train the change classifier"))

    p = sub.add_parser("baselines", help="train the classic ML baselines")
    p.add_argument("--features", choices=["code", "metadata", "both"], help="overrides baselines.features")
    p.add_argument("--models", nargs="+", help="subset of models, e.g. XGBoost LogisticRegression")
    add_experiment_args(p)

    p = sub.add_parser("grid", help="run the experiment grid")
    p.add_argument("--grid", default="configs/grid.yaml")
    p.add_argument("--blocks", nargs="+", help="only these blocks (e.g. main baselines)")
    p.add_argument("--folds", nargs="+", help="override the grid's folds (e.g. 1 for a quick pass)")
    p.add_argument("--dry-run", action="store_true", help="list the runs without training")
    p.add_argument("--force", action="store_true", help="redo finished runs")

    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    if args.command == "prepare":
        from commitclf.data import prepare, splits
        splits.run(cfg, prepare.run(cfg))
    elif args.command == "splits":
        from commitclf.data import splits
        splits.run(cfg)
    elif args.command == "embed":
        from commitclf import embeddings
        embeddings.run(cfg, only=args.only, overwrite=args.overwrite)
    elif args.command == "grid":
        from commitclf import grid
        grid.run(cfg, args.grid, blocks=args.blocks, dry_run=args.dry_run, force=args.force,
                 folds_override=args.folds)
    else:
        from commitclf import train
        spec = train.make_spec(cfg, model=args.model, split=args.split, subset=args.subset, code=args.code,
                               embed_pool=args.embed_pool, file_pool=args.file_pool, metadata=args.meta)
        folds = parse_folds(args.folds, cfg, spec["split"])
        if args.command == "train":
            train.run(cfg, folds, spec, name=args.name)
        else:
            from commitclf import baselines
            baselines.run(cfg, folds, spec, features=args.features, only=args.models, name=args.name)


if __name__ == "__main__":
    main()
