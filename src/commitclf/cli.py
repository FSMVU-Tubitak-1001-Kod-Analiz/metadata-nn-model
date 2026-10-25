"""Command line entry point:  python -m commitclf <command> [options]

    convert-colare  CoLaRe's 1793 commit set -> collector-format CSV + their CV folds
    prepare     collector dataset -> data/processed/{rows,units}.csv + split files
    splits      (re)build the split files from units.csv
    metatext    metadata written as text (short / long) per file and per change (see metatext.py)
    embed       CodeBERT / BERT vectors per file row (resumable)
    train       train/evaluate the change classifier
    finetune    same, with the message BERT and/or CodeBERT fine-tuned end to end (see finetune.py)
    colare      CoLaRe's own model and training recipe on our folds (see sota/colare.py)
    baselines   train/evaluate the classic ML models
    grid        run the experiment grid in configs/grid.yaml
    explain     shap | permutation | weights | tokens | occlusion on finished runs (see explain.py)

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
    p.add_argument("--sim", choices=["raw", "white"],
                   help="before/after cosine similarity per file (+ change aggregates); white = whitened")
    p.add_argument("--code", nargs="*", help="embedding inputs per file")
    p.add_argument("--embed-pool", choices=["mean", "cls", "pooler"])
    p.add_argument("--file-pool", choices=FILE_POOLS)
    p.add_argument("--meta", nargs="*", help="change-level metadata groups: change repo message")
    p.add_argument("--folds", nargs="+", default=["1"], help="fold numbers or 'all' (default: 1)")
    p.add_argument("--name", help="output folder name (default: derived from the settings)")
    p.add_argument("--no-facts", dest="facts", action="store_false", default=None,
                   help="no per-file facts on the file vectors (e.g. when the metadata comes as text)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="commitclf", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-c", "--config", default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("convert-colare", help="CoLaRe's 1793 Java commit set -> collector-format CSV")
    p.add_argument("--src", default="sotas/COLARE-main/dataset/1793/dataset.csv")
    p.add_argument("--out", default="data/external/colare1793")
    p.add_argument("--files", choices=["all", "java"], default="all",
                   help="all files with a text diff (default) or .java files only")

    sub.add_parser("prepare", help="build rows/units tables and splits")
    sub.add_parser("splits", help="rebuild the split files")

    p = sub.add_parser("metatext", help="write metadata as text (metatext.csv), needed by `text:` inputs")
    p.add_argument("--show", type=int, default=3, help="print this many random examples")

    p = sub.add_parser("embed", help="compute embeddings")
    p.add_argument("--only", nargs="+", help="inputs from embeddings.inputs (default: all except message)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--copy-from", metavar="CONFIG", help="copy finished embeddings from another config (this data must be a subset of its rows)")

    p = sub.add_parser("embed-windows", help="windowed embeddings of one input (see windows.py)")
    p.add_argument("name", help="an entry of embeddings.windows, e.g. patch_win")
    p.add_argument("--overwrite", action="store_true")

    add_experiment_args(sub.add_parser("train", help="train the change classifier"))
    p = sub.add_parser("finetune", help="train with fine-tuned encoders (message and/or code)")
    p.add_argument("--code-ft", action="store_true", help="also fine-tune CodeBERT on the files' code inputs")
    p.add_argument("--no-message", action="store_true", help="don't use the message (code fine-tuning only)")
    p.add_argument("--meta-text", metavar="COLUMN",
                   help="a metatext.csv column (e.g. change_short) instead of the message, read by the code encoder")
    add_experiment_args(p)

    p = sub.add_parser("colare", help="run CoLaRe's model (sotas/COLARE-main) on our folds")
    p.add_argument("--variant", default="CCModel",
                   choices=["CCModel", "CodeFeatModel", "MessageCodeModel", "MessageFeatModel"],
                   help="CCModel = full CoLaRe (message + code + counts); CodeFeatModel = no message")
    p.add_argument("--split", default="given", choices=SCHEMES)
    p.add_argument("--folds", nargs="+", default=["1"], help="fold numbers or 'all' (default: 1)")
    p.add_argument("--colare-dir", default="sotas/COLARE-main")
    p.add_argument("--src", default="sotas/COLARE-main/dataset/1793/dataset.csv")
    p.add_argument("--model-path", help="local CodeBERT to use instead of microsoft/codebert-base")
    p.add_argument("--epochs", type=int, help="default 10 (CoLaRe: epochs 0..9)")
    p.add_argument("--batch-size", type=int, help="default 3 (CoLaRe)")
    p.add_argument("--grad-acc", type=int, help="default 2 (CoLaRe)")
    p.add_argument("--file-limit", type=int, help="default 5 (CoLaRe, Java set)")
    p.add_argument("--hunk-limit", type=int, help="default 2 (CoLaRe, Java set)")
    p.add_argument("--name", help="output folder under train.out_dir (default colare/<variant>_<split>)")

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

    p = sub.add_parser("explain", help="explain finished runs")
    p.add_argument("analysis", choices=["shap", "permutation", "weights", "tokens", "occlusion"])
    p.add_argument("--run", help="finished run, e.g. main/attention_repo (permutation, weights, tokens, occlusion)")
    p.add_argument("--split", default="repo", choices=SCHEMES, help="split for shap (default: repo)")
    p.add_argument("--subset", choices=["single_commit"], help="subset for shap")
    p.add_argument("--folds", nargs="+", default=["all"], help="fold numbers or 'all' (default: all)")
    p.add_argument("--repeats", type=int, default=3, help="permutation: shuffles per feature")
    p.add_argument("--n", type=int, default=300, help="tokens / occlusion: number of test changes")
    p.add_argument("--steps", type=int, default=32, help="tokens: Integrated Gradients steps")
    p.add_argument("--max-files", type=int, default=5, help="tokens / occlusion: skip changes with more files")
    p.add_argument("--batch", type=int, default=64, help="occlusion: texts per CodeBERT batch")
    p.add_argument("--target", choices=["pred", "true"], default="pred", help="tokens: class to explain")
    p.add_argument("--ig-batch", type=int, default=32,
                   help="tokens: interpolation steps per GPU batch (32 x 512 tokens needs ~15 GB; lower if out of memory)")

    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    if args.command == "convert-colare":
        from commitclf.config import resolve
        from commitclf.data.external import convert_colare
        convert_colare(resolve(cfg, args.src), resolve(cfg, args.out), files=args.files)
    elif args.command == "prepare":
        from commitclf.data import prepare, splits
        splits.run(cfg, prepare.run(cfg))
    elif args.command == "splits":
        from commitclf.data import splits
        splits.run(cfg)
    elif args.command == "metatext":
        from commitclf import metatext
        metatext.run(cfg, show=args.show)
    elif args.command == "embed-windows":
        from commitclf import windows
        windows.run(cfg, args.name, overwrite=args.overwrite)
    elif args.command == "embed":
        from commitclf import embeddings
        if args.copy_from:
            embeddings.copy_from(cfg, args.copy_from, only=args.only)
        else:
            embeddings.run(cfg, only=args.only, overwrite=args.overwrite)
    elif args.command == "explain":
        from commitclf import explain
        if args.analysis != "shap" and not args.run:
            parser.error(f"explain {args.analysis} needs --run (e.g. main/attention_repo)")
        folds = None if args.folds == ["all"] else [int(f) for f in args.folds]
        if args.analysis == "shap":
            explain.run_shap(cfg, args.split, folds or parse_folds(["all"], cfg, args.split), args.subset)
        elif args.analysis == "permutation":
            explain.run_permutation(cfg, args.run, folds, repeats=args.repeats)
        elif args.analysis == "weights":
            explain.run_weights(cfg, args.run, folds)
        elif args.analysis == "occlusion":
            explain.run_occlusion(cfg, args.run, fold=(folds or [1])[0], n=args.n,
                                  max_files=args.max_files, batch_size=args.batch)
        else:
            explain.run_tokens(cfg, args.run, fold=(folds or [1])[0], n=args.n, steps=args.steps,
                               max_files=args.max_files, target=args.target, ig_batch=args.ig_batch)
    elif args.command == "colare":
        from commitclf.sota import colare
        colare.run(cfg, parse_folds(args.folds, cfg, args.split), args.variant, args.split, args.colare_dir,
                   args.src, model_path=args.model_path, name=args.name, epochs=args.epochs,
                   batch_size=args.batch_size, grad_acc=args.grad_acc, file_limit=args.file_limit,
                   hunk_limit=args.hunk_limit)
    elif args.command == "grid":
        from commitclf import grid
        grid.run(cfg, args.grid, blocks=args.blocks, dry_run=args.dry_run, force=args.force,
                 folds_override=args.folds)
    else:
        from commitclf import train
        spec = train.make_spec(cfg, model=args.model, split=args.split, subset=args.subset, code=args.code,
                               embed_pool=args.embed_pool, file_pool=args.file_pool, metadata=args.meta,
                               sim=args.sim, facts=args.facts)
        folds = parse_folds(args.folds, cfg, spec["split"])
        if args.command == "train":
            train.run(cfg, folds, spec, name=args.name)
        elif args.command == "finetune":
            from commitclf import finetune
            finetune.run(cfg, folds, spec, name=args.name, code=args.code_ft, message=not args.no_message,
                         meta_text=args.meta_text)
        else:
            from commitclf import baselines
            baselines.run(cfg, folds, spec, features=args.features, only=args.models, name=args.name)


if __name__ == "__main__":
    main()
