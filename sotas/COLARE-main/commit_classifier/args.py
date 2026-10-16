from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()

    # task name
    parser.add_argument("--name", type=str, default="nameless")

    # which model to select
    parser.add_argument("--model", type=str)

    # which device should the model running on
    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "gpu", "gpu2"]
    )

    # number of workers to process data loaders
    parser.add_argument("--num_workers", type=int, default=3)

    # global seed, for reproducing
    parser.add_argument("--seed", type=int, default=413)

    # gradient accumulate step
    parser.add_argument("--grad_acc", type=int, default=2)

    # batch size
    parser.add_argument("--batch_size", type=int, default=3)

    # learning rate
    parser.add_argument("--lr", type=float, default=3.5e-5)

    # max epochs
    parser.add_argument("--max_epochs", type=int, default=9)

    # patience
    parser.add_argument("--patience", type=int, default=9)

    # directory of train/eval data
    parser.add_argument("--data_dir", type=str)

    # directory of result saved
    parser.add_argument("--save_dir", type=str)

    # whether to enable checkpoints
    parser.add_argument("--enable_checkpoint", action="store_true", default=False)

    # whether to enable 10-fold cross validation
    parser.add_argument("--enable_cv", action="store_true", default=False)

    # do train
    parser.add_argument("--do_train", action="store_true", default=False)

    # do test
    parser.add_argument("--do_test", action="store_true", default=False)

    parser.add_argument(
        "--dataset",
        type=str,
        default="CommitDataset",
        choices=["CommitDataset", "CC2Vec"],
    )

    parser.add_argument("--file_num_limit", type=int, default=4)  # 79.94%

    parser.add_argument("--hunk_num_limit", type=int, default=3)  # 88.43%

    parser.add_argument("--use_roberta", action="store_true", default=False)

    args_parsed = parser.parse_args()
    if not args_parsed.do_train and not args_parsed.do_test:
        raise RuntimeError(f"should specify do_train / do_test")
    return args_parsed
