from .ccdataset import collate_fn as cc_collate_fn
from .cc2vec import collate_fn as cc2vec_collate_fn
import os
import functools
from registry import DATASET_REGISTRY
from torch.utils.data import DataLoader


def build_data_loaders(config, testing: bool = False):
    dataset = DATASET_REGISTRY.get_obj(config.dataset)

    test_dataset = dataset(
        os.path.join(config.data_dir, "test.csv"),
        config=config,
        use_roberta=config.use_roberta,
    )
    if config.dataset == "CC2Vec":
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=functools.partial(
                cc2vec_collate_fn,
                msg_collator=test_dataset.message_collator,
            ),
        )
    else:
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=functools.partial(
                cc_collate_fn,
                msg_collator=test_dataset.message_collator,
            ),
        )

    if testing:
        return test_loader, test_dataset

    train_dataset = dataset(
        os.path.join(config.data_dir, "train.csv"),
        config=config,
        use_roberta=config.use_roberta,
    )
    if config.dataset == "CC2Vec":
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            collate_fn=functools.partial(
                cc2vec_collate_fn,
                msg_collator=train_dataset.message_collator,
            ),
            drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            collate_fn=functools.partial(
                cc_collate_fn,
                msg_collator=train_dataset.message_collator,
            ),
            drop_last=True,
        )
    config.train_data_len = len(train_dataset)
    config.train_loader_len = len(train_loader)
    if config.enable_cv:
        return train_loader, test_loader

    evaluate_dataset = dataset(
        os.path.join(config.data_dir, "eval.csv"),
        config=config,
        use_roberta=config.use_roberta,
    )
    if config.dataset == "CC2Vec":
        eval_loader = DataLoader(
            evaluate_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=functools.partial(
                cc2vec_collate_fn,
                msg_collator=evaluate_dataset.message_collator,
            ),
        )
    else:
        eval_loader = DataLoader(
            evaluate_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=functools.partial(
                cc_collate_fn,
                msg_collator=evaluate_dataset.message_collator,
            ),
        )

    config.eval_data_len = len(evaluate_dataset)
    config.eval_loader_len = len(eval_loader)
    return train_loader, test_loader, eval_loader
