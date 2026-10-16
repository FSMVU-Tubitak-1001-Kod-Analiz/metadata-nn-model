import inspect
import torch.nn as nn
from registry import MODEL_REGISTRY
import model.models
import model.cc2vec


def build_model(config):
    if not MODEL_REGISTRY.has_obj(config.model):
        raise TypeError(f"there is no model register for {config.model}")

    obj = MODEL_REGISTRY.get_obj(config.model)

    if inspect.isclass(obj):
        m = obj(config)
        return m
    elif isinstance(obj, nn.Module):
        return obj
    else:
        return TypeError(
            f"expect the obj register for {config.model} is torch.nn.Module, but get {type(obj)}"
        )
