from .registry import Registry


MODEL_REGISTRY = Registry("model_registry")
DATASET_REGISTRY = Registry("dataset_registry")

__all__ = ["Registry", "MODEL_REGISTRY", "DATASET_REGISTRY"]
