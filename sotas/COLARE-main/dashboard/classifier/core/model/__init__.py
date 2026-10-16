import os
import torch
from pathlib import Path
from .model import CommitClassifier
from transformers import AutoTokenizer, DataCollatorWithPadding
from django.conf import settings

# load model from checkpoint
COMMIT_CLASSIFIER = CommitClassifier()
_MODEL_WEIGHT_PATH = Path(os.path.abspath(__file__)).parent.joinpath("model_weight.pt")

assert _MODEL_WEIGHT_PATH.exists(), "Core error: model_weight.pt not exist"
COMMIT_CLASSIFIER.load_state_dict(
    torch.load(_MODEL_WEIGHT_PATH, map_location=settings.DEPLOY_DEVICE)["state_dict"]
)
COMMIT_CLASSIFIER.to(settings.DEPLOY_DEVICE)

# load global tokenizer
_TOKENIZER = AutoTokenizer.from_pretrained("bert-base-uncased")
_COLLATOR = DataCollatorWithPadding(tokenizer=_TOKENIZER)


def _move_to_device(batch, device):
    if isinstance(batch, dict):
        return {k: _move_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [_move_to_device(v, device) for v in batch]
    else:
        return batch.to(device=device)


def model_classify(
    messages: list[str], numerical_features: list[list[int]]
) -> list[str]:
    inputs = _TOKENIZER(messages, truncation=True)
    inputs = _COLLATOR(inputs)
    inputs["numerical_features"] = torch.tensor(numerical_features, dtype=torch.float32)

    inputs = _move_to_device(inputs, settings.DEPLOY_DEVICE)
    outputs = COMMIT_CLASSIFIER(**inputs)
    pred = (
        torch.nn.functional.softmax(outputs, dim=1).argmax(dim=1).cpu().detach().numpy()
    )

    def num_to_label(num: int):
        if num == 0:
            return "Corrective"
        if num == 1:
            return "Adaptive"
        return "Perfective"

    return [num_to_label(num) for num in pred]
