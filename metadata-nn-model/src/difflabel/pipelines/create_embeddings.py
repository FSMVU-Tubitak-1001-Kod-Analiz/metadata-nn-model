"""Step 2: write per-row CodeBERT (or other HF encoder) embeddings to disk."""

from __future__ import annotations

from difflabel.config import CreateEmbeddingsConfig
from difflabel.embeddings.generate import run as _run


def run(cfg: CreateEmbeddingsConfig) -> None:
    _run(cfg)
