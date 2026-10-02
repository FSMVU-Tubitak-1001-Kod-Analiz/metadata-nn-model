#!/usr/bin/env python
"""Write per-row CodeBERT embeddings for one text column.

Usage:  python scripts/02_create_embeddings.py [--config configs/create_embeddings_content_before.json]
"""
from difflabel.cli import run_cli
from difflabel.config import CreateEmbeddingsConfig
from difflabel.pipelines import create_embeddings

if __name__ == "__main__":
    run_cli(CreateEmbeddingsConfig, create_embeddings.run, "configs/create_embeddings_content_before.json", "Write per-row CodeBERT embeddings for one text column.")
