"""Change classifier: per-file encoder -> pooling over files -> fusion with change metadata -> head.

model types
    code       files (code vectors) -> pool -> head
    metadata   files (per-file facts) -> pool, + change facts -> head
    concat     [code side, metadata side] -> head
    gated      g * tanh(Pc c) + (1 - g) * tanh(Pm m), g = sigmoid(W[c; m]), per dimension
    attention  softmax attention over the two sides (one weight per side and change)

file pooling
    mean | max | attention  (attention: a_i = softmax_i(w · tanh(V h_i)), Ilse et al. 2018)

When the model uses metadata, every file vector also carries the per-file facts
(status, churn, test file, path flags), so the pooling can use them.
"""
from __future__ import annotations

import torch
import torch.nn as nn

MODEL_TYPES = ("code", "metadata", "concat", "gated", "attention")
FILE_POOLS = ("mean", "max", "attention")


def mlp(in_dim: int, hidden: list[int], dropout: float) -> nn.Sequential:
    """Linear -> BatchNorm -> ReLU -> Dropout blocks."""
    layers = []
    for h in hidden:
        layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
        in_dim = h
    return nn.Sequential(*layers)


class FilePool(nn.Module):
    """Files of a change [B, F, D] (+ mask [B, F]) -> one vector [B, D]."""

    def __init__(self, kind: str, dim: int, attn_dim: int = 128):
        super().__init__()
        if kind not in FILE_POOLS:
            raise ValueError(f"file pooling must be one of {FILE_POOLS}, got '{kind}'")
        self.kind = kind
        if kind == "attention":
            self.score = nn.Sequential(nn.Linear(dim, attn_dim), nn.Tanh(), nn.Linear(attn_dim, 1))
        self.last_weights: torch.Tensor | None = None   # [B, F] attention weights of the last batch

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.unsqueeze(-1)
        if self.kind == "mean":
            return (h * m).sum(1) / m.sum(1).clamp(min=1)
        if self.kind == "max":
            return h.masked_fill(~m, float("-inf")).max(1).values
        scores = self.score(h).squeeze(-1).masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=1)
        self.last_weights = weights.detach()
        return (weights.unsqueeze(-1) * h).sum(1)


class FileBranch(nn.Module):
    """Encode every file with a shared MLP, then pool to one vector per change."""

    def __init__(self, in_dim: int, hidden: list[int], dropout: float, pool: str):
        super().__init__()
        self.encoder = mlp(in_dim, hidden, dropout)
        self.pool = FilePool(pool, hidden[-1])
        self.out_dim = hidden[-1]

    def forward(self, files: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, f, d = files.shape
        h = torch.zeros(b, f, self.out_dim, device=files.device, dtype=torch.float32)
        h[mask] = self.encoder(files[mask].float())   # only real files go through BatchNorm
        return self.pool(h, mask)


class GatedFusion(nn.Module):
    def __init__(self, a_dim: int, b_dim: int, dim: int):
        super().__init__()
        self.pa, self.pb = nn.Linear(a_dim, dim), nn.Linear(b_dim, dim)
        self.gate = nn.Linear(a_dim + b_dim, dim)
        self.last_weight: float | None = None   # mean code weight of the last batch

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gate(torch.cat((a, b), dim=1)))
        self.last_weight = g.mean().item()
        return g * torch.tanh(self.pa(a)) + (1 - g) * torch.tanh(self.pb(b))


class AttentionFusion(nn.Module):
    """Project each side to `dim`, score it, and take the softmax-weighted sum."""

    def __init__(self, a_dim: int, b_dim: int, dim: int, attn_dim: int = 128):
        super().__init__()
        self.pa, self.pb = nn.Linear(a_dim, dim), nn.Linear(b_dim, dim)
        self.score = nn.Sequential(nn.Linear(dim, attn_dim), nn.Tanh(), nn.Linear(attn_dim, 1))
        self.last_weight: float | None = None   # mean code weight of the last batch
        self.last_weights: torch.Tensor | None = None   # [B, 2] per-change weights (code, metadata)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        tokens = torch.stack((torch.tanh(self.pa(a)), torch.tanh(self.pb(b))), dim=1)   # [B, 2, dim]
        weights = torch.softmax(self.score(tokens).squeeze(-1), dim=1)                   # [B, 2]
        self.last_weights = weights.detach()
        self.last_weight = weights[:, 0].mean().item()
        return (weights.unsqueeze(-1) * tokens).sum(1)


class ChangeClassifier(nn.Module):
    def __init__(self, model_type: str, file_dim: int, meta_dim: int, n_classes: int, *,
                 file_hidden: list[int], metadata_hidden: list[int], head_hidden: list[int],
                 fusion_dim: int, file_pool: str, dropout: float = 0.2):
        super().__init__()
        if model_type not in MODEL_TYPES:
            raise ValueError(f"model type must be one of {MODEL_TYPES}, got '{model_type}'")
        self.model_type = model_type
        self.files = FileBranch(file_dim, file_hidden, dropout, file_pool)
        side_a = self.files.out_dim

        self.meta = mlp(meta_dim, metadata_hidden, dropout) if model_type != "code" and meta_dim else None
        side_b = metadata_hidden[-1] if self.meta is not None else 0

        self.fusion = None
        if model_type in ("gated", "attention"):
            if self.meta is None:
                raise ValueError(f"'{model_type}' fusion needs change-level metadata (got none)")
            cls = GatedFusion if model_type == "gated" else AttentionFusion
            self.fusion = cls(side_a, side_b, fusion_dim)
            head_in = fusion_dim
        else:
            head_in = side_a + side_b

        self.head = mlp(head_in, head_hidden, dropout)
        self.out = nn.Linear(head_hidden[-1], n_classes)

    def forward(self, files: torch.Tensor, mask: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        a = self.files(files, mask)
        b = self.meta(meta) if self.meta is not None else None
        if self.fusion is not None:
            x = self.fusion(a, b)
        else:
            x = a if b is None else torch.cat((a, b), dim=1)
        return self.out(self.head(x))
