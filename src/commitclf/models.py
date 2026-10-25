"""Change classifier: per-file encoder -> pooling over files -> fusion with change metadata -> head.

model types
    code       files (code vectors) -> pool -> head
    metadata   files (per-file facts) -> pool, + change facts -> head
    concat     [code side, metadata side] -> head
    gated      g * tanh(Pc c) + (1 - g) * tanh(Pm m), g = sigmoid(W[c; m]), per dimension
    attention  softmax attention over the two sides (one weight per side and change)
    interact   [c', m', c' - m', c' * m'] with c' = tanh(Pc c), m' = tanh(Pm m) (InferSent / SBERT style)
    mlb        tanh(Uc c) * tanh(Um m), low-rank bilinear pooling (Kim et al. 2017)
    mfb        factorized bilinear pooling (Yu et al. 2017): (Uc c * Um m) summed over k factors,
               signed square root, L2 normalised
    late       a classifier on each side, trained with one loss each; the prediction is the mean of
               their probabilities
    gated_aux  gated + an auxiliary classifier on each side (loss weight AUX_WEIGHT) + modality
               dropout (in training, one side of a change is zeroed with probability MODALITY_DROPOUT each)
    gated_clip gated, after CLIP-style contrastive pre-training of both sides (matching code / metadata
               of the same change close, the others apart; CLIP_EPOCHS epochs, see train.py)
    misa       shared and private subspaces (Hazarika et al. 2020): a shared and a private encoder per
               side, with similarity (CMD), difference (orthogonality) and reconstruction losses;
               the head reads [shared c, shared m, private c, private m]

file pooling
    mean | max | attention  (attention: a_i = softmax_i(w · tanh(V h_i)), Ilse et al. 2018)

When the model uses metadata, every file vector also carries the per-file facts
(status, churn, test file, path flags), so the pooling can use them.
"""
from __future__ import annotations

import torch
import torch.nn as nn

FUSIONS = ("gated", "attention", "interact", "mlb", "mfb", "late", "gated_aux", "gated_clip", "misa")
MODEL_TYPES = ("code", "metadata", "concat") + FUSIONS
AUX_WEIGHT = 0.3              # gated_aux: weight of each side's auxiliary loss
MODALITY_DROPOUT = 0.15       # gated_aux: probability, per change, of zeroing the code side (and, separately, the metadata side)
CLIP_EPOCHS, CLIP_LR, CLIP_DIM = 10, 1e-4, 128   # gated_clip pre-training
MISA_WEIGHTS = {"sim": 1.0, "diff": 0.3, "recon": 1.0}   # as in the MISA code (MOSI settings)
MFB_FACTORS = 5
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


class WindowPool(nn.Module):
    """Windows of each file [N, W, D] (+ mask [N, W]) -> one vector per file [N, D], by attention."""

    def __init__(self, dim: int, attn_dim: int = 128):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, attn_dim), nn.Tanh(), nn.Linear(attn_dim, 1))
        self.last_weights: torch.Tensor | None = None   # [N, W] of the last batch

    def forward(self, w: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        w = w.float()
        scores = self.score(w).squeeze(-1).masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=1)
        self.last_weights = weights.detach()
        return (weights.unsqueeze(-1) * w).sum(1)


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
        self.last_code_weights: torch.Tensor | None = None   # [B] per-change mean gate

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gate(torch.cat((a, b), dim=1)))
        self.last_code_weights = g.mean(1).detach()
        self.last_weight = self.last_code_weights.mean().item()
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
        self.last_code_weights = self.last_weights[:, 0]
        self.last_weight = self.last_code_weights.mean().item()
        return (weights.unsqueeze(-1) * tokens).sum(1)


class InteractionFusion(nn.Module):
    """[a', b', a' - b', a' * b'] after projecting both sides to the same size."""

    def __init__(self, a_dim: int, b_dim: int, dim: int):
        super().__init__()
        self.pa, self.pb = nn.Linear(a_dim, dim), nn.Linear(b_dim, dim)
        self.out_dim = 4 * dim

    def forward(self, a, b):
        a, b = torch.tanh(self.pa(a)), torch.tanh(self.pb(b))
        return torch.cat((a, b, a - b, a * b), dim=1)


class MLBFusion(nn.Module):
    def __init__(self, a_dim: int, b_dim: int, dim: int):
        super().__init__()
        self.ua, self.ub = nn.Linear(a_dim, dim), nn.Linear(b_dim, dim)
        self.out_dim = dim

    def forward(self, a, b):
        return torch.tanh(self.ua(a)) * torch.tanh(self.ub(b))


class MFBFusion(nn.Module):
    def __init__(self, a_dim: int, b_dim: int, dim: int, k: int = MFB_FACTORS, dropout: float = 0.1):
        super().__init__()
        self.ua, self.ub = nn.Linear(a_dim, dim * k), nn.Linear(b_dim, dim * k)
        self.k, self.out_dim, self.drop = k, dim, nn.Dropout(dropout)

    def forward(self, a, b):
        z = self.drop(self.ua(a) * self.ub(b)).view(len(a), self.out_dim, self.k).sum(2)
        z = torch.sign(z) * torch.sqrt(z.abs() + 1e-8)
        return nn.functional.normalize(z, dim=1)


def _cmd(x: torch.Tensor, y: torch.Tensor, k: int = 5) -> torch.Tensor:
    """Central moment discrepancy (Zellinger et al. 2017) between two batches of vectors."""
    mx, my = x.mean(0), y.mean(0)
    dx, dy = x - mx, y - my
    out = (mx - my).norm()
    for i in range(2, k + 1):
        out = out + ((dx ** i).mean(0) - (dy ** i).mean(0)).norm()
    return out


def _ortho(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Soft orthogonality of two batches of vectors (MISA's difference loss)."""
    x = nn.functional.normalize(x - x.mean(0), dim=1)
    y = nn.functional.normalize(y - y.mean(0), dim=1)
    return (x.T @ y).pow(2).mean()


class MISAFusion(nn.Module):
    def __init__(self, a_dim: int, b_dim: int, dim: int):
        super().__init__()
        self.pa = nn.Sequential(nn.Linear(a_dim, dim), nn.LayerNorm(dim))
        self.pb = nn.Sequential(nn.Linear(b_dim, dim), nn.LayerNorm(dim))
        self.shared = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        self.priv_a = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        self.priv_b = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        self.rec_a, self.rec_b = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.out_dim = 4 * dim
        self.reg_loss = torch.tensor(0.0)

    def forward(self, a, b):
        a, b = self.pa(a), self.pb(b)
        sa, sb, pa, pb = self.shared(a), self.shared(b), self.priv_a(a), self.priv_b(b)
        if self.training:
            w = MISA_WEIGHTS
            rec = nn.functional.mse_loss(self.rec_a(sa + pa), a) + nn.functional.mse_loss(self.rec_b(sb + pb), b)
            self.reg_loss = (w["sim"] * _cmd(sa, sb) + w["diff"] * (_ortho(sa, pa) + _ortho(sb, pb) + _ortho(pa, pb))
                             + w["recon"] * rec)
        return torch.cat((sa, sb, pa, pb), dim=1)


class ContrastiveAlign(nn.Module):
    """CLIP loss between the two sides of the same changes (other changes in the batch are negatives)."""

    def __init__(self, a_dim: int, b_dim: int, dim: int = CLIP_DIM):
        super().__init__()
        self.wa, self.wb = nn.Linear(a_dim, dim), nn.Linear(b_dim, dim)
        self.logit_scale = nn.Parameter(torch.tensor(2.659))   # log(1 / 0.07)

    def forward(self, a, b):
        za, zb = nn.functional.normalize(self.wa(a), dim=1), nn.functional.normalize(self.wb(b), dim=1)
        logits = self.logit_scale.exp().clamp(max=100) * za @ zb.T
        target = torch.arange(len(a), device=a.device)
        return (nn.functional.cross_entropy(logits, target) + nn.functional.cross_entropy(logits.T, target)) / 2


class ChangeClassifier(nn.Module):
    def __init__(self, model_type: str, file_dim: int, meta_dim: int, n_classes: int, *,
                 file_hidden: list[int], metadata_hidden: list[int], head_hidden: list[int],
                 fusion_dim: int, file_pool: str, dropout: float = 0.2, window_dims: list[int] | None = None):
        super().__init__()
        if model_type not in MODEL_TYPES:
            raise ValueError(f"model type must be one of {MODEL_TYPES}, got '{model_type}'")
        self.model_type = model_type
        # Windowed code inputs: attention over each file's windows, appended to the file vector.
        self.window_pools = nn.ModuleList([WindowPool(d) for d in (window_dims or [])])
        self.files = FileBranch(file_dim + sum(window_dims or []), file_hidden, dropout, file_pool)
        side_a = self.files.out_dim

        self.meta = mlp(meta_dim, metadata_hidden, dropout) if model_type != "code" and meta_dim else None
        side_b = metadata_hidden[-1] if self.meta is not None else 0

        self.fusion, self.aux_heads, self.clip, self.side_heads = None, None, None, None
        self.modality_dropout, self._aux = 0.0, []
        if model_type in FUSIONS and self.meta is None:
            raise ValueError(f"'{model_type}' fusion needs change-level metadata (got none)")
        if model_type in ("gated", "gated_aux", "gated_clip", "attention"):
            cls = AttentionFusion if model_type == "attention" else GatedFusion
            self.fusion = cls(side_a, side_b, fusion_dim)
            head_in = fusion_dim
        elif model_type in ("interact", "mlb", "misa"):
            cls = {"interact": InteractionFusion, "mlb": MLBFusion, "misa": MISAFusion}[model_type]
            self.fusion = cls(side_a, side_b, fusion_dim if model_type != "misa" else fusion_dim // 2)
            head_in = self.fusion.out_dim
        elif model_type == "mfb":
            self.fusion = MFBFusion(side_a, side_b, fusion_dim // 2)
            head_in = self.fusion.out_dim
        else:
            head_in = side_a + side_b
        if model_type == "gated_aux":
            self.aux_heads = nn.ModuleList([nn.Linear(side_a, n_classes), nn.Linear(side_b, n_classes)])
            self.modality_dropout = MODALITY_DROPOUT
        if model_type == "gated_clip":
            self.clip = ContrastiveAlign(side_a, side_b)

        if model_type == "late":   # one classifier per side, no shared head
            self.side_heads = nn.ModuleList([nn.Sequential(mlp(d, head_hidden, dropout), nn.Linear(head_hidden[-1], n_classes))
                                             for d in (side_a, side_b)])
        else:
            self.head = mlp(head_in, head_hidden, dropout)
            self.out = nn.Linear(head_hidden[-1], n_classes)

    def forward(self, files: torch.Tensor, mask: torch.Tensor, meta: torch.Tensor, windows=None) -> torch.Tensor:
        a, b = self.sides(files, mask, meta, windows)
        if self.side_heads is not None:   # late fusion: mean of the two classifiers' probabilities
            self._aux = [h(x) for h, x in zip(self.side_heads, (a, b))]
            return torch.log(torch.stack([torch.softmax(l, 1) for l in self._aux]).mean(0) + 1e-8)
        if self.aux_heads is not None:
            self._aux = [h(x) for h, x in zip(self.aux_heads, (a, b))]
        if self.training and self.modality_dropout:
            r = torch.rand(len(a), 1, device=a.device)
            p = self.modality_dropout
            a = a * (r >= p).float()                              # code side dropped
            b = b * ((r < p) | (r >= 2 * p)).float()              # metadata side dropped (never both)
        if self.fusion is not None:
            x = self.fusion(a, b)
        else:
            x = a if b is None else torch.cat((a, b), dim=1)
        return self.out(self.head(x))

    def sides(self, files, mask, meta, windows=None):
        """(code side, metadata side or None) of a batch."""
        if len(self.window_pools):
            parts = [files.float()]
            for pool, (w, wm) in zip(self.window_pools, windows):
                per_file = torch.zeros(*mask.shape, w.shape[-1], device=files.device, dtype=torch.float32)
                per_file[mask] = pool(w, wm)
                parts.append(per_file)
            files = torch.cat(parts, dim=2)
        return self.files(files, mask), (self.meta(meta) if self.meta is not None else None)

    def loss(self, logits: torch.Tensor, y: torch.Tensor, criterion) -> torch.Tensor:
        """Training loss of the last forward pass (adds the auxiliary / regularisation terms)."""
        if self.side_heads is not None:
            return sum(criterion(l, y) for l in self._aux)
        loss = criterion(logits, y)
        if self.aux_heads is not None:
            loss = loss + AUX_WEIGHT * sum(criterion(l, y) for l in self._aux)
        if isinstance(self.fusion, MISAFusion) and self.training:
            loss = loss + self.fusion.reg_loss
        return loss

    def clip_loss(self, files, mask, meta, windows=None) -> torch.Tensor:
        return self.clip(*self.sides(files, mask, meta, windows))
