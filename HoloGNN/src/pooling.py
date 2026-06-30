"""
src/pooling.py
==============
Mask-aware graph-level pooling for the Holo-GNN backbone.

  • ``AttentionPooling`` — a learned, mask-aware attention pool that computes a
    softmax over real residues only and returns their weighted sum.
  • ``masked_mean`` — a simple padding-aware mean.

Both honour the ``attention_mask`` so padding never contributes.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Padding-aware mean pool.

    Args:
        hidden : (B, L, D) residue embeddings.
        mask   : (B, L) with 1 for real tokens, 0 for padding.
    Returns:
        (B, D) mean over real residues only (falls back to a plain mean if a row
        has no valid tokens, which should not happen in practice).
    """
    m = mask.unsqueeze(-1).to(hidden.dtype)          # (B, L, 1)
    summed = (hidden * m).sum(dim=1)                  # (B, D)
    counts = m.sum(dim=1).clamp_min(1.0)              # (B, 1)
    return summed / counts


class AttentionPooling(nn.Module):
    """
    Learned, mask-aware attention pooling.

    A small scorer maps each residue embedding to a scalar logit; padding
    positions are masked to ``-inf`` before a softmax over the length dimension,
    and the graph embedding is the attention-weighted sum of residue embeddings.

    Args:
        dim : embedding size (D).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1),
        )

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden : (B, L, D) residue embeddings.
            mask   : (B, L) 1 = real token, 0 = padding.
        Returns:
            (B, D) attention-pooled graph embedding.
        """
        scores = self.scorer(hidden).squeeze(-1)          # (B, L)
        scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=1)            # (B, L)
        # Guard against an all-padding row producing NaNs from softmax(-inf).
        weights = torch.nan_to_num(weights, nan=0.0)
        return torch.bmm(weights.unsqueeze(1), hidden).squeeze(1)  # (B, D)
