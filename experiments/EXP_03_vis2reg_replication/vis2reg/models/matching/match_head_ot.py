from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn


class MatchHeadOT(nn.Module):
    """Attention-style matcher followed by log-domain Sinkhorn normalization."""

    def __init__(
        self,
        feature_dim: int,
        heads: int = 4,
        dropout: float = 0.0,
        temperature: Optional[float] = None,
        sinkhorn_iters: int = 20,
        dustbin_score: float = 1.0,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.heads = heads
        self.dim_per_head = feature_dim // heads
        self.temperature = temperature or math.sqrt(self.dim_per_head)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.dustbin_score = nn.Parameter(torch.tensor(float(dustbin_score)))

        self.q_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        self.k_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _masked_log_sinkhorn(logits: torch.Tensor, iters: int) -> torch.Tensor:
        log_p = logits
        for _ in range(max(1, iters)):
            log_p = log_p - torch.logsumexp(log_p, dim=2, keepdim=True)
            log_p = log_p - torch.logsumexp(log_p, dim=1, keepdim=True)
        return log_p

    def forward_from_logits(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, M = logits.shape
        dust_col = self.dustbin_score.expand(B, N, 1)
        dust_row = self.dustbin_score.expand(B, 1, M + 1)
        logits_aug = torch.cat([logits, dust_col], dim=2)
        logits_aug = torch.cat([logits_aug, dust_row], dim=1)
        logits_aug = torch.clamp(logits_aug, min=-1e3, max=1e3)
        log_prob = self._masked_log_sinkhorn(logits_aug, self.sinkhorn_iters)
        return logits, torch.exp(log_prob)

    def forward(
        self,
        pre_feats: torch.Tensor,
        tgt_feats: torch.Tensor,
        pre_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N_pre, _ = pre_feats.shape
        _, N_tgt, _ = tgt_feats.shape

        q = self.q_proj(pre_feats).view(B, N_pre, self.heads, self.dim_per_head).transpose(1, 2)
        k = self.k_proj(tgt_feats).view(B, N_tgt, self.heads, self.dim_per_head).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / self.temperature
        scores = torch.clamp(scores, min=-1e3, max=1e3)

        if tgt_mask is not None:
            scores = scores.masked_fill(~tgt_mask[:, None, None,].bool(), float("-inf"))
        if pre_mask is not None:
            scores = scores.masked_fill(~pre_mask[:, None, :, None].bool(), float("-inf"))

        logits = scores.mean(dim=1)
        if bias is not None:
            logits = logits + bias
        logits = self.dropout(logits)
        return self.forward_from_logits(logits)
