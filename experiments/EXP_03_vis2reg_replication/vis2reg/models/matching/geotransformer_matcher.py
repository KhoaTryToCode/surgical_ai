from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .match_head import MatchHead
from .match_head_ot import MatchHeadOT


class GeoTransformerMatcher(nn.Module):
    """Compact coarse-to-fine matcher with coarse OT and full-resolution refinement."""

    def __init__(
        self,
        feature_dim: int,
        heads: int = 4,
        dropout: float = 0.0,
        temperature: float = 0.1,
        coarse_points: int = 512,
        sinkhorn_iters: int = 20,
        coarse_bias_weight: float = 1.0,
    ):
        super().__init__()
        self.coarse_points = int(coarse_points)
        self.coarse_bias_weight = float(coarse_bias_weight)
        self.match_head = MatchHead(feature_dim, heads=heads, dropout=dropout, temperature=temperature)
        self.match_head_ot = MatchHeadOT(
            feature_dim,
            heads=heads,
            dropout=dropout,
            temperature=temperature,
            sinkhorn_iters=sinkhorn_iters,
        )

    @staticmethod
    def _linspace_idx(n: int, k: int, device: torch.device) -> torch.Tensor:
        k = max(1, min(int(k), int(n)))
        return torch.linspace(0, n - 1, k, device=device).round().long()

    @staticmethod
    def _full_to_coarse_map(n: int, k: int, device: torch.device) -> torch.Tensor:
        if k <= 1:
            return torch.zeros(n, device=device, dtype=torch.long)
        return torch.clamp(
            (torch.arange(n, device=device, dtype=torch.float32) * (k - 1) / max(n - 1, 1)).round().long(),
            0,
            k - 1,
        )

    def forward(
        self,
        pre_coords: torch.Tensor,
        tgt_coords: torch.Tensor,
        pre_feats: torch.Tensor,
        tgt_feats: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        B, N_pre, _ = pre_feats.shape
        N_tgt = tgt_feats.shape[1]
        Nc = min(self.coarse_points, N_pre, N_tgt)
        pre_idx = self._linspace_idx(N_pre, Nc, pre_feats.device)
        tgt_idx = self._linspace_idx(N_tgt, Nc, pre_feats.device)

        pre_coarse = pre_feats[:, pre_idx]
        tgt_coarse = tgt_feats[:, tgt_idx]
        coarse_logits, coarse_prob_full = self.match_head_ot(pre_coarse, tgt_coarse)
        coarse_prob = coarse_prob_full[:, :Nc, :Nc]

        pre_map = self._full_to_coarse_map(N_pre, Nc, pre_feats.device)
        tgt_map = self._full_to_coarse_map(N_tgt, Nc, pre_feats.device)
        coarse_bias = coarse_logits[:, pre_map][:, :, tgt_map] * self.coarse_bias_weight

        sim_logits, _ = self.match_head(pre_feats, tgt_feats)
        sim_logits = sim_logits + coarse_bias
        soft_match = F.softmax(sim_logits, dim=-1)
        coarse_topk = coarse_prob.topk(k=min(8, coarse_prob.shape[-1]), dim=-1).indices
        return (
            sim_logits,
            soft_match,
            coarse_prob_full,
            pre_idx.expand(B, -1),
            tgt_idx.expand(B, -1),
            pre_coords[:, pre_idx],
            tgt_coords[:, tgt_idx],
            coarse_prob,
            coarse_topk,
            True,
            coarse_logits,
            coarse_logits,
            None,
            None,
            pre_coords[:, pre_idx],
            tgt_coords[:, tgt_idx],
        )
