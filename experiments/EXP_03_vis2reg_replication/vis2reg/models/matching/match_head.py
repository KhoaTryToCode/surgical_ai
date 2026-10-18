import math
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class MatchHead(nn.Module):
    """Cross-attention based matching head.

    Takes preoperative and intraoperative point features and produces
    similarity logits plus row-softmax probabilities (source -> target).
    """

    def __init__(
        self,
        feature_dim: int,
        heads: int = 4,
        dropout: float = 0.0,
        temperature: Optional[float] = None,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.heads = heads
        self.dim_per_head = feature_dim // heads
        self.temperature = temperature or math.sqrt(self.dim_per_head)

        self.q_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        self.k_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        self.v_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        self.norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        pre_feats: torch.Tensor,
        tgt_feats: torch.Tensor,
        pre_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pre_feats: (B, N_pre, C)
            tgt_feats: (B, N_tgt, C)
            pre_mask:  (B, N_pre) boolean mask (True for valid points)
            tgt_mask:  (B, N_tgt) boolean mask (True for valid points)
        Returns:
            sim_logits: (B, N_pre, N_tgt) raw similarity logits
            soft_match: (B, N_pre, N_tgt) row-softmax probabilities
        """
        B, N_pre, _ = pre_feats.shape
        _, N_tgt, _ = tgt_feats.shape

        q = self.q_proj(pre_feats).view(B, N_pre, self.heads, self.dim_per_head).transpose(1, 2)
        k = self.k_proj(tgt_feats).view(B, N_tgt, self.heads, self.dim_per_head).transpose(1, 2)
        v = self.v_proj(tgt_feats).view(B, N_tgt, self.heads, self.dim_per_head).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) / self.temperature  # (B, H, N_pre, N_tgt)
        # clamp to avoid inf in softmax
        scores = torch.clamp(scores, min=-1e3, max=1e3)

        if tgt_mask is not None:
            tgt_mask_exp = tgt_mask[:, None, None, :].to(dtype=torch.bool)
            scores = scores.masked_fill(~tgt_mask_exp, float("-inf"))
        if pre_mask is not None:
            pre_mask_exp = pre_mask[:, None, :, None].to(dtype=torch.bool)
            scores = scores.masked_fill(~pre_mask_exp, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
        soft_match = attn.mean(dim=1)  # average heads for output probability

        attn = self.dropout(attn)
        context = torch.matmul(attn, v)  # (B, H, N_pre, dim_per_head)
        context = context.transpose(1, 2).contiguous().view(B, N_pre, self.feature_dim)
        out = self.out_proj(context)
        out = self.norm(pre_feats + out)  # residual for stability

        sim_logits = scores.mean(dim=1)
        return sim_logits, soft_match
