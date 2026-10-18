from __future__ import annotations

import math

import torch
from torch import nn


class FusionLayer(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_s = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.ffn_norm = nn.LayerNorm(dim)

    def forward(self, point_feats: torch.Tensor, visual_feats: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.cross_attn(point_feats, visual_feats, visual_feats)
        point_feats = self.norm_q(point_feats + attn_out)
        self_out, _ = self.self_attn(point_feats, point_feats, point_feats)
        point_feats = self.norm_s(point_feats + self_out)
        point_feats = self.ffn_norm(point_feats + self.ffn(point_feats))
        return point_feats


class StructureCorrelation(nn.Module):
    def __init__(self, dim: int, tokens: int = 128):
        super().__init__()
        self.tokens = tokens
        self.proj = nn.Linear(dim, tokens)

    def forward(self, feats: torch.Tensor):
        B, N, C = feats.shape
        if N == self.tokens:
            pooled = feats
        else:
            segments = torch.linspace(0, N, steps=self.tokens + 1, device=feats.device, dtype=torch.float32).round().long()
            pooled_chunks = []
            for i in range(self.tokens):
                start = segments[i].item()
                end = max(start + 1, segments[i + 1].item())
                chunk = feats[:, start:end]
                pooled_chunks.append(chunk.mean(dim=1, keepdim=True))
            pooled = torch.cat(pooled_chunks, dim=1)
        emb = self.proj(pooled)
        S = torch.matmul(emb, emb.transpose(1, 2)) / emb.shape[-1]
        return S


class CrossModalFusion(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.1,
        temporal_layers: int = 2,
    ):
        super().__init__()
        self.layers = nn.ModuleList([FusionLayer(dim, heads, dropout) for _ in range(layers)])
        self.structure = nn.ModuleList([StructureCorrelation(dim) for _ in range(layers)])
        if temporal_layers > 0:
            encoder = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, batch_first=True)
            self.temporal = nn.TransformerEncoder(encoder, num_layers=temporal_layers)
            self.temporal_norm = nn.LayerNorm(dim)
        else:
            self.temporal = None
        self.frame_proj = nn.Linear(dim, dim)
        self.contrast_proj = nn.Linear(dim, dim)
        self.anchor_mlp = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
        )

    def _reshape_frames(self, frame_feats: torch.Tensor):
        if frame_feats.dim() == 3:
            B, S, C = frame_feats.shape
            return frame_feats.unsqueeze(1), frame_feats
        if frame_feats.dim() == 4:
            B, F, S, C = frame_feats.shape
            tokens = frame_feats.view(B, F * S, C)
            return frame_feats, tokens
        raise ValueError(f'Unexpected frame feature shape: {frame_feats.shape}')

    def _compute_frame_globals(
        self,
        frame_seq: torch.Tensor,
        temporal_feats: torch.Tensor | None,
        render_contrast_feat: torch.Tensor | None,
    ):
        frame_globals = self.frame_proj(frame_seq.mean(dim=2))
        if render_contrast_feat is not None:
            contrast = render_contrast_feat
            if contrast.dim() == 3:
                # (B, F, 1) -> broadcast
                contrast = contrast.repeat(1, 1, frame_globals.shape[-1])
            elif contrast.dim() == 4:
                contrast = contrast.mean(dim=2)
            frame_globals = frame_globals + self.contrast_proj(contrast)
        if temporal_feats is not None and self.temporal is not None:
            temporal = self.temporal_norm(self.temporal(temporal_feats))
            frame_globals = frame_globals + temporal
        return frame_globals

    def _frame_attention(self, point_feats: torch.Tensor, frame_globals: torch.Tensor):
        B, F, C = frame_globals.shape
        point_global = point_feats.mean(dim=1, keepdim=True)
        logits = torch.matmul(frame_globals, point_global.transpose(-1, -2)).squeeze(-1) / math.sqrt(C)
        return torch.softmax(logits, dim=1)

    def forward(
        self,
        point_feats: torch.Tensor,
        frame_features: torch.Tensor,
        render_condition: torch.Tensor | None = None,
        temporal_feats: torch.Tensor | None = None,
        render_contrast_feat: torch.Tensor | None = None,
    ):
        B, N, C = point_feats.shape
        frame_seq, visual_tokens = self._reshape_frames(frame_features)
        frame_globals = self._compute_frame_globals(frame_seq, temporal_feats, render_contrast_feat)
        frame_weights = self._frame_attention(point_feats, frame_globals)
        context = torch.sum(frame_weights.unsqueeze(-1) * frame_globals, dim=1, keepdim=True)
        context = context.expand(-1, N, -1)
        fused = self.anchor_mlp(torch.cat([point_feats, context], dim=-1))
        structure_mats = []
        for layer, topo in zip(self.layers, self.structure):
            fused = layer(fused, visual_tokens)
            if render_condition is not None:
                fused = fused + render_condition
            structure_mats.append(topo(fused))
        return fused, structure_mats, frame_weights
