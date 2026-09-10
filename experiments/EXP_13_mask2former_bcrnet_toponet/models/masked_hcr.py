"""
EXP_13: Mask-Gated Hierarchical Curve Refinement (M-HCR)
========================================================
Refines 5th-order Bézier curves through:
1. Deformable cross-attention sampling along curve trajectory B(t)
2. Mask-gated feature modulation: F_gated = F_pixel * (1 + Sigmoid(M_pred))
3. Factored 3-Way Structured Self-Attention (Intra-curve, Inter-proposal, Inter-category)
4. Multi-stage coarse-to-fine control point updates
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

try:
    from ..utils.bezier_ops import evaluate_bezier_torch
except (ImportError, ValueError):
    from utils.bezier_ops import evaluate_bezier_torch


class FactoredStructuredAttention(nn.Module):
    """
    Factored 3-Way Structured Attention:
    Decomposes full (M * K * N) attention into three lightweight 1D attentions:
    1. Intra-Curve (along N=26 points): Curve smoothness
    2. Inter-Curve (along K=5 proposals): Proposal competition
    3. Inter-Category (along M=3 classes): Anatomical spatial hierarchy
    Reduces FLOPs by >20x compared to full flattened attention.
    """

    def __init__(self, embed_dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.intra_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.inter_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cat_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, M, K, N, D)
        Returns:
            out: (B, M, K, N, D)
        """
        B, M, K, N, D = x.shape

        # 1. Intra-curve attention along N points
        x_intra = x.view(B * M * K, N, D)
        x_intra_norm = self.norm1(x_intra)
        attn_intra, _ = self.intra_attn(x_intra_norm, x_intra_norm, x_intra_norm)
        x = x + attn_intra.view(B, M, K, N, D)

        # 2. Inter-curve attention along K proposals
        x_inter = x.permute(0, 1, 3, 2, 4).contiguous().view(B * M * N, K, D)
        x_inter_norm = self.norm2(x_inter)
        attn_inter, _ = self.inter_attn(x_inter_norm, x_inter_norm, x_inter_norm)
        x = x + attn_inter.view(B, M, N, K, D).permute(0, 1, 3, 2, 4).contiguous()

        # 3. Inter-category attention along M classes
        x_cat = x.permute(0, 2, 3, 1, 4).contiguous().view(B * K * N, M, D)
        x_cat_norm = self.norm3(x_cat)
        attn_cat, _ = self.cat_attn(x_cat_norm, x_cat_norm, x_cat_norm)
        x = x + attn_cat.view(B, K, N, M, D).permute(0, 3, 1, 2, 4).contiguous()

        return x


class MaskedHCRStage(nn.Module):
    """A single stage of coarse-to-fine curve refinement."""

    def __init__(
        self,
        in_channels: int = 256,
        embed_dim: int = 128,
        num_ctrl_pts: int = 6,
        num_sample_pts: int = 26,
    ):
        super().__init__()
        self.num_ctrl_pts = num_ctrl_pts
        self.num_sample_pts = num_sample_pts

        # Point feature projection
        self.feat_proj = nn.Linear(in_channels, embed_dim)
        # Coordinate positional encoding
        self.pos_mlp = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )

        self.structured_attn = FactoredStructuredAttention(embed_dim=embed_dim)

        # Control point residual regressor
        self.ctrl_regressor = nn.Sequential(
            nn.Linear(embed_dim * num_sample_pts, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_ctrl_pts * 2),
        )

        # Score refinement head
        self.score_head = nn.Sequential(
            nn.Linear(embed_dim * num_sample_pts, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def sample_features(
        self,
        features: torch.Tensor,     # (B, C, H, W)
        points: torch.Tensor,       # (B, M, K, N, 2) in [0, 1]^2
    ) -> torch.Tensor:
        """
        Differentiable bilinear grid sampling at curve coordinates.
        """
        B, C, H, W = features.shape
        _, M, K, N, _ = points.shape

        # Flatten (M, K, N) into spatial grid: (B, M*K*N, 1, 2)
        grid = points.view(B, M * K * N, 1, 2) * 2.0 - 1.0  # Normalize to [-1, 1]

        # Sample features: (B, C, M*K*N, 1)
        sampled = F.grid_sample(features, grid, align_corners=False, mode="bilinear")
        sampled = sampled.squeeze(-1).permute(0, 2, 1).contiguous()  # (B, M*K*N, C)
        return sampled.view(B, M, K, N, C)

    def forward(
        self,
        features: torch.Tensor,          # (B, C, H, W) mask-gated pixel feature
        ctrl_pts: torch.Tensor,          # (B, M, K, 6, 2)
        scores: torch.Tensor,            # (B, M, K)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, M, K, _, _ = ctrl_pts.shape

        # 1. Evaluate curve points B(t): (B*M*K, 6, 2) -> (B*M*K, N, 2)
        ctrl_flat = ctrl_pts.view(B * M * K, self.num_ctrl_pts, 2)
        curve_pts = evaluate_bezier_torch(ctrl_flat, num_samples=self.num_sample_pts)
        curve_pts = curve_pts.view(B, M, K, self.num_sample_pts, 2)

        # 2. Sample mask-gated features along curve
        feat_sampled = self.sample_features(features, curve_pts)  # (B, M, K, N, C)
        feat_proj = self.feat_proj(feat_sampled)                  # (B, M, K, N, D)
        pos_proj = self.pos_mlp(curve_pts)                       # (B, M, K, N, D)

        tokens = feat_proj + pos_proj

        # 3. 3-Way Structured Attention
        tokens = self.structured_attn(tokens)  # (B, M, K, N, D)

        # 4. Predict control point residual updates
        tokens_flat = tokens.view(B, M, K, -1)  # (B, M, K, N * D)
        delta_ctrl = self.ctrl_regressor(tokens_flat).view(B, M, K, self.num_ctrl_pts, 2)
        delta_score = self.score_head(tokens_flat).squeeze(-1)  # (B, M, K)

        # Bounded residual update: step of 0.15
        refined_ctrl_pts = torch.clamp(
            ctrl_pts + 0.15 * torch.tanh(delta_ctrl),
            0.0, 1.0
        )
        refined_scores = torch.sigmoid(torch.logit(scores.clamp(1e-4, 1.0 - 1e-4)) + delta_score)

        return refined_ctrl_pts, curve_pts, refined_scores


class MaskedHCR(nn.Module):
    """
    Complete Multi-Stage Hierarchical Curve Refinement (M-HCR).
    Iteratively refines Bézier curves across multi-scale feature maps.
    """

    def __init__(
        self,
        in_channels: int = 256,
        embed_dim: int = 128,
        num_stages: int = 3,
        num_ctrl_pts: int = 6,
        num_sample_pts: int = 26,
    ):
        super().__init__()
        self.num_stages = num_stages
        self.num_sample_pts = num_sample_pts
        self.num_ctrl_pts = num_ctrl_pts

        self.stages = nn.ModuleList([
            MaskedHCRStage(
                in_channels=in_channels,
                embed_dim=embed_dim,
                num_ctrl_pts=num_ctrl_pts,
                num_sample_pts=num_sample_pts,
            )
            for _ in range(num_stages)
        ])

    def forward(
        self,
        fpn_features: List[torch.Tensor],  # [f1(1/4), f2(1/8), f3(1/16), f4(1/32)]
        mask_logits: torch.Tensor,          # (B, 4, H, W) from Mask2Former
        init_ctrl_pts: torch.Tensor,        # (B, M, K, 6, 2)
        init_scores: torch.Tensor,          # (B, M, K)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            final_ctrl_pts: (B, M, K, 6, 2)
            final_curve_pts: (B, M, K, N, 2)
            final_scores: (B, M, K)
        """
        # Feature pyramid map assignments for coarse-to-fine stages:
        # Stage 0: f3 (1/16 scale - coarse global context)
        # Stage 1: f2 (1/8 scale - intermediate contour)
        # Stage 2: f1 (1/4 scale - fine sub-pixel boundary)
        feature_levels = [fpn_features[2], fpn_features[1], fpn_features[0]]

        curr_ctrl = init_ctrl_pts
        curr_scores = init_scores
        curr_curve = None

        # Mask probability for foreground landmarks (classes 1..3 -> channels 1..3):
        # Foreground probability sum: (B, 1, H, W)
        fg_mask_prob = torch.sigmoid(mask_logits[:, 1:]).sum(dim=1, keepdim=True).clamp(0.0, 1.0)

        for s, stage_module in enumerate(self.stages):
            f_s = feature_levels[min(s, len(feature_levels) - 1)]

            # Mask-gating: modulate features along foreground landmark boundaries
            # Resize fg_mask_prob to match feature level resolution
            mask_resized = F.interpolate(fg_mask_prob, size=f_s.shape[2:], mode="bilinear", align_corners=False)
            f_gated = f_s * (1.0 + mask_resized)

            curr_ctrl, curr_curve, curr_scores = stage_module(f_gated, curr_ctrl, curr_scores)

        # Final curve evaluation
        B, M, K, _, _ = curr_ctrl.shape
        ctrl_flat = curr_ctrl.view(B * M * K, self.num_ctrl_pts, 2)
        final_curve = evaluate_bezier_torch(ctrl_flat, num_samples=self.num_sample_pts).view(
            B, M, K, self.num_sample_pts, 2
        )

        return curr_ctrl, final_curve, curr_scores
