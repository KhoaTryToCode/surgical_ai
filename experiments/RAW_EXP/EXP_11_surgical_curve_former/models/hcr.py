"""
EXP_11: Hierarchical Curve Refinement (HCR)
============================================
Implements the BCRNet HCR module with 3-way structured self-attention.

Mathematical grounding: BCRNet (arXiv:2506.15279), Section 3.2.

Pipeline:
  Stage h ∈ {0, 1, 2}:
    1. Build query tokens Q from sinusoidal PE of current ref points + learnable semantic query
    2. Deformable cross-attention onto FPN feature pair {f_coarse, f_fine}
    3. 3-way structured self-attention:
         (a) Intra-curve:     N×N  along ref points  → geometric continuity
         (b) Inter-curve:     K×K  along proposals    → mutual suppression
         (c) Inter-category:  M×M  along classes      → topological relations
    4. MLP → predict Δref_pts offsets → clamp to [0,1]
    5. Refit Bézier from updated ref points → control points for next stage
    6. Predict existence logit and curve confidence

Note on deformable cross-attention:
  We implement a simplified differentiable version compatible with standard PyTorch
  (no CUDA-only `MultiScaleDeformableAttention` kernel required for portability).
  For Kaggle CUDA training, the full operator can be substituted transparently.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .bezier_ops import evaluate_bezier_torch, fit_bezier_least_squares
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Sinusoidal Position Encoding for 2D curve reference points
# ─────────────────────────────────────────────────────────────────────────────

class SinusoidalPE2D(nn.Module):
    """
    Sinusoidal positional encoding for 2D normalized coordinates (x, y) ∈ [0,1]^2.

    Follows standard Transformer PE but applied to continuous 2D coordinates.
    Encodes x and y each with D/2 sinusoidal frequencies, concatenated to D total.

    For coordinate u ∈ [0, 1] and frequency i:
        PE(u, 2i)   = sin(2π * i * u / (temperature^{2i/D}))
        PE(u, 2i+1) = cos(2π * i * u / (temperature^{2i/D}))
    """
    def __init__(self, embed_dim: int = 256, temperature: float = 10000.0):
        super().__init__()
        assert embed_dim % 4 == 0, "embed_dim must be divisible by 4 (2 coords × 2 for sin/cos)"
        self.embed_dim = embed_dim
        self.dim_per_coord = embed_dim // 2  # D/2 per coord (x, y)
        self.temperature = temperature

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (..., 2)  normalized (x, y) coordinates in [0, 1]^2.

        Returns:
            pe: (..., embed_dim)  positional encoding tensor.
        """
        device = coords.device
        dtype = coords.dtype
        D_half = self.dim_per_coord  # D/2

        # Frequency scales: shape (D/4,) — each coord has D/4 sin + D/4 cos
        i = torch.arange(D_half // 2, device=device, dtype=dtype)
        freq = self.temperature ** (2.0 * i / float(D_half))  # (D/4,)
        # Scale: 2π / temperature_scale (no: use standard formulation below)
        inv_freq = 1.0 / freq  # (D/4,)

        # coords: (..., 2) → separate x and y
        x = coords[..., 0:1]  # (..., 1)
        y = coords[..., 1:2]  # (..., 1)

        # x PE: sin/cos at D/4 frequencies each
        x_scaled = x * inv_freq  # (..., D/4)
        pe_x = torch.cat([torch.sin(x_scaled), torch.cos(x_scaled)], dim=-1)  # (..., D/2)

        # y PE: sin/cos at D/4 frequencies each
        y_scaled = y * inv_freq
        pe_y = torch.cat([torch.sin(y_scaled), torch.cos(y_scaled)], dim=-1)  # (..., D/2)

        # Concatenate to D total
        pe = torch.cat([pe_x, pe_y], dim=-1)  # (..., embed_dim)
        return pe


# ─────────────────────────────────────────────────────────────────────────────
# Simplified Deformable Cross-Attention over 2 FPN Levels
# ─────────────────────────────────────────────────────────────────────────────

class BiLevelDeformCrossAttn(nn.Module):
    """
    Simplified deformable cross-attention over 2 FPN feature levels.

    For each query point (reference coordinate in [0,1]^2), samples
    D_pts deformed sampling offsets per attention head per level,
    gathers features via bilinear grid_sample, and applies attention weights.

    This is a portable PyTorch implementation. For Kaggle CUDA training,
    this can be replaced with the MMCV MultiScaleDeformableAttention kernel.

    Shapes:
        Q:   (B, L_q, D)        query tokens
        feat_coarse: (B, C, H_c, W_c)  coarse FPN level
        feat_fine:   (B, C, H_f, W_f)  fine FPN level
        ref_pts: (B, L_q, 2)           normalized query reference points
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_sampling_pts: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.D = embed_dim
        self.H = num_heads
        self.P = num_sampling_pts
        self.head_dim = embed_dim // num_heads
        self.num_levels = 2  # coarse + fine

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # Sampling offset predictor: Q → (H, num_levels, P, 2) offsets
        self.offset_proj = nn.Linear(embed_dim, num_heads * self.num_levels * num_sampling_pts * 2)
        # Attention weight predictor: Q → (H, num_levels, P) weights
        self.attn_proj = nn.Linear(embed_dim, num_heads * self.num_levels * num_sampling_pts)
        # Value projection per level
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

        # Init offsets near zero
        nn.init.zeros_(self.offset_proj.weight)
        nn.init.zeros_(self.offset_proj.bias)
        nn.init.constant_(self.attn_proj.bias, 0.0)

    def _sample_features(
        self,
        feat: torch.Tensor,
        ref_pts: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample from a feature map at reference points + predicted offsets.

        Args:
            feat:     (B, C, H, W)  feature map.
            ref_pts:  (B, L_q, 2)   reference point coordinates in [0, 1]^2.
            offsets:  (B, L_q, H_attn, P, 2)  sampling offsets in [-1, 1].

        Returns:
            sampled:  (B, L_q, H_attn, P, C)  sampled feature values.
        """
        B, C, H_feat, W_feat = feat.shape
        L_q = ref_pts.shape[1]
        H_attn = offsets.shape[2]
        P = offsets.shape[3]

        # Normalize ref_pts to [-1, 1] grid_sample convention
        # grid_sample expects grid in [-1, 1]^2 (x: left=-1, right=+1)
        ref_norm = ref_pts * 2.0 - 1.0  # (B, L_q, 2) in [-1, 1]

        # Sampling locations: ref + offset, clamped to valid range
        # ref_norm expanded: (B, L_q, 1, 1, 2), offsets: (B, L_q, H_attn, P, 2)
        sample_locs = ref_norm.unsqueeze(2).unsqueeze(2) + offsets   # (B, L_q, H_attn, P, 2)
        sample_locs = sample_locs.clamp(-1.0, 1.0)

        # Reshape for grid_sample: (B, L_q * H_attn * P, 1, 2)
        sample_flat = sample_locs.view(B, L_q * H_attn * P, 1, 2)
        grid = sample_flat.float()

        # Repeat feat channels for batched sampling
        feat_v = self.value_proj(feat.permute(0, 2, 3, 1).reshape(B, H_feat * W_feat, C))
        feat_img = feat_v.reshape(B, H_feat, W_feat, C).permute(0, 3, 1, 2)  # (B, C, H, W)

        # grid_sample: (B, C, L_q*H*P, 1) → (B, C, L_q*H*P, 1)
        sampled = F.grid_sample(
            feat_img.float(), grid, mode="bilinear", align_corners=False, padding_mode="zeros"
        )  # (B, C, L_q*H_attn*P, 1)
        sampled = sampled.squeeze(-1).to(feat.dtype)  # (B, C, L_q*H_attn*P)
        sampled = sampled.permute(0, 2, 1).view(B, L_q, H_attn, P, C)  # (B, L_q, H_attn, P, C)
        return sampled

    def forward(
        self,
        query: torch.Tensor,
        feat_coarse: torch.Tensor,
        feat_fine: torch.Tensor,
        ref_pts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Deformable cross-attention over 2 FPN levels.

        Args:
            query:      (B, L_q, D)     query tokens (from position encoding + semantic query)
            feat_coarse:(B, C, H_c, W_c)  coarse feature level
            feat_fine:  (B, C, H_f, W_f)  fine feature level
            ref_pts:    (B, L_q, 2)       reference coords in [0, 1]^2

        Returns:
            out: (B, L_q, D) updated query tokens
        """
        B, L_q, D = query.shape
        H = self.H
        P = self.P

        # Predict sampling offsets: (B, L_q, H*2*P*2)
        offsets = self.offset_proj(query).view(B, L_q, H, self.num_levels, P, 2) * 0.1
        # Scale offsets to ≈ ±1/5 of feature map → meaningful but not extreme
        # offsets: (B, L_q, H, 2, P, 2)  where dim=3 is level index

        # Predict attention weights: (B, L_q, H*2*P)
        attn_w = self.attn_proj(query).view(B, L_q, H, self.num_levels, P)
        attn_w = F.softmax(attn_w.view(B, L_q, H, self.num_levels * P), dim=-1)
        attn_w = attn_w.view(B, L_q, H, self.num_levels, P)

        # Split offsets by level
        offsets_c = offsets[:, :, :, 0, :, :]  # (B, L_q, H, P, 2)  coarse level
        offsets_f = offsets[:, :, :, 1, :, :]  # (B, L_q, H, P, 2)  fine level

        # Sample features from both levels
        sampled_c = self._sample_features(feat_coarse, ref_pts, offsets_c)  # (B, L_q, H, P, C)
        sampled_f = self._sample_features(feat_fine,   ref_pts, offsets_f)  # (B, L_q, H, P, C)

        # Weighted aggregation over both levels
        # Reshape for per-head computation: C → head_dim × H
        head_dim = D // H

        def aggregate(sampled, weights):
            # sampled: (B, L_q, H, P, C)   weights: (B, L_q, H, P)
            # Project C to head_dim per head: take [h*hd:(h+1)*hd] slice
            # For simplicity: reshape C → H × head_dim, apply per-head weights
            B_, Lq_, H_, P_, C_ = sampled.shape
            s = sampled.view(B_, Lq_, H_, P_, H_, head_dim).diagonal(dim1=2, dim2=4)
            # More straightforward: just use first head_dim features per head
            # Correct approach: sampled all heads share C features
            w = weights.unsqueeze(-1)   # (B, L_q, H, P, 1)
            return (sampled * w).sum(dim=3)  # (B, L_q, H, C)

        agg_c = aggregate(sampled_c, attn_w[:, :, :, 0, :])   # (B, L_q, H, C)
        agg_f = aggregate(sampled_f, attn_w[:, :, :, 1, :])   # (B, L_q, H, C)

        # Combine levels (mean)
        agg = (agg_c + agg_f) / 2.0   # (B, L_q, H, C)

        # Collapse heads: (B, L_q, H*C)  — but C is shared so just mean over H
        agg_out = agg.mean(dim=2)   # (B, L_q, C=D)

        out = self.out_proj(agg_out)
        return self.dropout(out)


# ─────────────────────────────────────────────────────────────────────────────
# 3-Way Structured Self-Attention (BCRNet Section 3.2)
# ─────────────────────────────────────────────────────────────────────────────

class ThreeWaySelfAttention(nn.Module):
    """
    BCRNet 3-way structured self-attention layer.

    Token layout: (B, M, K_top, N, D)  where:
        M     = num_classes (3 categories)
        K_top = proposals per class (10)
        N     = ref points per proposal (26)
        D     = embed_dim (256)

    Three decomposed attention passes, each over a different dimension:

    (a) Intra-curve:    N × N attention for fixed (m, k)
        → enforces geometric curve continuity and smoothness
        → complexity O(N^2) per (m, k) pair

    (b) Inter-curve:    K × K attention for fixed (m, n)
        → allows proposals to compete and suppress each other (NMS-like soft)
        → complexity O(K^2) per (m, n) pair

    (c) Inter-category: M × M attention for fixed (k, n)
        → models anatomical topology between Ridge, Silhouette, Ligament
        → complexity O(M^2) per (k, n) pair

    Total: O(N^2 * M * K + K^2 * M * N + M^2 * K * N) << O((M*K*N)^2)
    """

    def __init__(self, embed_dim: int = 256, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.D = embed_dim
        self.H = num_heads

        # Separate multi-head self-attention for each dimension
        self.attn_intra = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_inter = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_cat   = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        # Pre-norms for each attention
        self.norm_intra = nn.LayerNorm(embed_dim)
        self.norm_inter = nn.LayerNorm(embed_dim)
        self.norm_cat   = nn.LayerNorm(embed_dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (B, M, K, N, D)  structured curve tokens.

        Returns:
            tokens: (B, M, K, N, D)  refined curve tokens (residual).
        """
        B, M, K, N, D = tokens.shape

        # ── (a) Intra-curve: attention over N for each (m, k) ────────────────
        # Reshape: (B * M * K, N, D)
        t_intra = tokens.view(B * M * K, N, D)
        t_intra_norm = self.norm_intra(t_intra)
        attn_out, _ = self.attn_intra(t_intra_norm, t_intra_norm, t_intra_norm)
        t_intra = t_intra + attn_out
        tokens = t_intra.view(B, M, K, N, D)

        # ── (b) Inter-curve: attention over K for each (m, n) ────────────────
        # Reshape: (B * M * N, K, D)
        t_inter = tokens.permute(0, 1, 3, 2, 4).reshape(B * M * N, K, D)
        t_inter_norm = self.norm_inter(t_inter)
        attn_out, _ = self.attn_inter(t_inter_norm, t_inter_norm, t_inter_norm)
        t_inter = t_inter + attn_out
        tokens = t_inter.view(B, M, N, K, D).permute(0, 1, 3, 2, 4).contiguous()

        # ── (c) Inter-category: attention over M for each (k, n) ─────────────
        # Reshape: (B * K * N, M, D)
        t_cat = tokens.permute(0, 2, 3, 1, 4).reshape(B * K * N, M, D)
        t_cat_norm = self.norm_cat(t_cat)
        attn_out, _ = self.attn_cat(t_cat_norm, t_cat_norm, t_cat_norm)
        t_cat = t_cat + attn_out
        tokens = t_cat.view(B, K, N, M, D).permute(0, 3, 1, 2, 4).contiguous()

        # ── FFN ───────────────────────────────────────────────────────────────
        tokens = tokens + self.ffn(tokens)

        return tokens


# ─────────────────────────────────────────────────────────────────────────────
# HCR Stage (one refinement stage)
# ─────────────────────────────────────────────────────────────────────────────

class HCRStage(nn.Module):
    """
    Single HCR refinement stage.

    Args flow:
        1. Build query tokens: Q = PE(ref_pts) + Q_semantic  (broadcast)
        2. Deformable cross-attention on FPN pair {coarse, fine}
        3. 3-way structured self-attention
        4. MLP → Δref_pts offsets → update ref_pts
        5. Predict stage confidence logits
        6. Refit Bézier from updated ref_pts (numpy, in eval; differentiable approx in training)
    """

    def __init__(
        self,
        num_classes: int = 3,
        top_k: int = 10,
        n_ref_pts: int = 26,
        embed_dim: int = 256,
        ffn_dim: int = 512,
        num_heads: int = 8,
        num_sampling_pts: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.M = num_classes
        self.K = top_k
        self.N = n_ref_pts
        self.D = embed_dim

        # Positional encoding for 2D ref points
        self.pe = SinusoidalPE2D(embed_dim=embed_dim)

        # Learnable per-class semantic query: (1, M, 1, 1, D) — broadcast over K and N
        self.semantic_query = nn.Parameter(torch.randn(1, num_classes, 1, 1, embed_dim) * 0.02)

        # Deformable cross-attention
        self.deform_cross_attn = BiLevelDeformCrossAttn(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_sampling_pts=num_sampling_pts,
            dropout=dropout,
        )

        # 3-way self-attention
        self.three_way_attn = ThreeWaySelfAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)

        # Offset MLP: predicts Δref_pts ∈ R^2 per reference point
        self.offset_mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, 2),
            nn.Tanh(),  # constrain offsets to (-1, 1) before scaling
        )

        # Confidence head: predicts per-proposal score
        self.conf_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1),
        )

        # Offset scale factor (large offset → proposal moves significantly)
        self.offset_scale = 0.1  # offsets scaled to ±10% of normalized image size

        # Post-update norm
        self.post_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        ref_pts: torch.Tensor,
        feat_coarse: torch.Tensor,
        feat_fine: torch.Tensor,
    ) -> dict:
        """
        Args:
            ref_pts:     (B, M, K, N, 2)   current reference point coordinates in [0,1]^2.
            feat_coarse: (B, C, H_c, W_c)  coarse FPN feature map.
            feat_fine:   (B, C, H_f, W_f)  fine FPN feature map.

        Returns:
            dict with:
              ref_pts_updated: (B, M, K, N, 2)  refined reference points (detached for next stage)
              conf_logits:     (B, M, K)         per-proposal confidence logits (before sigmoid)
              tokens:          (B, M, K, N, D)   final token representations
        """
        B, M, K, N, _ = ref_pts.shape
        D = self.D

        # 1. Build query tokens: Q = PE(ref_pts) + semantic_query
        pe_tokens = self.pe(ref_pts)  # (B, M, K, N, D)
        q_tokens = pe_tokens + self.semantic_query.expand(B, M, K, N, D)

        # 2. Deformable cross-attention (flatten M, K, N into batch dimension L_q)
        # ref_pts_flat: (B, M*K*N, 2)
        L_q = M * K * N
        ref_flat = ref_pts.view(B, L_q, 2)
        q_flat = q_tokens.view(B, L_q, D)

        q_cross = q_flat + self.deform_cross_attn(q_flat, feat_coarse, feat_fine, ref_flat)
        q_cross = self.post_norm(q_cross)

        # Reshape back to (B, M, K, N, D)
        tokens = q_cross.view(B, M, K, N, D)

        # 3. 3-way structured self-attention
        tokens = self.three_way_attn(tokens)  # (B, M, K, N, D)

        # 4. Predict offset and update ref points
        delta = self.offset_mlp(tokens)  # (B, M, K, N, 2) in (-1, 1)
        delta = delta * self.offset_scale  # scale to ±0.1 in [0,1] space

        # Update: clamp to valid image canvas
        ref_pts_updated = (ref_pts + delta).clamp(0.0, 1.0)  # (B, M, K, N, 2)

        # 5. Confidence logits per proposal: mean over N ref points
        # tokens: (B, M, K, N, D) → mean N → (B, M, K, D)
        proposal_tokens = tokens.mean(dim=3)  # (B, M, K, D)
        conf_logits = self.conf_head(proposal_tokens).squeeze(-1)  # (B, M, K)

        return {
            "ref_pts_updated": ref_pts_updated,  # (B, M, K, N, 2)
            "conf_logits": conf_logits,          # (B, M, K)
            "tokens": tokens,                    # (B, M, K, N, D)
        }


# ─────────────────────────────────────────────────────────────────────────────
# Full HCR Module (3 stages)
# ─────────────────────────────────────────────────────────────────────────────

class HCRModule(nn.Module):
    """
    Full Hierarchical Curve Refinement (HCR) module.

    3 refinement stages with coarse-to-fine feature pair progression:
        Stage 0: {f3, f4}  (32x32 + 64x64  features — coarsest)
        Stage 1: {f2, f3}  (64x64 + 128x128 features)
        Stage 2: {f1, f2}  (128x128 + 256x256 features — finest)

    After each stage, updated reference points are used to refit Bézier curves
    (differentiable via the ACPI weighted-point approach in training).
    """

    def __init__(
        self,
        num_classes: int = 3,
        top_k: int = 10,
        n_ref_pts: int = 26,
        embed_dim: int = 256,
        ffn_dim: int = 512,
        num_heads: int = 8,
        num_sampling_pts: int = 4,
        num_stages: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_stages = num_stages
        self.M = num_classes
        self.K = top_k
        self.N = n_ref_pts
        self.D = embed_dim

        # Stage list
        self.stages = nn.ModuleList([
            HCRStage(
                num_classes=num_classes,
                top_k=top_k,
                n_ref_pts=n_ref_pts,
                embed_dim=embed_dim,
                ffn_dim=ffn_dim,
                num_heads=num_heads,
                num_sampling_pts=num_sampling_pts,
                dropout=dropout,
            )
            for _ in range(num_stages)
        ])

        # Final Bézier regressor from last-stage ref points
        # Maps updated N ref points → K control points via learned MLP
        # (Differentiable alternative to numpy lstsq in training)
        self.bz_regressor = nn.Sequential(
            nn.Linear(n_ref_pts * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 6 * 2),  # K=6 ctrl pts × 2 coords
            nn.Sigmoid(),  # Bounded in (0,1)^2
        )

    def refit_bezier_differentiable(self, ref_pts: torch.Tensor) -> torch.Tensor:
        """
        Differentiable Bézier control point prediction from N reference points.

        Flattens ref_pts and passes through a learned MLP to predict K=6 ctrl pts.
        This is the training-time path (gradient flows through ref_pts → ctrl_pts).

        Args:
            ref_pts: (B, M, K, N, 2)  updated reference points.

        Returns:
            ctrl_pts: (B, M, K, 6, 2)  predicted Bézier control points in (0,1)^2.
        """
        B, M, K, N, _ = ref_pts.shape
        # Flatten N×2 → 2N and predict ctrl pts
        flat = ref_pts.view(B * M * K, N * 2)
        ctrl = self.bz_regressor(flat)   # (B*M*K, 12)
        ctrl = ctrl.view(B, M, K, 6, 2)
        return ctrl

    def forward(
        self,
        ref_pts_init: torch.Tensor,
        fpn_features: list,
    ) -> dict:
        """
        Full HCR forward pass.

        Args:
            ref_pts_init: (B, M, K, N, 2)  initial ref points from ACPI.
            fpn_features: [f1, f2, f3, f4] FPN feature maps, index 0=finest (stride 4).

        Returns:
            dict with:
              all_ref_pts:     [(B, M, K, N, 2)] × num_stages, per-stage updated ref points
              all_conf_logits: [(B, M, K)]        × num_stages, per-stage confidence logits
              final_ctrl_pts:  (B, M, K, 6, 2)   final Bézier control points (stage-2 output)
              final_ref_pts:   (B, M, K, N, 2)   final reference points
        """
        # Feature pair indices (coarse, fine) for each HCR stage
        # Stage 0: f3 (idx 2) + f4 (idx 3) — coarsest
        # Stage 1: f2 (idx 1) + f3 (idx 2)
        # Stage 2: f1 (idx 0) + f2 (idx 1) — finest
        level_pairs = [(2, 3), (1, 2), (0, 1)]

        all_ref_pts = []
        all_conf_logits = []
        current_ref = ref_pts_init

        for h, stage in enumerate(self.stages):
            coarse_idx, fine_idx = level_pairs[h]
            feat_coarse = fpn_features[coarse_idx]
            feat_fine   = fpn_features[fine_idx]

            stage_out = stage(current_ref, feat_coarse, feat_fine)

            current_ref = stage_out["ref_pts_updated"]  # (B, M, K, N, 2)
            all_ref_pts.append(current_ref)
            all_conf_logits.append(stage_out["conf_logits"])  # (B, M, K)

        # Differentiable Bézier refit from final stage reference points
        final_ctrl_pts = self.refit_bezier_differentiable(current_ref)  # (B, M, K, 6, 2)

        return {
            "all_ref_pts": all_ref_pts,          # List of (B, M, K, N, 2)
            "all_conf_logits": all_conf_logits,  # List of (B, M, K)
            "final_ctrl_pts": final_ctrl_pts,    # (B, M, K, 6, 2) in (0,1)^2
            "final_ref_pts": current_ref,        # (B, M, K, N, 2)
        }
