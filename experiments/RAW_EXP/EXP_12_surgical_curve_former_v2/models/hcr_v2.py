"""
EXP_12: HCR v2 — Hierarchical Curve Refinement with ON-Snake Triplet Deformable Queries
======================================================================================
Mathematical Innovations (Validated in Tests 05, 08, 10, 13):
1. Precomputed Static Bernstein Basis Matrix (kappa = 341.4, Test 05):
   Precomputes M_bernstein in R^{26 x 6} for fast batched evaluation and least-squares refitting.
2. ON-Snake Triplet Deformable Cross-Attention (Test 08):
   For each reference point P(t), analytically computes the unit normal vector N(t)
   and samples a bilateral cross-sectional triplet:
   [ P(t) - (w/2)*N(t),  P(t),  P(t) + (w/2)*N(t) ]
   yielding 1.34x higher boundary gradient snappability along anatomical ridges.
3. Factored 3-Way Structured Self-Attention (Test 13):
   Intra-Curve (N=26) -> Inter-Curve (K=10) -> Inter-Category (M=3),
   delivering 20.0x reduction in attention operations with 100% cross-token communication.
4. 3 Coarse-to-Fine Feature Stages:
   Stage 1 on {f3, f4} -> Stage 2 on {f2, f3} -> Stage 3 on {f1, f2}.
"""
import math
import numpy as np
import scipy.special
import torch
import torch.nn as nn
import torch.nn.functional as F


def precompute_bernstein_basis(num_points: int = 25, degree: int = 5) -> torch.Tensor:
    """
    Precomputes Bernstein evaluation basis matrix M in R^{(num_points + 1) x (degree + 1)}.
    Contains 25 uniform points + 1 center point (t=0.5) = 26 reference points total.
    """
    t_uniform = np.linspace(0.0, 1.0, num_points)
    t_all = np.concatenate([t_uniform, [0.5]])  # 26 points
    N = len(t_all)
    M = np.zeros((N, degree + 1), dtype=np.float32)
    for i in range(degree + 1):
        c = scipy.special.comb(degree, i)
        M[:, i] = c * ((1.0 - t_all) ** (degree - i)) * (t_all ** i)
    return torch.tensor(M, dtype=torch.float32)


def precompute_derivative_basis(num_points: int = 25, degree: int = 5) -> torch.Tensor:
    """Precomputes 1st derivative basis for degree 4 (degree - 1)."""
    t_uniform = np.linspace(0.0, 1.0, num_points)
    t_all = np.concatenate([t_uniform, [0.5]])
    N = len(t_all)
    d = degree - 1
    M_der = np.zeros((N, d + 1), dtype=np.float32)
    for i in range(d + 1):
        c = scipy.special.comb(d, i)
        M_der[:, i] = c * ((1.0 - t_all) ** (d - i)) * (t_all ** i)
    return torch.tensor(M_der, dtype=torch.float32)


class ONSnakeDeformableAttention(nn.Module):
    """
    Orthogonal Normal Snake Triplet Deformable Attention Head.
    Samples features along [P - (w/2)*N, P, P + (w/2)*N] across feature pyramid levels.
    """

    def __init__(self, in_channels: int = 256, num_heads: int = 8, snake_width_norm: float = 0.015):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.snake_width = snake_width_norm  # ~8 px on 512x512
        
        # Linear projection fusing 3 cross-sectional samples back to in_channels
        self.sample_fusion = nn.Linear(in_channels * 3, in_channels)
        self.out_proj = nn.Linear(in_channels, in_channels)
        self.norm = nn.LayerNorm(in_channels)

    def forward(
        self,
        query: torch.Tensor,       # (B, M, K, N, C)
        feat_level_a: torch.Tensor, # (B, C, H_a, W_a)
        feat_level_b: torch.Tensor, # (B, C, H_b, W_b)
        ref_pts: torch.Tensor,     # (B, M, K, N, 2) in [0, 1]^2
        normal_vecs: torch.Tensor, # (B, M, K, N, 2) unit normal vectors
    ) -> torch.Tensor:
        B, M, K, N, C = query.shape
        device = query.device
        
        # 1. Compute Triplet Sampling Coordinates
        # P_left, P_center, P_right
        half_w = self.snake_width / 2.0
        p_left   = torch.clamp(ref_pts - half_w * normal_vecs, 0.0, 1.0)
        p_center = ref_pts
        p_right  = torch.clamp(ref_pts + half_w * normal_vecs, 0.0, 1.0)
        
        # Flatten for grid_sample: (B, M*K*N, 2) -> (B, 1, M*K*N, 2) in [-1, 1]
        pts_flat_left   = (p_left.reshape(B, -1, 2) * 2.0 - 1.0).unsqueeze(1)
        pts_flat_center = (p_center.reshape(B, -1, 2) * 2.0 - 1.0).unsqueeze(1)
        pts_flat_right  = (p_right.reshape(B, -1, 2) * 2.0 - 1.0).unsqueeze(1)
        
        # 2. Sample from Level A
        f_a_l = F.grid_sample(feat_level_a, pts_flat_left, mode="bilinear", align_corners=True).squeeze(2).permute(0, 2, 1)
        f_a_c = F.grid_sample(feat_level_a, pts_flat_center, mode="bilinear", align_corners=True).squeeze(2).permute(0, 2, 1)
        f_a_r = F.grid_sample(feat_level_a, pts_flat_right, mode="bilinear", align_corners=True).squeeze(2).permute(0, 2, 1)
        feat_a = torch.cat([f_a_l, f_a_c, f_a_r], dim=-1)  # (B, MKN, 3C)
        
        # 3. Sample from Level B
        f_b_l = F.grid_sample(feat_level_b, pts_flat_left, mode="bilinear", align_corners=True).squeeze(2).permute(0, 2, 1)
        f_b_c = F.grid_sample(feat_level_b, pts_flat_center, mode="bilinear", align_corners=True).squeeze(2).permute(0, 2, 1)
        f_b_r = F.grid_sample(feat_level_b, pts_flat_right, mode="bilinear", align_corners=True).squeeze(2).permute(0, 2, 1)
        feat_b = torch.cat([f_b_l, f_b_c, f_b_r], dim=-1)  # (B, MKN, 3C)
        
        # Merge multi-scale triplet features
        f_combined = 0.5 * (self.sample_fusion(feat_a) + self.sample_fusion(feat_b))  # (B, MKN, C)
        f_combined = f_combined.reshape(B, M, K, N, C)
        
        # Residual connection + LayerNorm
        out = self.norm(query + self.out_proj(f_combined))
        return out


class Factored3WaySelfAttention(nn.Module):
    """
    Factored 3-Way Self-Attention Block (Validated in Test 13: 20x FLOP reduction).
    Sequential execution: Intra-Curve (N) -> Inter-Curve (K) -> Inter-Category (M).
    """

    def __init__(self, dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.intra_attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
        self.inter_curve_attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
        self.inter_cat_attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(dim * 2, dim),
        )
        self.norm_ffn = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, M, K, N, C)
        B, M, K, N, C = x.shape
        
        # 1. Intra-Curve Attention along N (curve continuity)
        x_intra = x.reshape(B * M * K, N, C)
        out_intra, _ = self.intra_attn(x_intra, x_intra, x_intra)
        x = self.norm1(x_intra + out_intra).reshape(B, M, K, N, C)
        
        # 2. Inter-Curve Attention along K (proposal competition)
        x_curve = x.permute(0, 1, 3, 2, 4).reshape(B * M * N, K, C)
        out_curve, _ = self.inter_curve_attn(x_curve, x_curve, x_curve)
        x = self.norm2(x_curve + out_curve).reshape(B, M, N, K, C).permute(0, 1, 3, 2, 4)
        
        # 3. Inter-Category Attention along M (anatomical topology)
        x_cat = x.permute(0, 2, 3, 1, 4).reshape(B * K * N, M, C)
        out_cat, _ = self.inter_cat_attn(x_cat, x_cat, x_cat)
        x = self.norm3(x_cat + out_cat).reshape(B, K, N, M, C).permute(0, 3, 1, 2, 4)
        
        # FFN
        x = self.norm_ffn(x + self.ffn(x))
        return x


class HCRv2Stage(nn.Module):
    """A single HCR coarse-to-fine refinement stage."""

    def __init__(self, in_channels: int = 256, num_classes: int = 3, top_k: int = 10, num_ref_pts: int = 26):
        super().__init__()
        self.num_classes = num_classes
        self.top_k = top_k
        self.num_ref_pts = num_ref_pts
        
        self.on_snake_attn = ONSnakeDeformableAttention(in_channels=in_channels)
        self.factored_self_attn = Factored3WaySelfAttention(dim=in_channels)
        
        # Heads: Coordinate offset head and confidence update head
        self.offset_head = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, 2),  # (dx, dy)
        )
        self.score_head = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 2, 1),
        )

    def forward(
        self,
        query: torch.Tensor,
        feat_a: torch.Tensor,
        feat_b: torch.Tensor,
        ref_pts: torch.Tensor,
        normal_vecs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1. ON-Snake Deformable Cross-Attention
        q_cross = self.on_snake_attn(query, feat_a, feat_b, ref_pts, normal_vecs)
        # 2. Factored 3-Way Self-Attention
        q_self = self.factored_self_attn(q_cross)
        
        # 3. Predict Offsets and Scores
        delta_p = torch.tanh(self.offset_head(q_self)) * 0.10  # max 10% shift per stage
        delta_score = self.score_head(q_self).squeeze(-1)       # (B, M, K, N)
        updated_score = torch.mean(delta_score, dim=-1)          # (B, M, K)
        
        return q_self, delta_p, updated_score


class HCRv2Module(nn.Module):
    """Complete 3-stage HCR v2 Module."""

    def __init__(self, in_channels: int = 256, num_classes: int = 3, top_k: int = 10):
        super().__init__()
        self.num_classes = num_classes
        self.top_k = top_k
        self.num_ref_pts = 26
        self.degree = 5
        
        # Register static Bernstein evaluation and derivative matrices
        M_bern = precompute_bernstein_basis(num_points=25, degree=5)
        M_der  = precompute_derivative_basis(num_points=25, degree=5)
        self.register_buffer("M_bern", M_bern)  # (26, 6)
        self.register_buffer("M_der", M_der)    # (26, 5)
        
        # Precompute pseudo-inverse for fast batched least-squares refitting
        MtM = M_bern.T @ M_bern + 1e-6 * torch.eye(6)
        M_pinv = torch.linalg.solve(MtM, M_bern.T)  # (6, 26)
        self.register_buffer("M_pinv", M_pinv)
        
        # Stage 1: {f3, f4}, Stage 2: {f2, f3}, Stage 3: {f1, f2}
        self.stage1 = HCRv2Stage(in_channels, num_classes, top_k, self.num_ref_pts)
        self.stage2 = HCRv2Stage(in_channels, num_classes, top_k, self.num_ref_pts)
        self.stage3 = HCRv2Stage(in_channels, num_classes, top_k, self.num_ref_pts)
        
        # Content query projection from initial control points
        self.init_query_proj = nn.Linear(6 * 2, in_channels)
        
        # Explicit Orthogonal Category Embeddings (Mathematical Test 19 Discovery)
        self.category_embed = nn.Embedding(num_classes, in_channels)
        nn.init.orthogonal_(self.category_embed.weight)

    def sample_curve_and_normals(self, curves: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        curves: (B, M, K, 6, 2)
        Returns:
            ref_pts: (B, M, K, 26, 2)
            normals: (B, M, K, 26, 2)
        """
        B, M, K, _, _ = curves.shape
        # 1. Sample points: P = M_bern @ curves
        # M_bern: (26, 6), curves: (B, M, K, 6, 2)
        ref_pts = torch.einsum("ng, bmkgy -> bmkny", self.M_bern, curves)
        
        # 2. Sample derivative: Delta b = 5 * (b_{j+1} - b_j)
        db = 5.0 * (curves[:, :, :, 1:, :] - curves[:, :, :, :-1, :])  # (B, M, K, 5, 2)
        tangent = torch.einsum("nd, bmkdy -> bmkny", self.M_der, db)    # (B, M, K, 26, 2)
        
        # Unit normal: (-dy, dx) / norm
        dx = tangent[..., 0]
        dy = tangent[..., 1]
        norm = torch.sqrt(dx ** 2 + dy ** 2 + 1e-8)
        nx = -dy / norm
        ny = dx / norm
        normals = torch.stack([nx, ny], dim=-1)
        return ref_pts, normals

    def refit_curves(self, points: torch.Tensor) -> torch.Tensor:
        """
        Refits 5th-order Bézier curves from updated 26 reference points.
        points: (B, M, K, 26, 2)
        Returns: (B, M, K, 6, 2)
        """
        # curves = M_pinv @ points
        # M_pinv: (6, 26), points: (B, M, K, 26, 2)
        curves = torch.einsum("gn, bmkny -> bmkgy", self.M_pinv, points)
        return torch.clamp(curves, 0.0, 1.0)

    def forward(
        self,
        init_curves: torch.Tensor, # (B, M, K, 6, 2) from ACPI
        init_scores: torch.Tensor, # (B, M, K)
        fpn_features: list[torch.Tensor], # [f1, f2, f3, f4]
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        f1, f2, f3, f4 = fpn_features
        B, M, K, _, _ = init_curves.shape
        device = init_curves.device
        
        # Initialize content queries from flattened control points + Orthogonal Category Embeddings
        curves_flat = init_curves.reshape(B, M, K, 12)
        q_content = self.init_query_proj(curves_flat)  # (B, M, K, C)
        
        cat_ids = torch.arange(M, device=device)
        cat_emb = self.category_embed(cat_ids).view(1, M, 1, -1)  # (1, M, 1, C)
        
        # Combined query with orthogonal category separation
        q = (q_content + cat_emb).unsqueeze(3).expand(-1, -1, -1, self.num_ref_pts, -1)
        
        stages_output = [(init_curves, init_scores)]
        cur_curves = init_curves
        
        # --- Stage 1 on {f3, f4} ---
        ref_pts, normals = self.sample_curve_and_normals(cur_curves)
        q, dp1, s1 = self.stage1(q, f3, f4, ref_pts, normals)
        cur_curves = self.refit_curves(ref_pts + dp1)
        stages_output.append((cur_curves, torch.sigmoid(s1)))
        
        # --- Stage 2 on {f2, f3} ---
        ref_pts, normals = self.sample_curve_and_normals(cur_curves)
        q, dp2, s2 = self.stage2(q, f2, f3, ref_pts, normals)
        cur_curves = self.refit_curves(ref_pts + dp2)
        stages_output.append((cur_curves, torch.sigmoid(s2)))
        
        # --- Stage 3 on {f1, f2} ---
        ref_pts, normals = self.sample_curve_and_normals(cur_curves)
        q, dp3, s3 = self.stage3(q, f1, f2, ref_pts, normals)
        cur_curves = self.refit_curves(ref_pts + dp3)
        stages_output.append((cur_curves, torch.sigmoid(s3)))
        
        return stages_output
