"""
EXP_11: Adaptive Curve Proposal Initialization (ACPI)
======================================================
Implements the BCRNet ACPI module.

Mathematical grounding: BCRNet (arXiv:2506.15279), Section 3.1.

For each pixel location i in the feature map f4 (stride 32),
the ACPI head predicts K=6 Bézier control point offsets Δb ∈ R^{12}
and a confidence score s ∈ (0, 1).

The bounded offset mapping guarantees all control points lie in (0, 1)^2:
    b_i^j = ( sigma(Δb_ix^j + logit(c_ix)),
              sigma(Δb_iy^j + logit(c_iy)) )

During training, an Induction Loss prevents cold-start collapse:
    L_ind = BCE(s_init, s*)
where s* is 1 at GT curve midpoints, 0 elsewhere.

Reference point construction for HCR:
    P_s ∪ {B(0.5)}  — 25 uniform + 1 center = 26 total
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .bezier_ops import (
    acpi_bounded_offset,
    build_coord_grid,
    build_acpi_reference_points,
    evaluate_bezier_torch,
)


class ACPIHead(nn.Module):
    """
    Per-class ACPI prediction head.

    Takes a single FPN feature map f_l (B, C_fpn, H_f, W_f) and predicts:
      - delta_b: (B, K*2, H_f, W_f)  — 12 Bézier offset channels
      - score:   (B, 1,   H_f, W_f)  — confidence score map

    The bounded offset mapping is applied jointly with the precomputed
    coordinate grid to yield control point maps (B, K, 2, H_f, W_f).
    """

    def __init__(
        self,
        in_channels: int = 256,
        hidden_channels: int = 256,
        num_ctrl_pts: int = 6,
        feature_map_h: int = 32,
        feature_map_w: int = 32,
    ):
        super().__init__()
        self.K = num_ctrl_pts  # 6 control points for degree-5 Bézier
        out_channels = num_ctrl_pts * 2 + 1  # 12 offsets + 1 score = 13

        # Lightweight conv head: 2 × 3×3 depthwise-separable + 1×1 projection
        self.conv_stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False),
            nn.GroupNorm(32, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            nn.GroupNorm(32, hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.out_conv = nn.Conv2d(hidden_channels, out_channels, 1)

        # Weight init: small init for stable early training
        nn.init.normal_(self.out_conv.weight, std=0.01)
        nn.init.zeros_(self.out_conv.bias)

        # Pre-computed normalized coordinate grid (c_ix, c_iy) for ACPI formula
        # Registered as buffer so it moves to the correct device automatically
        grid = build_coord_grid(feature_map_h, feature_map_w)  # (1, 1, 2, H, W)
        self.register_buffer("coord_grid", grid)

    def forward(self, feat: torch.Tensor) -> dict:
        """
        Args:
            feat: (B, C_fpn, H_f, W_f)  FPN feature map.

        Returns:
            dict with:
              ctrl_pts_map: (B, K, 2, H_f, W_f)  ACPI control points in (0,1)^2
              score_map:    (B, 1, H_f, W_f)      raw confidence logits (pre-sigmoid)
        """
        B, C, H_f, W_f = feat.shape

        # Adaptive resize coord_grid if feature map size changed (e.g. multi-scale)
        if self.coord_grid.shape[-2] != H_f or self.coord_grid.shape[-1] != W_f:
            grid = build_coord_grid(H_f, W_f).to(feat.device, feat.dtype)
        else:
            grid = self.coord_grid.to(feat.dtype)

        # Feature → raw predictions
        h = self.conv_stem(feat)
        raw = self.out_conv(h)  # (B, K*2+1, H_f, W_f)

        delta_b = raw[:, :-1, :, :]  # (B, K*2, H_f, W_f)
        score_logits = raw[:, -1:, :, :]  # (B, 1, H_f, W_f)

        # Expand delta_b for ACPI: (B, 1, K*2, H_f, W_f)
        # Note: ACPIHead is per-class, so M=1 here
        delta_b_m = delta_b.unsqueeze(1)  # (B, 1, K*2, H_f, W_f)

        # Apply ACPI bounded mapping: b = sigma(Δb + logit(c))
        # Returns: (B, 1, K, 2, H_f, W_f)
        ctrl_pts_m = acpi_bounded_offset(delta_b_m, grid)

        # Squeeze class dim (this head is per-class)
        ctrl_pts = ctrl_pts_m.squeeze(1)  # (B, K, 2, H_f, W_f)

        return {
            "ctrl_pts_map": ctrl_pts,   # (B, K, 2, H_f, W_f)  all in (0,1)^2
            "score_logits": score_logits,  # (B, 1, H_f, W_f) raw logits
        }


class ACPIModule(nn.Module):
    """
    Multi-class ACPI Module.

    Applies one ACPIHead per landmark class and produces the top-K curve proposals
    to be refined by HCR.

    Pipeline:
        f4 → [ACPIHead × M] → ctrl_pts_map (B, M, K, 2, H_f, W_f)
                             → score_map    (B, M, H_f, W_f)
                             → Top-K NMS    → proposals (B, M, K_top, K, 2)
    """

    def __init__(
        self,
        num_classes: int = 3,
        in_channels: int = 256,
        hidden_channels: int = 256,
        num_ctrl_pts: int = 6,
        feature_map_h: int = 32,
        feature_map_w: int = 32,
        top_k: int = 10,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.K = num_ctrl_pts
        self.top_k = top_k

        # One lightweight head per class
        self.heads = nn.ModuleList([
            ACPIHead(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                num_ctrl_pts=num_ctrl_pts,
                feature_map_h=feature_map_h,
                feature_map_w=feature_map_w,
            )
            for _ in range(num_classes)
        ])

    def forward(self, f4: torch.Tensor) -> dict:
        """
        Args:
            f4: (B, C_fpn, H_f, W_f)  ACPI feature map (stride-32 level).

        Returns:
            dict with:
              ctrl_pts_map:  (B, M, K, 2, H_f, W_f)  all proposals in (0,1)^2
              score_logits:  (B, M, H_f, W_f)          raw confidence logits
              proposals:     (B, M, K_top, K, 2)        top-K selected per class
              proposal_scores: (B, M, K_top)            corresponding scores (pre-sigmoid)
        """
        B, _, H_f, W_f = f4.shape
        K = self.K
        M = self.num_classes

        ctrl_maps_list = []
        score_maps_list = []

        for m, head in enumerate(self.heads):
            out = head(f4)
            ctrl_maps_list.append(out["ctrl_pts_map"])   # (B, K, 2, H_f, W_f)
            score_maps_list.append(out["score_logits"])  # (B, 1, H_f, W_f)

        # Stack across classes
        # ctrl_pts_map: (B, M, K, 2, H_f, W_f)
        ctrl_pts_map = torch.stack(ctrl_maps_list, dim=1)
        # score_logits: (B, M, H_f, W_f)
        score_logits = torch.cat(score_maps_list, dim=1)

        # ─── Top-K Proposal Selection ─────────────────────────────────────────
        # Flatten spatial dims for efficient top-K: (B, M, H_f * W_f)
        N_spatial = H_f * W_f
        scores_flat = score_logits.view(B, M, N_spatial)  # (B, M, N_spatial)

        # Top-K indices per class: (B, M, K_top)
        k_actual = min(self.top_k, N_spatial)
        topk_scores, topk_idx = torch.topk(scores_flat, k=k_actual, dim=-1)
        # topk_idx: (B, M, K_top)

        # Gather corresponding ctrl points from flattened spatial map
        # ctrl_pts_map: (B, M, K, 2, H_f*W_f)
        ctrl_flat = ctrl_pts_map.view(B, M, K, 2, N_spatial)

        # Expand topk_idx to gather: (B, M, K, 2, K_top)
        idx_expanded = topk_idx.unsqueeze(2).unsqueeze(2).expand(B, M, K, 2, k_actual)
        # proposals: (B, M, K, 2, K_top) → transpose → (B, M, K_top, K, 2)
        proposals_raw = ctrl_flat.gather(dim=-1, index=idx_expanded)   # (B, M, K, 2, K_top)
        proposals = proposals_raw.permute(0, 1, 4, 2, 3).contiguous()  # (B, M, K_top, K, 2)

        return {
            "ctrl_pts_map": ctrl_pts_map,      # (B, M, K, 2, H_f, W_f)
            "score_logits": score_logits,      # (B, M, H_f, W_f)
            "proposals": proposals,            # (B, M, K_top, K, 2)  in (0,1)^2
            "proposal_scores": topk_scores,   # (B, M, K_top)  raw logits
        }

    def build_hcr_reference_points(
        self,
        proposals: torch.Tensor,
        n_uniform: int = 25,
    ) -> torch.Tensor:
        """
        Constructs N = n_uniform + 1 HCR reference points from the top-K proposals.

        BCRNet: P_s = {B(t_1), ..., B(t_{N-1})} ∪ {B(0.5)}, N = 26 total.

        Args:
            proposals: (B, M, K_top, K, 2)  top-K Bézier control points in (0,1)^2.
            n_uniform: Number of uniform curve samples (default 25).

        Returns:
            ref_pts: (B, M, K_top, N, 2)  N=26 reference points per proposal.
        """
        B, M, K_top, K_ctrl, _ = proposals.shape
        # Flatten for batched curve evaluation
        ctrl_flat = proposals.view(B * M * K_top, K_ctrl, 2)
        ref_flat = build_acpi_reference_points(ctrl_flat, n_uniform=n_uniform)  # (B*M*K_top, 26, 2)
        ref_pts = ref_flat.view(B, M, K_top, n_uniform + 1, 2)
        return ref_pts
