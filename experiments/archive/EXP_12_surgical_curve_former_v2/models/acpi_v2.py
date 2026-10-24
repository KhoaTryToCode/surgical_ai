"""
EXP_12: ACPI v2 — Adaptive Curve Proposal Initialization with Bounded Tanh Residuals
====================================================================================
Mathematical Innovations (Validated in Test 09 & Test 12):
1. Bounded Tanh Residual Mapping:
   b_i^j = clamp( c_i + tanh(Delta b_i^j) * r_max, 0.0, 1.0 )
   Eliminates BCRNet's sigmoid-logit border saturation, providing 12.7x higher
   gradient sensitivity near image borders (constant 0.2500 vs 0.0196).
2. Predicts 5th-order Bézier curves (6 control points = 12 coordinate offsets per pixel).
3. Supervised with Weighted Proposal Induction Loss (pos_weight = 15.0) to resolve
   the 1:255 pixel class imbalance on feature map f_4.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ACPIv2Module(nn.Module):
    """
    Adaptive Curve Proposal Initialization v2.
    Operating on high-level feature pyramid map f_4 (Stride 32 or 16).
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 3,       # Falciform (0), Ridge (1), Silhouette (2)
        bezier_order: int = 5,      # 5th-order = 6 control points
        top_k: int = 10,            # Proposals selected per category
        r_max: float = 0.35,        # Maximum spatial displacement radius from anchor
    ):
        super().__init__()
        self.num_classes = num_classes
        self.bezier_order = bezier_order
        self.num_ctrl_pts = bezier_order + 1  # 6 control points
        self.top_k = top_k
        self.r_max = r_max
        self.out_dim = self.num_ctrl_pts * 2  # 12 coordinate offsets

        # Offset regression head: Shared trunk + class-specific 12D projection
        self.offset_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_classes * self.out_dim, kernel_size=1),
        )

        # Proposal confidence score head
        self.score_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_classes, kernel_size=1),
        )

        # Initialize score head with negative bias for stable early training
        nn.init.constant_(self.score_conv[-1].bias, -2.19)  # sigmoid(-2.19) ~ 0.10

    def forward(
        self, f4: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            f4: (B, C, H, W) high-level feature map (e.g. 16x16 or 32x32)

        Returns:
            proposals: (B, M, K, 6, 2) top-K 5th-order Bézier curves per class in [0, 1]^2
            scores:    (B, M, K) proposal confidence scores
            score_map: (B, M, H, W) raw logits for weighted proposal induction loss
        """
        B, C, H, W = f4.shape
        device = f4.device

        # 1. Predict Raw Residual Offsets and Confidence Logits
        raw_offsets = self.offset_conv(f4)  # (B, M * 12, H, W)
        score_logits = self.score_conv(f4)  # (B, M, H, W)

        # Reshape offsets to (B, M, H, W, 6, 2)
        offsets = raw_offsets.view(B, self.num_classes, self.num_ctrl_pts, 2, H, W)
        offsets = offsets.permute(0, 1, 4, 5, 2, 3)  # (B, M, H, W, 6, 2)

        # 2. Normalized Pixel Grid Anchors c_i = (c_ix, c_iy) in [0, 1]^2
        ys = (torch.arange(H, device=device, dtype=torch.float32) + 0.5) / H
        xs = (torch.arange(W, device=device, dtype=torch.float32) + 0.5) / W
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        anchors = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)
        anchors = anchors.view(1, 1, H, W, 1, 2)         # (1, 1, H, W, 1, 2)

        # 3. Bounded Tanh Residual Mapping (Validated in Test 09)
        # b_i^j = clamp( c_i + tanh(Delta b_i^j) * r_max, 0.0, 1.0 )
        bounded_curves = torch.clamp(
            anchors + torch.tanh(offsets) * self.r_max, 0.0, 1.0
        )  # (B, M, H, W, 6, 2)

        # 4. Top-K Candidate Selection per Category
        score_probs = torch.sigmoid(score_logits)  # (B, M, H, W)
        score_flat = score_probs.view(B, self.num_classes, H * W)  # (B, M, HW)
        curves_flat = bounded_curves.view(B, self.num_classes, H * W, self.num_ctrl_pts, 2)

        # Top-K indices per class
        topk_scores, topk_indices = torch.topk(score_flat, k=self.top_k, dim=-1)  # (B, M, K)

        # Gather Top-K curves
        # Expand indices for gathering: (B, M, K, 6, 2)
        gather_indices = topk_indices.unsqueeze(-1).unsqueeze(-1).expand(
            B, self.num_classes, self.top_k, self.num_ctrl_pts, 2
        )
        topk_curves = torch.gather(curves_flat, dim=2, index=gather_indices)  # (B, M, K, 6, 2)

        return topk_curves, topk_scores, score_logits
