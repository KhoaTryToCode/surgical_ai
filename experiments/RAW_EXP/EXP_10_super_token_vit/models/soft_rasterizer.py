import torch
import torch.nn as nn
import torch.nn.functional as F
from .spline_utils import evaluate_bezier_curve_torch


class SoftCurveRasterizer(nn.Module):
    """
    Differentiable Soft Line Rasterizer in PyTorch.
    
    Renders predicted continuous Bézier control points (B, C, K, 2) into
    soft probability masks (B, C, H_render, W_render) via vectorized
    Gaussian distance fields over sampled trajectory points.
    
    Enables end-to-end backpropagation of Soft Dice / Focal loss directly
    into the Bézier control point coordinates.
    """
    def __init__(
        self,
        render_size: int = 128,
        num_samples: int = 64,
        sigma_px: float = 1.5,
        target_size: int = 512
    ):
        super().__init__()
        self.render_size = render_size
        self.num_samples = num_samples
        self.sigma_px = sigma_px
        self.target_size = target_size
        
        # Precompute normalized coordinate grid in [0, 1]^2
        # Shape: (1, 1, render_size, render_size, 2)
        ys = torch.linspace(0.0, 1.0, render_size, dtype=torch.float32)
        xs = torch.linspace(0.0, 1.0, render_size, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([grid_x, grid_y], dim=-1)  # (render_size, render_size, 2)
        self.register_buffer("grid", grid.unsqueeze(0).unsqueeze(0))
        
        # Normalized sigma squared in [0, 1] coordinate system
        sigma_norm = sigma_px / float(target_size)
        self.inv_two_sigma_sq = 1.0 / (2.0 * (sigma_norm ** 2))

    def forward(
        self,
        ctrl_points: torch.Tensor,
        existence_probs: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward rendering pass.
        
        Args:
            ctrl_points: (B, C, K, 2) control points in [0, 1]^2
            existence_probs: Optional (B, C) or (B, C, 1) probability tensor
            
        Returns:
            soft_masks: (B, C, render_size, render_size) soft raster masks in [0, 1]
        """
        B, C, K, _ = ctrl_points.shape
        R = self.render_size
        
        # 1. Sample N dense points along the continuous Bézier curve
        # sampled_pts: (B, C, N, 2)
        sampled_pts = evaluate_bezier_curve_torch(ctrl_points, num_samples=self.num_samples)
        
        # 2. Vectorized Distance Computation to curve points
        # Grid shape: (1, 1, R, R, 1, 2)
        # Sampled pts: (B, C, 1, 1, N, 2)
        grid_exp = self.grid.unsqueeze(-2)  # (1, 1, R, R, 1, 2)
        pts_exp = sampled_pts.unsqueeze(2).unsqueeze(2)  # (B, C, 1, 1, N, 2)
        
        # Squared Euclidean distance from every grid pixel to all N curve samples: (B, C, R, R, N)
        diff = grid_exp - pts_exp
        dist_sq = diff.pow(2).sum(dim=-1)
        
        # Minimum distance to the curve: (B, C, R, R)
        min_dist_sq, _ = dist_sq.min(dim=-1)
        
        # 3. Gaussian Soft Falloff
        soft_mask = torch.exp(-min_dist_sq * self.inv_two_sigma_sq)
        
        # 4. Modulate by landmark visibility / existence probability if provided
        if existence_probs is not None:
            if existence_probs.dim() == 2:
                exist_mod = existence_probs.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
            elif existence_probs.dim() == 3:
                exist_mod = existence_probs.unsqueeze(-1)  # (B, C, 1, 1)
            else:
                exist_mod = existence_probs
            soft_mask = soft_mask * exist_mod
            
        return soft_mask

    def render_to_target_size(
        self,
        ctrl_points: torch.Tensor,
        existence_probs: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Renders soft masks and upsamples to target evaluation size (e.g. 512x512).
        """
        soft_masks = self.forward(ctrl_points, existence_probs)
        if self.render_size != self.target_size:
            soft_masks = F.interpolate(
                soft_masks,
                size=(self.target_size, self.target_size),
                mode="bilinear",
                align_corners=False
            )
        return soft_masks
