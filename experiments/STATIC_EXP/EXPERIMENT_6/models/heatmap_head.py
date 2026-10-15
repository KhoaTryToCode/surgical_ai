"""
Dynamic Spatial Heatmap Head for EXPERIMENT_6.
Replaces the decoupled MLP coordinate regression head.

Mechanism:
  1. The 4 refined junction vectors J_k (B, 4, 256) act as dynamic spatial filters.
  2. Projects J_k and the image feature map F (B, 256, 64, 64) to a common projection space (dim 128).
  3. Computes normalized spatial dot-product:
       heatmap_logits = (J_proj · F_proj) / sqrt(d)
       pred_heatmaps = sigmoid(heatmap_logits)  # (B, 4, 64, 64)
  4. Automatically decodes peak coordinates and existence flags without separate heads.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from experiments.EXPERIMENT_6.utils.heatmap_utils import extract_peak_coords


class DynamicHeatmapHead(nn.Module):
    def __init__(self, in_channels=256, proj_dim=128):
        super().__init__()
        self.proj_dim = proj_dim
        self.scale = 1.0 / np.sqrt(proj_dim)
        
        # Linear projection for 4 Junction vectors
        self.j_proj = nn.Sequential(
            nn.Linear(in_channels, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim)
        )
        
        # 1x1 Convolutional projection for 2D Feature Map
        self.f_proj = nn.Sequential(
            nn.Conv2d(in_channels, proj_dim, kernel_size=1),
            nn.GroupNorm(8, proj_dim),
            nn.GELU(),
            nn.Conv2d(proj_dim, proj_dim, kernel_size=1)
        )
        
        # Learnable temperature scalar initialized to 1.0
        self.temp = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        
        # Prior bias initialization for focal loss stability (CenterNet / RetinaNet style)
        # Initializes sigmoid(bias) ~ 0.1 to prevent loss explosion from background pixels at step 0
        self.bias = nn.Parameter(torch.tensor(-2.19, dtype=torch.float32))

    def forward(self, j_features, feature_map):
        """
        Args:
            j_features (Tensor): (B, 4, 256) refined junction tokens.
            feature_map (Tensor): (B, 256, H, W) stride-16 features (e.g. 64x64).
        Returns:
            pred_heatmaps (Tensor): (B, 4, H, W) continuous sigmoid heatmaps in [0, 1].
            pred_coords (Tensor): (B, 4, 2) decoded peak coordinates in [0, 1]^2.
            pred_vis (Tensor): (B, 4) decoded visibility flags {0.0, 1.0}.
        """
        B, K, C = j_features.shape
        _, _, H, W = feature_map.shape
        
        # 1. Projections
        q = self.j_proj(j_features)        # (B, 4, proj_dim)
        k = self.f_proj(feature_map)       # (B, proj_dim, H, W)
        
        # 2. Dynamic Spatial Dot-Product: (B, 4, C) x (B, C, H, W) -> (B, 4, H, W)
        dots = torch.einsum("bkc,bchw->bkhw", q, k) * (self.scale / (torch.clamp(self.temp, min=0.1, max=5.0)))
        dots = dots + self.bias
        
        # 3. Sigmoid activation
        pred_heatmaps = torch.sigmoid(dots) # (B, 4, H, W)
        
        # 4. Decode peak coordinates and visibility
        pred_coords, pred_vis, _ = extract_peak_coords(pred_heatmaps, threshold=0.3)
        
        return pred_heatmaps, pred_coords, pred_vis
