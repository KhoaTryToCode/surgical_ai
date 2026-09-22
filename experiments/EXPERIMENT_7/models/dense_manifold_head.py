import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseManifoldHead(nn.Module):
    """
    Auxiliary Dense Manifold Head for EXPERIMENT_7.
    
    Predicts continuous (u, v) in [0, 1]^2 coordinates across the liver parenchyma
    from stride-4 pixel decoder mask features. Acts as a topological regularizer
    forcing backbone representations to understand organ surface manifold depth.
    """
    def __init__(self, in_channels=256, hidden_dim=64):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 2, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, mask_features, target_size=(1024, 1024)):
        """
        Args:
            mask_features: Tensor (B, 256, H/4, W/4)
            target_size: (H_out, W_out)
        Returns:
            pred_uv: Tensor (B, 2, H_out, W_out) in [0, 1]^2
        """
        uv_low = self.head(mask_features)  # (B, 2, H/4, W/4)
        pred_uv = F.interpolate(uv_low, size=target_size, mode="bilinear", align_corners=False)
        return pred_uv
