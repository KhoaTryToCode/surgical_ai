import torch
import torch.nn as nn
import torch.nn.functional as F

class ProposalHead3D(nn.Module):
    """
    Option B Proposal Head:
    Predicts initial coarse 3D landmark anchor points p_anchor^{(0)} in [-1, 1]^3 for Layer 1
    from fused visual + 3D feature maps.
    Outputs:
      anchors_3d: (B, N, K, 3) initial 3D vertex positions in [-1, 1]^3.
    """
    def __init__(self, embed_dim: int = 256, num_instances: int = 10, num_points: int = 20):
        super().__init__()
        self.num_instances = num_instances
        self.num_points = num_points

        # Conv proposal classifier / regressor over stride-8 feature map
        self.proposal_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Regresses N instance center 3D points
        self.center_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_instances * 3),
            nn.Tanh() # Enforces [-1, 1]^3 bounds
        )

        # Offsets for K vertices along a small default canonical line around center
        # Point offsets from -0.1 to +0.1
        point_steps = torch.linspace(-0.1, 0.1, num_points)
        offsets = torch.stack([point_steps, torch.zeros(num_points), torch.zeros(num_points)], dim=1) # (K, 3)
        self.register_buffer("default_offsets", offsets)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        """
        feature_map: (B, embed_dim, H, W)
        Returns:
          anchors_3d: (B, N, K, 3) initial 3D anchor locations
        """
        B = feature_map.size(0)
        pooled = self.proposal_conv(feature_map).view(B, -1) # (B, embed_dim)
        
        centers = self.center_head(pooled).view(B, self.num_instances, 1, 3) # (B, N, 1, 3)
        
        # Add default point offsets along the curve
        offsets = self.default_offsets.view(1, 1, self.num_points, 3) # (1, 1, K, 3)
        anchors_3d = centers + offsets # (B, N, K, 3)
        anchors_3d = torch.clamp(anchors_3d, -1.0, 1.0)
        return anchors_3d
