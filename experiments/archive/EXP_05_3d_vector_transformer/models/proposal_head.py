import torch
import torch.nn as nn
import torch.nn.functional as F

class ProposalHead3D(nn.Module):
    """
    Dynamic 2D/3D Proposal Head:
    Predicts initial coarse 3D landmark anchor points p_anchor^{(0)} in [-1, 1]^3 for Layer 1.
    Predicts both 3D Center coordinates (X_c, Y_c, Z_c) AND 3D Orientation Direction Vectors (d_x, d_y, d_z)
    for each instance query, allowing 3D anchors to start at any 3D spatial orientation.
    Outputs:
      anchors_3d: (B, N, K, 3) initial 3D vertex positions in [-1, 1]^3.
    """
    def __init__(self, embed_dim: int = 256, num_instances: int = 10, num_points: int = 20):
        super().__init__()
        self.num_instances = num_instances
        self.num_points = num_points

        # Conv proposal classifier / regressor over feature map
        self.proposal_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Regresses N instance 3D center positions
        self.center_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_instances * 3),
            nn.Tanh() # Enforces [-1, 1]^3 bounds
        )

        # Regresses N instance 3D orientation direction vectors (unit 3D direction vectors)
        self.direction_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_instances * 3),
            nn.Tanh()
        )

        # Normalized 1D step scalar along polyline curve from -0.15 to +0.15
        point_steps = torch.linspace(-0.15, 0.15, num_points)
        self.register_buffer("point_steps", point_steps)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        """
        feature_map: (B, embed_dim, H, W)
        Returns:
          anchors_3d: (B, N, K, 3) initial 3D anchor locations in [-1, 1]^3
        """
        B = feature_map.size(0)
        pooled = self.proposal_conv(feature_map).view(B, -1) # (B, embed_dim)
        
        centers = self.center_head(pooled).view(B, self.num_instances, 1, 3) # (B, N, 1, 3)
        directions = self.direction_head(pooled).view(B, self.num_instances, 1, 3) # (B, N, 1, 3)
        
        # Normalize direction vectors
        dir_norm = F.normalize(directions, p=2, dim=-1, eps=1e-6) # (B, N, 1, 3)
        
        # Compute K 3D points along the predicted 3D direction vector
        steps = self.point_steps.view(1, 1, self.num_points, 1) # (1, 1, K, 1)
        anchors_3d = centers + steps * dir_norm # (B, N, K, 3)
        anchors_3d = torch.clamp(anchors_3d, -1.0, 1.0)
        return anchors_3d
