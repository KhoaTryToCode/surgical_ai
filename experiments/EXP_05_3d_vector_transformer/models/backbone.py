import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerConfig
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

class Sinusoidal3DPositionalEncoding(nn.Module):
    """
    Computes 3D Continuous Sinusoidal Positional Encoding PE_3D for (X, Y, Z) coordinates.
    Maps 3D spatial points (B, 3, H, W) to (B, pe_dim, H, W).
    """
    def __init__(self, num_pos_feats: int = 32):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        # Frequency bands
        freq_bands = 2.0 ** torch.linspace(0, num_pos_feats - 1, num_pos_feats)
        self.register_buffer("freq_bands", freq_bands)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        # xyz: (B, 3, H, W) in range [-1, 1]^3
        B, _, H, W = xyz.shape
        # Flatten spatial dims for frequency projection
        xyz_flat = xyz.unsqueeze(-1) # (B, 3, H, W, 1)
        freqs = self.freq_bands.view(1, 1, 1, 1, -1) # (1, 1, 1, 1, num_pos_feats)
        
        scaled = xyz_flat * math.pi * freqs # (B, 3, H, W, num_pos_feats)
        sin_feat = torch.sin(scaled)
        cos_feat = torch.cos(scaled)
        
        # Concat sin and cos across frequencies and 3 spatial channels
        pe = torch.cat([sin_feat, cos_feat], dim=-1) # (B, 3, H, W, 2 * num_pos_feats)
        pe = pe.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous() # (B, 6 * num_pos_feats, H, W)
        return pe

class SurgicalBackbone3DLifting(nn.Module):
    """
    Swin Transformer / ResNet 2D Backbone + Canonical 3D Pinhole Unprojection + PE_3D Injection.
    Outputs multi-scale fused feature pyramids {F_stride4, F_stride8, F_stride16, F_stride32}.
    """
    def __init__(self, embed_dim: int = 256, fov_degrees: float = 60.0, mask2former_model_name: str = "facebook/mask2former-swin-tiny-ade-semantic"):
        super().__init__()
        self.embed_dim = embed_dim
        self.fov_degrees = fov_degrees
        self.f_canon = 1.0 / math.tan(math.radians(fov_degrees / 2.0)) # Canonical focal length (~1.732)

        # 1. Load Mask2Former Pixel Decoder / Backbone
        if HAS_TRANSFORMERS:
            try:
                m2f = Mask2FormerForUniversalSegmentation.from_pretrained(mask2former_model_name)
                self.pixel_decoder = m2f.model.pixel_decoder
            except Exception as e:
                print(f"⚠️ Warning: Could not load '{mask2former_model_name}' directly ({e}). Initializing standard ConvNeXt/FPN fallback.")
                self.pixel_decoder = None
        else:
            self.pixel_decoder = None

        # 2. 3D Positional Encoder
        self.pe_3d_module = Sinusoidal3DPositionalEncoding(num_pos_feats=32)
        pe_dim = 6 * 32 # 192 channels

        # 3. 1x1 Convolutions for 2D + 3D Feature Fusion across FPN levels
        # Standard FPN channels C_2d = 256
        self.fuse_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256 + pe_dim, embed_dim, kernel_size=1),
                nn.GroupNorm(32, embed_dim),
                nn.GELU()
            ) for _ in range(4)
        ])

        # Fallback conv backbone if HuggingFace pretrained download fails
        self.fallback_backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )

    def _unproject_depth_to_3d(self, depth: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        """
        Unprojects (u, v, d) into canonical 3D camera coordinates (X_canon, Y_canon, Z_canon) in [-1, 1]^3.
        depth: (B, 1, 1024, 1024)
        Returns: xyz: (B, 3, target_h, target_w)
        """
        B = depth.size(0)
        depth_ds = F.interpolate(depth, size=(target_h, target_w), mode='bilinear', align_corners=False)
        
        # Grid of normalized image coordinates [-1, 1]
        v_coords = torch.linspace(-1.0, 1.0, target_h, device=depth.device)
        u_coords = torch.linspace(-1.0, 1.0, target_w, device=depth.device)
        grid_v, grid_u = torch.meshgrid(v_coords, u_coords, indexing='ij') # (target_h, target_w)
        
        u_norm = grid_u.view(1, 1, target_h, target_w).expand(B, 1, target_h, target_w)
        v_norm = grid_v.view(1, 1, target_h, target_w).expand(B, 1, target_h, target_w)

        # Scale depth to [0.1, 1.0] to prevent zero division
        z_canon = 0.1 + depth_ds * 0.9
        x_canon = (u_norm * z_canon) / self.f_canon
        y_canon = (v_norm * z_canon) / self.f_canon

        # Normalize X, Y, Z into [-1, 1] bounds
        x_norm = torch.clamp(x_canon, -1.0, 1.0)
        y_norm = torch.clamp(y_canon, -1.0, 1.0)
        z_norm = z_canon * 2.0 - 1.0

        xyz = torch.cat([x_norm, y_norm, z_norm], dim=1) # (B, 3, target_h, target_w)
        return xyz

    def forward(self, images: torch.Tensor, depth: torch.Tensor):
        """
        images: (B, 3, 1024, 1024)
        depth: (B, 1, 1024, 1024)
        Returns:
          fused_features: List of 4 feature maps at strides {4, 8, 16, 32}
        """
        B = images.size(0)
        
        # Extract 2D visual features
        if self.pixel_decoder is not None:
            try:
                pixel_decoder_outputs = self.pixel_decoder(images)
                # Multi-scale features from pixel decoder
                features_2d = pixel_decoder_outputs.feature_maps # List of 4 feature maps
            except Exception:
                # Fallback to local conv backbone if evaluation mode signature differs
                feat = self.fallback_backbone(images)
                features_2d = [
                    F.interpolate(feat, scale_factor=2.0, mode='bilinear', align_corners=False),
                    feat,
                    F.interpolate(feat, scale_factor=0.5, mode='bilinear', align_corners=False),
                    F.interpolate(feat, scale_factor=0.25, mode='bilinear', align_corners=False),
                ]
        else:
            feat = self.fallback_backbone(images)
            features_2d = [
                F.interpolate(feat, scale_factor=2.0, mode='bilinear', align_corners=False),
                feat,
                F.interpolate(feat, scale_factor=0.5, mode='bilinear', align_corners=False),
                F.interpolate(feat, scale_factor=0.25, mode='bilinear', align_corners=False),
            ]

        # Inject 3D Positional Encodings into each feature level
        fused_features = []
        for l, f_2d in enumerate(features_2d):
            _, C, H, W = f_2d.shape
            # Unproject depth at feature spatial resolution (H, W)
            xyz = self._unproject_depth_to_3d(depth, H, W)
            pe_3d = self.pe_3d_module(xyz) # (B, 192, H, W)
            
            # Concatenate 2D visual feature and 3D positional encoding
            cat_feat = torch.cat([f_2d, pe_3d], dim=1) # (B, C + 192, H, W)
            fused = self.fuse_proj[l](cat_feat) # (B, embed_dim, H, W)
            fused_features.append(fused)

        return fused_features
