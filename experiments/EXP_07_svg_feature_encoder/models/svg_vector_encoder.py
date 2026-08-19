import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .swin_standard_encoder import StandardSwinEncoder

class SVGVectorAwareFeatureEncoder(nn.Module):
    """
    SVG / Vector-Aware Feature Map Encoder for EXP_07.
    Transforms dense pixel features from Swin Backbone into continuous geometric vector primitives:
      1. Landmark Saliency & Skeleton Field: S(x, y) in [0, 1]
      2. 2D Tangent Flow Field: T(x, y) = (cos theta, sin theta)
      3. 2D Normal Gradient Field: N(x, y) = (-sin theta, cos theta)
      4. Local Curvature Field: kappa(x, y) (bending energy)
      5. Parametric Bézier Spline Primitives: B(t) = (1-t)^3 P0 + 3(1-t)^2 t C1 + 3(1-t) t^2 C2 + t^3 P3
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim

        # 1. Base Swin Feature Extractor
        self.swin_encoder = StandardSwinEncoder(config)

        # 2. Multi-Scale Geometric Fusion Neck
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(self.embed_dim, 128, kernel_size=1) for _ in range(4)
        ])
        
        self.smooth_conv = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU()
        )

        # 3. Geometric Vector Field Heads (Operating on High-Res 256x256 Feature Canvas)
        # Head A: Saliency & Skeleton Probability Field S(x, y)
        self.saliency_head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Head B: 2D Tangent Flow Field T(x, y) = (Tx, Ty)
        self.tangent_head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 2, kernel_size=1),
            nn.Tanh()
        )

        # Head C: Curvature Field kappa(x, y)
        self.curvature_head = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Head D: SVG Control Point Displacement Field
        self.control_disp_head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 4, kernel_size=1) # (dx1, dy1, dx2, dy2) for Bézier C1, C2
        )

    def forward(self, pixel_values: torch.Tensor) -> dict:
        """
        pixel_values: (B, 3, 1024, 1024)
        Returns: Comprehensive SVG / Vector Feature Representation.
        """
        B = pixel_values.shape[0]

        # 1. Extract base multi-scale Swin features
        swin_outputs = self.swin_encoder(pixel_values)
        pyramid = swin_outputs["fused_features"] # [Stride 4, Stride 8, Stride 16, Stride 32]

        # 2. Multi-Scale Top-Down Feature Aggregation
        p4 = self.lateral_convs[0](pyramid[0]) # (B, 128, 256, 256)
        p8 = F.interpolate(self.lateral_convs[1](pyramid[1]), size=(256, 256), mode='bilinear', align_corners=False)
        p16 = F.interpolate(self.lateral_convs[2](pyramid[2]), size=(256, 256), mode='bilinear', align_corners=False)
        p32 = F.interpolate(self.lateral_convs[3](pyramid[3]), size=(256, 256), mode='bilinear', align_corners=False)

        feat_geo = self.smooth_conv(p4 + p8 + p16 + p32) # (B, 128, 256, 256)

        # 3. Predict Geometric Vector Fields
        # Saliency Map: S(x, y) in [0, 1]
        saliency_map = self.saliency_head(feat_geo) # (B, 1, 256, 256)

        # Tangent Unit Vectors: T(x, y) = (Tx, Ty)
        tangent_raw = self.tangent_head(feat_geo) # (B, 2, 256, 256)
        tangent_norm = torch.sqrt(torch.sum(tangent_raw ** 2, dim=1, keepdim=True) + 1e-6)
        tangent_field = tangent_raw / tangent_norm # Normalized unit tangent vectors

        # Normal Unit Vectors: N(x, y) = (-Ty, Tx) (orthogonal to tangent)
        normal_field = torch.cat([-tangent_field[:, 1:2], tangent_field[:, 0:1]], dim=1)

        # Curvature Field: kappa(x, y)
        curvature_field = self.curvature_head(feat_geo) # (B, 1, 256, 256)

        # Control Point Displacements
        control_disp = self.control_disp_head(feat_geo) # (B, 4, 256, 256)

        # 4. Extract Parametric SVG Bézier Primitives from Vector Fields
        bezier_curves = self._extract_bezier_splines(
            saliency_map, tangent_field, control_disp,
            num_proposals=self.config.num_bezier_proposals,
            num_points=self.config.num_points_per_bezier
        )

        return {
            "swin_outputs": swin_outputs,
            "geometric_features": feat_geo,          # (B, 128, 256, 256)
            "saliency_field": saliency_map,           # (B, 1, 256, 256)
            "tangent_field": tangent_field,           # (B, 2, 256, 256)
            "normal_field": normal_field,             # (B, 2, 256, 256)
            "curvature_field": curvature_field,       # (B, 1, 256, 256)
            "bezier_curves": bezier_curves            # Analytical SVG Spline Primitives
        }

    def _extract_bezier_splines(self, saliency: torch.Tensor, tangent: torch.Tensor, control_disp: torch.Tensor,
                                num_proposals: int = 5, num_points: int = 30) -> dict:
        """
        Extracts continuous parametric cubic Bézier curves B(t) from the neural vector field.
        """
        B, _, H, W = saliency.shape
        device = saliency.device

        all_beziers = []

        for b in range(B):
            sal_np = saliency[b, 0].detach().cpu().numpy()
            
            # Find top landmark seeds by local maxima on saliency map
            flat_indices = torch.topk(saliency[b, 0].view(-1), k=min(num_proposals, H * W)).indices
            seeds_y = (flat_indices // W).float() / float(H) # in [0, 1]
            seeds_x = (flat_indices % W).float() / float(W)

            batch_beziers = []
            for i in range(len(seeds_x)):
                p0_x = seeds_x[i].item()
                p0_y = seeds_y[i].item()

                # Sample local tangent direction at seed
                grid_y = int(np.clip(p0_y * H, 0, H - 1))
                grid_x = int(np.clip(p0_x * W, 0, W - 1))
                tx = tangent[b, 0, grid_y, grid_x].item()
                ty = tangent[b, 1, grid_y, grid_x].item()

                # Sample control point offsets
                cd = control_disp[b, :, grid_y, grid_x].detach().cpu().numpy()
                scale = 0.15

                # Define SVG Control Points: P0 (Start), C1, C2, P3 (End)
                p0 = np.array([p0_x, p0_y])
                p3 = np.array([np.clip(p0_x + tx * scale * 2.0, 0.0, 1.0),
                               np.clip(p0_y + ty * scale * 2.0, 0.0, 1.0)])
                c1 = np.array([np.clip(p0_x + tx * scale + cd[0] * 0.05, 0.0, 1.0),
                               np.clip(p0_y + ty * scale + cd[1] * 0.05, 0.0, 1.0)])
                c2 = np.array([np.clip(p3[0] - tx * scale * 0.5 + cd[2] * 0.05, 0.0, 1.0),
                               np.clip(p3[1] - ty * scale * 0.5 + cd[3] * 0.05, 0.0, 1.0)])

                # Evaluate continuous Bézier curve B(t) for t in [0, 1]
                t_vals = np.linspace(0.0, 1.0, num_points).reshape(-1, 1)
                curve_pts = (
                    ((1.0 - t_vals) ** 3) * p0 +
                    3.0 * ((1.0 - t_vals) ** 2) * t_vals * c1 +
                    3.0 * (1.0 - t_vals) * (t_vals ** 2) * c2 +
                    (t_vals ** 3) * p3
                ) # (num_points, 2)

                batch_beziers.append({
                    "P0": p0,
                    "C1": c1,
                    "C2": c2,
                    "P3": p3,
                    "curve": curve_pts
                })

            all_beziers.append(batch_beziers)

        return all_beziers
