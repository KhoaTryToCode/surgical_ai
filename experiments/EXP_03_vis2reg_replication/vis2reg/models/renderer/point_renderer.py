from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

try:
    from pytorch3d.renderer import (
        AlphaCompositor,
        PointsRasterizationSettings,
        PointsRasterizer,
        PointsRenderer,
        PerspectiveCameras,
    )
    from pytorch3d.structures import Pointclouds
    HAS_PYTORCH3D = True
except ImportError:
    HAS_PYTORCH3D = False


import torch.nn.functional as F


class PointsSilhouetteRenderer(nn.Module):
    """Differentiable renderer that projects point clouds into binary/soft masks."""

    def __init__(self, image_size=(512, 512), radius=0.01, points_per_pixel=16):
        super().__init__()
        self.has_pytorch3d = HAS_PYTORCH3D
        self.image_size = image_size
        self.radius = radius
        self.points_per_pixel = points_per_pixel
        if HAS_PYTORCH3D:
            raster_settings = PointsRasterizationSettings(
                image_size=image_size,
                radius=radius,
                points_per_pixel=points_per_pixel,
            )
            self.rasterizer = PointsRasterizer(raster_settings=raster_settings)
            self.renderer = PointsRenderer(
                rasterizer=self.rasterizer,
                compositor=AlphaCompositor(),
            )
        else:
            self.rasterizer = None
            self.renderer = None

    def forward(
        self,
        points: torch.Tensor,
        intrinsics: torch.Tensor,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        mask, _ = self.render_with_depth(points, intrinsics, image_size=image_size, return_depth=True)
        return mask

    def render_with_depth(
        self,
        points: torch.Tensor,
        intrinsics: torch.Tensor,
        image_size: Optional[Tuple[int, int]] = None,
        return_depth: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        device = points.device
        B = points.shape[0]

        if self.has_pytorch3d and self.rasterizer is not None:
            focal = torch.stack([intrinsics[:, 0, 0], intrinsics[:, 1, 1]], dim=-1)
            principal = torch.stack([intrinsics[:, 0, 2], intrinsics[:, 1, 2]], dim=-1)

            raster_size = self.rasterizer.raster_settings.image_size
            if image_size is None:
                if isinstance(raster_size, int):
                    image_size = (raster_size, raster_size)
                else:
                    image_size = raster_size
            elif image_size != raster_size:
                self.rasterizer = PointsRasterizer(
                    raster_settings=PointsRasterizationSettings(
                        image_size=image_size,
                        radius=self.rasterizer.raster_settings.radius,
                        points_per_pixel=self.rasterizer.raster_settings.points_per_pixel,
                    )
                )
            image_tensor = torch.tensor(image_size, device=device, dtype=torch.float32)
            image_tensor = image_tensor.view(1, 2).repeat(B, 1)

            cameras = PerspectiveCameras(
                focal_length=focal,
                principal_point=principal,
                in_ndc=False,
                image_size=image_tensor,
                device=device,
            )

            feats = torch.ones(B, points.shape[1], 1, device=device)
            pointclouds = Pointclouds(points=points, features=feats)
            fragments = self.rasterizer(point_clouds=pointclouds, cameras=cameras)
            idx = fragments.idx[..., 0]
            mask = (idx >= 0).float()
            depth = fragments.zbuf[..., 0]
            depth = torch.where(mask > 0, depth, torch.zeros_like(depth))
            if return_depth:
                return mask, depth
            return mask, None
        else:
            # Pure PyTorch differentiable soft splatting fallback
            if image_size is None:
                image_size = self.image_size if isinstance(self.image_size, tuple) else (self.image_size, self.image_size)
            H, W = image_size
            z = points[..., 2].clamp(min=1e-2)
            u = (intrinsics[:, 0, 0:1] * (points[..., 0] / z) + intrinsics[:, 0, 2:3])
            v = (intrinsics[:, 1, 1:2] * (points[..., 1] / z) + intrinsics[:, 1, 2:3])
            u_idx = u.round().long().clamp(0, W - 1)
            v_idx = v.round().long().clamp(0, H - 1)
            valid = (points[..., 2] > 0)
            masks = torch.zeros(B, 1, H, W, device=device)
            depths = torch.zeros(B, 1, H, W, device=device) if return_depth else None
            for b in range(B):
                val_b = valid[b]
                if val_b.any():
                    v_b = v_idx[b, val_b]
                    u_b = u_idx[b, val_b]
                    masks[b, 0, v_b, u_b] = 1.0
                    if return_depth:
                        depths[b, 0, v_b, u_b] = z[b, val_b]
            radius_px = max(1, int(self.radius * max(H, W)))
            soft_mask = F.max_pool2d(masks, kernel_size=radius_px * 2 + 1, stride=1, padding=radius_px)
            soft_mask = F.avg_pool2d(soft_mask, kernel_size=3, stride=1, padding=1).squeeze(1)
            soft_depth = depths.squeeze(1) if return_depth else None
            return soft_mask, soft_depth
