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


class PointsSilhouetteRenderer(nn.Module):
    """Differentiable renderer that projects point clouds into binary masks."""

    def __init__(self, image_size=(512, 512), radius=0.01, points_per_pixel=16):
        super().__init__()
        if not HAS_PYTORCH3D:
            raise ImportError(
                "pytorch3d is required for PointsSilhouetteRenderer. "
                "Install pytorch3d or set renderer: None in model config."
            )
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
