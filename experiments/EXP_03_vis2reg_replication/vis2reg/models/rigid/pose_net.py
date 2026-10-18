from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from vis2reg.utils.geometry import se3_from_params


class RigidScaleHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, iters: int = 3, predict_scale: bool = True):
        super().__init__()
        self.predict_scale = predict_scale
        self.iters = iters
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 7 if predict_scale else 6)
        )

    def forward(self, fused_global: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        axis_angle = torch.zeros(fused_global.size(0), 3, device=fused_global.device)
        translation = torch.zeros_like(axis_angle)
        scale_param = torch.zeros(fused_global.size(0), 1, device=fused_global.device) if self.predict_scale else None
        for _ in range(self.iters):
            delta = self.mlp(fused_global)
            axis_angle = axis_angle + delta[:, :3]
            translation = translation + delta[:, 3:6]
            if self.predict_scale:
                scale_param = scale_param + delta[:, 6:7]
        return se3_from_params(axis_angle, translation, scale_param if self.predict_scale else None)
