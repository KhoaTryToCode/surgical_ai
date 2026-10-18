from __future__ import annotations

from typing import Tuple, Optional

import torch
from torch import nn

from .pose_net import RigidScaleHead


class RigidPoseNet(nn.Module):
    """
    Thin wrapper around the existing RigidScaleHead so that the rigid module
    can be referenced explicitly when running in decoupled mode.
    """

    def __init__(self, input_dim: int, pose_cfg):
        super().__init__()
        self.pose_head = RigidScaleHead(
            input_dim=input_dim,
            hidden_dim=pose_cfg.get('hidden_dim', 256),
            iters=pose_cfg.get('iters', 3),
            predict_scale=pose_cfg.get('predict_scale', True),
        )

    def forward(self, fused_global: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        return self.pose_head(fused_global)
