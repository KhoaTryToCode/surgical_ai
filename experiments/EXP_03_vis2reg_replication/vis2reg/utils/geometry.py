from __future__ import annotations

import torch


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle (B,3) to rotation matrices (B,3,3)."""
    angle = torch.norm(axis_angle + 1e-9, dim=-1, keepdim=True)
    axis = axis_angle / angle.clamp(min=1e-9)
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one_minus_cos = 1 - cos

    x, y, z = axis.unbind(-1)

    rot = torch.stack([
        cos + x * x * one_minus_cos,
        x * y * one_minus_cos - z * sin,
        x * z * one_minus_cos + y * sin,
        y * x * one_minus_cos + z * sin,
        cos + y * y * one_minus_cos,
        y * z * one_minus_cos - x * sin,
        z * x * one_minus_cos - y * sin,
        z * y * one_minus_cos + x * sin,
        cos + z * z * one_minus_cos,
    ], dim=-1)
    rot = rot.view(*axis_angle.shape[:-1], 3, 3)
    return rot


def build_se3_transform(R: torch.Tensor, t: torch.Tensor, scale: torch.Tensor | None = None) -> torch.Tensor:
    """Return homogeneous transform (B,4,4)."""
    B = R.shape[0]
    T = torch.eye(4, device=R.device, dtype=R.dtype).unsqueeze(0).repeat(B, 1, 1)
    T[:, :3, :3] = R * (scale.view(B, 1, 1) if scale is not None else 1.0)
    T[:, :3, 3] = t
    return T


def apply_transform(points: torch.Tensor, R: torch.Tensor, t: torch.Tensor, scale: torch.Tensor | None = None) -> torch.Tensor:
    if scale is None:
        scale = 1.0
    return (R @ points.transpose(1, 2)).transpose(1, 2) * scale + t.unsqueeze(1)


def se3_from_params(axis_angle: torch.Tensor, translation: torch.Tensor, scale_param: torch.Tensor | None = None):
    R = axis_angle_to_matrix(axis_angle)
    scale = None
    if scale_param is not None:
        clipped = torch.clamp(scale_param, min=-6.0, max=6.0)
        scale = torch.exp(clipped)
    return R, translation, scale
