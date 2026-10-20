from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from vis2reg.utils.geometry import apply_transform


class PositionalEncoding(nn.Module):
    def __init__(self, num_frequencies: int = 8, coord_scale: float = 100.0):
        super().__init__()
        self.coord_scale = coord_scale
        self.register_buffer('freq_bands', 2 ** torch.arange(num_frequencies).float(), persistent=False)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # Scale coordinates to nominal range so frequencies are well-conditioned
        scaled_coords = coords / self.coord_scale
        encodings = [scaled_coords]
        for freq in self.freq_bands:
            encodings.append(torch.sin(scaled_coords * freq))
            encodings.append(torch.cos(scaled_coords * freq))
        return torch.cat(encodings, dim=-1)


class SineLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, omega_0: float = 1.0, is_first: bool = False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.omega_0 = omega_0
        self.is_first = is_first
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / self.linear.in_features
                self.linear.weight.uniform_(-bound, bound)
            else:
                bound = (6.0 / self.linear.in_features) ** 0.5 / self.omega_0
                self.linear.weight.uniform_(-bound, bound)
            nn.init.zeros_(self.linear.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(input))


class ImplicitDeformationField(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 5,
        pe_frequencies: int = 8,
        skip_connections: Optional[list] = None,
        max_displacement: float = 30.0,
    ):
        super().__init__()
        self.pe = PositionalEncoding(pe_frequencies)
        self.max_displacement = max_displacement
        pe_dim = (1 + 2 * pe_frequencies) * 3
        layers = []
        in_dim = pe_dim + feature_dim
        self.skip_connections = set(skip_connections or [])
        self.skip_projections = nn.ModuleDict({
            str(idx): nn.Linear(in_dim, hidden_dim)
            for idx in self.skip_connections
        })
        for layer_idx in range(num_layers):
            if layer_idx == 0:
                layers.append(SineLayer(in_dim, hidden_dim, omega_0=1.0, is_first=True))
            else:
                layers.append(SineLayer(hidden_dim, hidden_dim, omega_0=1.0, is_first=False))
        self.mlp = nn.ModuleList(layers)
        self.output_layer = nn.Linear(hidden_dim, 3)
        self.feature_head = nn.Linear(hidden_dim, hidden_dim)

        # Zero-initialize output layer so initial deformation at Epoch 21 is strictly identity
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, points: torch.Tensor, fused_feats: torch.Tensor, pose: Optional[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]] = None):
        base = self.pe(points)
        x = torch.cat([base, fused_feats], dim=-1)
        residual = x
        for idx, layer in enumerate(self.mlp):
            x = layer(x)
            if idx in self.skip_connections:
                proj = self.skip_projections[str(idx)](residual)
                x = x + proj
        # Anatomically bounded displacement (max +/- 30mm)
        delta = torch.tanh(self.output_layer(x)) * self.max_displacement
        latent = self.feature_head(x)
        warped_points = points + delta
        regularization = (delta - delta.mean(dim=1, keepdim=True)).pow(2).mean()
        if pose is not None:
            R, t, scale = pose
            warped_points = apply_transform(warped_points, R, t, scale)
        return {
            'warped_points': warped_points,
            'delta': delta,
            'latent_features': latent,
            'regularization': regularization,
        }
