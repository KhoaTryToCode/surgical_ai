import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from experiments.EXPERIMENT_7.utils.atlas_extractor import CANONICAL_UV


def build_2d_sinusoidal_pe(coords_uv, embed_dim=256):
    """
    Builds fixed 2D sinusoidal positional encodings for the canonical (u, v) points.
    coords_uv: (K, 2) in [0, 1]^2
    Returns: Tensor (1, K, embed_dim)
    """
    K = len(coords_uv)
    dim_per_coord = embed_dim // 2
    pe = torch.zeros(K, embed_dim, dtype=torch.float32)
    
    # 1D sin/cos frequencies
    div_term = torch.exp(torch.arange(0, dim_per_coord, 2).float() * (-np.log(10000.0) / dim_per_coord))
    
    for k in range(K):
        u_val = float(coords_uv[k, 0])
        v_val = float(coords_uv[k, 1])
        
        # u encoding
        pe[k, 0:dim_per_coord:2] = torch.sin(u_val * 100.0 * div_term)
        pe[k, 1:dim_per_coord:2] = torch.cos(u_val * 100.0 * div_term)
        
        # v encoding
        pe[k, dim_per_coord::2] = torch.sin(v_val * 100.0 * div_term)
        pe[k, dim_per_coord+1::2] = torch.cos(v_val * 100.0 * div_term)
        
    return pe.unsqueeze(0)  # (1, K, embed_dim)


class CanonicalAtlasHead(nn.Module):
    """
    11-Query Canonical Atlas Head for EXPERIMENT_7.
    
    Anchored to 11 fixed canonical coordinates (u, v) on the 2-chart liver manifold.
    Cross-attends to stride-16 visual features to output:
      - (x_k, y_k): 2D screen coordinates in [0, 1]^2
      - v_k: Visibility logits in R
      - structural_features: Guidance embeddings (B, 11, embed_dim)
    """
    def __init__(self, in_dim=256, embed_dim=256, num_queries=11, num_layers=2, num_heads=8):
        super().__init__()
        self.num_queries = num_queries
        self.embed_dim = embed_dim

        # Canonical query embeddings initialized with 2D sinusoidal PE of canonical (u, v)
        pe = build_2d_sinusoidal_pe(CANONICAL_UV, embed_dim=embed_dim)
        self.register_buffer("canonical_pe", pe)  # (1, 11, embed_dim)

        # Learnable query content offsets
        self.query_embed = nn.Parameter(torch.zeros(1, num_queries, embed_dim))
        nn.init.normal_(self.query_embed, std=0.02)

        # 2-layer Transformer cross-attention decoder
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
                norm_first=True
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # 1. Coordinate regression head (MLP -> Sigmoid -> normalized coords)
        self.coord_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 2),
            nn.Sigmoid()
        )

        # 2. Visibility head (Linear -> logit)
        self.vis_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1)
        )
        
        # Initialize coord head near canonical priors
        with torch.no_grad():
            self.coord_head[-2].weight.zero_()
            # Bias maps to inverse sigmoid of canonical coords
            inv_sig = np.log(np.clip(CANONICAL_UV, 0.05, 0.95) / (1.0 - np.clip(CANONICAL_UV, 0.05, 0.95)))
            inv_sig[:, 1] = -inv_sig[:, 1]  # flip v to y
            self.coord_head[-2].bias.zero_()

    def forward(self, feature_map):
        """
        feature_map: (B, 256, H, W) typically stride-16 features (e.g. 64x64)
        Returns:
            pred_coords: (B, 11, 2) in [0, 1]^2
            pred_vis_logits: (B, 11) in R
            structural_features: (B, 11, embed_dim)
        """
        B, C, H, W = feature_map.shape
        memory = feature_map.flatten(2).transpose(1, 2)  # (B, H*W, C)

        # Queries = learnable content + fixed canonical positional encoding
        queries = self.query_embed.expand(B, -1, -1) + self.canonical_pe.to(feature_map.device)

        # Cross-attention decoding over visual features
        for layer in self.layers:
            queries = layer(tgt=queries, memory=memory)

        queries = self.norm(queries)  # (B, 11, embed_dim)

        # Predictions
        pred_coords = self.coord_head(queries)              # (B, 11, 2)
        pred_vis_logits = self.vis_head(queries).squeeze(-1) # (B, 11)

        return pred_coords, pred_vis_logits, queries
