#!/usr/bin/env python3
"""
EXPERIMENT_4: Landmark Master Tokens + Patch Bézier Decoder
Integrates 3 categorical Landmark Master Tokens (Ridge, Silhouette, Falciform) with 64 Spatial Patch Queries
into a unified 67-token Transformer Decoder sequence.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def build_2d_sinusoidal_pe(grid_size, embed_dim):
    """
    Builds fixed 2D sinusoidal positional encodings for an (grid_size x grid_size) grid.
    Returns tensor of shape (1, grid_size*grid_size, embed_dim).
    """
    pe = torch.zeros(grid_size * grid_size, embed_dim)
    d_model = embed_dim // 2
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    
    for r in range(grid_size):
        for c in range(grid_size):
            pos_idx = r * grid_size + c
            pe[pos_idx, 0:d_model:2]      = torch.sin(r * div_term)
            pe[pos_idx, 1:d_model:2]      = torch.cos(r * div_term)
            pe[pos_idx, d_model::2]       = torch.sin(c * div_term)
            pe[pos_idx, d_model + 1::2]   = torch.cos(c * div_term)
            
    return pe.unsqueeze(0)

class TransformerDecoderLayer(nn.Module):
    """
    Standard pre-norm Transformer Decoder Layer: Self-Attention -> Cross-Attention -> FFN.
    """
    def __init__(self, embed_dim=256, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, embed_dim)
        )
        
    def forward(self, queries, context):
        # 1. Multi-head self-attention among all tokens
        q_norm = self.norm1(queries)
        sa_out, _ = self.self_attn(q_norm, q_norm, q_norm)
        queries = queries + sa_out
        
        # 2. Multi-head cross-attention into image feature maps
        q_norm2 = self.norm2(queries)
        ca_out, _ = self.cross_attn(q_norm2, context, context)
        queries = queries + ca_out
        
        # 3. Feed-forward network
        q_norm3 = self.norm3(queries)
        ffn_out = self.ffn(q_norm3)
        queries = queries + ffn_out
        
        return queries

class LandmarkBezierPatchDecoder(nn.Module):
    """
    Decoder processing 67 tokens:
      - 3 Landmark Master Tokens: T_ridge (0), T_sil (1), T_falc (2)
      - 64 Patch Queries: Q_0 .. Q_63
    """
    def __init__(self, embed_dim=256, grid_size=8, num_classes=4, num_decoder_layers=6,
                 num_heads=8, num_landmarks=3, single_scale=True):
        super().__init__()
        self.grid_size = grid_size
        self.num_patches = grid_size * grid_size
        self.num_landmarks = num_landmarks
        self.single_scale = single_scale
        
        # 3 Category-level Landmark Master Tokens (Ridge, Silhouette, Falciform)
        # Coordinate-free learnable embeddings representing the 3 anatomical classes
        self.landmark_tokens = nn.Parameter(torch.randn(1, num_landmarks, embed_dim) * 0.02)
        
        # 2D sinusoidal positional encodings for the 64 spatial patch queries
        self.register_buffer('pos_enc', build_2d_sinusoidal_pe(grid_size, embed_dim))
        
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(embed_dim=embed_dim, num_heads=num_heads)
            for _ in range(num_decoder_layers)
        ])
        
        # Patch query projection
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        
        # Patch prediction heads (Tokens 3..66)
        self.class_head = nn.Linear(embed_dim, num_classes)
        self.bezier_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 8),
            nn.Sigmoid()
        )
        
        # Landmark macroscopic head (Tokens 0..2)
        # Outputs: [centroid_x, centroid_y, presence_logit] for each of the 3 landmarks
        self.landmark_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, 3)
        )
        
    def forward(self, multi_scale_features):
        """
        Args:
            multi_scale_features: list of 3 feature maps [stride-32, stride-16, stride-8]
        Returns:
            pred_class: (B, 64, num_classes)
            pred_bezier: (B, 64, 4, 2) normalized to [0, 1]
            pred_landmark_presence: (B, 3) raw logits
            pred_landmark_centroid: (B, 3, 2) normalized coordinates in [0, 1]^2
        """
        B = multi_scale_features[0].size(0)
        f_medium = multi_scale_features[1] # stride-16 features (B, 256, 64, 64)
        
        # 1. Initialize 64 spatial patch queries via adaptive average pooling + 2D sinusoidal PE
        f_pooled = F.adaptive_avg_pool2d(f_medium, (self.grid_size, self.grid_size))
        patch_queries = f_pooled.flatten(2).transpose(1, 2) # (B, 64, 256)
        patch_queries = self.query_proj(patch_queries) + self.pos_enc.to(patch_queries.device)
        
        # 2. Expand 3 Landmark Master Tokens
        landmark_tokens = self.landmark_tokens.expand(B, -1, -1) # (B, 3, 256)
        
        # 3. Concatenate to form unified 67-token sequence: [T_ridge, T_sil, T_falc, Q_0..Q_63]
        tokens = torch.cat([landmark_tokens, patch_queries], dim=1) # (B, 67, 256)
        
        # 4. Process across 6 Transformer Decoder Layers
        for i, layer in enumerate(self.decoder_layers):
            # Scale engine: lock to stride-16 if single_scale is True (prevents scale hopping)
            f_cur = multi_scale_features[1] if self.single_scale else multi_scale_features[i % 3]
            context = f_cur.flatten(2).transpose(1, 2)
            tokens = layer(tokens, context)
            
        # 5. Sliced predictions
        landmark_out = tokens[:, :self.num_landmarks, :] # (B, 3, 256)
        patch_out    = tokens[:, self.num_landmarks:, :] # (B, 64, 256)
        
        # Patch outputs
        pred_class  = self.class_head(patch_out)
        bezier_flat = self.bezier_head(patch_out)
        pred_bezier = bezier_flat.view(B, self.num_patches, 4, 2)
        
        # Landmark macroscopic outputs
        lm_raw = self.landmark_head(landmark_out) # (B, 3, 3)
        pred_landmark_centroid = torch.sigmoid(lm_raw[..., :2]) # (B, 3, 2) in [0, 1]^2
        pred_landmark_presence = lm_raw[..., 2]                 # (B, 3) raw presence logit
        
        return pred_class, pred_bezier, pred_landmark_presence, pred_landmark_centroid
