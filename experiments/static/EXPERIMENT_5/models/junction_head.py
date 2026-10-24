"""
Junction Anchor Head for EXPERIMENT_5.
Extracts 4 biological landmark vectors and predicts (x, y) coordinates + visibility flags:
  0: J_top (Falc-Sil Root)
  1: J_bottom (Umbilical Notch)
  2: J_lat_right (Right Lateral Tip)
  3: J_lat_left (Left Lateral Tip)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class JunctionDecoderLayer(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, context):
        """
        queries: (B, 4, C)
        context: (B, S, C)
        """
        # 1. Self-attention among the 4 junction nodes (anatomical graph reasoning)
        q_norm = self.norm1(queries)
        q_self, _ = self.self_attn(q_norm, q_norm, q_norm)
        queries = queries + self.dropout(q_self)
        
        # 2. Cross-attention from context feature map
        q_norm = self.norm2(queries)
        q_cross, _ = self.cross_attn(query=q_norm, key=context, value=context)
        queries = queries + self.dropout(q_cross)
        
        # 3. Feedforward
        queries = queries + self.ffn(self.norm3(queries))
        return queries

class JunctionAnchorHead(nn.Module):
    def __init__(self, embed_dim=256, num_layers=2, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_junctions = 4
        
        # 4 Learnable Anatomical Query Embeddings
        self.junction_queries = nn.Parameter(torch.randn(1, 4, embed_dim) * 0.02)
        
        # 2-Layer Transformer Decoder
        self.layers = nn.ModuleList([
            JunctionDecoderLayer(embed_dim=embed_dim, num_heads=num_heads)
            for _ in range(num_layers)
        ])
        
        # Coordinate Regression Head: (B, 4, 2) in [0, 1]^2
        self.coord_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, 2)
        )
        
        # Visibility Prediction Head: (B, 4) binary logit
        self.vis_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        
        # Anatomical Prior Biases (logits for initial coordinate sigmoid)
        # [0]: Top (0.50, 0.20), [1]: Bot (0.50, 0.70), [2]: Lat_R (0.85, 0.40), [3]: Lat_L (0.15, 0.60)
        priors = torch.tensor([
            [0.50, 0.20],
            [0.50, 0.70],
            [0.85, 0.40],
            [0.15, 0.60]
        ], dtype=torch.float32)
        # logit(p) = log(p / (1 - p))
        prior_logits = torch.log(priors / (1.0 - priors + 1e-6))
        self.register_buffer('prior_logits', prior_logits)

    def forward(self, feature_map):
        """
        Args:
            feature_map (Tensor): (B, 256, H, W) from Pixel Decoder (e.g. stride 16).
        Returns:
            coords (Tensor): (B, 4, 2) normalized to [0, 1]^2
            vis_logits (Tensor): (B, 4) unnormalized visibility logits
            features (Tensor): (B, 4, 256) junction feature tokens
        """
        B, C, H, W = feature_map.shape
        context = feature_map.flatten(2).transpose(1, 2) # (B, H*W, C)
        
        queries = self.junction_queries.expand(B, -1, -1) # (B, 4, C)
        
        for layer in self.layers:
            queries = layer(queries, context)
            
        # Predict coordinates with anatomical prior bias
        coord_deltas = self.coord_head(queries) # (B, 4, 2)
        coords = torch.sigmoid(coord_deltas + self.prior_logits.unsqueeze(0))
        
        # Predict visibility logits
        vis_logits = self.vis_head(queries).squeeze(-1) # (B, 4)
        
        return coords, vis_logits, queries
