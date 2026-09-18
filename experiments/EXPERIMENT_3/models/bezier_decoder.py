import torch
import torch.nn as nn
import torch.nn.functional as F

def build_2d_sinusoidal_pe(grid_size, embed_dim):
    """
    Builds 2D sinusoidal positional encodings.
    
    Args:
        grid_size: int
        embed_dim: int
        
    Returns:
        Tensor shape (1, grid_size*grid_size, embed_dim)
    """
    pe = torch.zeros(grid_size * grid_size, embed_dim)
    d_model = embed_dim // 2
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
    
    for r in range(grid_size):
        for c in range(grid_size):
            pos_idx = r * grid_size + c
            
            pe[pos_idx, 0:d_model:2] = torch.sin(r * div_term)
            pe[pos_idx, 1:d_model:2] = torch.cos(r * div_term)
            
            pe[pos_idx, d_model::2] = torch.sin(c * div_term)
            pe[pos_idx, d_model+1::2] = torch.cos(c * div_term)
            
    return pe.unsqueeze(0)

class TransformerDecoderLayer(nn.Module):
    """
    A single layer of the transformer decoder with full cross attention.
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
        
    def forward(self, queries, context, attn_mask=None):
        q_norm = self.norm1(queries)
        sa_out, _ = self.self_attn(q_norm, q_norm, q_norm)
        queries = queries + sa_out
        
        q_norm2 = self.norm2(queries)
        ca_out, _ = self.cross_attn(q_norm2, context, context)
        queries = queries + ca_out
        
        q_norm3 = self.norm3(queries)
        ffn_out = self.ffn(q_norm3)
        queries = queries + ffn_out
        
        return queries

class BezierPatchDecoder(nn.Module):
    """
    Bezier patch decoder module.
    """
    def __init__(self, embed_dim=256, grid_size=8, num_classes=4, num_decoder_layers=6, num_heads=8):
        super().__init__()
        self.grid_size = grid_size
        self.num_patches = grid_size * grid_size
        
        self.register_buffer('pos_enc', build_2d_sinusoidal_pe(grid_size, embed_dim))
        
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(embed_dim=embed_dim, num_heads=num_heads)
            for _ in range(num_decoder_layers)
        ])
        
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.class_head = nn.Linear(embed_dim, num_classes)
        self.bezier_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 8),
            nn.Sigmoid()
        )
        
    def forward(self, multi_scale_features):
        B = multi_scale_features[0].size(0)
        f_medium = multi_scale_features[1]
        
        f_pooled = F.adaptive_avg_pool2d(f_medium, (self.grid_size, self.grid_size))
        queries = f_pooled.flatten(2).transpose(1, 2)
        queries = self.query_proj(queries)
        queries = queries + self.pos_enc.to(queries.device)
        
        for i, layer in enumerate(self.decoder_layers):
            f_cur = multi_scale_features[i % 3]
            context = f_cur.flatten(2).transpose(1, 2)
            queries = layer(queries, context)
            
        pred_class = self.class_head(queries)
        bezier_flat = self.bezier_head(queries)
        pred_bezier = bezier_flat.view(B, self.num_patches, 4, 2)
        
        return pred_class, pred_bezier
