"""
Lightweight Spatiotemporal Query Cross-Attention Block for EXPERIMENT_15.
Allows queries at frame t to attend to query states across the temporal clip (t-2, t-1, t).
Uses learnable residual gating (gamma initialized small) to ensure training stability and
smooth convergence from static Mask2Former weights.
"""
import torch
import torch.nn as nn

class SpatiotemporalQueryBlock(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, dropout=0.1, gamma_init=0.05):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Cross-Attention over temporal query context
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # Temporal Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Learnable gating parameter initialized small (ensures smooth departure from static baseline)
        self.gamma = nn.Parameter(torch.tensor([gamma_init], dtype=torch.float32))

    def forward(self, queries_clip):
        """
        Args:
            queries_clip: (B, T, N, C) - Queries for each of the T frames.
        Returns:
            steered_queries: (B, T, N, C) - Temporally enriched queries.
        """
        B, T, N, C = queries_clip.shape
        
        # Flatten all temporal queries to form Key and Value context: (B, T*N, C)
        kv_context = queries_clip.view(B, T * N, C)
        
        out_queries = []
        for t in range(T):
            q_curr = queries_clip[:, t, :, :] # (B, N, C)
            
            # Cross-attend to all temporal queries in the clip
            attn_out, _ = self.cross_attn(query=q_curr, key=kv_context, value=kv_context)
            q_res = self.norm1(q_curr + self.gamma * attn_out)
            
            # FFN with residual gating
            ffn_out = self.ffn(q_res)
            q_steered = self.norm2(q_res + self.gamma * ffn_out)
            out_queries.append(q_steered)
            
        return torch.stack(out_queries, dim=1) # (B, T, N, C)
