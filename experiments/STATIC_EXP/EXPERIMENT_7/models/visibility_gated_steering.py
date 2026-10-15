import torch
import torch.nn as nn
import torch.nn.functional as F

class VisibilityGatedSteering(nn.Module):
    """
    Visibility-Gated Steering (VGS) Block for EXPERIMENT_7.
    
    Allows the 100 base Mask2Former queries to cross-attend to the 11 structural atlas tokens,
    while strictly silencing inactive/invisible queries using logarithmic additive bias
    and value gating.
    """
    def __init__(self, embed_dim=256, num_heads=8, init_alpha=0.2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_struct = nn.LayerNorm(embed_dim)
        self.norm_out = nn.LayerNorm(embed_dim)

        # Learnable steering gating coefficient (initialized gently to 0.2)
        self.alpha = nn.Parameter(torch.tensor(init_alpha, dtype=torch.float32))

    def forward(self, base_queries, structural_tokens, vis_logits):
        """
        Args:
            base_queries: (B, 100, 256) Base Mask2Former query embeddings
            structural_tokens: (B, 11, 256) Output from Canonical Atlas Head
            vis_logits: (B, 11) Visibility logits from Canonical Atlas Head
            
        Returns:
            steered_queries: (B, 100, 256) Deformation-aware, steered queries
        """
        B, N_q, C = base_queries.shape
        _, N_s, _ = structural_tokens.shape

        q = self.norm_q(base_queries)
        s = self.norm_struct(structural_tokens)

        # Visibility probability in [0, 1]
        vis_prob = torch.sigmoid(vis_logits)  # (B, 11)

        # 1. Logarithmic Additive Attention Bias
        # When vis_prob ~ 0, attn_bias ~ -50.0 (strictly zeroes attention weight in softmax)
        # When vis_prob ~ 1, attn_bias ~ 0.0 (full standard attention)
        attn_bias = torch.log(vis_prob.clamp(min=1e-5))  # (B, 11)
        attn_bias = attn_bias.view(B, 1, 1, N_s).expand(-1, self.num_heads, N_q, -1)  # (B, H, 100, 11)

        # 2. Multi-Head Projections
        Q = self.q_proj(q).view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, 100, D)
        K = self.k_proj(s).view(B, N_s, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, 11, D)
        
        # 3. Double-Gate Values: scale values by visibility to guarantee zero gradient on absent tokens
        s_gated = s * vis_prob.unsqueeze(-1)
        V = self.v_proj(s_gated).view(B, N_s, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, 11, D)

        # 4. Scaled Dot-Product Attention with Visibility Gating
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, H, 100, 11)
        scores = scores + attn_bias

        attn_weights = F.softmax(scores, dim=-1)  # (B, H, 100, 11)
        delta_q = torch.matmul(attn_weights, V)   # (B, H, 100, D)

        delta_q = delta_q.transpose(1, 2).contiguous().view(B, N_q, C)  # (B, 100, 256)
        delta_q = self.out_proj(delta_q)

        # 5. Gated Residual Update
        steered_queries = self.norm_out(base_queries + self.alpha * delta_q)
        return steered_queries
