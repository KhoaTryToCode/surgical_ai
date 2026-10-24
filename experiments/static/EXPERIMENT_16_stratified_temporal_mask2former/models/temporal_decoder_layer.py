import torch
import torch.nn as nn


class TemporalQueryInterleavedBlock(nn.Module):
    """
    Interleaved temporal cross-attention module.
    Enables target frame queries to attend to past landmark prototypes.
    Uses a learnable residual gate parameter initialized small (e.g. 0.1)
    to smoothly admit temporal conditioning without perturbing initial convergence.
    """
    def __init__(self, hidden_dim=256, num_heads=8, dropout=0.1, gate_init=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.gate = nn.Parameter(torch.tensor([gate_init], dtype=torch.float32))
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, past_prototypes):
        """
        Args:
            queries: (B, num_queries, hidden_dim)
            past_prototypes: (B, num_past_tokens, hidden_dim)
            
        Returns:
            conditioned_queries: (B, num_queries, hidden_dim)
        """
        if past_prototypes is None:
            return queries

        attn_out, _ = self.cross_attn(
            query=queries,
            key=past_prototypes,
            value=past_prototypes
        )
        out = self.norm(queries + self.gate * self.dropout(attn_out))
        return out
