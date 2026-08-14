import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPE1D(nn.Module):
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, embed_dim, 2).float() / embed_dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        sin_inp = torch.einsum("i,j->ij", pos.float(), self.inv_freq)
        emb = torch.cat([sin_inp.sin(), sin_inp.cos()], dim=-1)
        return emb

class HierarchicalDecoderLayer(nn.Module):
    """
    Single Layer of the Hierarchical Transformer Decoder.
    Performs:
      1. Intra-Curve Point Self-Attention across K points within each instance
      2. Memory-Efficient Masked Cross-Attention to 2D visual feature map
      3. FFN update
    """
    def __init__(self, embed_dim: int = 256, num_heads: int = 8, feedforward_dim: int = 1024):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Self-attention across point sequence
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)

        # Cross-attention to 2D feature map
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Feedforward Network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            nn.GELU(),
            nn.Linear(feedforward_dim, embed_dim)
        )
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, queries: torch.Tensor, memory: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        """
        queries: (B, N * K, embed_dim) - all point queries in batch
        memory: (B, H_f*W_f, embed_dim) - 2D visual feature map
        attn_mask: (B * num_heads, N * K, H_f*W_f) boolean attention mask derived from M_{l-1}
        """
        B, NK, C = queries.shape

        # 1. Intra-Curve Self-Attention (Reshaped to B*N, K, C)
        # Assuming NK = N * K
        q_norm = self.norm1(queries)
        sa_out, _ = self.self_attn(q_norm, q_norm, q_norm)
        queries = queries + sa_out

        # 2. Memory-Efficient Shared Cross-Attention (B, NK, C) x (B, H_f*W_f, C)
        q_norm2 = self.norm2(queries)
        ca_out, _ = self.cross_attn(q_norm2, memory, memory, attn_mask=attn_mask)
        queries = queries + ca_out

        # 3. FFN
        queries = queries + self.ffn(self.norm3(queries))
        return queries

class HierarchicalMaskedDecoder3D(nn.Module):
    """
    Multi-Layer Hierarchical Masked Decoder with Memory-Optimized Dual Prediction Heads.
    Uses Stride-16 feature maps (64x64 = 4096 spatial tokens) for 4x cross-attention memory reduction.
    """
    def __init__(self, embed_dim: int = 256, num_instances: int = 10, num_points: int = 20, 
                 num_classes: int = 4, num_layers: int = 6):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_instances = num_instances
        self.num_points = num_points
        self.num_classes = num_classes
        self.num_layers = num_layers

        # Embeddings
        self.instance_embed = nn.Embedding(num_instances, embed_dim)
        self.pe1d = SinusoidalPE1D(embed_dim)
        self.pe3d_proj = nn.Linear(3, embed_dim)

        # Decoder layers
        self.layers = nn.ModuleList([
            HierarchicalDecoderLayer(embed_dim=embed_dim, num_heads=8) for _ in range(num_layers)
        ])

        # Dual Prediction Heads per layer
        self.class_heads = nn.ModuleList([
            nn.Linear(embed_dim, num_classes + 1) for _ in range(num_layers)
        ])
        
        self.point_3d_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, 3),
                nn.Tanh() # Offset Δp in [-0.2, 0.2]
            ) for _ in range(num_layers)
        ])

        self.mask_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(embed_dim, 1, kernel_size=1)
            ) for _ in range(num_layers)
        ])

    def forward(self, fused_features: list, initial_anchors_3d: torch.Tensor, stride_idx: int = 1):
        """
        fused_features: List of 4 feature maps at strides {4, 8, 16, 32}
        initial_anchors_3d: (B, N, K, 3) initial 3D positions from proposal head
        stride_idx: Feature map stride index (1: Stride-8 128x128 = 16,384 tokens for High Precision A100)
        """
        B, N, K, _ = initial_anchors_3d.shape
        C = self.embed_dim
        
        # Select high-resolution feature map (stride_idx=1 for Stride-8 = 128x128 = 16,384 tokens)
        feat_map = fused_features[stride_idx] if stride_idx < len(fused_features) else fused_features[1]
        _, _, H_f, W_f = feat_map.shape

        # Flatten visual feature map for cross-attention memory: (B, H_f*W_f, C)
        memory = feat_map.flatten(2).permute(0, 2, 1).contiguous()

        # Initialize Dual-Index Query Tokens Q_{i,j}
        inst_indices = torch.arange(N, device=initial_anchors_3d.device)
        order_indices = torch.arange(K, device=initial_anchors_3d.device)

        e_inst = self.instance_embed(inst_indices).view(1, N, 1, C) # (1, N, 1, C)
        e_order = self.pe1d(order_indices).view(1, 1, K, C) # (1, 1, K, C)

        current_anchors = initial_anchors_3d
        current_mask_logits = None

        outputs_cls = []
        outputs_polylines = []
        outputs_masks = []

        for l, decoder_layer in enumerate(self.layers):
            # Compute 3D Positional Query Embedding PE_3D(p_anchor)
            e_3d = self.pe3d_proj(current_anchors) # (B, N, K, C)
            queries = e_inst + e_order + e_3d # (B, N, K, C)
            queries_flat = queries.view(B, N * K, C) # (B, N*K, C)

            # Construct Masked Attention Mask from M_{l-1}
            attn_mask = None
            if current_mask_logits is not None:
                # Downsample mask to feature map resolution (H_f, W_f)
                mask_ds = F.interpolate(current_mask_logits, size=(H_f, W_f), mode='bilinear', align_corners=False)
                # Expand instance mask to all K points per instance: (B, N, H_f*W_f) -> (B, N, K, H_f*W_f)
                mask_bool = (mask_ds < 0.0).flatten(2).unsqueeze(2).repeat(1, 1, K, 1) # (B, N, K, H_f*W_f)
                mask_bool = mask_bool.view(B, N * K, H_f * W_f) # (B, N*K, H_f*W_f)

                # Safeguard: If all positions are masked out for a query, unmask them to prevent softmax([-inf]) = NaN
                all_masked = mask_bool.all(dim=-1, keepdim=True)
                mask_bool = mask_bool & (~all_masked)

                # Repeat for num_heads=8
                attn_mask = mask_bool.repeat_interleave(8, dim=0) # (B * 8, N*K, H_f*W_f)

            # Run Decoder Layer
            queries_out = decoder_layer(queries_flat, memory, attn_mask=attn_mask) # (B, N*K, C)
            queries_reshaped = queries_out.view(B, N, K, C)

            # 1. Instance Classification Head (Pooling point queries across instance)
            inst_feat = queries_reshaped.mean(dim=2) # (B, N, C)
            cls_logits = self.class_heads[l](inst_feat) # (B, N, num_classes+1)

            # 2. 3D Point Head (Predicting residual displacement Δp)
            delta_p = self.point_3d_heads[l](queries_reshaped) * 0.2 # (B, N, K, 3)
            current_anchors = torch.clamp(current_anchors + delta_p, -1.0, 1.0)

            # 3. 2D Mask Head (Generating 2D mask M_l)
            inst_feat_expanded = inst_feat.view(B * N, C, 1, 1).expand(-1, -1, H_f, W_f)
            mask_logits_stride = self.mask_heads[l](inst_feat_expanded).view(B, N, H_f, W_f)
            current_mask_logits = F.interpolate(mask_logits_stride, size=(1024, 1024), mode='bilinear', align_corners=False) # (B, N, 1024, 1024)

            outputs_cls.append(cls_logits)
            outputs_polylines.append(current_anchors)
            outputs_masks.append(current_mask_logits)

        return outputs_cls, outputs_polylines, outputs_masks
