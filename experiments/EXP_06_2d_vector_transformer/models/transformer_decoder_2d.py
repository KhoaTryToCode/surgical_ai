import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Point2DPositionalEncoding(nn.Module):
    """
    Continuous Sinusoidal Positional Encoding for 2D polyline vertices (u, v) in [0.0, 1.0]^2.
    Maps 2D coordinates into embed_dim features.
    """
    def __init__(self, embed_dim: int = 256, temperature: float = 10000.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.half_dim = embed_dim // 4 # 64 per coordinate frequency
        self.temperature = temperature
        
        freq = torch.exp(torch.arange(0, self.half_dim, dtype=torch.float32) * -(math.log(temperature) / self.half_dim))
        self.register_buffer("freq", freq)

    def forward(self, pts_2d: torch.Tensor) -> torch.Tensor:
        """
        pts_2d: (B, N, K, 2) in [0.0, 1.0]^2
        Returns: (B, N, K, embed_dim)
        """
        u = pts_2d[..., 0:1] * 2.0 * math.pi # (B, N, K, 1)
        v = pts_2d[..., 1:2] * 2.0 * math.pi
        
        u_emb = u * self.freq.view(1, 1, 1, -1) # (B, N, K, half_dim)
        v_emb = v * self.freq.view(1, 1, 1, -1)
        
        sin_u, cos_u = u_emb.sin(), u_emb.cos()
        sin_v, cos_v = v_emb.sin(), v_emb.cos()
        
        emb = torch.cat([sin_u, cos_u, sin_v, cos_v], dim=-1) # (B, N, K, embed_dim)
        return emb

class HierarchicalDecoderLayer2D(nn.Module):
    """
    Single Layer of the Hierarchical 2D Transformer Decoder.
    Performs:
      1. Intra-Curve Point Self-Attention across K points within each instance
      2. Memory-Efficient Masked Cross-Attention to 2D visual feature map
      3. FFN update with Dropout regularization
    """
    def __init__(self, embed_dim: int = 256, num_heads: int = 8, feedforward_dim: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # 1. Intra-curve self-attention
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        # 2. Masked cross-attention
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

        # 3. Feedforward Network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, queries: torch.Tensor, memory: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        """
        queries: (B, N * K, embed_dim) - all point queries in batch
        memory: (B, H_f*W_f, embed_dim) - 2D visual feature map
        attn_mask: (B * num_heads, N * K, H_f*W_f) boolean attention mask derived from M_{l-1}
        """
        # 1. Intra-Curve Point Self-Attention
        q_norm = self.norm1(queries)
        sa_out, _ = self.self_attn(q_norm, q_norm, q_norm)
        queries = queries + self.dropout1(sa_out)

        # 2. Masked Shared Cross-Attention
        q_norm2 = self.norm2(queries)
        ca_out, _ = self.cross_attn(q_norm2, memory, memory, attn_mask=attn_mask)
        queries = queries + self.dropout2(ca_out)

        # 3. FFN
        queries = queries + self.ffn(self.norm3(queries))
        return queries

class HierarchicalMaskedDecoder2D(nn.Module):
    """
    6-Layer Hierarchical 2D Transformer Decoder with Learned Query Embeddings for EXP_06.
    Operates natively in normalized image space (u, v) in [0.0, 1.0]^2.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_instances = config.num_instances
        self.num_points = config.num_points
        self.embed_dim = config.embed_dim
        self.num_layers = config.decoder_layers
        self.num_classes = config.num_classes
        self.stride_idx = config.decoder_stride_idx

        # 1. Learned Query Embeddings (DETR / Mask2Former style priors)
        # Learnable initial 2D coordinates in [0.0, 1.0]^2
        initial_coords = torch.rand(self.num_instances, self.num_points, 2)
        # Evenly spread initial queries across horizontal and vertical bands
        for i in range(self.num_instances):
            y_base = 0.2 + 0.6 * (i / max(self.num_instances - 1, 1))
            x_steps = torch.linspace(0.2, 0.8, self.num_points)
            initial_coords[i, :, 0] = x_steps
            initial_coords[i, :, 1] = y_base
        self.query_polylines = nn.Parameter(initial_coords) # (N, K, 2)

        self.instance_embed = nn.Embedding(self.num_instances, self.embed_dim)
        self.point_embed = nn.Embedding(self.num_points, self.embed_dim)
        self.pos_encoder_2d = Point2DPositionalEncoding(embed_dim=self.embed_dim)

        # 2. Stack of 6 Hierarchical Decoder Layers
        self.layers = nn.ModuleList([
            HierarchicalDecoderLayer2D(
                embed_dim=self.embed_dim,
                num_heads=config.num_heads,
                feedforward_dim=config.feedforward_dim,
                dropout=config.dropout
            ) for _ in range(self.num_layers)
        ])

        # 3. Deep Supervision Prediction Heads per Layer
        self.cls_heads = nn.ModuleList([
            nn.Linear(self.embed_dim, self.num_classes + 1) for _ in range(self.num_layers)
        ])
        
        # 2D Point Head: predicts bounded coordinate displacement (Δu, Δv) in [-0.2, 0.2]^2
        self.point_2d_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.GELU(),
                nn.Linear(self.embed_dim, 2),
                nn.Tanh()
            ) for _ in range(self.num_layers)
        ])

        # Dot-Product Mask Embedding Heads (Mask2Former paradigm: M_i = MLP(q_i) • F_stride4)
        self.mask_embed_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.GELU(),
                nn.Linear(self.embed_dim, self.embed_dim)
            ) for _ in range(self.num_layers)
        ])

    def forward(self, fused_features: list) -> dict:
        """
        fused_features: List of multi-scale feature maps from SurgicalBackbone2D
        Returns dict containing layer-by-layer predictions for deep supervision.
        """
        # Selected visual memory (Stride-4 at 256x256)
        feat_memory = fused_features[self.stride_idx] # (B, 256, H_f, W_f)
        feat_stride4 = fused_features[0]              # (B, 256, H_s4, W_s4)

        B, C, H_f, W_f = feat_memory.shape
        memory = feat_memory.flatten(2).permute(0, 2, 1) # (B, H_f*W_f, C)

        N = self.num_instances
        K = self.num_points

        # Initialize query positions from learned parameters
        current_anchors = self.query_polylines.unsqueeze(0).repeat(B, 1, 1, 1) # (B, N, K, 2)

        # Build initial query content embeddings
        inst_idx = torch.arange(N, device=feat_memory.device).unsqueeze(1).repeat(1, K) # (N, K)
        pt_idx = torch.arange(K, device=feat_memory.device).unsqueeze(0).repeat(N, 1)   # (N, K)
        
        base_content = (self.instance_embed(inst_idx) + self.point_embed(pt_idx)).unsqueeze(0).repeat(B, 1, 1, 1) # (B, N, K, C)

        outputs_cls = []
        outputs_polylines = []
        outputs_masks = []

        attn_mask = None # Layer 0 attends globally

        for l in range(self.num_layers):
            # 1. Fuse content with current 2D coordinate positional encoding
            pos_emb = self.pos_encoder_2d(current_anchors) # (B, N, K, C)
            queries = (base_content + pos_emb).view(B, N * K, C)

            # 2. Forward through Hierarchical Decoder Layer
            queries = self.layers[l](queries, memory, attn_mask=attn_mask)
            base_content = queries.view(B, N, K, C)

            # Instance-level pooled query representation
            instance_queries = base_content.mean(dim=2) # (B, N, C)

            # 3. Class Logits Prediction
            cls_logits = self.cls_heads[l](instance_queries) # (B, N, num_classes+1)
            outputs_cls.append(cls_logits)

            # 4. 2D Coordinate Displacement Prediction (bounded in [-0.2, 0.2]^2)
            delta_p = self.point_2d_heads[l](base_content) * 0.2 # (B, N, K, 2)
            current_anchors = torch.clamp(current_anchors + delta_p, 0.0, 1.0)
            outputs_polylines.append(current_anchors)

            # 5. Dot-Product 2D Mask Head (Mask2Former Formulation: M_i = MLP(q_i) • F_stride4)
            mask_embed = self.mask_embed_heads[l](instance_queries) # (B, N, C)
            mask_logits = torch.einsum("bnc,bchw->bnhw", mask_embed, feat_stride4) # (B, N, H_s4, W_s4)

            # Upsample mask to 1024x1024
            mask_logits_up = F.interpolate(mask_logits, size=(1024, 1024), mode='bilinear', align_corners=False)
            outputs_masks.append(mask_logits_up)

            # 6. Construct Masked Attention for Layer l+1
            with torch.no_grad():
                mask_ds = F.interpolate(mask_logits, size=(H_f, W_f), mode='bilinear', align_corners=False)
                mask_ds = mask_ds.unsqueeze(2).repeat(1, 1, K, 1, 1).view(B, N * K, H_f * W_f)
                
                # Boolean mask: True = masked out (background)
                mask_bool = (mask_ds < 0.0)
                
                # Safeguard: if an entire row is masked, unmask it to prevent softmax(-inf) = NaN
                all_masked = mask_bool.all(dim=-1, keepdim=True)
                mask_bool = mask_bool & (~all_masked)
                
                # Repeat across attention heads
                attn_mask = mask_bool.unsqueeze(1).repeat(1, self.config.num_heads, 1, 1).view(
                    B * self.config.num_heads, N * K, H_f * W_f
                )

        return {
            "pred_cls": outputs_cls[-1],
            "pred_polylines": outputs_polylines[-1],
            "pred_masks": outputs_masks[-1],
            "aux_cls": outputs_cls,
            "aux_polylines": outputs_polylines,
            "aux_masks": outputs_masks
        }
