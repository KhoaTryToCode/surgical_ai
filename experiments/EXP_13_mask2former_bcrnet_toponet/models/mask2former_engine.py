"""
EXP_13: Mask2Former Engine for Laparoscopic Landmark Detection
==============================================================
Implements:
1. 4-Channel RGB-D ResNet-50 Backbone with FPN (levels f1..f4, C=256)
2. High-resolution Pixel Embedding Generator
3. Multi-Layer Transformer Decoder with Masked Cross-Attention
4. Per-Query Binary Mask & Multi-Class Semantic Segmentation Heads
"""
import math
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class MaskedCrossAttention(nn.Module):
    """
    Masked Cross-Attention Module.
    Queries only attend to spatial locations where the predicted mask is active.
    """

    def __init__(self, embed_dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        q: torch.Tensor,              # (B, N, D)
        kv: torch.Tensor,             # (B, S, D)
        attn_mask: Optional[torch.Tensor] = None,  # (B, N, S) float with -inf for masked
    ) -> torch.Tensor:
        B, N, _ = q.shape
        _, S, _ = kv.shape

        q_proj = self.q_proj(q).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)   # (B, H, N, d)
        k_proj = self.k_proj(kv).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, S, d)
        v_proj = self.v_proj(kv).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, S, d)

        # Scaled dot-product attention
        scores = torch.matmul(q_proj, k_proj.transpose(-2, -1)) * self.scale  # (B, H, N, S)

        if attn_mask is not None:
            # attn_mask: (B, N, S) -> broadcast across heads: (B, 1, N, S)
            scores = scores + attn_mask.unsqueeze(1)

        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v_proj)  # (B, H, N, d)
        out = out.transpose(1, 2).contiguous().view(B, N, self.embed_dim)
        return self.out_proj(out)


class Mask2FormerDecoderLayer(nn.Module):
    """Single layer of Mask2Former Transformer Decoder."""

    def __init__(self, embed_dim: int = 256, num_heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.cross_attn = MaskedCrossAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)

        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, embed_dim),
        )
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        queries: torch.Tensor,        # (B, N, D)
        features: torch.Tensor,       # (B, S, D)
        attn_mask: Optional[torch.Tensor] = None,  # (B, N, S)
    ) -> torch.Tensor:
        # 1. Self-Attention
        q_norm = self.norm1(queries)
        sa_out, _ = self.self_attn(q_norm, q_norm, q_norm)
        queries = queries + sa_out

        # 2. Masked Cross-Attention
        q_norm = self.norm2(queries)
        ca_out = self.cross_attn(q_norm, features, attn_mask=attn_mask)
        queries = queries + ca_out

        # 3. FFN
        q_norm = self.norm3(queries)
        queries = queries + self.mlp(q_norm)
        return queries


class ResNetFPN4Ch(nn.Module):
    """ResNet-50 FPN taking 4-channel RGB-D input."""

    def __init__(self, in_channels: int = 4, out_channels: int = 256):
        super().__init__()
        base = resnet50(weights=ResNet50_Weights.DEFAULT)

        # Initialize first convolution for 4 channels
        w_orig = base.conv1.weight.data
        new_conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        new_conv1.weight.data[:, :3] = w_orig
        new_conv1.weight.data[:, 3:4] = w_orig[:, :1]  # Initialize depth from grayscale average
        self.conv1 = new_conv1

        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool

        self.layer1 = base.layer1  # 256, stride 4
        self.layer2 = base.layer2  # 512, stride 8
        self.layer3 = base.layer3  # 1024, stride 16
        self.layer4 = base.layer4  # 2048, stride 32

        # Lateral 1x1 convolutions
        self.lat4 = nn.Conv2d(2048, out_channels, 1)
        self.lat3 = nn.Conv2d(1024, out_channels, 1)
        self.lat2 = nn.Conv2d(512, out_channels, 1)
        self.lat1 = nn.Conv2d(256, out_channels, 1)

        # Smoothing 3x3 convolutions
        self.smooth4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth1 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        p4 = self.lat4(c4)
        p3 = self.lat3(c3) + F.interpolate(p4, size=c3.shape[2:], mode="nearest")
        p2 = self.lat2(c2) + F.interpolate(p3, size=c2.shape[2:], mode="nearest")
        p1 = self.lat1(c1) + F.interpolate(p2, size=c1.shape[2:], mode="nearest")

        f4 = self.smooth4(p4)  # (B, 256, H/32, W/32)
        f3 = self.smooth3(p3)  # (B, 256, H/16, W/16)
        f2 = self.smooth2(p2)  # (B, 256, H/8, W/8)
        f1 = self.smooth1(p1)  # (B, 256, H/4, W/4)
        return [f1, f2, f3, f4]


class Mask2FormerEngine(nn.Module):
    """
    Complete Mask2Former Engine:
    - 4-Channel RGB-D ResNet-50 FPN
    - High-Resolution Pixel Embedding
    - Masked Attention Transformer Decoder
    - Co-Supervised Semantic Segmentation Map (Background + 3 Landmarks)
    """

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 3,
        num_queries_per_class: int = 5,
        fpn_dim: int = 256,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries_per_class = num_queries_per_class
        self.total_queries = num_classes * num_queries_per_class
        self.fpn_dim = fpn_dim
        self.num_decoder_layers = num_decoder_layers

        # 1. Multi-scale feature backbone
        self.backbone = ResNetFPN4Ch(in_channels=in_channels, out_channels=fpn_dim)

        # 2. Pixel embedding generator (projects 1/4 resolution feature to pixel embedding)
        self.pixel_embed_proj = nn.Sequential(
            nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1),
            nn.GroupNorm(32, fpn_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_dim, fpn_dim, 1),
        )

        # 3. Learnable landmark queries (initialized per class)
        self.query_embeddings = nn.Parameter(torch.randn(self.total_queries, fpn_dim) * 0.02)
        self.query_pos_embeddings = nn.Parameter(torch.randn(self.total_queries, fpn_dim) * 0.02)

        # 4. Decoder layers
        self.decoder_layers = nn.ModuleList([
            Mask2FormerDecoderLayer(embed_dim=fpn_dim, num_heads=num_heads)
            for _ in range(num_decoder_layers)
        ])

        # 5. Class prediction MLP: outputs 4 classes (0: bg, 1: ridge, 2: silhouette, 3: ligament)
        self.class_head = nn.Linear(fpn_dim, num_classes + 1)

        # 6. Dense Semantic Fusion Head: Produces 4-channel semantic map (for TopoNet clDice)
        self.semantic_head = nn.Sequential(
            nn.Conv2d(fpn_dim, fpn_dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_dim // 2, num_classes + 1, 1),
        )

    def forward(
        self,
        rgbd: torch.Tensor,
        target_size: Tuple[int, int] = (512, 512),
    ) -> dict:
        B, _, H, W = rgbd.shape

        # 1. Extract multi-scale FPN features: [f1(1/4), f2(1/8), f3(1/16), f4(1/32)]
        fpn_features = self.backbone(rgbd)
        f1, f2, f3, f4 = fpn_features

        # 2. Generate high-resolution pixel embedding: (B, 256, H/4, W/4)
        pixel_embeddings = self.pixel_embed_proj(f1)
        B_pe, C_pe, H_pe, W_pe = pixel_embeddings.shape

        # Flatten features from f3 (stride 16) for Transformer Decoder cross-attention
        # Shape: (B, H_f3 * W_f3, 256)
        f_dec = f3.flatten(2).transpose(1, 2)
        S = f_dec.shape[1]
        H_f3, W_f3 = f3.shape[2:]

        # 3. Initialize queries
        queries = self.query_embeddings.unsqueeze(0).expand(B, -1, -1)  # (B, N, 256)
        pos = self.query_pos_embeddings.unsqueeze(0).expand(B, -1, -1)
        queries = queries + pos

        # Iterative Masked Attention Decoder
        mask_logits = None
        for i, layer in enumerate(self.decoder_layers):
            attn_mask = None
            if mask_logits is not None:
                # Downsample predicted mask logits to match decoder feature resolution (H_f3, W_f3)
                mask_down = F.interpolate(
                    mask_logits, size=(H_f3, W_f3), mode="bilinear", align_corners=False
                )  # (B, N, H_f3, W_f3)
                mask_down = mask_down.flatten(2)  # (B, N, S)
                # Binarize threshold: -inf where sigmoid(logit) < 0.5 (i.e. logit < 0.0)
                attn_mask = torch.zeros_like(mask_down)
                attn_mask[mask_down < 0.0] = -1e9

            # Layer forward
            queries = layer(queries, f_dec, attn_mask=attn_mask)

            # Compute new mask logits: Einsum(queries, pixel_embeddings) -> (B, N, H/4, W/4)
            mask_logits = torch.einsum("bnd,bdhw->bnhw", queries, pixel_embeddings)

        # 4. Final Query Outputs
        # Interpolate query mask logits to target_size (e.g. 512x512)
        query_masks_full = F.interpolate(
            mask_logits, size=target_size, mode="bilinear", align_corners=False
        )  # (B, N, H, W)
        query_class_logits = self.class_head(queries)  # (B, N, 4)

        # 5. Full Semantic Map Prediction (for TopoNet clDice loss)
        # Direct pixel-level projection from pixel_embeddings:
        semantic_logits_low = self.semantic_head(pixel_embeddings)  # (B, 4, H/4, W/4)
        semantic_logits = F.interpolate(
            semantic_logits_low, size=target_size, mode="bilinear", align_corners=False
        )  # (B, 4, H, W)

        return {
            "fpn_features": fpn_features,               # [f1, f2, f3, f4]
            "pixel_embeddings": pixel_embeddings,       # (B, 256, H/4, W/4)
            "query_embeddings": queries,                 # (B, N, 256)
            "query_masks": query_masks_full,             # (B, N, H, W)
            "query_classes": query_class_logits,         # (B, N, 4)
            "semantic_logits": semantic_logits,          # (B, 4, H, W)
        }
