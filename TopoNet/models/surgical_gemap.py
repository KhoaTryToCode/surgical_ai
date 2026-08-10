"""
Surgical-GeMap: Vectorized Landmark Segmentation Model.

Swin-Tiny backbone + FPN pixel decoder + Transformer decoder with
Geometry-Decoupled Attention (GDA). Predicts N polyline queries,
each with K=20 ordered 2D control points and a class label.

Adapted from GeMap (ECCV 2024) for 2D single-view laparoscopic images.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ──────────────────────────────────────────────
#  Swin-Tiny Backbone
# ──────────────────────────────────────────────

class SwinTinyBackbone(nn.Module):
    """
    Swin-Tiny backbone for multi-scale feature extraction.
    Outputs features at 4 scales: stride 4, 8, 16, 32.
    Channel dims: [96, 192, 384, 768] for Swin-Tiny.
    """

    def __init__(self, pretrained=True, img_size=1024):
        super().__init__()
        self.model = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=pretrained,
            features_only=True,
            img_size=img_size,
        )
        self.out_channels = [96, 192, 384, 768]

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) input images

        Returns:
            List of 4 feature maps at increasing strides.
        """
        features = self.model(x)
        return features  # [C2, C3, C4, C5]


# ──────────────────────────────────────────────
#  Feature Pyramid Network (FPN) Pixel Decoder
# ──────────────────────────────────────────────

class FPNPixelDecoder(nn.Module):
    """
    Standard FPN that fuses multi-scale backbone features into a single
    high-resolution feature map for the transformer decoder.
    """

    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        self.out_channels = out_channels

        # Lateral connections (1x1 conv to unify channels)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1) for in_ch in in_channels_list
        ])

        # Output convolutions (3x3 after feature fusion)
        self.output_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.GroupNorm(32, out_channels),
                nn.ReLU(inplace=True),
            ) for _ in in_channels_list
        ])

    def forward(self, features):
        """
        Args:
            features: list of 4 feature maps [C2, C3, C4, C5]

        Returns:
            fused: (B, out_channels, H/4, W/4) fused feature map
        """
        # Top-down pathway
        laterals = [lat(f) for lat, f in zip(self.lateral_convs, features)]

        for i in range(len(laterals) - 2, -1, -1):
            laterals[i] = laterals[i] + F.interpolate(
                laterals[i + 1],
                size=laterals[i].shape[-2:],
                mode='bilinear',
                align_corners=False
            )

        # Output convolutions
        outs = [conv(lat) for conv, lat in zip(self.output_convs, laterals)]

        # Return finest-resolution feature map (stride 4)
        return outs[0]


# ──────────────────────────────────────────────
#  Positional Encoding
# ──────────────────────────────────────────────

class PositionalEncoding2D(nn.Module):
    """Sinusoidal 2D positional encoding for spatial tokens."""

    def __init__(self, d_model, max_h=256, max_w=256):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(d_model, max_h, max_w)
        half_d = d_model // 2

        # Height encoding
        pos_h = torch.arange(0, max_h).unsqueeze(1).float()
        div_h = torch.exp(torch.arange(0, half_d, 2).float()
                          * -(math.log(10000.0) / half_d))
        pe[0:half_d:2, :, :] = torch.sin(pos_h * div_h).transpose(0, 1).unsqueeze(2).expand(-1, -1, max_w)
        pe[1:half_d:2, :, :] = torch.cos(pos_h * div_h).transpose(0, 1).unsqueeze(2).expand(-1, -1, max_w)

        # Width encoding
        pos_w = torch.arange(0, max_w).unsqueeze(1).float()
        div_w = torch.exp(torch.arange(0, half_d, 2).float()
                          * -(math.log(10000.0) / half_d))
        pe[half_d::2, :, :] = torch.sin(pos_w * div_w).transpose(0, 1).unsqueeze(1).expand(-1, max_h, -1)
        pe[half_d + 1::2, :, :] = torch.cos(pos_w * div_w).transpose(0, 1).unsqueeze(1).expand(-1, max_h, -1)

        self.register_buffer('pe', pe.unsqueeze(0))  # (1, D, max_h, max_w)

    def forward(self, x):
        """x: (B, D, H, W)"""
        return self.pe[:, :, :x.shape[2], :x.shape[3]]


# ──────────────────────────────────────────────
#  Geometry-Decoupled Attention (GDA)
# ──────────────────────────────────────────────

class GeometryDecoupledAttention(nn.Module):
    """
    Separates self-attention into:
    - Intra-instance: attention among points within the same polyline
    - Inter-instance: attention across different polylines (via centroids)

    Adapted from GeMap's GDA module.
    """

    def __init__(self, num_instances=30, num_pts=20, embed_dim=256,
                 num_heads=8, dropout=0.1):
        super().__init__()
        self.num_instances = num_instances
        self.num_pts = num_pts

        # Intra-instance attention (points within same polyline)
        self.intra_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.intra_norm1 = nn.LayerNorm(embed_dim)
        self.intra_ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.intra_norm2 = nn.LayerNorm(embed_dim)

        # Inter-instance attention (between polylines)
        self.inter_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.inter_norm1 = nn.LayerNorm(embed_dim)
        self.inter_ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.inter_norm2 = nn.LayerNorm(embed_dim)

    def forward(self, query):
        """
        Args:
            query: (B, N*K, D) flattened polyline queries

        Returns:
            (B, N*K, D) updated queries
        """
        B, NK, D = query.shape
        N = self.num_instances
        K = self.num_pts

        # ── Intra-instance attention ──
        # Reshape to (B*N, K, D) so each polyline attends only within itself
        q_intra = query.reshape(B * N, K, D)
        out_intra, _ = self.intra_attn(q_intra, q_intra, q_intra)
        out_intra = self.intra_norm1(out_intra + q_intra)
        out_intra = self.intra_norm2(out_intra + self.intra_ffn(out_intra))

        # ── Inter-instance attention ──
        # Pool each polyline to a single centroid token: (B, N, D)
        q_inter = out_intra.reshape(B, N, K, D).mean(dim=2)
        out_inter, _ = self.inter_attn(q_inter, q_inter, q_inter)
        out_inter = self.inter_norm1(out_inter + q_inter)
        out_inter = self.inter_norm2(out_inter + self.inter_ffn(out_inter))

        # Broadcast inter-instance update back to all points
        # (B, N, D) → (B, N, K, D) → (B, N*K, D)
        inter_broadcast = out_inter.unsqueeze(2).expand(-1, -1, K, -1)
        result = out_intra.reshape(B, N, K, D) + inter_broadcast
        result = result.reshape(B, NK, D)

        return result


# ──────────────────────────────────────────────
#  Transformer Decoder Layer
# ──────────────────────────────────────────────

class SurgicalGeMapDecoderLayer(nn.Module):
    """
    Single decoder layer:
    1. Geometry-Decoupled Self-Attention (GDA)
    2. Cross-Attention to spatial tokens
    3. FFN
    """

    def __init__(self, embed_dim=256, num_heads=8, num_instances=30,
                 num_pts=20, dropout=0.1):
        super().__init__()

        self.gda = GeometryDecoupledAttention(
            num_instances=num_instances,
            num_pts=num_pts,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Cross-attention to spatial features
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)

    def forward(self, query, spatial_tokens, spatial_pos=None):
        """
        Args:
            query: (B, N*K, D)
            spatial_tokens: (B, HW, D) flattened spatial features
            spatial_pos: (B, HW, D) positional encoding for spatial tokens

        Returns:
            (B, N*K, D) updated queries
        """
        # 1. Geometry-Decoupled Self-Attention
        query = self.gda(query)

        # 2. Cross-Attention
        k = spatial_tokens + spatial_pos if spatial_pos is not None else spatial_tokens
        cross_out, _ = self.cross_attn(query, k, spatial_tokens)
        query = self.cross_norm(query + cross_out)

        # 3. FFN
        query = self.ffn_norm(query + self.ffn(query))

        return query


# ──────────────────────────────────────────────
#  Point Refinement Head
# ──────────────────────────────────────────────

class PointRefinementHead(nn.Module):
    """Predicts residual offsets for iterative point refinement."""

    def __init__(self, embed_dim=256, num_pts=20):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, 2),  # predict (dx, dy) offset
        )

    def forward(self, query_per_point):
        """
        Args:
            query_per_point: (B, N*K, D)

        Returns:
            offsets: (B, N*K, 2)
        """
        return self.mlp(query_per_point)


# ──────────────────────────────────────────────
#  Full Surgical-GeMap Model
# ──────────────────────────────────────────────

class SurgicalGeMap(nn.Module):
    """
    Vectorized landmark segmentation model.

    Input: RGB image (B, 3, H, W)
    Output:
        pred_logits: (B, N, num_classes) classification logits
        pred_pts: (B, N, K, 2) normalized polyline coordinates [0, 1]
    """

    def __init__(self,
                 img_size=1024,
                 num_classes=4,
                 N=30,
                 K=20,
                 embed_dim=256,
                 num_heads=8,
                 num_decoder_layers=6,
                 dropout=0.1,
                 pretrained_backbone=True):
        super().__init__()

        self.N = N
        self.K = K
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # ── Backbone ──
        self.backbone = SwinTinyBackbone(
            pretrained=pretrained_backbone,
            img_size=img_size
        )

        # ── FPN Pixel Decoder ──
        self.pixel_decoder = FPNPixelDecoder(
            in_channels_list=self.backbone.out_channels,
            out_channels=embed_dim
        )

        # ── Positional Encoding ──
        self.pos_enc = PositionalEncoding2D(embed_dim, max_h=img_size // 4,
                                            max_w=img_size // 4)

        # ── Learnable polyline queries ──
        # Instance-level queries: (N, D)
        self.instance_query = nn.Embedding(N, embed_dim)
        # Point-level queries: (K, D) — shared across all instances
        self.point_query = nn.Embedding(K, embed_dim)

        # ── Initial reference points ──
        # Learnable initial positions for each query point
        self.reference_points = nn.Embedding(N * K, 2)
        nn.init.uniform_(self.reference_points.weight, 0.0, 1.0)

        # ── Transformer Decoder ──
        self.decoder_layers = nn.ModuleList([
            SurgicalGeMapDecoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_instances=N,
                num_pts=K,
                dropout=dropout,
            )
            for _ in range(num_decoder_layers)
        ])

        # ── Point refinement heads (one per decoder layer) ──
        self.point_refine_heads = nn.ModuleList([
            PointRefinementHead(embed_dim, K)
            for _ in range(num_decoder_layers)
        ])

        # ── Classification head ──
        self.cls_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize classification and regression heads."""
        for module in [self.cls_head]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        for head in self.point_refine_heads:
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def _build_query(self, B, device):
        """
        Construct initial polyline queries by combining instance-level
        and point-level embeddings.

        Returns:
            query: (B, N*K, D)
            ref_pts: (B, N*K, 2) initial reference point positions
        """
        # Instance embedding: (N, D) → (N, 1, D) → (N, K, D)
        inst_emb = self.instance_query.weight.unsqueeze(1).expand(-1, self.K, -1)
        # Point embedding: (K, D) → (1, K, D) → (N, K, D)
        pt_emb = self.point_query.weight.unsqueeze(0).expand(self.N, -1, -1)
        # Combined: (N, K, D) → (N*K, D)
        query = (inst_emb + pt_emb).reshape(self.N * self.K, self.embed_dim)
        query = query.unsqueeze(0).expand(B, -1, -1)  # (B, N*K, D)

        # Reference points: (N*K, 2) → (B, N*K, 2)
        ref_pts = self.reference_points.weight.sigmoid()
        ref_pts = ref_pts.unsqueeze(0).expand(B, -1, -1)

        return query, ref_pts

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) input images

        Returns:
            pred_logits: (B, N, num_classes)
            pred_pts: (B, N, K, 2) normalized coordinates [0, 1]
        """
        B = x.shape[0]
        device = x.device

        # ── Backbone ──
        features = self.backbone(x)

        # ── FPN ──
        fused = self.pixel_decoder(features)  # (B, D, H/4, W/4)

        # ── Spatial tokens ──
        H_feat, W_feat = fused.shape[-2:]
        spatial_tokens = fused.flatten(2).permute(0, 2, 1)  # (B, HW, D)
        spatial_pos = self.pos_enc(fused)  # (1, D, H, W)
        spatial_pos = spatial_pos.flatten(2).permute(0, 2, 1).expand(B, -1, -1)

        # ── Build queries ──
        query, ref_pts = self._build_query(B, device)

        # ── Decoder layers with iterative refinement ──
        for layer, refine_head in zip(self.decoder_layers, self.point_refine_heads):
            query = layer(query, spatial_tokens, spatial_pos)

            # Point refinement
            offsets = refine_head(query)  # (B, N*K, 2)
            # inverse_sigmoid for numerically stable update
            ref_pts_raw = self._inverse_sigmoid(ref_pts)
            new_ref_pts = (ref_pts_raw + offsets).sigmoid()
            ref_pts = new_ref_pts.detach()  # detach for next layer

        # ── Final predictions ──
        # Classification: pool each instance's K point queries → single vector
        query_per_instance = query.reshape(B, self.N, self.K, self.embed_dim)
        instance_features = query_per_instance.mean(dim=2)  # (B, N, D)
        pred_logits = self.cls_head(instance_features)       # (B, N, num_classes)

        # Point coordinates: use refined reference points
        pred_pts = ref_pts.reshape(B, self.N, self.K, 2)  # (B, N, K, 2)

        return pred_logits, pred_pts

    @staticmethod
    def _inverse_sigmoid(x, eps=1e-5):
        x = x.clamp(eps, 1 - eps)
        return torch.log(x / (1 - x))
