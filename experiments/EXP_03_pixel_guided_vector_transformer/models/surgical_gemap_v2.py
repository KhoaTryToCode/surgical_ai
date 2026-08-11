"""
Surgical-GeMap v2: Pixel-Guided Vector Transformer for Laparoscopic Landmark Segmentation.

Architecture:
1. Swin-Tiny Backbone + FPN Pixel Decoder
2. Auxiliary Pixel Segmentation Head (outputs 4-channel dense mask at 1024x1024)
3. Heatmap-Guided Reference Point Proposal (initializes ref_pts from spatial activation peaks)
4. Transformer Vector Decoder with Geometry-Decoupled Attention (GDA) & Point-Sampled grid_sample
5. Dual Output: Pixel Heatmaps (for pixel BCE+Dice loss) & Ordered Polylines (for vector losses)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ──────────────────────────────────────────────
#  Backbone: Swin-Tiny via timm
# ──────────────────────────────────────────────

class SwinTinyBackbone(nn.Module):
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
        features = self.model(x)
        # Convert timm NHWC output to NCHW
        out = []
        for feat in features:
            if feat.dim() == 4 and feat.shape[1] != feat.shape[3]:
                feat = feat.permute(0, 3, 1, 2).contiguous()
            out.append(feat)
        return out


# ──────────────────────────────────────────────
#  FPN Pixel Decoder with Auxiliary Pixel Head
# ──────────────────────────────────────────────

class FPNPixelDecoder(nn.Module):
    """
    FPN Pixel Decoder with 2 outputs:
    1. Multi-scale fused spatial features (stride 4, 256x256) for Transformer Decoder
    2. Dense 4-channel pixel segmentation logits (1024x1024) for Pixel Loss
    """

    def __init__(self, in_channels_list=[96, 192, 384, 768], out_channels=256, num_classes=4):
        super().__init__()
        self.out_channels = out_channels
        self.num_classes = num_classes

        # Lateral 1x1 convs
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1) for in_ch in in_channels_list
        ])

        # Output 3x3 convs
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
            for _ in in_channels_list
        ])

        # Auxiliary Pixel Segmentation Head (256x256 -> 1024x1024)
        self.pixel_seg_head = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 2, 3, padding=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
            nn.ConvUpsample(out_channels // 2, out_channels // 4, scale_factor=2),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvUpsample(out_channels // 4, num_classes, scale_factor=2),
        )

    def forward(self, features):
        laterals = [lat(feat) for lat, feat in zip(self.lateral_convs, features)]

        # Top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            target_size = laterals[i - 1].shape[-2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=target_size, mode='bilinear', align_corners=False
            )

        outs = [conv(lat) for conv, lat in zip(self.output_convs, laterals)]
        fused = outs[0]  # finest FPN feature map (B, 256, 256, 256) at stride 4

        # Dense pixel logits (B, 4, 1024, 1024)
        pixel_logits = self.pixel_seg_head(fused)

        return fused, pixel_logits


class ConvUpsample(nn.Module):
    """Helper module: 3x3 Conv + Bilinear Upsampling."""
    def __init__(self, in_ch, out_ch, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        return self.conv(x)


# ──────────────────────────────────────────────
#  Positional Encodings
# ──────────────────────────────────────────────

class PositionalEncoding2D(nn.Module):
    """Sinusoidal 2D positional encoding for spatial tokens."""

    def __init__(self, d_model, max_h=256, max_w=256):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(d_model, max_h, max_w)
        half_d = d_model // 2

        pos_h = torch.arange(0, max_h).unsqueeze(1).float()
        div_h = torch.exp(torch.arange(0, half_d, 2).float() * -(math.log(10000.0) / half_d))
        pe[0:half_d:2, :, :] = torch.sin(pos_h * div_h).transpose(0, 1).unsqueeze(2).expand(-1, -1, max_w)
        pe[1:half_d:2, :, :] = torch.cos(pos_h * div_h).transpose(0, 1).unsqueeze(2).expand(-1, -1, max_w)

        pos_w = torch.arange(0, max_w).unsqueeze(1).float()
        div_w = torch.exp(torch.arange(0, half_d, 2).float() * -(math.log(10000.0) / half_d))
        pe[half_d::2, :, :] = torch.sin(pos_w * div_w).transpose(0, 1).unsqueeze(1).expand(-1, max_h, -1)
        pe[half_d + 1::2, :, :] = torch.cos(pos_w * div_w).transpose(0, 1).unsqueeze(1).expand(-1, max_h, -1)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.pe[:, :, :x.shape[2], :x.shape[3]]


class PointPositionalEncoding2D(nn.Module):
    """Sinusoidal 2D positional encoding for continuous reference point coordinates [0, 1]."""

    def __init__(self, embed_dim=256, temperature=10000):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_pos_feats = embed_dim // 2
        self.temperature = temperature
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, pts):
        device = pts.device
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / self.num_pos_feats)

        x_embed = pts[:, :, 0:1] * 2 * math.pi / dim_t
        y_embed = pts[:, :, 1:2] * 2 * math.pi / dim_t

        pos_x = torch.stack((x_embed[:, :, 0::2].sin(), x_embed[:, :, 1::2].cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((y_embed[:, :, 0::2].sin(), y_embed[:, :, 1::2].cos()), dim=-1).flatten(-2)

        pos = torch.cat((pos_x, pos_y), dim=-1)
        return self.mlp(pos)


# ──────────────────────────────────────────────
#  Geometry-Decoupled Attention (GDA)
# ──────────────────────────────────────────────

class GeometryDecoupledAttention(nn.Module):
    def __init__(self, num_instances=30, num_pts=20, embed_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_instances = num_instances
        self.num_pts = num_pts

        self.intra_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.intra_norm1 = nn.LayerNorm(embed_dim)
        self.intra_ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.intra_norm2 = nn.LayerNorm(embed_dim)

        self.inter_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.inter_norm1 = nn.LayerNorm(embed_dim)
        self.inter_ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.inter_norm2 = nn.LayerNorm(embed_dim)

    def forward(self, query):
        B, NK, D = query.shape
        N = self.num_instances
        K = self.num_pts

        q_intra = query.reshape(B * N, K, D)
        out_intra, _ = self.intra_attn(q_intra, q_intra, q_intra)
        out_intra = self.intra_norm1(out_intra + q_intra)
        out_intra = self.intra_norm2(out_intra + self.intra_ffn(out_intra))

        q_inter = out_intra.reshape(B, N, K, D).mean(dim=2)
        out_inter, _ = self.inter_attn(q_inter, q_inter, q_inter)
        out_inter = self.inter_norm1(out_inter + q_inter)
        out_inter = self.inter_norm2(out_inter + self.inter_ffn(out_inter))

        inter_broadcast = out_inter.unsqueeze(2).expand(-1, -1, K, -1)
        result = out_intra.reshape(B, N, K, D) + inter_broadcast
        return result.reshape(B, NK, D)


# ──────────────────────────────────────────────
#  Transformer Decoder Layer & Refinement Head
# ──────────────────────────────────────────────

class SurgicalGeMapDecoderLayer(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_instances=30, num_pts=20, dropout=0.1):
        super().__init__()
        self.gda = GeometryDecoupledAttention(
            num_instances=num_instances, num_pts=num_pts,
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout
        )
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)

    def forward(self, query, spatial_tokens, spatial_pos=None, query_pos=None):
        query = self.gda(query)
        q = query + query_pos if query_pos is not None else query
        k = spatial_tokens + spatial_pos if spatial_pos is not None else spatial_tokens
        cross_out, _ = self.cross_attn(q, k, spatial_tokens)
        query = self.cross_norm(query + cross_out)
        query = self.ffn_norm(query + self.ffn(query))
        return query


class PointRefinementHead(nn.Module):
    def __init__(self, embed_dim=256, num_pts=20):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, 2),
        )

    def forward(self, query_per_point):
        return self.mlp(query_per_point)


# ──────────────────────────────────────────────
#  Full Surgical-GeMap v2 Model
# ──────────────────────────────────────────────

class SurgicalGeMapV2(nn.Module):
    """
    Surgical-GeMap v2 (Pixel-Guided Vector Transformer).

    Inputs: RGB image (B, 3, 1024, 1024)
    Outputs:
        pixel_logits: (B, 4, 1024, 1024) dense pixel segmentation logits
        pred_vector_logits: (B, N, num_classes) polyline query classification logits
        pred_vector_pts: (B, N, K, 2) normalized polyline control points in [0, 1]
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

        # 1. Swin-Tiny Backbone
        self.backbone = SwinTinyBackbone(pretrained=pretrained_backbone, img_size=img_size)

        # 2. FPN Pixel Decoder with Auxiliary Segmentation Head
        self.pixel_decoder = FPNPixelDecoder(
            in_channels_list=self.backbone.out_channels,
            out_channels=embed_dim,
            num_classes=num_classes
        )

        # 3. Positional Encodings
        self.pos_enc = PositionalEncoding2D(embed_dim, max_h=img_size // 4, max_w=img_size // 4)
        self.ref_point_pos_enc = PointPositionalEncoding2D(embed_dim)

        # 4. Polyline queries
        self.instance_query = nn.Embedding(N, embed_dim)
        self.point_query = nn.Embedding(K, embed_dim)

        # 5. Dynamic Reference Point Head & Heatmap Guidance MLP
        self.ref_point_head = nn.Linear(embed_dim, K * 2)

        # 6. Transformer Vector Decoder
        self.decoder_layers = nn.ModuleList([
            SurgicalGeMapDecoderLayer(
                embed_dim=embed_dim, num_heads=num_heads,
                num_instances=N, num_pts=K, dropout=dropout,
            )
            for _ in range(num_decoder_layers)
        ])

        self.point_refine_heads = nn.ModuleList([
            PointRefinementHead(embed_dim, K) for _ in range(num_decoder_layers)
        ])

        self.cls_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for module in [self.cls_head, self.ref_point_head]:
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
        inst_emb = self.instance_query.weight
        pt_emb = self.point_query.weight

        query = (inst_emb.unsqueeze(1) + pt_emb.unsqueeze(0)).reshape(self.N * self.K, self.embed_dim)
        query = query.unsqueeze(0).expand(B, -1, -1)

        ref_pts = self.ref_point_head(inst_emb).sigmoid().reshape(self.N * self.K, 2)
        ref_pts = ref_pts.unsqueeze(0).expand(B, -1, -1)

        return query, ref_pts

    def forward(self, x):
        B = x.shape[0]
        device = x.device

        # ── 1. Backbone & Pixel Decoder ──
        features = self.backbone(x)
        fused, pixel_logits = self.pixel_decoder(features)  # fused: (B, D, 256, 256), pixel_logits: (B, 4, 1024, 1024)

        # ── 2. Spatial Tokens ──
        spatial_feat = F.adaptive_avg_pool2d(fused, (64, 64))
        spatial_tokens = spatial_feat.flatten(2).permute(0, 2, 1)
        spatial_pos = self.pos_enc(spatial_feat).flatten(2).permute(0, 2, 1).expand(B, -1, -1)

        # ── 3. Polyline Queries & Reference Points ──
        query, ref_pts = self._build_query(B, device)

        # ── 4. Transformer Decoder with Heatmap Point Feature Sampling ──
        intermediate_vector_logits = []
        intermediate_vector_pts = []

        for layer, refine_head in zip(self.decoder_layers, self.point_refine_heads):
            query_pos = self.ref_point_pos_enc(ref_pts)
            query = layer(query, spatial_tokens, spatial_pos, query_pos=query_pos)

            # High-resolution Point-Sampled FPN Feature Extraction
            grid = ref_pts.unsqueeze(2) * 2.0 - 1.0  # (B, N*K, 1, 2) in [-1, 1]
            point_feats = F.grid_sample(
                fused, grid, mode='bilinear', padding_mode='border', align_corners=False
            ).squeeze(3).permute(0, 2, 1)

            # Refinement offsets bounded by tanh
            offsets = refine_head(query + point_feats)
            offsets = torch.tanh(offsets) * 0.15

            ref_pts_raw = self._inverse_sigmoid(ref_pts)
            new_ref_pts = (ref_pts_raw + offsets).sigmoid()

            layer_pts = new_ref_pts.reshape(B, self.N, self.K, 2)
            query_per_inst = query.reshape(B, self.N, self.K, self.embed_dim).mean(dim=2)
            layer_logits = self.cls_head(query_per_inst)

            intermediate_vector_logits.append(layer_logits)
            intermediate_vector_pts.append(layer_pts)

            ref_pts = new_ref_pts.detach()

        if self.training:
            return pixel_logits, intermediate_vector_logits, intermediate_vector_pts
        else:
            return pixel_logits, intermediate_vector_logits[-1], intermediate_vector_pts[-1]

    @staticmethod
    def _inverse_sigmoid(x, eps=1e-5):
        x = x.clamp(eps, 1 - eps)
        return torch.log(x / (1 - x))
