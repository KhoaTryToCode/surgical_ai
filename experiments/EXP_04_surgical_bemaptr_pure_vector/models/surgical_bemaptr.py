"""
Surgical-BeMapTR: Pure Vectorized Landmark Segmentation Model for Laparoscopic Surgery.

Combines:
1. BeMapNet (CVPR 2023): Piecewise Bézier Curve parameterization <k=3, n=3> (10 control handles)
   with End-to-End Endpoint Linear Interpolation (lerp) + Curvature Offsets,
   restored via Bernstein basis matrix multiplication (P = B * C).
2. MapTRv2 (TPAMI 2024): Hierarchical Queries (q_ins + q_pt), Decoupled Self-Attention (GDA),
   2D Sinusoidal Point Positional Encoding (query_pos), and Point-Sampled grid_sample FPN feature extraction.

Guarantees smooth, ordered, continuous anatomical vector curves (no zig-zag star loops!).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ──────────────────────────────────────────────
#  Bernstein Basis Matrix Generator
# ──────────────────────────────────────────────

def compute_bernstein_basis(n=3, m=20, device=None):
    """
    Computes Bernstein Basis Coefficient Matrix B of shape (m, n + 1) for a single Bézier curve of degree n.
    b_{i, n}(t) = binom(n, i) * t^i * (1 - t)^(n - i),  t in [0, 1]
    """
    t = torch.linspace(0.0, 1.0, m, device=device).unsqueeze(1)  # (m, 1)
    B = []
    for i in range(n + 1):
        binom = math.comb(n, i)
        poly = binom * (t ** i) * ((1.0 - t) ** (n - i))
        B.append(poly)
    return torch.cat(B, dim=1)  # (m, n + 1)


def compute_piecewise_bernstein_matrix(k=3, n=3, m_total=20, device=None):
    """
    Computes global Bernstein matrix B_global of shape (m_total, n*k + 1) for a piecewise Bézier curve <k, n>.
    With k=3 pieces of degree n=3, there are 10 control points in total.
    """
    pts_per_piece = max(2, m_total // k)
    B_single = compute_bernstein_basis(n=n, m=pts_per_piece, device=device)  # (m_piece, n+1)

    num_control_pts = n * k + 1
    B_global = torch.zeros(k * pts_per_piece, num_control_pts, device=device)

    for piece_idx in range(k):
        start_ctrl = piece_idx * n
        end_ctrl = start_ctrl + n + 1
        row_start = piece_idx * pts_per_piece
        row_end = row_start + pts_per_piece
        B_global[row_start:row_end, start_ctrl:end_ctrl] = B_single

    if B_global.shape[0] != m_total:
        indices = torch.linspace(0, B_global.shape[0] - 1, m_total, device=device).long()
        B_global = B_global[indices]

    return B_global  # (m_total, n*k + 1)


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
        out = []
        for feat in features:
            if feat.dim() == 4 and feat.shape[1] != feat.shape[3]:
                feat = feat.permute(0, 3, 1, 2).contiguous()
            out.append(feat)
        return out


# ──────────────────────────────────────────────
#  FPN Pixel Decoder
# ──────────────────────────────────────────────

class FPNPixelDecoder(nn.Module):
    def __init__(self, in_channels_list=[96, 192, 384, 768], out_channels=256):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1) for in_ch in in_channels_list
        ])
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
            for _ in in_channels_list
        ])

    def forward(self, features):
        laterals = [lat(feat) for lat, feat in zip(self.lateral_convs, features)]
        for i in range(len(laterals) - 1, 0, -1):
            target_size = laterals[i - 1].shape[-2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=target_size, mode='bilinear', align_corners=False
            )
        outs = [conv(lat) for conv, lat in zip(self.output_convs, laterals)]
        return outs[0]  # finest FPN map (B, 256, 256, 256) at stride 4


# ──────────────────────────────────────────────
#  Positional Encodings
# ──────────────────────────────────────────────

class PositionalEncoding2D(nn.Module):
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
    def __init__(self, num_instances=30, num_pts=10, embed_dim=256, num_heads=8, dropout=0.1):
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
#  Transformer Decoder Layer & Piecewise Head
# ──────────────────────────────────────────────

class SurgicalBeMapTRDecoderLayer(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_instances=30, num_pts=10, dropout=0.1):
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


class PiecewiseBezierHead(nn.Module):
    """Predicts curvature offsets for Piecewise Bézier control points."""
    def __init__(self, embed_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, 2),
        )

    def forward(self, query_per_point):
        return self.mlp(query_per_point)


# ──────────────────────────────────────────────
#  Full Surgical-BeMapTR Model
# ──────────────────────────────────────────────

class SurgicalBeMapTR(nn.Module):
    """
    Pure Vectorized Surgical Landmark Segmentation Model (Surgical-BeMapTR).
    Inputs: RGB image (B, 3, 1024, 1024)
    Outputs:
        pred_logits: (B, N, num_classes) classification logits
        pred_bezier_ctrl: (B, N, num_ctrl_pts, 2) Piecewise Bézier control points in [0, 1]
        pred_restored_pts: (B, N, K_dense, 2) Bernstein restored dense curve points in [0, 1]
    """

    def __init__(self,
                 img_size=1024,
                 num_classes=4,
                 N=30,
                 K_dense=20,
                 bezier_k=3,
                 bezier_n=3,
                 embed_dim=256,
                 num_heads=8,
                 num_decoder_layers=6,
                 dropout=0.1,
                 pretrained_backbone=True):
        super().__init__()

        self.N = N
        self.K_dense = K_dense
        self.bezier_k = bezier_k
        self.bezier_n = bezier_n
        self.num_ctrl_pts = bezier_n * bezier_k + 1  # 10 control points for <3, 3>
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # 1. Swin-Tiny Backbone + FPN
        self.backbone = SwinTinyBackbone(pretrained=pretrained_backbone, img_size=img_size)
        self.pixel_decoder = FPNPixelDecoder(
            in_channels_list=self.backbone.out_channels, out_channels=embed_dim
        )

        # 2. Positional Encodings
        self.pos_enc = PositionalEncoding2D(embed_dim, max_h=img_size // 4, max_w=img_size // 4)
        self.ref_point_pos_enc = PointPositionalEncoding2D(embed_dim)

        # 3. Hierarchical Queries
        self.instance_query = nn.Embedding(N, embed_dim)
        self.point_query = nn.Embedding(self.num_ctrl_pts, embed_dim)

        # 4. Endpoint Head: Predicts (x_start, y_start, x_end, y_end) per instance
        self.endpoint_head = nn.Linear(embed_dim, 4)

        # 5. Precomputed lerp time steps t in [0, 1] for 10 control points
        t_steps = torch.linspace(0.0, 1.0, self.num_ctrl_pts).unsqueeze(1)  # (10, 1)
        self.register_buffer('t_steps', t_steps)

        # 6. Transformer Decoder Layers
        self.decoder_layers = nn.ModuleList([
            SurgicalBeMapTRDecoderLayer(
                embed_dim=embed_dim, num_heads=num_heads,
                num_instances=N, num_pts=self.num_ctrl_pts, dropout=dropout,
            )
            for _ in range(num_decoder_layers)
        ])

        self.point_refine_heads = nn.ModuleList([
            PiecewiseBezierHead(embed_dim) for _ in range(num_decoder_layers)
        ])

        self.cls_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, num_classes),
        )

        # 7. Precomputed Bernstein Basis Matrix B of shape (K_dense, num_ctrl_pts)
        B_matrix = compute_piecewise_bernstein_matrix(
            k=self.bezier_k, n=self.bezier_n, m_total=self.K_dense
        )
        self.register_buffer('B_matrix', B_matrix)

        self._init_weights()

    def _init_weights(self):
        for module in [self.cls_head, self.endpoint_head]:
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

    def _build_initial_control_points(self, inst_emb, B):
        """
        Predicts ordered initial control points C^0 via lerp(P_start, P_end)
        Returns:
            ref_pts: (B, N * num_ctrl_pts, 2) in [0, 1]
        """
        endpoints = self.endpoint_head(inst_emb).sigmoid()  # (N, 4) -> [x_start, y_start, x_end, y_end]
        p_start = endpoints[:, :2]  # (N, 2)
        p_end = endpoints[:, 2:]    # (N, 2)

        # lerp: C_j = (1 - t_j) * P_start + t_j * P_end
        # t_steps is (10, 1), p_start is (N, 2) -> ctrl_pts is (N, 10, 2)
        ctrl_base = (1.0 - self.t_steps.unsqueeze(0)) * p_start.unsqueeze(1) + self.t_steps.unsqueeze(0) * p_end.unsqueeze(1)

        ref_pts = ctrl_base.reshape(self.N * self.num_ctrl_pts, 2)
        return ref_pts.unsqueeze(0).expand(B, -1, -1)

    def restore_curve(self, ctrl_pts):
        """
        Multiplies control points by Bernstein basis matrix: P = B * C.
        ctrl_pts: (..., num_ctrl_pts, 2) -> returns (..., K_dense, 2)
        """
        return torch.matmul(self.B_matrix, ctrl_pts)

    def forward(self, x):
        B = x.shape[0]
        device = x.device

        # ── 1. Backbone & FPN ──
        features = self.backbone(x)
        fused = self.pixel_decoder(features)  # (B, 256, 256, 256)

        # ── 2. Spatial Tokens ──
        spatial_feat = F.adaptive_avg_pool2d(fused, (64, 64))
        spatial_tokens = spatial_feat.flatten(2).permute(0, 2, 1)
        spatial_pos = self.pos_enc(spatial_feat).flatten(2).permute(0, 2, 1).expand(B, -1, -1)

        # ── 3. Build Hierarchical Queries & Ordered Base Control Points ──
        inst_emb = self.instance_query.weight
        pt_emb = self.point_query.weight

        query = (inst_emb.unsqueeze(1) + pt_emb.unsqueeze(0)).reshape(self.N * self.num_ctrl_pts, self.embed_dim)
        query = query.unsqueeze(0).expand(B, -1, -1)

        ref_pts = self._build_initial_control_points(inst_emb, B)

        # ── 4. Decoder Layers with Iterative Refinement & Bernstein Restoration ──
        intermediate_logits = []
        intermediate_ctrl_pts = []
        intermediate_restored_pts = []

        for layer, refine_head in zip(self.decoder_layers, self.point_refine_heads):
            query_pos = self.ref_point_pos_enc(ref_pts)
            query = layer(query, spatial_tokens, spatial_pos, query_pos=query_pos)

            # High-resolution Point-Sampled FPN Feature Extraction
            grid = ref_pts.unsqueeze(2) * 2.0 - 1.0  # (B, N*num_ctrl_pts, 1, 2) in [-1, 1]
            point_feats = F.grid_sample(
                fused, grid, mode='bilinear', padding_mode='border', align_corners=False
            ).squeeze(3).permute(0, 2, 1)

            # Curvature offsets bounded by tanh
            offsets = refine_head(query + point_feats)
            offsets = torch.tanh(offsets) * 0.10

            ref_pts_raw = self._inverse_sigmoid(ref_pts)
            new_ref_pts = (ref_pts_raw + offsets).sigmoid()

            layer_ctrl_pts = new_ref_pts.reshape(B, self.N, self.num_ctrl_pts, 2)
            layer_restored_pts = self.restore_curve(layer_ctrl_pts)

            query_per_inst = query.reshape(B, self.N, self.num_ctrl_pts, self.embed_dim).mean(dim=2)
            layer_logits = self.cls_head(query_per_inst)

            intermediate_logits.append(layer_logits)
            intermediate_ctrl_pts.append(layer_ctrl_pts)
            intermediate_restored_pts.append(layer_restored_pts)

            ref_pts = new_ref_pts

        if self.training:
            return intermediate_logits, intermediate_ctrl_pts, intermediate_restored_pts
        else:
            return intermediate_logits[-1], intermediate_ctrl_pts[-1], intermediate_restored_pts[-1]

    @staticmethod
    def _inverse_sigmoid(x, eps=1e-5):
        x = x.clamp(eps, 1 - eps)
        return torch.log(x / (1 - x))
