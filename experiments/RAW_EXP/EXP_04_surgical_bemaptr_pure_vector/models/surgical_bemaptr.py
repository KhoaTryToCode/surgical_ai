"""
Surgical-BeMapTR v2: Pure Vectorized Landmark Segmentation Model for Laparoscopic Surgery.

Comprehensive architecture addressing three root causes of poor predictions:
1. BeMapNet Spatial Centroid Coordinate Head (einsum + GAP) — replaces MLP offset regression
2. PyTorch-native Deformable Cross-Attention (grid_sample around ref points) — replaces vanilla attention
3. Multi-scale FPN features for rich spatial context — replaces single-scale 64x64 pooling

Architecture references:
- BeMapNet (CVPR 2023): Spatial voting coordinate prediction via coords_head + einsum + GAP
  [repos/BeMapNet/bemapnet/models/output_head/bezier_outputs.py]
- MapTRv2 (TPAMI 2024): Hierarchical queries, GDA, deformable attention, iterative refinement
  [repos/MapTR/projects/mmdet3d_plugin/maptr/modules/geometry_kernel_attention.py]
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
    Computes Bernstein Basis Coefficient Matrix B of shape (m, n+1).
    b_{i,n}(t) = C(n,i) * t^i * (1-t)^(n-i), t in [0, 1]
    """
    t = torch.linspace(0.0, 1.0, m, device=device).unsqueeze(1)  # (m, 1)
    B = []
    for i in range(n + 1):
        binom = math.comb(n, i)
        poly = binom * (t ** i) * ((1.0 - t) ** (n - i))
        B.append(poly)
    return torch.cat(B, dim=1)  # (m, n+1)


def compute_piecewise_bernstein_matrix(k=3, n=3, m_total=20, device=None):
    """
    Computes global Bernstein matrix B_global of shape (m_total, n*k+1)
    for piecewise Bezier curve <k, n>.
    """
    pts_per_piece = max(2, m_total // k)
    B_single = compute_bernstein_basis(n=n, m=pts_per_piece, device=device)

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

    return B_global  # (m_total, n*k+1)


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
#  FPN Pixel Decoder (Multi-Scale Output)
# ──────────────────────────────────────────────

class FPNPixelDecoder(nn.Module):
    """FPN that returns ALL scale levels for multi-scale deformable attention."""

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
        return outs  # ALL levels: [P2, P3, P4, P5]


# ──────────────────────────────────────────────
#  Point Positional Encoding (2D Sinusoidal)
# ──────────────────────────────────────────────

class PointPositionalEncoding2D(nn.Module):
    """Encodes 2D reference point (x, y) in [0,1] as sinusoidal positional embedding."""

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
#  Deformable Cross-Attention (PyTorch-native)
# ──────────────────────────────────────────────

class DeformableCrossAttention(nn.Module):
    """
    PyTorch-native deformable cross-attention via F.grid_sample.

    Instead of attending to all spatial tokens equally (vanilla attention),
    each query samples K features at learned offsets around its reference point,
    from multiple FPN scale levels. This provides focused, high-resolution
    local context — matching MapTR's GeometryKernelAttention mechanism.

    Reference: repos/MapTR/.../geometry_kernel_attention.py
    """

    def __init__(self, embed_dim=256, num_levels=3, num_points=8, dropout=0.1):
        super().__init__()
        self.num_levels = num_levels
        self.num_points = num_points
        total_pts = num_levels * num_points

        # Per-query predicted sampling offsets around reference point
        self.sampling_offsets = nn.Linear(embed_dim, total_pts * 2)
        # Per-query attention weights over all sampled features
        self.attention_weights = nn.Linear(embed_dim, total_pts)
        # Per-level value projection (1x1 conv on FPN feature maps)
        self.value_projs = nn.ModuleList([
            nn.Conv2d(embed_dim, embed_dim, 1) for _ in range(num_levels)
        ])
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.sampling_offsets.weight)
        nn.init.zeros_(self.sampling_offsets.bias)
        nn.init.zeros_(self.attention_weights.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, query, multi_scale_feats, ref_pts):
        """
        Args:
            query: (B, Q, D)
            multi_scale_feats: list of (B, D, H_l, W_l) for each FPN level
            ref_pts: (B, Q, 2) in [0, 1]
        Returns:
            (B, Q, D)
        """
        B, Q, D = query.shape
        L = min(self.num_levels, len(multi_scale_feats))
        P = self.num_points

        # Predict sampling offsets: small displacements around reference point
        offsets = self.sampling_offsets(query).reshape(B, Q, L, P, 2)
        offsets = offsets.tanh() * 0.1  # local neighborhood +/-10% of image

        # Attention weights over all (level x point) samples
        weights = self.attention_weights(query)
        weights = weights.reshape(B, Q, L * P).softmax(-1)

        # Sample features from each FPN level
        all_samples = []
        for lvl in range(L):
            feat = self.value_projs[lvl](multi_scale_feats[lvl])  # (B, D, H_l, W_l)

            # Sampling locations: ref_pt + offset, clamped to [0,1], converted to [-1,1]
            locs = ref_pts.unsqueeze(2) + offsets[:, :, lvl, :, :]  # (B, Q, P, 2)
            grid = locs.clamp(0, 1) * 2.0 - 1.0  # [0,1] -> [-1,1] for grid_sample
            grid = grid.reshape(B, Q * P, 1, 2)

            s = F.grid_sample(
                feat, grid, mode='bilinear',
                padding_mode='border', align_corners=False
            )  # (B, D, Q*P, 1)
            s = s.squeeze(-1).permute(0, 2, 1)  # (B, Q*P, D)
            s = s.reshape(B, Q, P, D)
            all_samples.append(s)

        # Concatenate across levels: (B, Q, L*P, D)
        sampled = torch.cat(all_samples, dim=2)

        # Weighted aggregation
        weights = weights.unsqueeze(-1)  # (B, Q, L*P, 1)
        out = (sampled * weights).sum(dim=2)  # (B, Q, D)

        return self.dropout(self.output_proj(out))


# ──────────────────────────────────────────────
#  Geometry-Decoupled Attention (GDA)
# ──────────────────────────────────────────────

class GeometryDecoupledAttention(nn.Module):
    """
    MapTRv2 Geometry-Decoupled Attention.
    Intra-instance attention: points within the same polyline communicate.
    Inter-instance attention: polyline-level features communicate across instances.
    """

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
#  Transformer Decoder Layer
# ──────────────────────────────────────────────

class SurgicalBeMapTRDecoderLayer(nn.Module):
    """
    Single decoder layer combining:
    1. GDA self-attention (intra + inter instance reasoning)
    2. Deformable cross-attention (focused sampling around reference points)
    3. FFN
    """

    def __init__(self, embed_dim=256, num_heads=8, num_instances=30, num_pts=10,
                 num_levels=3, num_sample_points=8, dropout=0.1):
        super().__init__()
        self.gda = GeometryDecoupledAttention(
            num_instances=num_instances, num_pts=num_pts,
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout
        )
        self.deform_cross_attn = DeformableCrossAttention(
            embed_dim=embed_dim, num_levels=num_levels,
            num_points=num_sample_points, dropout=dropout
        )
        self.cross_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)

    def forward(self, query, multi_scale_feats, ref_pts, query_pos=None):
        # 1. GDA self-attention
        query = self.gda(query)
        # 2. Deformable cross-attention (ref-point-guided feature sampling)
        q = query + query_pos if query_pos is not None else query
        cross_out = self.deform_cross_attn(q, multi_scale_feats, ref_pts)
        query = self.cross_norm(query + cross_out)
        # 3. FFN
        query = self.ffn_norm(query + self.ffn(query))
        return query


# ──────────────────────────────────────────────
#  Spatial Centroid Coordinate Head (BeMapNet)
# ──────────────────────────────────────────────

class SpatialCentroidCoordHead(nn.Module):
    """
    BeMapNet-style spatial voting coordinate prediction.

    Instead of regressing (x, y) from a 1D vector (MLP approach),
    predicts coordinates via spatial voting over a 2D coordinate feature map:

    1. Coordinate feature map: learnable encoding of normalized (x, y) grid positions
    2. Voting features: per-point query -> spatial weighting features
    3. einsum(voting_feats, coords_feats) -> spatial heatmap
    4. GAP(heatmap) -> centroid (x, y)

    Reference: repos/BeMapNet/bemapnet/models/output_head/bezier_outputs.py (Lines 57-84)
    """

    def __init__(self, coord_dim=64, feat_h=64, feat_w=64):
        super().__init__()
        self.coord_dim = coord_dim

        # Normalized coordinate grid: (1, 2, H, W) with x in [0,1] and y in [0,1]
        coords = self._make_coordinate_grid(feat_h, feat_w)
        self.register_buffer('coords', coords)

        # Learnable coordinate encoder: 2D position -> coord_dim features
        # Exactly mirrors BeMapNet's coords_head = FFN(2, 256, _C, 3, 'conv')
        self.coords_encoder = nn.Sequential(
            nn.Conv2d(2, coord_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(coord_dim, coord_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(coord_dim, coord_dim, 1),
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    @staticmethod
    def _make_coordinate_grid(h, w):
        """Creates normalized coordinate grid, exactly as in BeMapNet compute_locations()."""
        xs = torch.linspace(0, 1, w)
        ys = torch.linspace(0, 1, h)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        return torch.stack([xx, yy], dim=0).unsqueeze(0)  # (1, 2, H, W)

    def forward(self, vote_feats):
        """
        Args:
            vote_feats: (B, Q_xy, C) where Q_xy = num_points * 2
                        (pairs of voting features for x and y per control point)
        Returns:
            coords: (B, num_points, 2) — predicted (x, y) for each control point
        """
        B = vote_feats.shape[0]
        Q_xy = vote_feats.shape[1]

        # Coordinate feature map: (B, C, H, W) — encodes spatial position
        coords_feats = self.coords_encoder(self.coords.expand(B, -1, -1, -1))

        # Spatial voting: dot product of voting features with coordinate features
        # -> heatmap indicating where each control point should be
        heatmap = torch.einsum("bqc,bchw->bqhw", vote_feats, coords_feats)

        # GAP: extract centroid from heatmap
        centroid = self.gap(heatmap).reshape(B, Q_xy)

        # Reshape: pair consecutive scalars as (x, y)
        return centroid.reshape(B, Q_xy // 2, 2)


# ──────────────────────────────────────────────
#  Full Surgical-BeMapTR v2 Model
# ──────────────────────────────────────────────

class SurgicalBeMapTR(nn.Module):
    """
    Pure Vectorized Surgical Landmark Segmentation Model (Surgical-BeMapTR v2).

    Inputs: RGB image (B, 3, 1024, 1024)
    Outputs:
        pred_logits: (B, N, num_classes) classification logits
        pred_ctrl_pts: (B, N, num_ctrl_pts, 2) Piecewise Bezier control points in [0, 1]
        pred_restored_pts: (B, N, K_dense, 2) Bernstein-restored dense curve points in [0, 1]
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
                 num_deform_levels=3,
                 num_sample_points=8,
                 coord_feat_dim=64,
                 coord_feat_size=64,
                 dropout=0.1,
                 pretrained_backbone=True):
        super().__init__()

        self.N = N
        self.K_dense = K_dense
        self.bezier_k = bezier_k
        self.bezier_n = bezier_n
        self.num_ctrl_pts = bezier_n * bezier_k + 1  # 10 for <3,3>
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_deform_levels = num_deform_levels

        # ── 1. Backbone + Multi-Scale FPN ──
        self.backbone = SwinTinyBackbone(pretrained=pretrained_backbone, img_size=img_size)
        self.pixel_decoder = FPNPixelDecoder(
            in_channels_list=self.backbone.out_channels, out_channels=embed_dim
        )

        # ── 2. Positional Encoding for reference points ──
        self.ref_point_pos_enc = PointPositionalEncoding2D(embed_dim)

        # ── 3. Hierarchical Queries (MapTRv2) ──
        self.instance_query = nn.Embedding(N, embed_dim)
        self.point_query = nn.Embedding(self.num_ctrl_pts, embed_dim)

        # ── 4. Initial Reference Point Head ──
        # Each (instance, point) pair predicts its initial (x, y) location
        self.ref_point_init = nn.Linear(embed_dim, 2)

        # ── 5. Transformer Decoder Layers ──
        self.decoder_layers = nn.ModuleList([
            SurgicalBeMapTRDecoderLayer(
                embed_dim=embed_dim, num_heads=num_heads,
                num_instances=N, num_pts=self.num_ctrl_pts,
                num_levels=num_deform_levels, num_sample_points=num_sample_points,
                dropout=dropout,
            )
            for _ in range(num_decoder_layers)
        ])

        # ── 6. Per-Layer Voting Heads (query -> spatial voting features) ──
        # Each layer has its own voting head (like MapTR's per-layer reg_branches)
        self.voting_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dim, 2 * coord_feat_dim),
            )
            for _ in range(num_decoder_layers)
        ])

        # ── 7. Shared Spatial Centroid Head (coordinate feature map + GAP) ──
        self.spatial_coord_head = SpatialCentroidCoordHead(
            coord_dim=coord_feat_dim,
            feat_h=coord_feat_size,
            feat_w=coord_feat_size,
        )

        # ── 8. Classification Head ──
        self.cls_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, num_classes),
        )

        # ── 9. Bernstein Basis Matrix ──
        B_matrix = compute_piecewise_bernstein_matrix(
            k=self.bezier_k, n=self.bezier_n, m_total=self.K_dense
        )
        self.register_buffer('B_matrix', B_matrix)

        self._init_weights()

    def _init_weights(self):
        # Classification head
        for m in self.cls_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Reference point init head
        nn.init.xavier_uniform_(self.ref_point_init.weight)
        nn.init.zeros_(self.ref_point_init.bias)
        # Voting heads: xavier init
        for voting_head in self.voting_heads:
            for m in voting_head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def _build_initial_ref_pts(self, B, device):
        """
        Predict initial reference points for each (instance, control_point) pair.
        Each point gets a unique initial (x, y) position from the learned embeddings.
        """
        inst_emb = self.instance_query.weight       # (N, D)
        pt_emb = self.point_query.weight             # (K, D)
        combined = inst_emb.unsqueeze(1) + pt_emb.unsqueeze(0)  # (N, K, D)
        ref_pts = self.ref_point_init(combined).sigmoid()       # (N, K, 2) in [0,1]
        ref_pts = ref_pts.reshape(self.N * self.num_ctrl_pts, 2)
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

        # ── 1. Backbone + Multi-Scale FPN ──
        features = self.backbone(x)
        fpn_levels = self.pixel_decoder(features)  # [P2, P3, P4, P5]
        deform_feats = fpn_levels[:self.num_deform_levels]  # First 3 levels

        # ── 2. Build Hierarchical Queries ──
        inst_emb = self.instance_query.weight   # (N, D)
        pt_emb = self.point_query.weight        # (K, D)
        query = (inst_emb.unsqueeze(1) + pt_emb.unsqueeze(0))
        query = query.reshape(self.N * self.num_ctrl_pts, self.embed_dim)
        query = query.unsqueeze(0).expand(B, -1, -1)   # (B, N*K, D)

        # ── 3. Initial Reference Points ──
        ref_pts = self._build_initial_ref_pts(B, device)  # (B, N*K, 2)

        # ── 4. Decoder Layers with Spatial Centroid Coordinate Prediction ──
        intermediate_logits = []
        intermediate_ctrl_pts = []
        intermediate_restored_pts = []

        for i, layer in enumerate(self.decoder_layers):
            # Positional encoding from current reference points
            query_pos = self.ref_point_pos_enc(ref_pts)

            # Decoder layer: GDA self-attention + Deformable cross-attention + FFN
            query = layer(query, deform_feats, ref_pts, query_pos=query_pos)

            # ── Spatial Centroid Coordinate Prediction ──
            # Per-point voting features: (B, N*K, 2*C)
            vote_feats = self.voting_heads[i](query)
            C = vote_feats.shape[-1] // 2
            # Reshape for spatial centroid: (B, N*K*2, C)
            vote_feats = vote_feats.reshape(B, self.N * self.num_ctrl_pts * 2, C)
            # Spatial voting -> control point coordinates
            point_coords = self.spatial_coord_head(vote_feats)  # (B, N*K, 2)

            # Reshape and Bernstein restore
            layer_ctrl = point_coords.reshape(B, self.N, self.num_ctrl_pts, 2)
            layer_restored = self.restore_curve(layer_ctrl)  # (B, N, K_dense, 2)

            # Classification (mean-pool over control points per instance)
            query_inst = query.reshape(
                B, self.N, self.num_ctrl_pts, self.embed_dim
            ).mean(dim=2)  # (B, N, D)
            layer_logits = self.cls_head(query_inst)  # (B, N, num_classes)

            intermediate_logits.append(layer_logits)
            intermediate_ctrl_pts.append(layer_ctrl)
            intermediate_restored_pts.append(layer_restored)

            # Update reference points for next layer (detach like official MapTR)
            ref_pts = point_coords.detach()

        if self.training:
            return intermediate_logits, intermediate_ctrl_pts, intermediate_restored_pts
        else:
            return intermediate_logits[-1], intermediate_ctrl_pts[-1], intermediate_restored_pts[-1]
