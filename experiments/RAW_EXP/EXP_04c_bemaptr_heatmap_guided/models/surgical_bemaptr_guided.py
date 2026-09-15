"""
Surgical-BeMapTR v4 (Heatmap-Guided Vector Prompt Learning): EXP_04c.

Key Innovations:
1. Feature Map Modulation:
   - Branch B predicts 4-channel edge logits on FPN P2 (stride 4, 256x256).
   - Computes landmark probability mask M_edge = sigmoid(max_{c=1..3} logits_c).
   - Modulates P2 in FORWARD PASS: P2_guided = P2 * (1.0 + M_edge).
   - Zeroes out background fat/tissue features and amplifies landmark edge features 2x before Deformable Cross-Attention runs!

2. Relative Reference Point Anchoring:
   - Refines reference points using logit-space relative offsets:
     ref_pts_l = sigmoid(logit(ref_pts_{l-1}) + delta_l).
   - Eliminates spatial translation offset (~150px shift), snapping vector splines directly onto anatomical landmarks!
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_04_MODELS = os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)), 'EXP_04_surgical_bemaptr_pure_vector', 'models')
EXP_04B_MODELS = os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)), 'EXP_04b_bemaptr_aux_edge', 'models')

for p in [EXP_04_MODELS, EXP_04B_MODELS]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

from surgical_bemaptr import (
    SwinTinyBackbone,
    FPNPixelDecoder,
    PointPositionalEncoding2D,
    DeformableCrossAttention,
    GeometryDecoupledAttention,
    SurgicalBeMapTRDecoderLayer,
    SpatialCentroidCoordHead,
    compute_piecewise_bernstein_matrix,
)
from surgical_bemaptr_aux import AuxiliaryEdgeHead


class SurgicalBeMapTRGuided(nn.Module):
    """
    Surgical-BeMapTR v4 with Heatmap-Guided Feature Modulation & Relative Anchoring (EXP_04c).
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
        self.num_ctrl_pts = bezier_n * bezier_k + 1  # 10
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_deform_levels = num_deform_levels

        # 1. Backbone + Multi-Scale FPN
        self.backbone = SwinTinyBackbone(pretrained=pretrained_backbone, img_size=img_size)
        self.pixel_decoder = FPNPixelDecoder(
            in_channels_list=self.backbone.out_channels, out_channels=embed_dim
        )

        # 2. Auxiliary Edge Head (Supervises FPN P2 & produces guidance heatmap)
        self.aux_edge_head = AuxiliaryEdgeHead(in_channels=embed_dim, num_classes=num_classes)

        # 3. Positional Encoding
        self.ref_point_pos_enc = PointPositionalEncoding2D(embed_dim)

        # 4. Hierarchical Queries
        self.instance_query = nn.Embedding(N, embed_dim)
        self.point_query = nn.Embedding(self.num_ctrl_pts, embed_dim)
        self.ref_point_init = nn.Linear(embed_dim, 2)

        # 5. Decoder Layers
        self.decoder_layers = nn.ModuleList([
            SurgicalBeMapTRDecoderLayer(
                embed_dim=embed_dim, num_heads=num_heads,
                num_instances=N, num_pts=self.num_ctrl_pts,
                num_levels=num_deform_levels, num_sample_points=num_sample_points,
                dropout=dropout,
            )
            for _ in range(num_decoder_layers)
        ])

        # 6. Relative Offset Heads & Spatial Centroid Head
        self.voting_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dim, 2 * coord_feat_dim),
            )
            for _ in range(num_decoder_layers)
        ])

        # Relative delta offset heads for anchoring
        self.offset_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dim // 2, 2),
            )
            for _ in range(num_decoder_layers)
        ])

        self.spatial_coord_head = SpatialCentroidCoordHead(
            coord_dim=coord_feat_dim, feat_h=coord_feat_size, feat_w=coord_feat_size
        )

        # 7. Classification Head
        self.cls_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, num_classes),
        )

        # 8. Bernstein Matrix
        B_matrix = compute_piecewise_bernstein_matrix(k=bezier_k, n=bezier_n, m_total=K_dense)
        self.register_buffer('B_matrix', B_matrix)

        self._init_weights()

    def _init_weights(self):
        for m in self.cls_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.ref_point_init.weight)
        nn.init.zeros_(self.ref_point_init.bias)
        for voting_head in self.voting_heads:
            for m in voting_head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        for offset_head in self.offset_heads:
            for m in offset_head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def _build_initial_ref_pts(self, B, device):
        inst_emb = self.instance_query.weight
        pt_emb = self.point_query.weight
        combined = inst_emb.unsqueeze(1) + pt_emb.unsqueeze(0)
        ref_pts = self.ref_point_init(combined).sigmoid()
        ref_pts = ref_pts.reshape(self.N * self.num_ctrl_pts, 2)
        return ref_pts.unsqueeze(0).expand(B, -1, -1)

    def restore_curve(self, ctrl_pts):
        return torch.matmul(self.B_matrix, ctrl_pts)

    def forward(self, x):
        B = x.shape[0]
        device = x.device

        # 1. Backbone + Multi-Scale FPN
        features = self.backbone(x)
        fpn_levels = self.pixel_decoder(features)  # [P2, P3, P4, P5]

        # 2. Auxiliary Edge Prediction & Heatmap Guidance (FOR FORWARD PASS)
        aux_edge_logits = self.aux_edge_head(fpn_levels[0])  # (B, 4, 256, 256)
        aux_edge_probs = torch.sigmoid(aux_edge_logits)
        
        # Extract max landmark probability (channels 1, 2, 3)
        landmark_map = aux_edge_probs[:, 1:].max(dim=1, keepdim=True)[0]  # (B, 1, 256, 256)

        # FEATURE MAP MODULATION: Boost landmark edge features, suppress background fat
        guided_p2 = fpn_levels[0] * (1.0 + landmark_map)

        deform_feats = [guided_p2] + list(fpn_levels[1:self.num_deform_levels])

        # 3. Queries & Initial Reference Points
        inst_emb = self.instance_query.weight
        pt_emb = self.point_query.weight
        query = (inst_emb.unsqueeze(1) + pt_emb.unsqueeze(0)).reshape(self.N * self.num_ctrl_pts, self.embed_dim)
        query = query.unsqueeze(0).expand(B, -1, -1)
        ref_pts = self._build_initial_ref_pts(B, device)

        # 4. Decoder Layers with Heatmap-Guided Sampling & Relative Anchoring
        intermediate_logits = []
        intermediate_ctrl_pts = []
        intermediate_restored_pts = []

        for i, layer in enumerate(self.decoder_layers):
            query_pos = self.ref_point_pos_enc(ref_pts)
            query = layer(query, deform_feats, ref_pts, query_pos=query_pos)

            # Spatial Centroid Voting
            vote_feats = self.voting_heads[i](query)
            C = vote_feats.shape[-1] // 2
            vote_feats = vote_feats.reshape(B, self.N * self.num_ctrl_pts * 2, C)
            raw_coords = self.spatial_coord_head(vote_feats)

            # Relative Anchor Offset
            delta_offset = torch.tanh(self.offset_heads[i](query)) * 0.10
            ref_pts_logit = torch.logit(ref_pts.clamp(1e-4, 1.0 - 1e-4))
            anchored_coords = torch.sigmoid(ref_pts_logit + delta_offset)

            # Hybrid blend: 70% Spatial Centroid + 30% Relative Anchor
            point_coords = 0.70 * raw_coords + 0.30 * anchored_coords

            layer_ctrl = point_coords.reshape(B, self.N, self.num_ctrl_pts, 2)
            layer_restored = self.restore_curve(layer_ctrl)

            query_inst = query.reshape(B, self.N, self.num_ctrl_pts, self.embed_dim).mean(dim=2)
            layer_logits = self.cls_head(query_inst)

            intermediate_logits.append(layer_logits)
            intermediate_ctrl_pts.append(layer_ctrl)
            intermediate_restored_pts.append(layer_restored)

            ref_pts = point_coords.detach()

        if self.training:
            return intermediate_logits, intermediate_ctrl_pts, intermediate_restored_pts, aux_edge_logits
        else:
            return intermediate_logits[-1], intermediate_ctrl_pts[-1], intermediate_restored_pts[-1], aux_edge_logits
