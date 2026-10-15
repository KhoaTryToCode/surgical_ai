"""
Depth-Geometric Junction-Steered Mask2Former Model for EXPERIMENT_10.

Combines:
  1. Pretrained Swin-Tiny Backbone (strictly 3-channel RGB, preserving 100% ImageNet-1K weights)
  2. Multi-Scale Deformable Attention Pixel Decoder (strides 4, 8, 16, 32)
  3. Lightweight 4-Stage DepthGeometryEncoder (downsampling monocular depth to stride 16)
  4. Gated Stride-16 RGB-D Feature Fusion Layer
  5. Continuous 4-Query Junction Anchor Head (from EXPERIMENT_5)
  6. Dynamic Junction Cross-Attention Query Steering Block
  7. 9-Layer Mask2Former Transformer Decoder
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure workspace root is on sys.path
_WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)

from experiments.EXPERIMENT_10.models.depth_geometry_encoder import DepthGeometryEncoder
from experiments.EXPERIMENT_10.models.junction_head import JunctionAnchorHead

try:
    from transformers import Mask2FormerForUniversalSegmentation
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class JunctionQuerySteering(nn.Module):
    """
    Dynamic Query Steering Block:
    Allows 100 Mask2Former queries to cross-attend to the 4 depth-informed continuous junction tokens.
    """
    def __init__(self, embed_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.proj_q = nn.Linear(embed_dim, embed_dim)
        self.proj_k = nn.Linear(embed_dim, embed_dim)
        self.proj_v = nn.Linear(embed_dim, embed_dim)
        
        # Learnable gating scalar initialized gently to 0.1 for a smooth gradient highway
        self.gate = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, junction_features, junction_coords=None):
        """
        Args:
            queries (Tensor): (B, 100, C) base query features
            junction_features (Tensor): (B, 4, C) continuous junction anchor tokens
            junction_coords (Tensor, optional): (B, 4, 2) normalized coordinates
        Returns:
            steered_queries (Tensor): (B, 100, C)
            attn_weights (Tensor): (B, 100, 4)
        """
        q = self.proj_q(queries)
        k = self.proj_k(junction_features)
        v = self.proj_v(junction_features)
        
        # Cross-attention: 100 queries attend to 4 continuous junction tokens
        delta_q, attn_weights = self.cross_attn(query=q, key=k, value=v)
        
        # Gated residual update: Q_steered = Norm(Q + gate * delta_Q)
        steered_queries = self.norm(queries + self.gate * self.dropout(delta_q))
        
        return steered_queries, attn_weights


class DepthJunctionSteeredMask2Former(nn.Module):
    """
    Integrated Depth-Geometric Junction-Steered Mask2Former.
    """
    def __init__(self, model_name="facebook/mask2former-swin-tiny-ade-semantic", num_labels=4):
        super().__init__()
        if not HAS_TRANSFORMERS:
            raise ImportError("HuggingFace 'transformers' library is required. Please install it in your PyTorch environment.")
            
        print(f"📦 Building DepthJunctionSteeredMask2Former (EXP_10) with base: '{model_name}'...")
        try:
            self.m2f = Mask2FormerForUniversalSegmentation.from_pretrained(
                model_name,
                num_labels=num_labels,
                ignore_mismatched_sizes=True,
                local_files_only=True
            )
        except Exception:
            self.m2f = Mask2FormerForUniversalSegmentation.from_pretrained(
                model_name,
                num_labels=num_labels,
                ignore_mismatched_sizes=True
            )
        
        embed_dim = 256
        
        # 1. Depth Geometry Encoder (1024x1024 -> 64x64 at stride 16)
        self.depth_encoder = DepthGeometryEncoder(in_channels=1, out_channels=embed_dim)
        
        # 2. Stride-16 RGB-D Gated Fusion Layer
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1, bias=False),
            nn.GroupNorm(32, embed_dim),
            nn.GELU()
        )
        self.fusion_gate = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        
        # 3. Continuous Junction Anchor Head (4 queries from EXPERIMENT_5)
        self.junction_head = JunctionAnchorHead(embed_dim=embed_dim, num_layers=2, num_heads=8)
        
        # 4. Query Steering Block
        self.query_steering = JunctionQuerySteering(embed_dim=embed_dim, num_heads=8)
        
    def forward_transformer_decoder(self, multi_scale_features, mask_features, steered_query_feat):
        """
        Runs the Mask2Former Transformer Decoder directly with sample-specific steered queries.
        """
        tm = self.m2f.model.transformer_module
        multi_stage_features = []
        multi_stage_positional_embeddings = []
        size_list = []

        for i in range(tm.num_feature_levels):
            size_list.append(multi_scale_features[i].shape[-2:])
            multi_stage_positional_embeddings.append(
                tm.position_embedder(
                    multi_scale_features[i].shape, multi_scale_features[i].device, multi_scale_features[i].dtype, None
                ).flatten(2)
            )
            multi_stage_features.append(
                tm.input_projections[i](multi_scale_features[i]).flatten(2)
                + tm.level_embed.weight[i][None, :, None]
            )
            # Permute NxCxHW to HWxNxC
            multi_stage_positional_embeddings[-1] = multi_stage_positional_embeddings[-1].permute(2, 0, 1)
            multi_stage_features[-1] = multi_stage_features[-1].permute(2, 0, 1)

        _, batch_size, _ = multi_stage_features[0].shape

        # [num_queries, batch_size, num_channels]
        query_embeddings = tm.queries_embedder.weight.unsqueeze(1).repeat(1, batch_size, 1)
        query_features = steered_query_feat.permute(1, 0, 2)

        decoder_output = tm.decoder(
            inputs_embeds=query_features,
            multi_stage_positional_embeddings=multi_stage_positional_embeddings,
            pixel_embeddings=mask_features,
            encoder_hidden_states=multi_stage_features,
            query_position_embeddings=query_embeddings,
            feature_size_list=size_list,
            output_hidden_states=False,
            output_attentions=False,
            return_dict=True,
        )
        return decoder_output

    def forward(self, pixel_values, depth_map, mask_labels=None, class_labels=None):
        """
        Args:
            pixel_values (Tensor): (B, 3, H, W) RGB normalized with ImageNet stats
            depth_map (Tensor): (B, 1, H, W) Monocular relative depth map from Depth Anything v2
            mask_labels (list of Tensors, optional): GT binary masks per image for training
            class_labels (list of Tensors, optional): GT class IDs per image for training
        Returns:
            dict containing:
                - 'masks_queries_logits': (B, 100, H/4, W/4)
                - 'class_queries_logits': (B, 100, num_classes + 1)
                - 'pred_junction_coords': (B, 4, 2) in [0, 1]^2
                - 'pred_junction_vis': (B, 4) unnormalized logits for auxiliary BCE loss
                - 'pred_junction_vis_probs': (B, 4) sigmoid probabilities in [0, 1]
                - 'junction_attn_weights': (B, 100, 4)
                - 'm2f_loss': standard Hungarian loss (if training)
        """
        B = pixel_values.shape[0]
        
        # 1. Swin-Tiny Backbone + MSDeformAttn Pixel Decoder (Standard 3-ch RGB)
        pixel_level_outputs = self.m2f.model.pixel_level_module(pixel_values, output_hidden_states=True)
        multi_scale_features = list(pixel_level_outputs.decoder_hidden_states)
        mask_features = pixel_level_outputs.decoder_last_hidden_state # (B, 256, H/4, W/4)
        
        # Stride-16 RGB feature map: (B, 256, H/16, W/16)
        f_rgb_16 = multi_scale_features[1]
        
        # 2. Extract 3D Surface Geometry from Depth Map
        f_depth_16 = self.depth_encoder(depth_map) # (B, 256, H/16, W/16)
        
        # 3. Gated Residual RGB-D Fusion at Stride 16
        rgbd_cat = torch.cat([f_rgb_16, f_depth_16], dim=1) # (B, 512, H/16, W/16)
        f_fused = f_rgb_16 + self.fusion_gate * self.fusion_conv(rgbd_cat)
        
        # 4. Extract 4 Continuous Biological Junctions over Fused Features
        pred_j_coords, pred_j_vis, pred_j_vis_probs, j_features = self.junction_head(f_fused)
        
        # 5. Dynamic Query Steering:
        tm = self.m2f.model.transformer_module
        base_query_feat = tm.queries_features.weight.unsqueeze(0).repeat(B, 1, 1) # (B, 100, 256)
        
        # 100 Mask2Former queries attend to the 4 continuous depth-steered junction tokens
        steered_query_feat, j_attn_weights = self.query_steering(base_query_feat, j_features, pred_j_coords)
        
        # 6. Forward through Mask2Former Transformer Decoder with steered queries
        decoder_output = self.forward_transformer_decoder(
            multi_scale_features,
            mask_features,
            steered_query_feat
        )
        
        # Extract class & mask logits
        class_queries_logits = self.m2f.class_predictor(decoder_output.last_hidden_state) # (B, 100, num_classes + 1)
        masks_queries_logits = decoder_output.masks_queries_logits[-1]                    # (B, 100, H/4, W/4)
        
        output = {
            'masks_queries_logits': masks_queries_logits,
            'class_queries_logits': class_queries_logits,
            'pred_junction_coords': pred_j_coords,
            'pred_junction_vis': pred_j_vis,
            'pred_junction_vis_probs': pred_j_vis_probs,
            'junction_attn_weights': j_attn_weights
        }
        
        # 7. Compute Hungarian Matching Loss if GT labels are provided
        if mask_labels is not None and class_labels is not None:
            loss_dict = self.m2f.criterion(
                masks_queries_logits=masks_queries_logits,
                class_queries_logits=class_queries_logits,
                mask_labels=mask_labels,
                class_labels=class_labels
            )
            weight_dict = self.m2f.criterion.weight_dict
            m2f_loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            output['m2f_loss'] = m2f_loss
            output['m2f_loss_dict'] = loss_dict
            
        return output


def load_depth_junction_steered_model(checkpoint_path=None, model_name="facebook/mask2former-swin-tiny-ade-semantic", device="cpu"):
    """
    Helper to instantiate model and load checkpoint weights with key compatibility.
    """
    model = DepthJunctionSteeredMask2Former(model_name=model_name, num_labels=4)
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"✅ Loaded checkpoint from: {checkpoint_path} (missing: {len(missing)}, unexpected: {len(unexpected)})")
    return model.to(device)
