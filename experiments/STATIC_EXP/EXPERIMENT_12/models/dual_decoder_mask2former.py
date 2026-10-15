"""
Dual-Decoder Mask2Former Model for EXPERIMENT_12.
Multi-task architecture combining:
  1. Shared Swin-Tiny Backbone + MSDeformAttn Pixel Decoder
  2. Scene Decoder (CholecSeg8k): 100 queries -> 9-layer Transformer Decoder -> 13 classes (num_labels=13)
  3. Landmark Decoder (L3D): 4 continuous Junction Queries + JunctionQuerySteering -> 100 queries -> 9-layer Transformer Decoder -> 3 landmark classes (num_labels=4)
"""
import os
import sys
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure workspace root is on sys.path
_WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)

from experiments.EXPERIMENT_12.models.junction_head import JunctionAnchorHead

try:
    from transformers import Mask2FormerForUniversalSegmentation
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class JunctionQuerySteering(nn.Module):
    """
    Dynamic Query Steering Block:
    Allows 100 Mask2Former queries to cross-attend to the 4 junction anchor vectors.
    """
    def __init__(self, embed_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.proj_q = nn.Linear(embed_dim, embed_dim)
        self.proj_k = nn.Linear(embed_dim, embed_dim)
        self.proj_v = nn.Linear(embed_dim, embed_dim)
        
        # Learnable gating scalar initialized to 0.1 for smooth gradient highway
        self.gate = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, junction_features, junction_coords=None):
        """
        Args:
            queries (Tensor): (B, 100, C) base query features
            junction_features (Tensor): (B, 4, C) junction anchor tokens
            junction_coords (Tensor, optional): (B, 4, 2) normalized coordinates
        Returns:
            steered_queries (Tensor): (B, 100, C)
            attn_weights (Tensor): (B, 100, 4)
        """
        q = self.proj_q(queries)
        k = self.proj_k(junction_features)
        v = self.proj_v(junction_features)
        
        delta_q, attn_weights = self.cross_attn(query=q, key=k, value=v)
        steered_queries = self.norm(queries + self.gate * self.dropout(delta_q))
        return steered_queries, attn_weights


def _load_hf_mask2former(model_name, num_labels):
    """
    Safely loads HuggingFace Mask2Former checking local offline cache first.
    """
    try:
        return Mask2FormerForUniversalSegmentation.from_pretrained(
            model_name,
            num_labels=num_labels,
            local_files_only=True,
            ignore_mismatched_sizes=True
        )
    except Exception:
        return Mask2FormerForUniversalSegmentation.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )


class DualDecoderMask2Former(nn.Module):
    """
    Dual-Decoder Mask2Former for Multi-Dataset Surgical Foundation Learning:
      - Shared: Swin-Tiny Backbone + MSDeformAttn Pixel Decoder
      - Decoder 1 (CholecSeg8k): 100 queries, 9-layer Transformer Decoder for 13 scene classes
      - Decoder 2 (L3D): 4 junction queries + 100 steered queries for 3 landmark classes + coordinates
    """
    def __init__(self, model_name="facebook/mask2former-swin-tiny-ade-semantic",
                 num_landmark_labels=4, num_scene_labels=13, embed_dim=256):
        super().__init__()
        if not HAS_TRANSFORMERS:
            raise ImportError("HuggingFace 'transformers' library is required.")

        print(f"📦 Building DualDecoderMask2Former from '{model_name}'...")
        print(f"   Landmark Decoder: {num_landmark_labels} classes (L3D)")
        print(f"   Scene Decoder   : {num_scene_labels} classes (CholecSeg8k)")

        # 1. Base model for Landmark Decoder (provides shared encoder & landmark decoder)
        self.m2f_landmark = _load_hf_mask2former(model_name, num_labels=num_landmark_labels)
        
        # 2. Base model for Scene Decoder (provides independent scene transformer decoder)
        m2f_scene_full = _load_hf_mask2former(model_name, num_labels=num_scene_labels)
        
        # Extract Scene Decoder components
        self.scene_transformer = m2f_scene_full.model.transformer_module
        self.scene_class_predictor = m2f_scene_full.class_predictor
        self.scene_criterion = m2f_scene_full.criterion
        
        # Free redundant pixel level module from the scene model to eliminate duplicate parameters
        del m2f_scene_full
        
        # 3. Landmark Specialization Modules (EXP_11 identical design)
        self.junction_head = JunctionAnchorHead(embed_dim=embed_dim, num_layers=2, num_heads=8)
        self.query_steering = JunctionQuerySteering(embed_dim=embed_dim, num_heads=8)
        
    @property
    def shared_pixel_level(self):
        return self.m2f_landmark.model.pixel_level_module

    @property
    def landmark_transformer(self):
        return self.m2f_landmark.model.transformer_module

    @property
    def landmark_class_predictor(self):
        return self.m2f_landmark.class_predictor

    @property
    def landmark_criterion(self):
        return self.m2f_landmark.criterion

    def extract_shared_features(self, pixel_values):
        """
        Runs the shared Swin-Tiny Backbone + MSDeformAttn Pixel Decoder.
        Returns:
            multi_scale_features: list of 3 tensors [stride 32, stride 16, stride 8]
            mask_features: tensor (B, 256, H/4, W/4)
        """
        pixel_level_outputs = self.shared_pixel_level(pixel_values, output_hidden_states=True)
        multi_scale_features = list(pixel_level_outputs.decoder_hidden_states)
        mask_features = pixel_level_outputs.decoder_last_hidden_state
        return multi_scale_features, mask_features

    def _run_transformer_decoder(self, tm, multi_scale_features, mask_features, query_features):
        """
        Executes a Mask2Former Transformer Decoder module given query features and feature pyramid.
        Args:
            tm: Mask2FormerTransformerModule
            multi_scale_features: list of 3 tensors [stride 32, stride 16, stride 8]
            mask_features: (B, 256, H/4, W/4)
            query_features: (B, 100, 256)
        """
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
            multi_stage_positional_embeddings[-1] = multi_stage_positional_embeddings[-1].permute(2, 0, 1)
            multi_stage_features[-1] = multi_stage_features[-1].permute(2, 0, 1)

        _, batch_size, _ = multi_stage_features[0].shape

        query_embeddings = tm.queries_embedder.weight.unsqueeze(1).repeat(1, batch_size, 1)
        # Permute (B, 100, 256) -> (100, B, 256)
        q_feat = query_features.permute(1, 0, 2)

        decoder_output = tm.decoder(
            inputs_embeds=q_feat,
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

    def forward_landmark(self, pixel_values, mask_labels=None, class_labels=None):
        """
        Forward pass through Shared Encoder + Landmark Decoder (L3D).
        """
        B = pixel_values.shape[0]
        multi_scale_features, mask_features = self.extract_shared_features(pixel_values)
        
        # 1. Continuous Junction Anchor Head on stride-16 features
        stride_16_feat = multi_scale_features[1]
        pred_j_coords, pred_j_vis, j_features = self.junction_head(stride_16_feat)
        
        # 2. Dynamic Query Steering: 100 queries attend to 4 junctions
        tm = self.landmark_transformer
        base_query_feat = tm.queries_features.weight.unsqueeze(0).repeat(B, 1, 1)
        steered_query_feat, j_attn_weights = self.query_steering(base_query_feat, j_features, pred_j_coords)
        
        # 3. Transformer Decoder
        decoder_output = self._run_transformer_decoder(
            tm, multi_scale_features, mask_features, steered_query_feat
        )
        
        class_queries_logits = self.landmark_class_predictor(decoder_output.last_hidden_state)
        masks_queries_logits = decoder_output.masks_queries_logits[-1]
        
        output = {
            'masks_queries_logits': masks_queries_logits,
            'class_queries_logits': class_queries_logits,
            'pred_junction_coords': pred_j_coords,
            'pred_junction_vis': pred_j_vis,
            'junction_attn_weights': j_attn_weights
        }
        
        # Hungarian matching loss if GT provided
        if mask_labels is not None and class_labels is not None:
            loss_dict = self.landmark_criterion(
                masks_queries_logits=masks_queries_logits,
                class_queries_logits=class_queries_logits,
                mask_labels=mask_labels,
                class_labels=class_labels
            )
            weight_dict = self.landmark_criterion.weight_dict
            m2f_loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            output['m2f_loss'] = m2f_loss
            output['m2f_loss_dict'] = loss_dict
            
        return output

    def forward_scene(self, pixel_values, mask_labels=None, class_labels=None):
        """
        Forward pass through Shared Encoder + Scene Decoder (CholecSeg8k).
        """
        B = pixel_values.shape[0]
        multi_scale_features, mask_features = self.extract_shared_features(pixel_values)
        
        # Standard unconditioned queries for scene segmentation
        tm = self.scene_transformer
        base_query_feat = tm.queries_features.weight.unsqueeze(0).repeat(B, 1, 1)
        
        decoder_output = self._run_transformer_decoder(
            tm, multi_scale_features, mask_features, base_query_feat
        )
        
        class_queries_logits = self.scene_class_predictor(decoder_output.last_hidden_state)
        masks_queries_logits = decoder_output.masks_queries_logits[-1]
        
        output = {
            'masks_queries_logits': masks_queries_logits,
            'class_queries_logits': class_queries_logits
        }
        
        # Hungarian matching loss if GT provided
        if mask_labels is not None and class_labels is not None:
            loss_dict = self.scene_criterion(
                masks_queries_logits=masks_queries_logits,
                class_queries_logits=class_queries_logits,
                mask_labels=mask_labels,
                class_labels=class_labels
            )
            weight_dict = self.scene_criterion.weight_dict
            scene_loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            output['scene_loss'] = scene_loss
            output['scene_loss_dict'] = loss_dict
            
        return output

    def forward(self, pixel_values, domain='landmark', mask_labels=None, class_labels=None):
        """
        Unified dispatch: routes to either landmark decoder or scene decoder.
        """
        if domain == 'landmark':
            return self.forward_landmark(pixel_values, mask_labels=mask_labels, class_labels=class_labels)
        elif domain == 'scene' or domain == 'cholec':
            return self.forward_scene(pixel_values, mask_labels=mask_labels, class_labels=class_labels)
        else:
            raise ValueError(f"Unknown domain: '{domain}'. Expected 'landmark' or 'scene'.")


def load_dual_decoder_model(checkpoint_path=None, model_name="facebook/mask2former-swin-tiny-ade-semantic", device="cpu"):
    """
    Instantiates and loads DualDecoderMask2Former checkpoint weights.
    """
    model = DualDecoderMask2Former(model_name=model_name, num_landmark_labels=4, num_scene_labels=13)
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(state_dict)
        print(f"✅ Loaded checkpoint from: {checkpoint_path}")
    return model.to(device)
