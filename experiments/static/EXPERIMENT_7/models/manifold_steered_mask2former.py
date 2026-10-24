import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure workspace root is on sys.path
_WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)

from experiments.EXPERIMENT_7.models.manifold_atlas_head import CanonicalAtlasHead
from experiments.EXPERIMENT_7.models.visibility_gated_steering import VisibilityGatedSteering
from experiments.EXPERIMENT_7.models.dense_manifold_head import DenseManifoldHead

try:
    from transformers import Mask2FormerForUniversalSegmentation
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class ManifoldSteeredMask2Former(nn.Module):
    """
    Manifold-Steered Mask2Former for EXPERIMENT_7.
    
    Combines:
      1. Pretrained Swin-Tiny Backbone + MSDeformAttn Pixel Decoder
      2. 11-Query Canonical Atlas Head (2-chart liver manifold skeleton)
      3. Visibility-Gated Steering (VGS) with logarithmic attention silencing
      4. Auxiliary Dense Manifold Head for continuous (u, v) parameterization
      5. 9-Layer Mask2Former Transformer Decoder
    """
    def __init__(self, model_name="facebook/mask2former-swin-tiny-ade-semantic", num_labels=4, embed_dim=256):
        super().__init__()
        if not HAS_TRANSFORMERS:
            raise ImportError("HuggingFace 'transformers' library is required.")

        print(f"📦 Building ManifoldSteeredMask2Former with base: '{model_name}'...")
        self.m2f = Mask2FormerForUniversalSegmentation.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )

        # 1. 11-Query Canonical Atlas Head
        self.atlas_head = CanonicalAtlasHead(
            in_dim=embed_dim,
            embed_dim=embed_dim,
            num_queries=11,
            num_layers=2,
            num_heads=8
        )

        # 2. Visibility-Gated Steering Block (VGS)
        self.query_steering = VisibilityGatedSteering(
            embed_dim=embed_dim,
            num_heads=8,
            init_alpha=0.2
        )

        # 3. Auxiliary Dense Manifold Head
        self.dense_manifold_head = DenseManifoldHead(
            in_channels=embed_dim,
            hidden_dim=64
        )

    def forward_transformer_decoder(self, multi_scale_features, mask_features, steered_query_feat):
        """
        Executes the Mask2Former Transformer Decoder with sample-specific steered queries.
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
            multi_stage_positional_embeddings[-1] = multi_stage_positional_embeddings[-1].permute(2, 0, 1)
            multi_stage_features[-1] = multi_stage_features[-1].permute(2, 0, 1)

        _, batch_size, _ = multi_stage_features[0].shape
        query_embeddings = tm.queries_embedder.weight.unsqueeze(1).repeat(1, batch_size, 1)
        query_features = steered_query_feat.permute(1, 0, 2)  # (100, B, 256)

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

    def forward(self, pixel_values, mask_labels=None, class_labels=None):
        """
        Args:
            pixel_values: Tensor (B, 3, H, W)
            mask_labels: List of binary mask tensors (for training)
            class_labels: List of class index tensors (for training)
        Returns:
            dict containing:
              - 'masks_queries_logits': (B, 100, H/4, W/4)
              - 'class_queries_logits': (B, 100, num_classes + 1)
              - 'pred_coords': (B, 11, 2)
              - 'pred_vis': (B, 11)
              - 'pred_uv': (B, 2, H, W)
              - 'm2f_loss': Hungarian matching loss (if training)
        """
        B = pixel_values.shape[0]

        # 1. Swin-Tiny Backbone + MSDeformAttn Pixel Decoder
        pixel_level_outputs = self.m2f.model.pixel_level_module(pixel_values, output_hidden_states=True)
        multi_scale_features = list(pixel_level_outputs.decoder_hidden_states)
        mask_features = pixel_level_outputs.decoder_last_hidden_state  # (B, 256, H/4, W/4)

        # 2. Extract 11-Query Structural Atlas from stride-16 features
        stride_16_feat = multi_scale_features[1]  # (B, 256, H_16, W_16)
        pred_coords, pred_vis_logits, structural_features = self.atlas_head(stride_16_feat)

        # 3. Visibility-Gated Query Steering (VGS)
        tm = self.m2f.model.transformer_module
        base_query_feat = tm.queries_features.weight.unsqueeze(0).repeat(B, 1, 1)  # (B, 100, 256)
        steered_query_feat = self.query_steering(base_query_feat, structural_features, pred_vis_logits)

        # 4. Dense Manifold Head (Continuous u, v regression)
        pred_uv = self.dense_manifold_head(mask_features, target_size=pixel_values.shape[-2:])  # (B, 2, H, W)

        # 5. Mask2Former Transformer Decoder
        decoder_output = self.forward_transformer_decoder(
            multi_scale_features,
            mask_features,
            steered_query_feat
        )

        class_queries_logits = self.m2f.class_predictor(decoder_output.last_hidden_state)  # (B, 100, num_classes + 1)
        masks_queries_logits = decoder_output.masks_queries_logits[-1]                     # (B, 100, H/4, W/4)

        output = {
            'masks_queries_logits': masks_queries_logits,
            'class_queries_logits': class_queries_logits,
            'pred_coords': pred_coords,
            'pred_vis': pred_vis_logits,
            'pred_uv': pred_uv
        }

        # 6. Compute Hungarian Matching Loss if GT labels provided
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


def load_manifold_steered_model(checkpoint_path=None, model_name="facebook/mask2former-swin-tiny-ade-semantic", device="cpu"):
    """
    Helper to instantiate and load checkpoint weights.
    """
    model = ManifoldSteeredMask2Former(model_name=model_name, num_labels=4)
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(state_dict)
        print(f"✅ Loaded checkpoint from: {checkpoint_path}")
    return model.to(device)
