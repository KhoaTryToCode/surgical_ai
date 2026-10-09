"""
Temporal-Augmented Mask2Former Model for EXPERIMENT_15.
Base: facebook/mask2former-swin-tiny-ade-semantic (pure vanilla base).
Augmentation: SpatiotemporalQueryBlock performing cross-attention across T=3 chronological clip queries.
"""
import os
import sys
import torch
import torch.nn as nn
from pathlib import Path

# Ensure workspace root is on sys.path
_WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)

from experiments.EXPERIMENT_15_temporal_augmented_mask2former.models.temporal_query_block import SpatiotemporalQueryBlock

try:
    from transformers import Mask2FormerForUniversalSegmentation
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

class TemporalMask2Former(nn.Module):
    def __init__(self, model_name="facebook/mask2former-swin-tiny-ade-semantic", num_labels=4, clip_len=3):
        super().__init__()
        if not HAS_TRANSFORMERS:
            raise ImportError("HuggingFace 'transformers' library is required.")
            
        print(f"📦 Building TemporalMask2Former with base: '{model_name}', clip_len={clip_len}...")
        is_offline = (os.environ.get("HF_HUB_OFFLINE", "0") == "1")
        try:
            self.m2f = Mask2FormerForUniversalSegmentation.from_pretrained(
                model_name,
                num_labels=num_labels,
                ignore_mismatched_sizes=True,
                local_files_only=is_offline
            )
        except Exception:
            self.m2f = Mask2FormerForUniversalSegmentation.from_pretrained(
                model_name,
                num_labels=num_labels,
                ignore_mismatched_sizes=True,
                local_files_only=True
            )
            
        self.clip_len = clip_len
        self.embed_dim = 256
        
        # Temporal Query Interaction Block
        self.temporal_query_block = SpatiotemporalQueryBlock(
            embed_dim=self.embed_dim,
            num_heads=8,
            dropout=0.1,
            gamma_init=0.05
        )

    def forward_transformer_decoder(self, multi_scale_features, mask_features, steered_query_feat):
        """
        Executes Mask2Former Transformer Decoder with temporally steered queries.
        steered_query_feat shape: (B_total, 100, 256)
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

        # Positional queries: (100, batch_size, 256)
        query_embeddings = tm.queries_embedder.weight.unsqueeze(1).repeat(1, batch_size, 1)
        # Content queries: (100, batch_size, 256)
        query_features = steered_query_feat.permute(1, 0, 2)

        decoder_output = tm.decoder(
            inputs_embeds=query_features,
            multi_stage_positional_embeddings=multi_stage_positional_embeddings,
            pixel_embeddings=mask_features,
            encoder_hidden_states=multi_stage_features,
            query_position_embeddings=query_position_embeddings,
            feature_size_list=size_list,
            output_hidden_states=False,
            output_attentions=False,
            return_dict=True,
        )
        return decoder_output

    def forward(self, pixel_values_clip, mask_labels_clip=None, class_labels_clip=None):
        """
        Args:
            pixel_values_clip: (B, T, 3, H, W) tensor of video frames
            mask_labels_clip: optional list of length B, where each element is a list of length T of GT masks
            class_labels_clip: optional list of length B, where each element is a list of length T of GT class IDs
        Returns:
            dict containing:
                - 'masks_queries_logits': (B, T, 100, H/4, W/4)
                - 'class_queries_logits': (B, T, 100, num_classes + 1)
                - 'loss': total clip loss (if labels provided)
                - 'gamma': current value of temporal gating scalar
        """
        B, T, C, H, W = pixel_values_clip.shape
        B_total = B * T
        
        # 1. Sequential spatial encoding per frame (drastically cuts peak VRAM from 14 GB to ~5 GB)
        multi_scale_features_list = []
        mask_features_list = []
        for t in range(T):
            p_out = self.m2f.model.pixel_level_module(pixel_values_clip[:, t], output_hidden_states=True)
            multi_scale_features_list.append(list(p_out.decoder_hidden_states))
            mask_features_list.append(p_out.decoder_last_hidden_state)
            
        # 2. Extract base query features: (B*T, 100, 256)
        tm = self.m2f.model.transformer_module
        base_queries = tm.queries_features.weight.unsqueeze(0).repeat(B_total, 1, 1)
        
        # Reshape into clip tensor: (B, T, 100, 256)
        base_queries_clip = base_queries.view(B, T, 100, self.embed_dim)
        
        # 3. Apply Spatiotemporal Query Cross-Attention Block
        steered_queries_clip = self.temporal_query_block(base_queries_clip) # (B, T, 100, 256)
        
        # 4. Pass through Mask2Former Transformer Decoder sequentially per frame
        # (Drastically cuts peak VRAM from 14 GB to ~5.5 GB by avoiding 3 parallel 9-layer decoder graphs)
        masks_queries_logits_list = []
        class_queries_logits_list = []
        
        for t in range(T):
            q_t = steered_queries_clip[:, t] # (B, 100, 256)
            dec_out = self.forward_transformer_decoder(
                multi_scale_features_list[t],
                mask_features_list[t],
                q_t
            )
            c_logits_t = self.m2f.class_predictor(dec_out.last_hidden_state) # (B, 100, num_classes + 1)
            m_logits_t = dec_out.masks_queries_logits[-1]                    # (B, 100, H/4, W/4)
            
            masks_queries_logits_list.append(m_logits_t)
            class_queries_logits_list.append(c_logits_t)
            
        masks_queries_logits = torch.stack(masks_queries_logits_list, dim=1) # (B, T, 100, H/4, W/4)
        class_queries_logits = torch.stack(class_queries_logits_list, dim=1) # (B, T, 100, num_classes + 1)
        
        output = {
            'masks_queries_logits': masks_queries_logits,
            'class_queries_logits': class_queries_logits,
            'gamma': self.temporal_query_block.gamma.item()
        }
        
        # 5. Compute Mask2Former Hungarian Loss across all clip frames
        if mask_labels_clip is not None and class_labels_clip is not None:
            mask_labels_flat = []
            class_labels_flat = []
            for b in range(B):
                for t in range(T):
                    mask_labels_flat.append(mask_labels_clip[b][t])
                    class_labels_flat.append(class_labels_clip[b][t])
                    
            H_m, W_m = masks_queries_logits.shape[-2:]
            masks_logits_flat = masks_queries_logits.view(B_total, 100, H_m, W_m)
            class_logits_flat = class_queries_logits.view(B_total, 100, -1)
            
            loss_dict = self.m2f.criterion(
                masks_queries_logits=masks_logits_flat,
                class_queries_logits=class_logits_flat,
                mask_labels=mask_labels_flat,
                class_labels=class_labels_flat
            )
            weight_dict = self.m2f.criterion.weight_dict
            total_loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            output['loss'] = total_loss
            output['loss_dict'] = loss_dict
            
        return output

def load_temporal_mask2former(checkpoint_path=None, model_name="facebook/mask2former-swin-tiny-ade-semantic", clip_len=3, device="cpu"):
    """
    Helper to instantiate TemporalMask2Former and load checkpoint weights.
    """
    model = TemporalMask2Former(model_name=model_name, num_labels=4, clip_len=clip_len)
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(state_dict)
        print(f"✅ Loaded checkpoint from: {checkpoint_path}")
    return model.to(device)
