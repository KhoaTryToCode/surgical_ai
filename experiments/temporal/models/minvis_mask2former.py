"""
MinVIS Mask2Former Model (Experiment 17).
Pure PyTorch implementation based on facebook/mask2former-swin-tiny-ade-semantic.
Extracts query embeddings from the transformer decoder for online MinVIS Hungarian tracking.
"""

import os
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation


class MinVISMask2Former(nn.Module):
    """
    MinVIS Mask2Former:
    - Standard Swin-Tiny + MSDeformAttn Pixel Decoder + Masked Transformer Decoder.
    - Output signature conforming to the unified template:
        Input:  pixel_values: (B, 3, H, W) or (B, T, 3, H, W)
        Output: dict with:
          'sem_seg': (B, 3, H, W) continuous sigmoid probabilities in [0, 1]
          'loss': Hungarian matching loss (if target_masks provided)
          'class_queries_logits': (B, Q, 4)
          'masks_queries_logits': (B, Q, H/4, W/4)
          'query_embeddings': (B, Q, 256) normalized query embeddings for temporal matching
    """
    def __init__(
        self,
        base_checkpoint: str = "facebook/mask2former-swin-tiny-ade-semantic",
        config: Optional[Mask2FormerConfig] = None,
        num_classes: int = 3,
        hidden_dim: int = 256,
        no_object_weight: float = 0.1
    ):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        if config is not None:
            self.base_model = Mask2FormerForUniversalSegmentation(config)
        else:
            try:
                # Fast path: check local cache first without triggering HTTP retries
                self.base_model = Mask2FormerForUniversalSegmentation.from_pretrained(
                    base_checkpoint,
                    num_labels=num_classes,
                    ignore_mismatched_sizes=True,
                    output_hidden_states=True,
                    local_files_only=True
                )
            except Exception:
                try:
                    self.base_model = Mask2FormerForUniversalSegmentation.from_pretrained(
                        base_checkpoint,
                        num_labels=num_classes,
                        ignore_mismatched_sizes=True,
                        output_hidden_states=True
                    )
                except Exception as e:
                    print(f"Initializing from offline config fallback ({e})...")
                    cfg = Mask2FormerConfig(
                        num_queries=100,
                        num_labels=num_classes,
                        output_hidden_states=True
                    )
                    self.base_model = Mask2FormerForUniversalSegmentation(cfg)

        # Ensure output_hidden_states is True
        self.base_model.config.output_hidden_states = True

        # Layer normalization for query embeddings matching MinVIS decoder_norm
        self.query_norm = nn.LayerNorm(hidden_dim)

    def prepare_targets(
        self,
        target_masks: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Converts (B, 3, H, W) binary masks into HuggingFace format:
        mask_labels: List[Tensor(K, H, W)]
        class_labels: List[Tensor(K,)]
        """
        bs, num_cls, h, w = target_masks.shape
        mask_labels = []
        class_labels = []

        for b in range(bs):
            b_masks = []
            b_classes = []
            for c in range(num_cls):
                m = target_masks[b, c]
                if m.sum() > 0:
                    b_masks.append(m)
                    b_classes.append(c)

            if len(b_masks) == 0:
                # Dummy background to prevent Hungarian matcher crash
                b_masks.append(torch.zeros((h, w), device=target_masks.device))
                b_classes.append(0)

            mask_labels.append(torch.stack(b_masks, dim=0))
            class_labels.append(torch.tensor(b_classes, dtype=torch.long, device=target_masks.device))

        return mask_labels, class_labels

    def forward(
        self,
        pixel_values: torch.Tensor,
        target_masks: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pixel_values: (B, 3, H, W) or (B, T, 3, H, W)
            target_masks: Optional (B, 3, H, W) float32 ground-truth masks
        """
        orig_shape = pixel_values.shape
        is_video = (pixel_values.ndim == 5)

        if is_video:
            b, t, c, h, w = orig_shape
            x = pixel_values.view(b * t, c, h, w)
        else:
            b, c, h, w = orig_shape
            t = 1
            x = pixel_values

        loss = None
        if target_masks is not None and self.training:
            # target_masks: (B, 3, H, W)
            mask_labels, class_labels = self.prepare_targets(target_masks)
            outputs = self.base_model(
                pixel_values=x,
                mask_labels=mask_labels,
                class_labels=class_labels,
                output_hidden_states=True
            )
            loss = outputs.loss
        else:
            outputs = self.base_model(
                pixel_values=x,
                output_hidden_states=True
            )

        class_logits = outputs.class_queries_logits  # (BT, Q, K+1)
        mask_logits = outputs.masks_queries_logits    # (BT, Q, H/4, W/4)

        # Extract query embeddings: ensure shape is (BT, Q, C)
        raw_hidden = outputs.transformer_decoder_last_hidden_state
        if raw_hidden.shape[0] == 100 and raw_hidden.shape[1] != 100:
            query_embds = raw_hidden.permute(1, 0, 2)
        else:
            query_embds = raw_hidden
        query_embds = self.query_norm(query_embds)

        # Compute continuous semantic segmentation probability map
        mask_probs = F.interpolate(
            mask_logits.sigmoid(),
            size=(h, w),
            mode='bilinear',
            align_corners=False
        )  # (BT, Q, H, W)

        class_probs = class_logits.softmax(dim=-1)[..., :self.num_classes]  # (BT, Q, 3)
        sem_seg = torch.einsum('bqc, bqhw -> bchw', class_probs, mask_probs)  # (BT, 3, H, W)

        if is_video:
            sem_seg = sem_seg.view(b, t, self.num_classes, h, w)
            class_logits = class_logits.view(b, t, -1, class_logits.shape[-1])
            query_embds = query_embds.view(b, t, -1, query_embds.shape[-1])

        res = {
            'sem_seg': sem_seg,
            'class_queries_logits': class_logits,
            'masks_queries_logits': mask_logits,
            'query_embeddings': query_embds,
        }
        if loss is not None:
            res['loss'] = loss

        return res
