"""
Dual-Decoder Mask2Former Model for EXPERIMENT_12.
Multi-task architecture combining:
  1. Shared Pretrained Swin-Tiny Backbone + MSDeformAttn Pixel Decoder
  2. Scene Decoder (CholecSeg8k): 100 queries -> 9-layer Transformer Decoder -> 13 Surgical Scene Classes
  3. Landmark Decoder (L3D): 4 continuous Junction Queries + JunctionQuerySteering -> 100 queries -> 9-layer Transformer Decoder -> 3 Anatomical Landmark Lines (Ridge, Silhouette, Falciform) + (x, y, v)
"""
from .junction_head import JunctionAnchorHead
from .dual_decoder_mask2former import DualDecoderMask2Former, load_dual_decoder_model
from .losses import MultiTaskDualLoss

__all__ = [
    'JunctionAnchorHead',
    'DualDecoderMask2Former',
    'load_dual_decoder_model',
    'MultiTaskDualLoss'
]
