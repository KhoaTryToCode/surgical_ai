from __future__ import annotations

from typing import Optional, List

import torch
from torch import nn

from torch.nn import functional as F

try:
    import timm
except Exception:  # pragma: no cover
    timm = None


class VisualEncoder(nn.Module):
    def __init__(
        self,
        backbone: str = 'resnet50',
        pretrained: bool = True,
        feature_dim: int = 256,
        num_frames: int = 1,
        temporal_layers: int = 2,
        aux_channels: int = 0,
        temporal_heads: int = 4,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.num_frames = num_frames
        self.aux_channels = max(0, aux_channels)
        available_timm = timm.list_models() if timm is not None else []
        self.using_timm = timm is not None and (backbone in available_timm) and ('swin' in backbone.lower())
        self.from_swin_t = False
        if self.using_timm:
            self.encoder = timm.create_model(backbone, pretrained=pretrained, features_only=False)
            if hasattr(self.encoder, 'head') and hasattr(self.encoder.head, 'in_features'):
                out_dim = self.encoder.head.in_features
            else:
                out_dim = self.encoder.num_features
        else:
            try:
                from torchvision.models import swin_t
                model = swin_t(weights='DEFAULT' if pretrained else None)
                self.encoder = model.features
                out_dim = 768
                self.from_swin_t = True
            except Exception:
                from torchvision import models
                cnn = getattr(models, backbone)(pretrained=pretrained)
                body = nn.Sequential(*list(cnn.children())[:-2])
                self.encoder = body
                out_dim = 2048
        input_c = 3
        if self.aux_channels > 0:
            self.input_adapter = nn.Sequential(
                nn.Conv2d(input_c + self.aux_channels, input_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(input_c),
                nn.ReLU(inplace=True),
            )
        else:
            self.input_adapter = nn.Identity()
        self.proj = nn.Linear(out_dim, feature_dim)
        self.norm = nn.LayerNorm(feature_dim)
        if temporal_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=temporal_heads, batch_first=True)
            self.temporal = nn.TransformerEncoder(encoder_layer, num_layers=temporal_layers)
        else:
            self.temporal = None
        self.temporal_cls = nn.Parameter(torch.zeros(1, 1, feature_dim))
        nn.init.trunc_normal_(self.temporal_cls, std=0.02)
        self.contrast_pool = nn.AdaptiveAvgPool2d((1, 1))

    def _build_aux(self, render_depths: Optional[torch.Tensor], render_masks: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if self.aux_channels <= 0:
            return None
        aux_list: List[torch.Tensor] = []
        if render_depths is not None:
            aux_list.append(render_depths)
        if render_masks is not None:
            aux_list.append(render_masks)
        if not aux_list:
            return None
        aux = torch.cat(aux_list, dim=2)
        return aux

    def forward(self, images: torch.Tensor, render_depths: Optional[torch.Tensor] = None, render_masks: Optional[torch.Tensor] = None) -> dict:
        """images: (B, T, C, H, W)."""
        B, T, C, H, W = images.shape
        aux = self._build_aux(render_depths, render_masks)
        if aux is not None and self.aux_channels > 0:
            inputs = torch.cat([images, aux], dim=2)
        else:
            inputs = images
        flat = inputs.view(B * T, inputs.shape[2], H, W)
        flat = self.input_adapter(flat)
        if self.using_timm:
            flat = torch.nn.functional.interpolate(flat, size=(224, 224), mode='bilinear', align_corners=False)
            features = self.encoder.forward_features(flat)
            tokens = features['x'] if isinstance(features, dict) else features
            if tokens.dim() == 4:
                tokens = tokens.view(B * T, -1, tokens.shape[-1])
            cls_feature = tokens.mean(dim=1, keepdim=True)
            fused = torch.cat([cls_feature, tokens], dim=1)
            fused = self.proj(fused)
            fused = self.norm(fused)
            feats = fused.view(B, T, fused.shape[1], fused.shape[2])
        else:
            feats = self.encoder(flat)
            if hasattr(self, 'from_swin_t') and self.from_swin_t and feats.dim() == 4:
                feats = feats.view(B * T, -1, feats.shape[-1])
            else:
                feats = feats.flatten(2).transpose(1, 2)
            feats = self.proj(feats)
            feats = self.norm(feats)
            feats = feats.view(B, T, feats.shape[1], feats.shape[2])
        frame_tokens = feats
        per_frame_global = frame_tokens.mean(dim=2)
        if self.temporal is not None:
            cls_token = self.temporal_cls.expand(B, -1, -1)
            temporal_in = torch.cat([cls_token, per_frame_global], dim=1)
            temporal_out = self.temporal(temporal_in)
            global_features = temporal_out[:, 0]
            temporal_frames = temporal_out[:, 1:]
        else:
            global_features = per_frame_global.mean(dim=1)
            temporal_frames = per_frame_global
        render_contrast_feat = None
        if render_depths is not None:
            depth = render_depths
            depth = F.interpolate(depth.view(B * T, *depth.shape[2:]), size=(H, W), mode='bilinear', align_corners=False)
            depth = depth.view(B, T, 1, H, W)
            norm_depth = (depth - depth.mean(dim=(-1, -2, -3), keepdim=True)) / (depth.std(dim=(-1, -2, -3), keepdim=True) + 1e-6)
            gray = images.mean(dim=2, keepdim=True)
            contrast = torch.abs(gray - norm_depth)
            if render_masks is not None:
                mask = render_masks
                mask = F.interpolate(mask.view(B * T, *mask.shape[2:]), size=(H, W), mode='nearest').view(B, T, 1, H, W)
                contrast = contrast * mask
            pooled = self.contrast_pool(contrast.view(B * T, 1, H, W)).view(B, T, -1)
            render_contrast_feat = pooled
        return {
            'frame_features': frame_tokens,
            'temporal_features': temporal_frames,
            'global_features': global_features,
            'render_contrast_feat': render_contrast_feat,
        }
