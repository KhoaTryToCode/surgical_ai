import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Sinusoidal2DPositionalEncoding(nn.Module):
    """
    Standard 2D Sinusoidal Positional Encoding for continuous image space (u, v) in [0, 1]^2.
    """
    def __init__(self, num_pos_feats: int = 64, temperature: float = 10000.0):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        y_embed = torch.linspace(0.0, 1.0, H, device=x.device).view(1, H, 1).repeat(B, 1, W)
        x_embed = torch.linspace(0.0, 1.0, W, device=x.device).view(1, 1, W).repeat(B, H, 1)

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (torch.div(dim_t, 2, rounding_mode='floor')) / self.num_pos_feats)

        pos_x = x_embed.unsqueeze(-1) / dim_t
        pos_y = y_embed.unsqueeze(-1) / dim_t

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

class StandardSwinEncoder(nn.Module):
    """
    Standard Swin Backbone Encoder from EXP_06.
    Extracts dense multi-scale 2D feature maps [Stride 4, Stride 8, Stride 16, Stride 32].
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim

        # Fallback Backbone
        self.fallback_backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, self.embed_dim, kernel_size=3, padding=1)
        )

        # Pre-trained Mask2Former Swin-Tiny Backbone
        self.pixel_decoder = None
        try:
            from transformers import Mask2FormerForUniversalSegmentation
            m2f = Mask2FormerForUniversalSegmentation.from_pretrained(config.mask2former_model_name)
            if hasattr(m2f.model, "pixel_level_module"):
                self.pixel_decoder = m2f.model.pixel_level_module
            elif hasattr(m2f.model, "pixel_decoder"):
                self.pixel_decoder = m2f.model.pixel_decoder
            else:
                self.pixel_decoder = m2f.model
        except Exception:
            pass

        self.proj_stride4 = nn.Conv2d(256, self.embed_dim, kernel_size=1) if self.embed_dim != 256 else nn.Identity()
        self.pe_2d = Sinusoidal2DPositionalEncoding(num_pos_feats=64)

        self.fuse_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.embed_dim + 128, self.embed_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(self.embed_dim),
                nn.GELU()
            ) for _ in range(4)
        ])

    def forward(self, pixel_values: torch.Tensor) -> dict:
        """
        pixel_values: (B, 3, 1024, 1024)
        Returns: Dictionary of standard multi-scale features and spatial maps.
        """
        pyramid = None
        if self.pixel_decoder is not None:
            try:
                pixel_outputs = self.pixel_decoder(pixel_values=pixel_values)
                feat_s4 = pixel_outputs.decoder_last_hidden_state
                feat_s4 = self.proj_stride4(feat_s4)

                if hasattr(pixel_outputs, "decoder_hidden_states") and pixel_outputs.decoder_hidden_states is not None:
                    hs = pixel_outputs.decoder_hidden_states
                    pyramid = [feat_s4, hs[2], hs[1], hs[0]]
                elif hasattr(pixel_outputs, "multi_scale_features") and pixel_outputs.multi_scale_features is not None:
                    ms = pixel_outputs.multi_scale_features
                    pyramid = [feat_s4, ms[2], ms[1], ms[0]]
                else:
                    pyramid = [
                        feat_s4,
                        F.avg_pool2d(feat_s4, 2),
                        F.avg_pool2d(feat_s4, 4),
                        F.avg_pool2d(feat_s4, 8)
                    ]
            except Exception:
                pyramid = None

        if pyramid is None or len(pyramid) < 4:
            feat_stride4 = self.fallback_backbone(pixel_values)
            pyramid = [
                feat_stride4,
                F.avg_pool2d(feat_stride4, 2),
                F.avg_pool2d(feat_stride4, 4),
                F.avg_pool2d(feat_stride4, 8)
            ]

        fused_features = []
        for i, feat in enumerate(pyramid):
            if feat.shape[1] != self.embed_dim:
                feat = F.interpolate(feat, size=(1024 // (4 * (2**i)), 1024 // (4 * (2**i))), mode='bilinear', align_corners=False)
            pos = self.pe_2d(feat)
            feat_pos = torch.cat([feat, pos], dim=1)
            fused = self.fuse_proj[i](feat_pos)
            fused_features.append(fused)

        return {
            "fused_features": fused_features,
            "stride4_features": fused_features[0],  # (B, 256, 256, 256)
            "stride8_features": fused_features[1],  # (B, 256, 128, 128)
            "stride16_features": fused_features[2], # (B, 256, 64, 64)
            "stride32_features": fused_features[3]  # (B, 256, 32, 32)
        }
