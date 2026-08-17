import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Sinusoidal2DPositionalEncoding(nn.Module):
    """
    Standard 2D Sinusoidal Positional Encoding for continuous image space (u, v) in [0, 1]^2.
    Generates multi-frequency sine and cosine waves: (B, 2 * num_pos_feats, H, W).
    """
    def __init__(self, num_pos_feats: int = 64, temperature: float = 10000.0):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W) feature map
        Returns: (B, 2 * num_pos_feats, H, W) 2D positional encoding
        """
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

class SurgicalBackbone2D(nn.Module):
    """
    Pretrained Swin-Tiny Backbone for EXP_06 Direct 2D Vector Space Transformer.
    Extracts multi-scale 2D features and fuses them with 2D Sinusoidal Positional Encoding.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        
        # Load pre-trained Mask2Former Swin-Tiny backbone
        try:
            from transformers import Mask2FormerForUniversalSegmentation
            m2f = Mask2FormerForUniversalSegmentation.from_pretrained(config.mask2former_model_name)
            self.pixel_level_module = m2f.model.pixel_level_module
            self.proj_stride4 = nn.Conv2d(self.pixel_level_module.decoder.conv_dim, self.embed_dim, kernel_size=1)
            self.use_mock = False
            print(f"✅ Loaded Mask2Former 2D backbone from '{config.mask2former_model_name}'")
        except Exception as e:
            print(f"⚠️ Transformers not available locally ({e}), initializing standard lightweight Swin FPN emulation for testing.")
            self.use_mock = True
            self.mock_encoder = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=4, padding=3),
                nn.BatchNorm2d(64),
                nn.GELU(),
                nn.Conv2d(64, self.embed_dim, kernel_size=3, padding=1)
            )

        # 2D Positional Encoding
        self.pe_2d = Sinusoidal2DPositionalEncoding(num_pos_feats=64) # 128 channels

        # Linear projections for multi-scale feature levels (Stride 4, 8, 16, 32)
        self.fuse_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.embed_dim + 128, self.embed_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(self.embed_dim),
                nn.GELU()
            ) for _ in range(4)
        ])

    def forward(self, pixel_values: torch.Tensor) -> list:
        """
        pixel_values: (B, 3, 1024, 1024)
        Returns: List of 4 fused 2D feature maps [Stride 4, Stride 8, Stride 16, Stride 32]
        """
        if getattr(self, 'use_mock', False):
            feat_stride4 = self.mock_encoder(pixel_values) # (B, 256, 256, 256)
            feat_stride8 = F.avg_pool2d(feat_stride4, 2)
            feat_stride16 = F.avg_pool2d(feat_stride8, 2)
            feat_stride32 = F.avg_pool2d(feat_stride16, 2)
            pyramid = [feat_stride4, feat_stride8, feat_stride16, feat_stride32]
        else:
            pixel_outputs = self.pixel_level_module(pixel_values=pixel_values)
            multi_scale_features = list(pixel_outputs.multi_scale_features) # Stride 32, 16, 8
            feat_stride4 = self.proj_stride4(pixel_outputs.decoder_last_hidden_state) # Stride 4 (256x256)
            pyramid = [feat_stride4, multi_scale_features[2], multi_scale_features[1], multi_scale_features[0]]
        
        fused_features = []
        for l, feat in enumerate(pyramid):
            pos_2d = self.pe_2d(feat) # (B, 128, H, W)
            fused = self.fuse_proj[l](torch.cat([feat, pos_2d], dim=1)) # (B, 256, H, W)
            fused_features.append(fused)

        return fused_features
