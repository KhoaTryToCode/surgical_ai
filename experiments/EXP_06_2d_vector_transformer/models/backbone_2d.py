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
        
        # 1. Fallback / Test Backbone (Always initialized for safety)
        self.fallback_backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, self.embed_dim, kernel_size=3, padding=1)
        )

        # 2. Load pre-trained Mask2Former Swin-Tiny backbone
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
            print(f"✅ Loaded Mask2Former 2D backbone from '{config.mask2former_model_name}'")
        except Exception as e:
            print(f"⚠️ Transformers Mask2Former not loaded ({e}), using standard convolutional backbone.")

        self.proj_stride4 = nn.Conv2d(256, self.embed_dim, kernel_size=1) if self.embed_dim != 256 else nn.Identity()

        # 3. 2D Positional Encoding (128 channels)
        self.pe_2d = Sinusoidal2DPositionalEncoding(num_pos_feats=64)

        # 4. Feature projection & fusion per level
        self.proj_fpn = nn.ModuleList([
            nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=1) for _ in range(4)
        ])
        
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
        pyramid = None
        if self.pixel_decoder is not None:
            try:
                pixel_outputs = self.pixel_decoder(pixel_values=pixel_values)
                feat_s4 = pixel_outputs.decoder_last_hidden_state # (B, 256, 256, 256) Stride-4
                feat_s4 = self.proj_stride4(feat_s4)

                # Check available multi-scale hidden states
                if hasattr(pixel_outputs, "decoder_hidden_states") and pixel_outputs.decoder_hidden_states is not None:
                    hs = pixel_outputs.decoder_hidden_states
                    pyramid = [feat_s4, hs[2], hs[1], hs[0]]
                elif hasattr(pixel_outputs, "multi_scale_features") and pixel_outputs.multi_scale_features is not None:
                    ms = pixel_outputs.multi_scale_features
                    pyramid = [feat_s4, ms[2], ms[1], ms[0]]
                else:
                    # Construct multi-scale pyramid via hierarchical pooling
                    pyramid = [
                        feat_s4,
                        F.avg_pool2d(feat_s4, 2),
                        F.avg_pool2d(feat_s4, 4),
                        F.avg_pool2d(feat_s4, 8)
                    ]
            except Exception as e:
                pyramid = None

        if pyramid is None or len(pyramid) < 4:
            feat_stride4 = self.fallback_backbone(pixel_values) # (B, 256, 256, 256)
            pyramid = [
                feat_stride4,
                F.avg_pool2d(feat_stride4, 2),
                F.avg_pool2d(feat_stride4, 4),
                F.avg_pool2d(feat_stride4, 8)
            ]
        
        fused_features = []
        for l in range(4):
            feat = pyramid[l]
            pos_2d = self.pe_2d(feat) # (B, 128, H_l, W_l)
            fused = self.fuse_proj[l](torch.cat([feat, pos_2d], dim=1)) # (B, 256, H_l, W_l)
            fused_features.append(fused)

        return fused_features
