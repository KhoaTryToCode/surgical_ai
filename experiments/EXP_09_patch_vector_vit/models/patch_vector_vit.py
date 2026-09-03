import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class PatchBezierViT(nn.Module):
    """
    Patch-Level Bézier Vector Vision Transformer (EXP_09).
    
    Architecture:
    1. Standard ViT Backbone (e.g. vit_tiny_patch16_224) extracting 1024 patch tokens (32x32) at 512x512 resolution.
    2. Per-patch Dual Prediction Heads:
       - Classification Head: Linear(D, num_classes + 1) -> (B, 1024, C+1)
       - Bézier Head: MLP(D -> 256 -> 8) + Sigmoid -> (B, 1024, 4, 2) [P0, P1, P2, P3] in [0, 1]^2
    """
    def __init__(
        self,
        backbone_name: str = "vit_tiny_patch16_224",
        in_chans: int = 4,
        pretrained: bool = False,
        image_size: int = 512,
        patch_size: int = 16,
        num_classes: int = 4,
        embed_dim: int = 192,
        hidden_dim: int = 256,
        dropout: float = 0.0
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size  # 32
        self.num_patches = self.grid_size * self.grid_size  # 1024
        self.num_classes = num_classes  # 1..num_classes (0 is Background)
        self.in_chans = in_chans
        
        # 1. Load ViT Backbone via timm with dynamic image size and 4-channel RGB-D input
        try:
            self.backbone = timm.create_model(
                backbone_name,
                in_chans=in_chans,
                pretrained=pretrained,
                num_classes=0,
                dynamic_img_size=True,
                drop_rate=dropout
            )
            # Infer actual embed_dim from backbone
            if hasattr(self.backbone, 'embed_dim'):
                self.embed_dim = self.backbone.embed_dim
            elif hasattr(self.backbone, 'num_features'):
                self.embed_dim = self.backbone.num_features
            else:
                self.embed_dim = embed_dim
        except Exception as e:
            print(f"⚠️ [ViT WARNING] Could not initialize {backbone_name} with in_chans={in_chans}, pretrained={pretrained} ({e}). Building modular ViT fallback.")
            self.backbone = ModularViTBackbone(
                img_size=image_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
                depth=12,
                num_heads=3
            )
            self.embed_dim = embed_dim
        
        # 2. Patch Classification Head (0 = background, 1..num_classes = surgical landmarks)
        self.class_head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes + 1)
        )
        
        # 3. Patch Bézier Control Point Head (4 control points x 2 coords = 8 values, bounded in [0, 1])
        self.bezier_head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 8),
            nn.Sigmoid()
        )
        
        # Initialize heads
        nn.init.normal_(self.class_head[-1].weight, std=0.01)
        nn.init.constant_(self.class_head[-1].bias, 0.0)
        # Prior bias for background class to prevent exploding gradients early
        self.class_head[-1].bias.data[0] = 2.0  # Background class prior
        
        nn.init.normal_(self.bezier_head[-2].weight, std=0.01)
        nn.init.constant_(self.bezier_head[-2].bias, 0.0)

    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass.
        
        Args:
            x: Input image tensor of shape (B, 3, H, W), typically (B, 3, 512, 512)
            
        Returns:
            dict containing:
                patch_logits: (B, grid_h, grid_w, num_classes + 1)
                patch_beziers: (B, grid_h, grid_w, 4, 2) in local patch [0, 1]^2
                flat_logits: (B, num_patches, num_classes + 1)
                flat_beziers: (B, num_patches, 4, 2)
        """
        B, C, H, W = x.shape
        grid_h = H // self.patch_size
        grid_w = W // self.patch_size
        num_patches = grid_h * grid_w
        
        # 1. Forward through ViT backbone
        if hasattr(self.backbone, 'forward_features'):
            feats = self.backbone.forward_features(x)  # (B, 1 + num_patches, D)
            # Remove [CLS] token if present
            if feats.shape[1] == num_patches + 1:
                patch_tokens = feats[:, 1:, :]  # (B, num_patches, D)
            else:
                patch_tokens = feats
        else:
            patch_tokens = self.backbone(x)
            
        # 2. Per-patch predictions
        flat_logits = self.class_head(patch_tokens)  # (B, num_patches, C+1)
        flat_beziers = self.bezier_head(patch_tokens)  # (B, num_patches, 8)
        flat_beziers = flat_beziers.view(B, num_patches, 4, 2)  # (B, num_patches, 4, 2)
        
        # 3. Spatial 2D Grid Reshape
        patch_logits = flat_logits.view(B, grid_h, grid_w, self.num_classes + 1)
        patch_beziers = flat_beziers.view(B, grid_h, grid_w, 4, 2)
        
        return {
            "patch_logits": patch_logits,
            "patch_beziers": patch_beziers,
            "flat_logits": flat_logits,
            "flat_beziers": flat_beziers
        }


class ModularViTBackbone(nn.Module):
    """
    Clean, self-contained Vision Transformer encoder fallback.
    Guarantees offline functionality without external weight downloads.
    """
    def __init__(
        self,
        img_size: int = 512,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 192,
        depth: int = 8,
        num_heads: int = 3,
        mlp_ratio: float = 4.0
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Learnable positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Transformer blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        # (B, C, H, W) -> (B, D, Gh, Gw) -> (B, Gh*Gw, D)
        x_patches = self.proj(x).flatten(2).transpose(1, 2)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_tok = torch.cat((cls_tokens, x_patches), dim=1)
        
        # Handle dynamic pos embed if size doesn't match exactly
        if x_tok.shape[1] == self.pos_embed.shape[1]:
            x_tok = x_tok + self.pos_embed
        else:
            # Interpolate pos embed
            x_tok = x_tok + F.interpolate(
                self.pos_embed.transpose(1, 2),
                size=x_tok.shape[1],
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
            
        x_out = self.transformer(x_tok)
        return self.norm(x_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.forward_features(x)
        return feats[:, 1:, :]
