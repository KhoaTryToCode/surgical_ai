import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class MacroPatchViT(nn.Module):
    """
    EXP_10: Macro-Patch Geometric Vision Transformer (Way A).
    
    Pipeline:
    1. ViT-Base Backbone (in_chans=4 RGB-D) -> Micro-tokens: (B, 1024, 768) on 32x32 grid (16px patches)
    2. Spatial Macro-Merge: 4x4 grouping conv -> Macro-tokens: (B, 64, 768) on 8x8 grid (64px patches)
    3. Inter-Macro Relational Transformer: 2 layers of all-to-all self-attention for global organ pose
    4. Per-Macro Dual Heads:
       - Classification Head: Linear(D -> 256 -> 5) -> (B, 8, 8, 5)
       - Bézier Head: MLP(D -> 256 -> 8) + Sigmoid -> (B, 8, 8, 4, 2) in local patch [0, 1]^2
    """
    def __init__(
        self,
        backbone_name: str = "vit_base_patch16_224",
        in_chans: int = 4,
        pretrained: bool = True,
        image_size: int = 512,
        micro_patch_size: int = 16,
        macro_patch_size: int = 64,
        num_classes: int = 4,
        embed_dim: int = 768,
        hidden_dim: int = 256,
        macro_depth: int = 2,
        macro_heads: int = 8,
        dropout: float = 0.0
    ):
        super().__init__()
        self.image_size = image_size
        self.micro_patch_size = micro_patch_size
        self.macro_patch_size = macro_patch_size
        self.merge_factor = macro_patch_size // micro_patch_size  # 64 / 16 = 4
        
        self.micro_grid_size = image_size // micro_patch_size    # 512 / 16 = 32
        self.macro_grid_size = image_size // macro_patch_size    # 512 / 64 = 8
        self.num_macro_patches = self.macro_grid_size * self.macro_grid_size  # 64
        self.num_classes = num_classes
        self.in_chans = in_chans
        
        # 1. ViT Backbone
        try:
            self.backbone = timm.create_model(
                backbone_name,
                in_chans=in_chans,
                pretrained=pretrained,
                num_classes=0,
                dynamic_img_size=True,
                drop_rate=dropout
            )
            if hasattr(self.backbone, 'embed_dim'):
                self.embed_dim = self.backbone.embed_dim
            elif hasattr(self.backbone, 'num_features'):
                self.embed_dim = self.backbone.num_features
            else:
                self.embed_dim = embed_dim
        except Exception as e:
            print(f"⚠️ [ViT WARNING] Could not initialize {backbone_name} ({e}). Building modular ViT fallback.")
            self.backbone = ModularViTBackbone(
                img_size=image_size,
                patch_size=micro_patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
                depth=12,
                num_heads=12
            )
            self.embed_dim = embed_dim
            
        # 2. Spatial Macro-Merge Layer (4x4 grouping conv with normalization)
        self.macro_merge = nn.Sequential(
            nn.Conv2d(
                self.embed_dim,
                self.embed_dim,
                kernel_size=self.merge_factor,
                stride=self.merge_factor,
                padding=0
            ),
            nn.GroupNorm(num_groups=32, num_channels=self.embed_dim),
            nn.GELU()
        )
        
        # Macro positional embeddings for 8x8 grid
        self.macro_pos_embed = nn.Parameter(torch.randn(1, self.num_macro_patches, self.embed_dim) * 0.02)
        
        # 3. Inter-Macro Relational Transformer (All-to-all self-attention for global organ pose)
        macro_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=macro_heads,
            dim_feedforward=self.embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.relational_transformer = nn.TransformerEncoder(macro_encoder_layer, num_layers=macro_depth)
        self.macro_norm = nn.LayerNorm(self.embed_dim)
        
        # 4. Macro Classification Head (0 = background, 1..num_classes = surgical landmarks)
        self.class_head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes + 1)
        )
        
        # 5. Macro Bézier Control Point Head (4 control points x 2 coords = 8 values in [0, 1])
        self.bezier_head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 8),
            nn.Sigmoid()
        )
        
        # Weight initialization
        nn.init.normal_(self.class_head[-1].weight, std=0.01)
        nn.init.constant_(self.class_head[-1].bias, 0.0)
        self.class_head[-1].bias.data[0] = 0.5  # Balanced background class prior
        
        nn.init.normal_(self.bezier_head[-2].weight, std=0.01)
        nn.init.constant_(self.bezier_head[-2].bias, 0.0)

    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass.
        
        Args:
            x: Input RGB-D tensor of shape (B, 4, 512, 512)
            
        Returns:
            dict containing:
                macro_logits:  (B, 8, 8, num_classes + 1)
                macro_beziers: (B, 8, 8, 4, 2) in local patch [0, 1]^2
                flat_logits:   (B, 64, num_classes + 1)
                flat_beziers:  (B, 64, 4, 2)
                macro_tokens:  (B, 64, embed_dim)
        """
        B, C_in, H, W = x.shape
        micro_g = self.micro_grid_size  # 32
        macro_g = self.macro_grid_size  # 8
        num_micro = micro_g * micro_g   # 1024
        
        # 1. ViT Backbone feature extraction
        if hasattr(self.backbone, 'forward_features'):
            feats = self.backbone.forward_features(x)  # (B, 1 + 1024, D) or (B, 1024, D)
            if feats.shape[1] == num_micro + 1:
                micro_tokens = feats[:, 1:, :]         # (B, 1024, D)
            else:
                micro_tokens = feats
        else:
            micro_tokens = self.backbone(x)
            
        # 2. Spatial Macro-Merge: Reshape to 2D grid and pool 4x4 micro-patches
        # (B, 1024, D) -> (B, 32, 32, D) -> (B, D, 32, 32)
        micro_spatial = micro_tokens.view(B, micro_g, micro_g, self.embed_dim).permute(0, 3, 1, 2)
        
        # Grouping Conv: (B, D, 32, 32) -> (B, D, 8, 8)
        macro_spatial = self.macro_merge(micro_spatial)
        
        # Reshape to token sequence: (B, D, 8, 8) -> (B, 64, D)
        macro_tokens = macro_spatial.permute(0, 2, 3, 1).contiguous().view(B, self.num_macro_patches, self.embed_dim)
        macro_tokens = macro_tokens + self.macro_pos_embed
        
        # 3. Inter-Macro Relational Transformer (All-to-All Self-Attention)
        macro_context = self.relational_transformer(macro_tokens)
        macro_context = self.macro_norm(macro_context)
        
        # 4. Predictions on the 64 Macro-Patches
        flat_logits = self.class_head(macro_context)                    # (B, 64, C+1)
        flat_beziers = self.bezier_head(macro_context)                  # (B, 64, 8)
        flat_beziers = flat_beziers.view(B, self.num_macro_patches, 4, 2)  # (B, 64, 4, 2)
        
        # 5. Reshape to 2D Macro-Grid (8, 8)
        macro_logits = flat_logits.view(B, macro_g, macro_g, self.num_classes + 1)
        macro_beziers = flat_beziers.view(B, macro_g, macro_g, 4, 2)
        
        return {
            "macro_logits": macro_logits,
            "macro_beziers": macro_beziers,
            "flat_logits": flat_logits,
            "flat_beziers": flat_beziers,
            "macro_tokens": macro_context
        }


class ModularViTBackbone(nn.Module):
    """
    Self-contained Vision Transformer encoder fallback for offline execution.
    """
    def __init__(
        self,
        img_size: int = 512,
        patch_size: int = 16,
        in_chans: int = 4,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.embed_dim = embed_dim
        
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)
        
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        patches = self.patch_embed(x).flatten(2).transpose(1, 2)
        tokens = patches + self.pos_embed
        feats = self.transformer(tokens)
        feats = self.norm(feats)
        return feats
