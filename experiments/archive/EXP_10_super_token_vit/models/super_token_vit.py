import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from .curve_decoder import GlobalCurveDecoder
from .soft_rasterizer import SoftCurveRasterizer


class MultiHeadSuperTokenAttention(nn.Module):
    """
    Multi-Head Cross-Attention module for Super-Token Aggregation.
    
    Queries:   (B, C, D)  - Conditioned Landmark Semantic Queries
    Keys/Vals: (B, N, D)  - Spatial Patch Tokens (N = 1024)
    
    Outputs:
        super_tokens:  (B, C, D)      - Aggregated landmark features
        attn_heatmaps: (B, C, G, G)   - Spatial attention heatmaps (G = 32)
    """
    def __init__(self, embed_dim: int = 768, num_heads: int = 8, grid_size: int = 32):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / (self.head_dim ** 0.5)
        self.grid_size = grid_size
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)

    def forward(self, queries: torch.Tensor, patch_tokens: torch.Tensor) -> tuple:
        B, C, _ = queries.shape
        _, N, _ = patch_tokens.shape
        G = self.grid_size
        
        q = self.norm_q(queries)
        kv = self.norm_kv(patch_tokens)
        
        # Project and reshape for multi-head attention: (B, H, L, head_dim)
        Q = self.q_proj(q).view(B, C, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(kv).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(kv).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Cross-attention weights: (B, H, C, N)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, H, C, N)
        
        # Weighted value aggregation: (B, H, C, head_dim) -> (B, C, D)
        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(B, C, self.embed_dim)
        super_tokens = self.out_proj(out)
        
        # Mean attention weights across heads for 2D spatial heatmap: (B, C, G, G)
        mean_attn = attn_weights.mean(dim=1)  # (B, C, N)
        attn_heatmaps = mean_attn.view(B, C, G, G)
        
        return super_tokens, attn_heatmaps


class SuperTokenGeometricViT(nn.Module):
    """
    EXP_10: Super-Token Geometric Vision Transformer.
    
    Pipeline:
    1. ViT-Base Backbone (in_chans=4 RGB-D) -> Spatial tokens (B, 1024, D) + [CLS] Pose (B, D)
    2. Pose Conditioning: Modulate C=4 semantic landmark queries with [CLS] organ pose embedding
    3. Super-Token Cross-Attention: Softly gather active patches into C=4 super-tokens (B, C, D)
    4. Global Curve Decoder: Predicts existence probability and K=6 control points (B, C, K, 2) in [0, 1]^2
    5. Soft Rasterizer: Fully differentiable curve rendering to (B, C, 128, 128) soft masks for Dice loss
    """
    def __init__(
        self,
        backbone_name: str = "vit_base_patch16_224",
        in_chans: int = 4,
        pretrained: bool = True,
        image_size: int = 512,
        patch_size: int = 16,
        num_classes: int = 4,
        num_ctrl_points: int = 6,
        embed_dim: int = 768,
        hidden_dim: int = 512,
        render_size: int = 128,
        cross_attn_heads: int = 8,
        dropout: float = 0.0
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size  # 32
        self.num_patches = self.grid_size * self.grid_size  # 1024
        self.num_classes = num_classes
        self.num_ctrl_points = num_ctrl_points
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
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
                depth=12,
                num_heads=12
            )
            self.embed_dim = embed_dim
            
        # 2. Base Semantic Landmark Queries (C=4 queries for Ridge, Silhouette, Ligament, Gallbladder)
        self.base_queries = nn.Parameter(torch.randn(1, num_classes, self.embed_dim) * 0.02)
        
        # 3. Global Organ Pose Projector: Projects [CLS] token to modulate base queries
        self.pose_proj = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )
        
        # 4. Multi-Head Super-Token Cross-Attention
        self.cross_attention = MultiHeadSuperTokenAttention(
            embed_dim=self.embed_dim,
            num_heads=cross_attn_heads,
            grid_size=self.grid_size
        )
        
        # 5. Global Curve Decoder
        self.decoder = GlobalCurveDecoder(
            embed_dim=self.embed_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_ctrl_points=num_ctrl_points,
            dropout=dropout
        )
        
        # 6. Differentiable Soft Line Rasterizer
        self.rasterizer = SoftCurveRasterizer(
            render_size=render_size,
            num_samples=64,
            sigma_px=1.5,
            target_size=image_size
        )

    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass.
        
        Args:
            x: Input RGB-D tensor of shape (B, 4, 512, 512)
            
        Returns:
            dict containing:
                exist_logits:  (B, C)
                exist_probs:   (B, C)
                ctrl_points:   (B, C, K, 2) in [0, 1]^2
                attn_heatmaps: (B, C, 32, 32)
                soft_masks:    (B, C, 128, 128)
                super_tokens:  (B, C, D)
                cls_token:     (B, D)
        """
        B, C_in, H, W = x.shape
        num_patches = self.num_patches
        
        # 1. Forward through ViT backbone
        if hasattr(self.backbone, 'forward_features'):
            feats = self.backbone.forward_features(x)  # (B, 1 + num_patches, D)
            if feats.shape[1] == num_patches + 1:
                cls_token = feats[:, 0, :]           # (B, D) - Global Organ Pose Descriptor
                patch_tokens = feats[:, 1:, :]       # (B, num_patches, D)
            else:
                cls_token = feats.mean(dim=1)
                patch_tokens = feats
        else:
            cls_token, patch_tokens = self.backbone(x)
            
        # 2. Pose-Conditioning: Modulate base landmark queries with global organ pose
        # pose_delta: (B, 1, D)
        pose_delta = self.pose_proj(cls_token).unsqueeze(1)
        conditioned_queries = self.base_queries.expand(B, -1, -1) + pose_delta  # (B, C, D)
        
        # 3. Super-Token Cross-Attention Aggregation
        super_tokens, attn_heatmaps = self.cross_attention(conditioned_queries, patch_tokens)
        
        # 4. Global Curve & Existence Decoding
        dec_out = self.decoder(super_tokens)
        exist_logits = dec_out["exist_logits"]  # (B, C)
        exist_probs = dec_out["exist_probs"]    # (B, C)
        ctrl_points = dec_out["ctrl_points"]    # (B, C, K, 2)
        
        # 5. Differentiable Soft Line Rendering for Soft Dice Loss
        soft_masks = self.rasterizer(ctrl_points, existence_probs=exist_probs)  # (B, C, R, R)
        
        return {
            "exist_logits": exist_logits,
            "exist_probs": exist_probs,
            "ctrl_points": ctrl_points,
            "attn_heatmaps": attn_heatmaps,
            "soft_masks": soft_masks,
            "super_tokens": super_tokens,
            "cls_token": cls_token
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
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + self.num_patches, embed_dim) * 0.02)
        
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

    def forward(self, x: torch.Tensor) -> tuple:
        B = x.shape[0]
        # Patch embedding: (B, D, G, G) -> (B, num_patches, D)
        patches = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, patches], dim=1) + self.pos_embed
        
        feats = self.transformer(tokens)
        feats = self.norm(feats)
        
        cls_out = feats[:, 0, :]
        patch_out = feats[:, 1:, :]
        return cls_out, patch_out
