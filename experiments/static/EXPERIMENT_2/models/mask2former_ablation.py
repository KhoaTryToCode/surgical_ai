import math
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

def fallback_linear_sum_assignment(cost_matrix):
    """Greedy fallback matching when scipy is unavailable (e.g. lightweight local environments)."""
    cost = cost_matrix.copy()
    num_rows, num_cols = cost.shape
    row_ind = []
    col_ind = []
    for _ in range(min(num_rows, num_cols)):
        min_idx = np.unravel_index(np.argmin(cost), cost.shape)
        r, c = int(min_idx[0]), int(min_idx[1])
        row_ind.append(r)
        col_ind.append(c)
        cost[r, :] = 1e9
        cost[:, c] = 1e9
    return np.array(row_ind, dtype=np.int64), np.array(col_ind, dtype=np.int64)

def solve_linear_assignment(cost_matrix):
    if HAS_SCIPY:
        return linear_sum_assignment(cost_matrix)
    return fallback_linear_sum_assignment(cost_matrix)




class MaskedCrossAttention(nn.Module):
    """
    Masked Cross-Attention Module.
    If attn_mask is provided, queries only attend to spatial locations
    where the predicted foreground mask is active.
    """
    def __init__(self, embed_dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        q: torch.Tensor,                           # (B, N, D)
        kv: torch.Tensor,                          # (B, S, D)
        attn_mask: Optional[torch.Tensor] = None,  # (B, N, S) float with -inf for masked
    ) -> torch.Tensor:
        B, N, _ = q.shape
        _, S, _ = kv.shape

        q_proj = self.q_proj(q).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)   # (B, H, N, d)
        k_proj = self.k_proj(kv).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, S, d)
        v_proj = self.v_proj(kv).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, S, d)

        scores = torch.matmul(q_proj, k_proj.transpose(-2, -1)) * self.scale  # (B, H, N, S)

        if attn_mask is not None:
            # Broadcast mask across attention heads: (B, 1, N, S)
            scores = scores + attn_mask.unsqueeze(1)

        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v_proj)  # (B, H, N, d)
        out = out.transpose(1, 2).contiguous().view(B, N, self.embed_dim)
        return self.out_proj(out)


class TransformerDecoderLayer(nn.Module):
    """
    Transformer Decoder Layer supporting clean ablation toggles:
    - enable_masked_attention: True uses dynamic spatial mask; False uses global attention.
    - enable_query_self_attention: True performs QxQ self-attention; False bypasses it.
    """
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        enable_query_self_attention: bool = True,
    ):
        super().__init__()
        self.enable_query_self_attention = enable_query_self_attention

        if self.enable_query_self_attention:
            self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
            self.norm1 = nn.LayerNorm(embed_dim)

        self.cross_attn = MaskedCrossAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)

        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, embed_dim),
        )
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        queries: torch.Tensor,                     # (B, N, D)
        features: torch.Tensor,                    # (B, S, D)
        attn_mask: Optional[torch.Tensor] = None,   # (B, N, S)
    ) -> torch.Tensor:
        # 1. Query Self-Attention (bypassed if disabled)
        if self.enable_query_self_attention:
            q_norm = self.norm1(queries)
            sa_out, _ = self.self_attn(q_norm, q_norm, q_norm)
            queries = queries + sa_out

        # 2. Cross-Attention (Masked or Global)
        q_norm = self.norm2(queries)
        ca_out = self.cross_attn(q_norm, features, attn_mask=attn_mask)
        queries = queries + ca_out

        # 3. Feed-Forward Network
        q_norm = self.norm3(queries)
        queries = queries + self.mlp(q_norm)
        return queries


class ResNet50FPN(nn.Module):
    """3-Channel Standard RGB ResNet-50 with Feature Pyramid Network (FPN)."""
    def __init__(self, out_channels: int = 256):
        super().__init__()
        base = resnet50(weights=ResNet50_Weights.DEFAULT)

        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool

        self.layer1 = base.layer1  # 256, stride 4
        self.layer2 = base.layer2  # 512, stride 8
        self.layer3 = base.layer3  # 1024, stride 16
        self.layer4 = base.layer4  # 2048, stride 32

        # Lateral 1x1 convolutions
        self.lat4 = nn.Conv2d(2048, out_channels, 1)
        self.lat3 = nn.Conv2d(1024, out_channels, 1)
        self.lat2 = nn.Conv2d(512, out_channels, 1)
        self.lat1 = nn.Conv2d(256, out_channels, 1)

        # Smoothing 3x3 convolutions
        self.smooth4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth1 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        p4 = self.lat4(c4)
        p3 = self.lat3(c3) + F.interpolate(p4, size=c3.shape[2:], mode="nearest")
        p2 = self.lat2(c2) + F.interpolate(p3, size=c2.shape[2:], mode="nearest")
        p1 = self.lat1(c1) + F.interpolate(p2, size=c1.shape[2:], mode="nearest")

        f4 = self.smooth4(p4)  # stride 32 (B, 256, H/32, W/32)
        f3 = self.smooth3(p3)  # stride 16 (B, 256, H/16, W/16)
        f2 = self.smooth2(p2)  # stride 8  (B, 256, H/8, W/8)
        f1 = self.smooth1(p1)  # stride 4  (B, 256, H/4, W/4)
        return [f1, f2, f3, f4]


class Mask2FormerAblationModel(nn.Module):
    """
    Unified Mask2Former Architecture with Exact Toggle Controls:
    1. 'baseline'       : Full Mask2Former (Masked Attention + Multi-Scale + Query Self-Attn)
    2. 'wo_masked_attn' : Full Global Attention (attn_mask = None across all layers)
    3. 'wo_multiscale'  : Single-Scale Decoder (stride 16 only, no 1/4 pixel embedding pyramid)
    4. 'wo_self_attn'   : Independent Queries (bypasses Query Self-Attention sublayer)
    """
    def __init__(
        self,
        ablation_mode: str = 'baseline',
        num_classes: int = 3,               # 3 foreground classes: Ridge, Silhouette, Falciform
        num_queries_per_class: int = 5,     # 15 queries total
        fpn_dim: int = 256,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
    ):
        super().__init__()
        self.ablation_mode = ablation_mode.lower()
        self.num_classes = num_classes
        self.num_queries = num_classes * num_queries_per_class
        self.fpn_dim = fpn_dim
        self.num_decoder_layers = num_decoder_layers

        # Mode Toggles
        self.enable_masked_attention = (self.ablation_mode != 'wo_masked_attn')
        self.enable_multiscale = (self.ablation_mode != 'wo_multiscale')
        self.enable_query_self_attention = (self.ablation_mode != 'wo_self_attn')

        # 1. 3-Channel RGB Backbone with FPN
        self.backbone = ResNet50FPN(out_channels=fpn_dim)

        # 2. Pixel Embedding Generator
        if self.enable_multiscale:
            # Projects high-resolution stride-4 feature f1 to pixel embedding
            self.pixel_embed_proj = nn.Sequential(
                nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1),
                nn.GroupNorm(32, fpn_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(fpn_dim, fpn_dim, 1),
            )
        else:
            # Single-scale: projects stride-16 feature directly without multi-scale pyramid
            self.pixel_embed_proj = nn.Sequential(
                nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1),
                nn.GroupNorm(32, fpn_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(fpn_dim, fpn_dim, 1),
            )

        # 3. Learnable Queries
        self.query_embeddings = nn.Parameter(torch.randn(self.num_queries, fpn_dim) * 0.02)
        self.query_pos_embeddings = nn.Parameter(torch.randn(self.num_queries, fpn_dim) * 0.02)

        # 4. Decoder Layers
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(
                embed_dim=fpn_dim,
                num_heads=num_heads,
                enable_query_self_attention=self.enable_query_self_attention,
            )
            for _ in range(num_decoder_layers)
        ])

        # 5. Class Head: 4 classes (0: background/no-object, 1: ridge, 2: silhouette, 3: falciform)
        self.class_head = nn.Linear(fpn_dim, num_classes + 1)

    def forward(
        self,
        images: torch.Tensor,                       # (B, 3, H, W)
        target_size: Tuple[int, int] = (1024, 1024)
    ) -> Dict[str, torch.Tensor]:
        B, _, H, W = images.shape

        # 1. Extract Backbone Features: [f1(1/4), f2(1/8), f3(1/16), f4(1/32)]
        fpn_features = self.backbone(images)
        f1, f2, f3, f4 = fpn_features

        # 2. Pixel Embeddings
        if self.enable_multiscale:
            pixel_embeddings = self.pixel_embed_proj(f1)  # (B, 256, H/4, W/4)
            # Multi-scale feature cycling list: f4 (stride 32), f3 (stride 16), f2 (stride 8)
            feature_levels = [f4, f3, f2]
        else:
            pixel_embeddings = self.pixel_embed_proj(f3)  # (B, 256, H/16, W/16)
            # Single-scale: only stride 16
            feature_levels = [f3]

        # 3. Initialize Queries
        queries = self.query_embeddings.unsqueeze(0).expand(B, -1, -1)
        queries = queries + self.query_pos_embeddings.unsqueeze(0).expand(B, -1, -1)

        intermediate_class_logits = []
        intermediate_mask_logits = []

        mask_logits = None

        # 4. Transformer Decoder Layers
        for i, layer in enumerate(self.decoder_layers):
            # Select feature level (cycle across 1/32, 1/16, 1/8 if multiscale, or keep single scale)
            f_cur = feature_levels[i % len(feature_levels)]
            B_f, C_f, H_f, W_f = f_cur.shape
            f_seq = f_cur.flatten(2).transpose(1, 2)  # (B, H_f * W_f, 256)

            attn_mask = None
            if self.enable_masked_attention and mask_logits is not None:
                # Downsample predicted mask logits to current feature map resolution
                mask_down = F.interpolate(
                    mask_logits, size=(H_f, W_f), mode="bilinear", align_corners=False
                ).flatten(2)  # (B, N, S)
                attn_mask = torch.zeros_like(mask_down)
                attn_mask[mask_down < 0.0] = -1e9

            queries = layer(queries, f_seq, attn_mask=attn_mask)

            # Compute intermediate mask logits: (B, N, H_embed, W_embed)
            mask_logits = torch.einsum("bnd,bdhw->bnhw", queries, pixel_embeddings)
            class_logits = self.class_head(queries)

            intermediate_mask_logits.append(mask_logits)
            intermediate_class_logits.append(class_logits)

        # 5. Final Predictions Interpolated to native target_size (1024, 1024)
        final_query_masks = F.interpolate(
            mask_logits, size=target_size, mode="bilinear", align_corners=False
        )  # (B, N, H, W)
        final_query_classes = intermediate_class_logits[-1]  # (B, N, 4)

        # 6. Full Multi-Class Semantic Map Projection (Standard Mask2Former Formulation)
        # S_{c, h, w} = sum_q ( p_q(c) * sigma(m_q(h, w)) )
        query_probs = F.softmax(final_query_classes, dim=-1)  # (B, N, 4)
        query_mask_probs = torch.sigmoid(final_query_masks)    # (B, N, H, W)

        # Foreground probability maps for classes 1, 2, 3:
        # Einsum: (B, N, 4) x (B, N, H, W) -> (B, 4, H, W)
        semantic_probs = torch.einsum("bnc,bnhw->bchw", query_probs, query_mask_probs)
        # Background class probability: 1 - sum(foreground)
        bg_prob = (1.0 - semantic_probs[:, 1:].sum(dim=1, keepdim=True)).clamp(min=0.0, max=1.0)
        semantic_probs[:, 0:1] = bg_prob
        semantic_logits = torch.log(semantic_probs.clamp(min=1e-7, max=1.0))

        return {
            "query_classes": final_query_classes,                      # (B, N, 4)
            "query_masks": final_query_masks,                          # (B, N, H, W)
            "semantic_logits": semantic_logits,                        # (B, 4, H, W)
            "intermediate_classes": intermediate_class_logits,
            "intermediate_masks": intermediate_mask_logits,
        }


class Mask2FormerLoss(nn.Module):
    """
    Standard Hungarian Bipartite Matching Loss for Pure Mask2Former:
    L = 2.0 * L_cls + 5.0 * L_bce + 5.0 * L_dice
    """
    def __init__(
        self,
        lambda_cls: float = 2.0,
        lambda_bce: float = 5.0,
        lambda_dice: float = 5.0,
        num_classes: int = 3,
    ):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_bce = lambda_bce
        self.lambda_dice = lambda_dice
        self.num_classes = num_classes

    @staticmethod
    def soft_dice_loss(pred_logits: torch.Tensor, target_masks: torch.Tensor, smooth: float = 1e-5):
        """Soft Dice Loss for binary mask logits."""
        pred = torch.sigmoid(pred_logits)
        intersection = (pred * target_masks).sum(dim=(-2, -1))
        union = pred.sum(dim=(-2, -1)) + target_masks.sum(dim=(-2, -1))
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - dice

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        gt_masks: torch.Tensor,  # (B, 4, H, W) one-hot: 0: BG, 1: Ridge, 2: Sil, 3: Falc
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        query_classes = outputs["query_classes"]  # (B, N, 4)
        query_masks = outputs["query_masks"]      # (B, N, H, W)
        B, N, C = query_classes.shape
        _, _, H, W = query_masks.shape

        total_loss = 0.0
        total_cls_loss = 0.0
        total_bce_loss = 0.0
        total_dice_loss = 0.0

        for b in range(B):
            q_cls = query_classes[b]  # (N, 4)
            q_mask = query_masks[b]   # (N, H, W)
            gt_m = gt_masks[b]        # (4, H, W)

            # Identify target foreground components present in this image
            tgt_classes = []
            tgt_masks = []
            for c in range(1, 4):
                if gt_m[c].sum() > 0:
                    tgt_classes.append(c)
                    tgt_masks.append(gt_m[c])

            if len(tgt_classes) == 0:
                # No foreground present: all queries should predict background class 0
                target_labels = torch.zeros(N, dtype=torch.long, device=q_cls.device)
                cls_loss = F.cross_entropy(q_cls, target_labels)
                total_loss += self.lambda_cls * cls_loss
                total_cls_loss += cls_loss.item()
                continue

            num_targets = len(tgt_classes)
            tgt_masks_t = torch.stack(tgt_masks, dim=0)  # (M, H, W)
            tgt_classes_t = torch.tensor(tgt_classes, dtype=torch.long, device=q_cls.device)

            # --- Pairwise Cost Matrix for Hungarian Matching ---
            # 1. Classification cost: -log softmax probability
            cls_probs = F.softmax(q_cls, dim=-1)  # (N, 4)
            cost_cls = -cls_probs[:, tgt_classes_t]  # (N, M)

            # 2. Mask BCE and Dice costs
            # Expand to (N, M, H, W) for vectorized cost calculation
            q_mask_exp = q_mask.unsqueeze(1).expand(-1, num_targets, -1, -1)
            tgt_exp = tgt_masks_t.unsqueeze(0).expand(N, -1, -1, -1)

            cost_bce = F.binary_cross_entropy_with_logits(
                q_mask_exp, tgt_exp, reduction="none"
            ).mean(dim=(-2, -1))  # (N, M)

            cost_dice = self.soft_dice_loss(q_mask_exp, tgt_exp)  # (N, M)

            total_cost = (
                self.lambda_cls * cost_cls
                + self.lambda_bce * cost_bce
                + self.lambda_dice * cost_dice
            )

            # Solve assignment via Hungarian Algorithm on CPU (with fallback if scipy missing)
            cost_matrix_np = total_cost.detach().cpu().numpy()
            q_idx, tgt_idx = solve_linear_assignment(cost_matrix_np)

            # --- Compute Target Losses ---
            # Matched queries
            target_labels = torch.zeros(N, dtype=torch.long, device=q_cls.device)
            target_labels[q_idx] = tgt_classes_t[tgt_idx]

            cls_loss = F.cross_entropy(q_cls, target_labels)

            matched_q_masks = q_mask[q_idx]
            matched_tgt_masks = tgt_masks_t[tgt_idx]

            bce_loss = F.binary_cross_entropy_with_logits(matched_q_masks, matched_tgt_masks)
            dice_loss = self.soft_dice_loss(matched_q_masks, matched_tgt_masks).mean()

            sample_loss = (
                self.lambda_cls * cls_loss
                + self.lambda_bce * bce_loss
                + self.lambda_dice * dice_loss
            )

            total_loss += sample_loss
            total_cls_loss += cls_loss.item()
            total_bce_loss += bce_loss.item()
            total_dice_loss += dice_loss.item()

        batch_loss = total_loss / B
        loss_dict = {
            "loss": batch_loss.item(),
            "cls_loss": total_cls_loss / B,
            "bce_loss": total_bce_loss / B,
            "dice_loss": total_dice_loss / B,
        }
        return batch_loss, loss_dict
