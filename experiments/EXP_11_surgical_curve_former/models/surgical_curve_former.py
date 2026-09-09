"""
EXP_11: SurgicalCurveFormer — Full Model Assembly
===================================================
Combines:
  1. Multimodal Backbone: SAM-ViT-B (frozen) + ResNet-50 (RGB-D, trainable)
  2. FPN Neck: 4-level pyramid {f1..f4}, C=256 channels
  3. CNN Decoder: Multi-level pixel head for deep supervision (L_s)
  4. ACPI: Per-class pixel-aligned Bézier proposal initialization
  5. HCR: 3-stage coarse-to-fine deformable curve refinement
  6. Existence Gate: CLS-conditioned super-token cross-attention (EXP_10)
  7. Soft Rasterizer: Differentiable Gaussian mask for end-to-end Dice (EXP_10)
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import timm for ViT backbone
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

# Import torchvision for ResNet backbone
from torchvision.models import resnet50, ResNet50_Weights

from .acpi import ACPIModule
from .hcr import HCRModule
from .losses import SoftRasterizer


# ─────────────────────────────────────────────────────────────────────────────
# SAM Feature Extractor (Frozen)
# ─────────────────────────────────────────────────────────────────────────────

class SAMFeatureExtractor(nn.Module):
    """
    Frozen SAM-ViT-B feature extractor for rich anatomical semantic priors.

    Uses timm's vit_base_patch16_224 as a SAM-compatible backbone surrogate
    (same ViT-B architecture). In a full deployment, replace with the actual
    SAM image encoder (segment_anything.build_sam_vit_b).

    Output: patch token feature map projected to C_fpn=256 channels.
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        image_size: int = 512,
        out_channels: int = 256,
        pretrained: bool = True,
    ):
        super().__init__()
        self.image_size = image_size
        self.out_channels = out_channels

        if TIMM_AVAILABLE:
            try:
                self.backbone = timm.create_model(
                    model_name,
                    in_chans=3,
                    pretrained=pretrained,
                    num_classes=0,
                    dynamic_img_size=True,
                )
                embed_dim = getattr(self.backbone, "embed_dim", 768)
            except Exception as e:
                print(f"[SAM] timm unavailable ({e}). Using identity placeholder.")
                self.backbone = None
                embed_dim = 768
        else:
            self.backbone = None
            embed_dim = 768

        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(embed_dim, out_channels, kernel_size=1)

        # Freeze SAM backbone entirely
        if self.backbone is not None:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb: (B, 3, H, W)  RGB image, normalized.

        Returns:
            f_sam: (B, out_channels, H/16, W/16)  = (B, 256, 32, 32) for 512px input.
        """
        B, _, H, W = rgb.shape
        G = H // 16  # grid size = 32 for 512px

        if self.backbone is not None:
            # Resize to 224 for timm ViT-B (or use dynamic_img_size=True for 512)
            rgb_resized = F.interpolate(rgb, size=(self.image_size, self.image_size),
                                        mode="bilinear", align_corners=False)
            feats = self.backbone.forward_features(rgb_resized)  # (B, 1+G*G, D)
            patch_feats = feats[:, 1:, :]  # (B, G*G, D)  — drop CLS token
            G_feat = int(patch_feats.shape[1] ** 0.5)
            feat_map = patch_feats.view(B, G_feat, G_feat, self.embed_dim)
            feat_map = feat_map.permute(0, 3, 1, 2)  # (B, D, G, G)
        else:
            # Fallback zeros if SAM is unavailable
            G_feat = G
            feat_map = torch.zeros(B, self.embed_dim, G_feat, G_feat,
                                   device=rgb.device, dtype=rgb.dtype)

        f_sam = self.proj(feat_map)  # (B, 256, G, G)
        return f_sam


# ─────────────────────────────────────────────────────────────────────────────
# ResNet-50 Backbone + FPN (RGB-D, trainable)
# ─────────────────────────────────────────────────────────────────────────────

class ResNet50FPN(nn.Module):
    """
    ResNet-50 backbone with Feature Pyramid Network (FPN) for RGB-D input.

    Modifications vs. standard ResNet-50:
    - Input conv: 4-channel (RGB + depth) by extending the first conv weight.
      New weight[:, :3, :, :] = pretrained RGB weights (ImageNet)
      New weight[:,  3, :, :] = mean of RGB weights (depth channel warm init)
    - FPN: lateral connections from C2, C3, C4, C5 → {f1, f2, f3, f4}, all C=256.

    FPN output scales (for 512×512 input):
        f1: (B, 256, 128, 128)  stride 4   ← finest
        f2: (B, 256,  64,  64)  stride 8
        f3: (B, 256,  32,  32)  stride 16
        f4: (B, 256,  16,  16)  stride 32  ← ACPI level  [NOTE: 512/32=16 not 32]
    """

    def __init__(
        self,
        in_chans: int = 4,
        fpn_channels: int = 256,
        pretrained: bool = True,
    ):
        super().__init__()
        self.fpn_channels = fpn_channels

        # Load pretrained ResNet-50
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        resnet = resnet50(weights=weights)

        # ── 4-channel input patch ──────────────────────────────────────────
        if in_chans != 3:
            orig_conv = resnet.conv1  # (64, 3, 7, 7)
            new_conv = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                new_conv.weight[:, :3, :, :] = orig_conv.weight.clone()
                if in_chans > 3:
                    # Initialize extra channels as mean of RGB channels
                    extra = orig_conv.weight.mean(dim=1, keepdim=True)
                    for c in range(3, in_chans):
                        new_conv.weight[:, c:c+1, :, :] = extra
            resnet.conv1 = new_conv

        # ── Extract backbone layers ────────────────────────────────────────
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1  # (B, 256, H/4, W/4)   — C2
        self.layer2 = resnet.layer2  # (B, 512, H/8, W/8)   — C3
        self.layer3 = resnet.layer3  # (B, 1024, H/16, W/16)— C4
        self.layer4 = resnet.layer4  # (B, 2048, H/32, W/32)— C5

        # ── FPN Lateral convs (1×1): C_i → fpn_channels ──────────────────
        self.lat1 = nn.Conv2d(256,  fpn_channels, 1)   # C2
        self.lat2 = nn.Conv2d(512,  fpn_channels, 1)   # C3
        self.lat3 = nn.Conv2d(1024, fpn_channels, 1)   # C4
        self.lat4 = nn.Conv2d(2048, fpn_channels, 1)   # C5

        # ── FPN Smoothing convs (3×3) ──────────────────────────────────────
        self.smooth1 = nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1)
        self.smooth2 = nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1)
        self.smooth3 = nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1)
        self.smooth4 = nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1)

        # Weight init for FPN heads
        for m in [self.lat1, self.lat2, self.lat3, self.lat4,
                  self.smooth1, self.smooth2, self.smooth3, self.smooth4]:
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> list:
        """
        Args:
            x: (B, 4, H, W)  RGB-D image (or 3-channel if in_chans=3).

        Returns:
            fpn: [f1, f2, f3, f4]  FPN feature maps, finest to coarsest.
                 f1: (B, 256, H/4, W/4)
                 f2: (B, 256, H/8, W/8)
                 f3: (B, 256, H/16, W/16)
                 f4: (B, 256, H/32, W/32)
        """
        # Bottom-up pathway
        c1 = self.stem(x)     # (B, 64, H/4, W/4)  — not FPN level
        c2 = self.layer1(c1)  # (B, 256, H/4, W/4)   C2
        c3 = self.layer2(c2)  # (B, 512, H/8, W/8)   C3
        c4 = self.layer3(c3)  # (B, 1024, H/16, W/16) C4
        c5 = self.layer4(c4)  # (B, 2048, H/32, W/32) C5

        # Top-down FPN with lateral connections
        p4 = self.lat4(c5)                                              # (B, 256, H/32, W/32)
        p3 = self.lat3(c4) + F.interpolate(p4, size=c4.shape[-2:], mode="nearest")
        p2 = self.lat2(c3) + F.interpolate(p3, size=c3.shape[-2:], mode="nearest")
        p1 = self.lat1(c2) + F.interpolate(p2, size=c2.shape[-2:], mode="nearest")

        # Smooth each level
        f4 = self.smooth4(p4)  # (B, 256, H/32, W/32)
        f3 = self.smooth3(p3)  # (B, 256, H/16, W/16)
        f2 = self.smooth2(p2)  # (B, 256, H/8, W/8)
        f1 = self.smooth1(p1)  # (B, 256, H/4, W/4)

        return [f1, f2, f3, f4]   # index 0 = finest


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Level CNN Segmentation Decoder (Deep Supervision L_s)
# ─────────────────────────────────────────────────────────────────────────────

class CNNSegDecoder(nn.Module):
    """
    Lightweight segmentation head applied to each FPN level.

    Output: M binary logit maps per FPN level, used for multi-level
    deep supervision (BCRNet L_s = mean Dice across 4 levels).

    Architecture per level:
        FPN_l → Conv3×3 → GroupNorm → ReLU → Conv1×1 → (B, M, H_l, W_l)
    """

    def __init__(self, in_channels: int = 256, num_classes: int = 3, hidden: int = 128):
        super().__init__()
        self.num_classes = num_classes
        # One head per FPN level
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, hidden, 3, padding=1, bias=False),
                nn.GroupNorm(16, hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, num_classes, 1),
            )
            for _ in range(4)
        ])

    def forward(self, fpn_features: list) -> list:
        """
        Args:
            fpn_features: [f1, f2, f3, f4]

        Returns:
            seg_logits_list: [(B, M, H_l, W_l)] × 4  raw logits per level.
        """
        return [head(feat) for head, feat in zip(self.heads, fpn_features)]


# ─────────────────────────────────────────────────────────────────────────────
# Super-Token Existence Gate (EXP_10)
# ─────────────────────────────────────────────────────────────────────────────

class SuperTokenExistenceGate(nn.Module):
    """
    EXP_10 Super-Token: CLS-pose-conditioned landmark existence classifier.

    Uses the global CLS token from SAM/ViT-B as an organ pose descriptor
    to condition M per-class semantic queries, then cross-attends over
    all spatial patch tokens to produce an existence logit per class.

    Input:
        cls_token:     (B, D_vit=768)  global organ pose token
        patch_tokens:  (B, N_patch, D_vit=768)  spatial patch tokens
    Output:
        exist_logits:  (B, M)          raw existence logits per class
        exist_probs:   (B, M)          sigmoid probabilities
    """

    def __init__(
        self,
        num_classes: int = 3,
        vit_embed_dim: int = 768,
        cross_attn_heads: int = 8,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.M = num_classes
        self.D = vit_embed_dim

        # Learnable per-class base semantic queries: (1, M, D)
        self.base_queries = nn.Parameter(torch.randn(1, num_classes, vit_embed_dim) * 0.02)

        # CLS pose projector: (B, D) → (B, 1, D)
        self.pose_proj = nn.Sequential(
            nn.LayerNorm(vit_embed_dim),
            nn.Linear(vit_embed_dim, vit_embed_dim),
            nn.GELU(),
            nn.Linear(vit_embed_dim, vit_embed_dim),
        )

        # Cross-attention: queries (B, M, D) over patch_tokens (B, N, D)
        self.cross_attn = nn.MultiheadAttention(
            vit_embed_dim, cross_attn_heads, batch_first=True, dropout=0.0
        )
        self.q_norm = nn.LayerNorm(vit_embed_dim)
        self.kv_norm = nn.LayerNorm(vit_embed_dim)

        # Existence head: (B, M, D) → (B, M)
        self.exist_head = nn.Sequential(
            nn.LayerNorm(vit_embed_dim),
            nn.Linear(vit_embed_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        cls_token: torch.Tensor,
        patch_tokens: torch.Tensor,
    ) -> dict:
        """
        Args:
            cls_token:    (B, D)     global organ pose from ViT CLS token.
            patch_tokens: (B, N, D)  spatial patch tokens from ViT backbone.

        Returns:
            dict with 'exist_logits' (B, M) and 'exist_probs' (B, M).
        """
        B = cls_token.shape[0]

        # Pose conditioning: modulate base queries with global organ pose
        pose_delta = self.pose_proj(cls_token).unsqueeze(1)  # (B, 1, D)
        conditioned_q = self.base_queries.expand(B, -1, -1) + pose_delta  # (B, M, D)

        # Cross-attention over spatial patch tokens
        q_normed = self.q_norm(conditioned_q)
        kv_normed = self.kv_norm(patch_tokens)
        super_tokens, _ = self.cross_attn(q_normed, kv_normed, kv_normed)  # (B, M, D)

        # Existence prediction
        exist_logits = self.exist_head(super_tokens).squeeze(-1)  # (B, M)
        exist_probs = torch.sigmoid(exist_logits)

        return {"exist_logits": exist_logits, "exist_probs": exist_probs}


# ─────────────────────────────────────────────────────────────────────────────
# Full SurgicalCurveFormer Model
# ─────────────────────────────────────────────────────────────────────────────

class SurgicalCurveFormer(nn.Module):
    """
    EXP_11: SurgicalCurveFormer — Full Architecture.

    Pipeline (see synthesis document for full data flow):
        Stage 1: Multimodal Backbone (SAM frozen + ResNet-50 RGB-D) → FPN {f1..f4}
        Stage 2: ACPI → pixel-aligned Bézier proposals → Top-K per class
        Stage 3: HCR → 3-stage coarse-to-fine deformable curve refinement
        Stage 4: Super-Token Existence Gate (EXP_10) → per-class existence gating
        Stage 5: Soft Rasterizer → for visualization / Dice evaluation
    """

    def __init__(
        self,
        num_classes: int = 3,
        image_size: int = 512,
        in_chans: int = 4,
        fpn_channels: int = 256,
        bezier_ctrl_pts: int = 6,
        acpi_top_k: int = 10,
        hcr_stages: int = 3,
        hcr_embed_dim: int = 256,
        hcr_heads: int = 8,
        hcr_n_ref_pts: int = 26,
        hcr_deform_pts: int = 4,
        vit_backbone: str = "vit_base_patch16_224",
        vit_embed_dim: int = 768,
        vit_pretrained: bool = True,
        exist_attn_heads: int = 8,
        raster_render_size: int = 128,
        raster_num_samples: int = 64,
        raster_sigma_px: float = 2.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.M = num_classes
        self.image_size = image_size
        self.K = bezier_ctrl_pts
        self.K_top = acpi_top_k

        # ── Stage 1: Multimodal Backbone ────────────────────────────────────
        self.sam_extractor = SAMFeatureExtractor(
            model_name=vit_backbone,
            image_size=image_size,
            out_channels=fpn_channels,
            pretrained=vit_pretrained,
        )
        self.resnet_fpn = ResNet50FPN(
            in_chans=in_chans,
            fpn_channels=fpn_channels,
            pretrained=True,
        )

        # FPN feature fusion: SAM f4 → merged with ResNet f4
        # Simple 1×1 conv to fuse SAM semantic features into ResNet FPN level f4
        self.sam_fusion = nn.Conv2d(fpn_channels * 2, fpn_channels, 1)

        # ── CNN Segmentation Decoder (deep supervision L_s) ─────────────────
        self.seg_decoder = CNNSegDecoder(fpn_channels, num_classes)

        # ── Stage 2: ACPI ────────────────────────────────────────────────────
        # ACPI operates on f4 (H/32 × W/32 = 16×16 for 512px input)
        acpi_h = image_size // 32
        acpi_w = image_size // 32
        self.acpi = ACPIModule(
            num_classes=num_classes,
            in_channels=fpn_channels,
            hidden_channels=fpn_channels,
            num_ctrl_pts=bezier_ctrl_pts,
            feature_map_h=acpi_h,
            feature_map_w=acpi_w,
            top_k=acpi_top_k,
        )

        # ── Stage 3: HCR ─────────────────────────────────────────────────────
        self.hcr = HCRModule(
            num_classes=num_classes,
            top_k=acpi_top_k,
            n_ref_pts=hcr_n_ref_pts,
            embed_dim=hcr_embed_dim,
            ffn_dim=hcr_embed_dim * 2,
            num_heads=hcr_heads,
            num_sampling_pts=hcr_deform_pts,
            num_stages=hcr_stages,
            dropout=dropout,
        )

        # ── Stage 4: Super-Token Existence Gate (EXP_10) ─────────────────────
        self.existence_gate = SuperTokenExistenceGate(
            num_classes=num_classes,
            vit_embed_dim=vit_embed_dim,
            cross_attn_heads=exist_attn_heads,
        )

        # ── Stage 5: Soft Rasterizer (EXP_10) ────────────────────────────────
        self.rasterizer = SoftRasterizer(
            render_size=raster_render_size,
            num_samples=raster_num_samples,
            sigma_px=raster_sigma_px,
            target_size=image_size,
        )

    def forward(self, x: torch.Tensor) -> dict:
        """
        Full SurgicalCurveFormer forward pass.

        Args:
            x: (B, 4, H, W)  RGB-D input (or 3-channel if no depth).

        Returns:
            dict with all intermediate and final predictions:
              seg_logits_list:  [(B, M, H_l, W_l)] × 4
              score_logits:     (B, M, H_f, W_f)   ACPI confidence map
              ctrl_pts_map:     (B, M, K, 2, H_f, W_f)
              proposals:        (B, M, K_top, 6, 2)
              proposal_scores:  (B, M, K_top)
              all_ref_pts:      [(B, M, K_top, N, 2)] × 3
              all_conf_logits:  [(B, M, K_top)] × 3
              final_ctrl_pts:   (B, M, K_top, 6, 2)
              exist_logits:     (B, M)
              exist_probs:      (B, M)
              soft_masks:       (B, M, R, R)
        """
        B = x.shape[0]

        # ─── Stage 1: Backbone + FPN ─────────────────────────────────────────
        rgb = x[:, :3, :, :]   # (B, 3, H, W)

        # SAM features: (B, 256, H/16, W/16)
        f_sam = self.sam_extractor(rgb)

        # ResNet-50 FPN features: [f1, f2, f3, f4]
        fpn_feats = self.resnet_fpn(x)   # index 0=finest (H/4), 3=coarsest (H/32)
        f1, f2, f3, f4 = fpn_feats

        # Resize SAM features to match f4 resolution and fuse
        f4_h, f4_w = f4.shape[-2], f4.shape[-1]
        f_sam_resized = F.interpolate(f_sam, size=(f4_h, f4_w), mode="bilinear", align_corners=False)
        f4_fused = self.sam_fusion(torch.cat([f4, f_sam_resized], dim=1))  # (B, 256, H/32, W/32)
        fpn_feats_fused = [f1, f2, f3, f4_fused]

        # ─── CNN Segmentation Decoder ─────────────────────────────────────────
        seg_logits_list = self.seg_decoder(fpn_feats_fused)  # 4 × (B, M, H_l, W_l)

        # ─── Stage 2: ACPI ────────────────────────────────────────────────────
        acpi_out = self.acpi(f4_fused)
        proposals     = acpi_out["proposals"]       # (B, M, K_top, 6, 2)
        score_logits  = acpi_out["score_logits"]    # (B, M, H_f, W_f)
        ctrl_pts_map  = acpi_out["ctrl_pts_map"]    # (B, M, K, 2, H_f, W_f)
        proposal_scores = acpi_out["proposal_scores"]  # (B, M, K_top)

        # Build HCR initial reference points from proposals.
        # n_uniform = self.hcr.N - 1  (e.g. N=26 → 25 uniform + 1 center)
        # MUST match what HCRModule.bz_regressor expects: Linear(N*2, ...)
        n_uniform = self.hcr.N - 1
        ref_pts_init = self.acpi.build_hcr_reference_points(proposals, n_uniform=n_uniform)
        # ref_pts_init: (B, M, K_top, self.hcr.N, 2)

        # ─── Stage 3: HCR ─────────────────────────────────────────────────────
        hcr_out = self.hcr(ref_pts_init, fpn_feats_fused)
        final_ctrl_pts  = hcr_out["final_ctrl_pts"]   # (B, M, K_top, 6, 2)
        all_ref_pts     = hcr_out["all_ref_pts"]       # list of (B, M, K_top, 26, 2)
        all_conf_logits = hcr_out["all_conf_logits"]   # list of (B, M, K_top)

        # ─── Stage 4: Super-Token Existence Gate (EXP_10) ────────────────────
        # Use SAM backbone's CLS token and patch tokens for pose-conditioned gating
        if hasattr(self.sam_extractor, 'backbone') and self.sam_extractor.backbone is not None:
            with torch.no_grad():
                rgb_512 = F.interpolate(rgb, size=(self.image_size, self.image_size),
                                        mode="bilinear", align_corners=False)
                vit_feats = self.sam_extractor.backbone.forward_features(rgb_512)
            # vit_feats: (B, 1 + N_patch, D)
            if vit_feats.shape[1] > 1:
                cls_token = vit_feats[:, 0, :]       # (B, D)
                patch_tokens = vit_feats[:, 1:, :]   # (B, N_patch, D)
            else:
                cls_token = vit_feats.mean(dim=1)
                patch_tokens = vit_feats
        else:
            D_vit = self.existence_gate.D
            cls_token = torch.zeros(B, D_vit, device=x.device, dtype=x.dtype)
            patch_tokens = torch.zeros(B, 1024, D_vit, device=x.device, dtype=x.dtype)

        exist_out = self.existence_gate(cls_token, patch_tokens)
        exist_logits = exist_out["exist_logits"]  # (B, M)
        exist_probs  = exist_out["exist_probs"]   # (B, M)

        # ─── Stage 5: Soft Rasterizer ─────────────────────────────────────────
        # Render the final predicted curves into soft masks for visualization / Dice
        soft_masks = self.rasterizer(final_ctrl_pts, exist_probs)  # (B, M, R, R)

        return {
            # Dense supervision outputs
            "seg_logits_list":  seg_logits_list,   # [(B, M, H_l, W_l)] × 4
            # ACPI outputs
            "score_logits":     score_logits,      # (B, M, H_f, W_f)
            "ctrl_pts_map":     ctrl_pts_map,      # (B, M, K, 2, H_f, W_f)
            "proposals":        proposals,         # (B, M, K_top, 6, 2)
            "proposal_scores":  proposal_scores,   # (B, M, K_top)
            # HCR outputs
            "all_ref_pts":      all_ref_pts,       # list of (B, M, K_top, 26, 2)
            "all_conf_logits":  all_conf_logits,   # list of (B, M, K_top)
            "final_ctrl_pts":   final_ctrl_pts,    # (B, M, K_top, 6, 2)
            # Existence gate (EXP_10)
            "exist_logits":     exist_logits,      # (B, M)
            "exist_probs":      exist_probs,       # (B, M)
            # Soft rasterization (EXP_10)
            "soft_masks":       soft_masks,        # (B, M, R, R)
        }

    def get_param_groups(self, base_lr: float, backbone_lr_mult: float = 0.1) -> list:
        """
        Return parameter groups with different learning rates.

        Frozen:         SAM backbone (no grad)
        Slow LR:        ResNet-50 pretrained layers (lr × backbone_lr_mult)
        Normal LR:      ACPI, HCR, FPN heads, segmentation decoder, existence gate

        Args:
            base_lr:           Base learning rate (e.g., 1e-5).
            backbone_lr_mult:  Multiplier for ResNet backbone (e.g., 0.1 → lr = 1e-6).

        Returns:
            List of param group dicts for the optimizer.
        """
        resnet_params = list(self.resnet_fpn.parameters())
        resnet_ids = set(id(p) for p in resnet_params)

        head_params = [
            p for p in self.parameters()
            if p.requires_grad and id(p) not in resnet_ids
        ]

        return [
            {"params": [p for p in resnet_params if p.requires_grad],
             "lr": base_lr * backbone_lr_mult, "name": "resnet_backbone"},
            {"params": head_params,
             "lr": base_lr, "name": "heads"},
        ]
