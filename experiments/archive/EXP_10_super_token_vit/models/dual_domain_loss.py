import torch
import torch.nn as nn
import torch.nn.functional as F


class DualDomainGeometricLoss(nn.Module):
    """
    Dual-Domain Supervision Loss for EXP_10 Super-Token Geometric ViT.
    
    Combines:
    1. L_attn:   Focal BCE on patch-level attention heatmaps (32x32)
    2. L_vector: Smooth L1 on global 6-control-point Bézier coordinates (JSON domain)
    3. L_dice:   Differentiable Soft Dice loss on rendered masks (GT Mask domain)
    4. L_exist:  BCE on landmark visibility/existence
    """
    def __init__(
        self,
        lambda_attn: float = 2.0,
        lambda_vector: float = 5.0,
        lambda_dice: float = 5.0,
        lambda_exist: float = 1.5,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.75,
        eps: float = 1e-6
    ):
        super().__init__()
        self.lambda_attn = lambda_attn
        self.lambda_vector = lambda_vector
        self.lambda_dice = lambda_dice
        self.lambda_exist = lambda_exist
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.eps = eps

    def forward(
        self,
        pred_dict: dict,
        target_dict: dict
    ) -> dict:
        """
        Args:
            pred_dict containing:
                exist_logits:  (B, C)
                exist_probs:   (B, C)
                ctrl_points:   (B, C, K, 2)
                attn_heatmaps: (B, C, 32, 32)
                soft_masks:    (B, C, R, R)
                
            target_dict containing:
                target_exists:      (B, C) float in {0, 1}
                target_ctrl_points: (B, C, K, 2) float in [0, 1]^2
                target_attn_masks:  (B, C, 32, 32) float in {0, 1}
                target_render_masks:(B, C, R, R) float in {0, 1}
                
        Returns:
            dict with individual loss components and total loss
        """
        device = pred_dict["exist_logits"].device
        
        exist_logits = pred_dict["exist_logits"]             # (B, C)
        ctrl_points = pred_dict["ctrl_points"]               # (B, C, K, 2)
        attn_heatmaps = pred_dict["attn_heatmaps"]           # (B, C, 32, 32)
        soft_masks = pred_dict["soft_masks"]                 # (B, C, R, R)
        
        target_exists = target_dict["target_exists"].to(device).float()              # (B, C)
        target_ctrl_pts = target_dict["target_ctrl_points"].to(device).float()      # (B, C, K, 2)
        target_attn_masks = target_dict["target_attn_masks"].to(device).float()      # (B, C, 32, 32)
        target_render_masks = target_dict["target_render_masks"].to(device).float()  # (B, C, R, R)
        
        B, C = target_exists.shape
        active_mask = (target_exists > 0.5)  # (B, C)
        
        # -------------------------------------------------------------
        # 1. Existence Loss (BCEWithLogits)
        # -------------------------------------------------------------
        loss_exist = F.binary_cross_entropy_with_logits(exist_logits, target_exists)
        
        # -------------------------------------------------------------
        # 2. Patch-Level Attention Heatmap Loss (Focal BCE)
        # -------------------------------------------------------------
        p = torch.clamp(attn_heatmaps, self.eps, 1.0 - self.eps)
        y = target_attn_masks
        
        # Focal BCE: -alpha * y * (1-p)^gamma * log(p) - (1-alpha) * (1-y) * p^gamma * log(1-p)
        focal_pos = -self.focal_alpha * y * ((1.0 - p) ** self.focal_gamma) * torch.log(p)
        focal_neg = -(1.0 - self.focal_alpha) * (1.0 - y) * (p ** self.focal_gamma) * torch.log(1.0 - p)
        loss_attn = (focal_pos + focal_neg).mean()
        
        # -------------------------------------------------------------
        # 3. Vector Domain Loss (Smooth L1 on Active Control Points)
        # -------------------------------------------------------------
        if active_mask.sum() > 0:
            active_pred_pts = ctrl_points[active_mask]       # (M, K, 2)
            active_target_pts = target_ctrl_pts[active_mask] # (M, K, 2)
            loss_vector = F.smooth_l1_loss(active_pred_pts, active_target_pts, beta=0.02)
        else:
            loss_vector = torch.tensor(0.0, device=device)
            
        # -------------------------------------------------------------
        # 4. Raster Domain Loss (Differentiable Soft Dice on Masks)
        # -------------------------------------------------------------
        # soft_masks: (B, C, R, R), target_render_masks: (B, C, R, R)
        # Compute soft dice per class
        pred_flat = soft_masks.view(B, C, -1)
        target_flat = target_render_masks.view(B, C, -1)
        
        intersection = (pred_flat * target_flat).sum(dim=-1)
        cardinality = (pred_flat.pow(2) + target_flat.pow(2)).sum(dim=-1)
        dice_score = (2.0 * intersection + self.eps) / (cardinality + self.eps)  # (B, C)
        
        # Loss is 1 - dice on active landmarks, plus small empty penalty on inactive landmarks
        if active_mask.sum() > 0:
            loss_dice_active = (1.0 - dice_score[active_mask]).mean()
        else:
            loss_dice_active = torch.tensor(0.0, device=device)
            
        inactive_mask = ~active_mask
        if inactive_mask.sum() > 0:
            # Inactive penalty: suppress false positives
            loss_dice_inactive = pred_flat[inactive_mask].mean()
        else:
            loss_dice_inactive = torch.tensor(0.0, device=device)
            
        loss_dice = loss_dice_active + 0.5 * loss_dice_inactive
        
        # -------------------------------------------------------------
        # Total Weighted Loss
        # -------------------------------------------------------------
        total_loss = (
            self.lambda_attn * loss_attn +
            self.lambda_vector * loss_vector +
            self.lambda_dice * loss_dice +
            self.lambda_exist * loss_exist
        )
        
        return {
            "loss": total_loss,
            "loss_exist": loss_exist,
            "loss_attn": loss_attn,
            "loss_vector": loss_vector,
            "loss_dice": loss_dice
        }
