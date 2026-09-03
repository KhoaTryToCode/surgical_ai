import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from models.bezier_utils import sample_cubic_bezier_torch
except ImportError:
    from .bezier_utils import sample_cubic_bezier_torch


class FocalLoss(nn.Module):
    """
    Multi-Class Focal Loss to address extreme background patch imbalance (~90% background).
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (N, C) class logits
            targets: (N,) ground truth class indices in 0..C-1
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # probability of true class
        
        # Alpha weight: down-weight background (class 0)
        alpha_factor = torch.ones_like(targets, dtype=torch.float32) * self.alpha
        alpha_factor[targets == 0] = (1.0 - self.alpha)
        
        focal_loss = alpha_factor * ((1.0 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class PatchBezierLoss(nn.Module):
    """
    Multi-task Loss Suite for EXP_09 Patch-Bézier ViT:
    1. L_cls: Multi-Class Focal Loss across all 1024 patches
    2. L_ctrl: Smooth L1 Loss on 4 Bézier Control Points (active patches only)
    3. L_sample: Differentiable Sampled Points L1 Loss along curve B(t)
    4. L_tan: Tangent direction cosine alignment at entry and exit
    5. L_cont: Inter-patch boundary continuity loss
    """
    def __init__(
        self,
        lambda_cls: float = 2.0,
        lambda_ctrl: float = 5.0,
        lambda_sample: float = 5.0,
        lambda_tan: float = 1.0,
        lambda_cont: float = 0.5,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        num_samples: int = 10
    ):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_ctrl = lambda_ctrl
        self.lambda_sample = lambda_sample
        self.lambda_tan = lambda_tan
        self.lambda_cont = lambda_cont
        self.num_samples = num_samples
        
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction='mean')

    def forward(
        self,
        pred_dict: dict,
        target_classes: torch.Tensor,
        target_beziers: torch.Tensor,
        active_mask: torch.Tensor
    ) -> dict:
        """
        Args:
            pred_dict: Model output containing:
                flat_logits: (B, num_patches, C+1)
                flat_beziers: (B, num_patches, 4, 2)
                patch_beziers: (B, G, G, 4, 2)
            target_classes: (B, G, G) long class IDs
            target_beziers: (B, G, G, 4, 2) float in [0, 1]^2
            active_mask: (B, G, G) boolean indicator for active landmark patches
            
        Returns:
            dict with total_loss and individual loss metrics
        """
        flat_logits = pred_dict["flat_logits"]
        flat_beziers = pred_dict["flat_beziers"]
        patch_beziers = pred_dict["patch_beziers"]
        
        B, G, G_w = target_classes.shape
        num_patches = G * G_w
        
        flat_targets = target_classes.view(B * num_patches)
        flat_logits_2d = flat_logits.view(B * num_patches, -1)
        flat_active = active_mask.view(B * num_patches)
        
        # 1. Classification Loss (all patches)
        l_cls = self.focal_loss(flat_logits_2d, flat_targets)
        
        # Active patch regression components
        num_active = flat_active.sum().item()
        
        if num_active > 0:
            act_pred_beziers = flat_beziers.view(B * num_patches, 4, 2)[flat_active]  # (K, 4, 2)
            act_tgt_beziers = target_beziers.view(B * num_patches, 4, 2)[flat_active]  # (K, 4, 2)
            
            # 2. Control Point Regression Loss (Smooth L1)
            l_ctrl = F.smooth_l1_loss(act_pred_beziers, act_tgt_beziers, beta=0.02)
            
            # 3. Sampled Curve L1 Loss
            pred_samples = sample_cubic_bezier_torch(act_pred_beziers, num_samples=self.num_samples)  # (K, S, 2)
            tgt_samples = sample_cubic_bezier_torch(act_tgt_beziers, num_samples=self.num_samples)    # (K, S, 2)
            l_sample = F.l1_loss(pred_samples, tgt_samples)
            
            # 4. Tangent Cosine Alignment Loss (Entry & Exit)
            pred_t0 = act_pred_beziers[:, 1] - act_pred_beziers[:, 0]  # Entry tangent
            tgt_t0 = act_tgt_beziers[:, 1] - act_tgt_beziers[:, 0]
            pred_t1 = act_pred_beziers[:, 3] - act_pred_beziers[:, 2]  # Exit tangent
            tgt_t1 = act_tgt_beziers[:, 3] - act_tgt_beziers[:, 2]
            
            cos_sim0 = F.cosine_similarity(pred_t0, tgt_t0, dim=-1, eps=1e-6)
            cos_sim1 = F.cosine_similarity(pred_t1, tgt_t1, dim=-1, eps=1e-6)
            l_tan = (2.0 - cos_sim0.mean() - cos_sim1.mean()) * 0.5
            
            # 5. Inter-Patch Continuity Loss
            l_cont = self._compute_continuity_loss(patch_beziers, active_mask)
        else:
            dummy = flat_beziers.sum() * 0.0
            l_ctrl = dummy
            l_sample = dummy
            l_tan = dummy
            l_cont = dummy
            
        total_loss = (
            self.lambda_cls * l_cls +
            self.lambda_ctrl * l_ctrl +
            self.lambda_sample * l_sample +
            self.lambda_tan * l_tan +
            self.lambda_cont * l_cont
        )
        
        return {
            "loss": total_loss,
            "loss_cls": l_cls,
            "loss_ctrl": l_ctrl,
            "loss_sample": l_sample,
            "loss_tan": l_tan,
            "loss_cont": l_cont,
            "num_active_patches": num_active
        }

    def _compute_continuity_loss(self, patch_beziers: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
        """
        Penalizes endpoint mismatch between adjacent active patches across shared boundaries.
        """
        B, G_h, G_w, _, _ = patch_beziers.shape
        loss = torch.tensor(0.0, device=patch_beziers.device)
        count = 0
        
        # Horizontal neighbors: exit of patch (r, c) should align with entry of patch (r, c+1)
        # In global patch space: P3_x - 1.0 ≈ P0_x_next - 0.0 => (P3_x - 1.0) - P0_x_next ≈ 0
        act_left = active_mask[:, :, :-1]
        act_right = active_mask[:, :, 1:]
        act_h_pair = act_left & act_right
        
        if act_h_pair.sum() > 0:
            p3_h = patch_beziers[:, :, :-1, 3, :][act_h_pair]  # (N_pair, 2)
            p0_h = patch_beziers[:, :, 1:, 0, :][act_h_pair]   # (N_pair, 2)
            # Global offset in patch coordinate units: left patch is at c, right is at c+1
            # Distance in (u+c, v+r):
            # dx = (p3_h[:, 0] + c) - (p0_h[:, 0] + c + 1) = p3_h[:, 0] - p0_h[:, 0] - 1.0
            dx = p3_h[:, 0] - p0_h[:, 0] - 1.0
            dy = p3_h[:, 1] - p0_h[:, 1]
            dist_h = torch.sqrt(dx ** 2 + dy ** 2 + 1e-6)
            loss = loss + dist_h.mean()
            count += 1
            
        # Vertical neighbors: exit of patch (r, c) should align with entry of patch (r+1, c)
        act_top = active_mask[:, :-1, :]
        act_bot = active_mask[:, 1:, :]
        act_v_pair = act_top & act_bot
        
        if act_v_pair.sum() > 0:
            p3_v = patch_beziers[:, :-1, :, 3, :][act_v_pair]
            p0_v = patch_beziers[:, 1:, :, 0, :][act_v_pair]
            dx_v = p3_v[:, 0] - p0_v[:, 0]
            dy_v = p3_v[:, 1] - p0_v[:, 1] - 1.0
            dist_v = torch.sqrt(dx_v ** 2 + dy_v ** 2 + 1e-6)
            loss = loss + dist_v.mean()
            count += 1
            
        if count > 0:
            return loss / float(count)
        return patch_beziers.sum() * 0.0
