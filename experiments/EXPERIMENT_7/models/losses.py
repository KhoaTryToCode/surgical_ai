import torch
import torch.nn as nn
import torch.nn.functional as F

class ManifoldMultiTaskLoss(nn.Module):
    """
    Multi-Task Loss for EXPERIMENT_7:
      L_total = L_m2f + lambda_vis * L_vis + lambda_coord * L_coord + lambda_uv * L_uv
      
    Features:
      - Visibility-Masked Coordinate Loss (only supervises on-screen queries)
      - Conditional Manifold Masking (disables u-loss when Falciform is absent)
    """
    def __init__(self, lambda_vis=1.0, lambda_coord=5.0, lambda_uv=1.0, coord_beta=0.02):
        super().__init__()
        self.lambda_vis = lambda_vis
        self.lambda_coord = lambda_coord
        self.lambda_uv = lambda_uv
        self.coord_beta = coord_beta

    def forward(self, m2f_loss, pred_coords, pred_vis, pred_uv,
                gt_coords, gt_vis, gt_uv, gt_liver_mask, has_falc):
        """
        Args:
            m2f_loss: Tensor scalar (Hungarian matching loss)
            pred_coords: Tensor (B, 11, 2)
            pred_vis: Tensor (B, 11) logits
            pred_uv: Tensor (B, 2, H, W)
            gt_coords: Tensor (B, 11, 2)
            gt_vis: Tensor (B, 11) binary in {0, 1}
            gt_uv: Tensor (B, 2, H, W)
            gt_liver_mask: Tensor (B, H, W)
            has_falc: Tensor (B,) float in {0.0, 1.0}
        """
        device = pred_coords.device
        
        # ── 1. Visibility Loss (BCEWithLogits across all 11 queries) ──────────
        loss_vis = F.binary_cross_entropy_with_logits(pred_vis, gt_vis)

        # ── 2. Visibility-Masked Coordinate Loss (Smooth-L1) ─────────────────
        vis_mask = (gt_vis > 0.5)  # (B, 11)
        if vis_mask.sum() > 0:
            p_c = pred_coords[vis_mask]
            g_c = gt_coords[vis_mask]
            loss_coord = F.smooth_l1_loss(p_c, g_c, beta=self.coord_beta, reduction="mean")
        else:
            loss_coord = torch.tensor(0.0, device=device)

        # ── 3. Masked Dense Manifold Loss ────────────────────────────────────
        # Only evaluate on liver parenchyma pixels
        fg_mask = (gt_liver_mask > 0.5)  # (B, H, W)
        if fg_mask.sum() > 0:
            # v-coordinate is always physically grounded (Anterior Ridge to Silhouette)
            v_err = F.l1_loss(pred_uv[:, 1][fg_mask], gt_uv[:, 1][fg_mask], reduction="mean")
            
            # u-coordinate is ONLY evaluated when Falciform is physically present
            # has_falc: (B,) -> expand to mask
            falc_per_pixel = has_falc.view(-1, 1, 1).expand_as(gt_liver_mask)[fg_mask]
            if falc_per_pixel.sum() > 0:
                u_diff = torch.abs(pred_uv[:, 0][fg_mask] - gt_uv[:, 0][fg_mask])
                u_err = (u_diff * falc_per_pixel).sum() / (falc_per_pixel.sum() + 1e-6)
            else:
                u_err = torch.tensor(0.0, device=device)
                
            loss_uv = v_err + u_err
        else:
            loss_uv = torch.tensor(0.0, device=device)

        # Total combined loss
        total_loss = (
            m2f_loss
            + self.lambda_vis * loss_vis
            + self.lambda_coord * loss_coord
            + self.lambda_uv * loss_uv
        )

        loss_breakdown = {
            "total_loss": total_loss.item(),
            "m2f_loss": m2f_loss.item(),
            "loss_vis": loss_vis.item(),
            "loss_coord": loss_coord.item(),
            "loss_uv": loss_uv.item()
        }

        return total_loss, loss_breakdown
