"""
Multi-Task Losses for EXPERIMENT_6 (Heatmap-Guided Junction-Steered Mask2Former).
Combines:
  1. Standard Mask2Former Hungarian Matching Loss (Cross-Entropy + Dice + BCE).
  2. Continuous Gaussian Focal Loss on 2D Spatial Heatmaps (CenterNet style).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianFocalLoss(nn.Module):
    """
    Modified Focal Loss for continuous Gaussian heatmaps (Law & Deng / CenterNet).
    Penalty-reduced focal loss for near-misses; hard suppression for empty background.
    """
    def __init__(self, alpha=2.0, beta=4.0, eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, pred, target):
        """
        Args:
            pred (Tensor): (B, 4, H, W) sigmoid predictions in [0, 1].
            target (Tensor): (B, 4, H, W) ground-truth Gaussian heatmaps in [0, 1].
        """
        pred = torch.clamp(pred, min=self.eps, max=1.0 - self.eps)
        
        # Positive locations (peaks where target == 1.0)
        pos_mask = target.eq(1.0)
        neg_mask = ~pos_mask
        
        pos_loss = -torch.pow(1.0 - pred[pos_mask], self.alpha) * torch.log(pred[pos_mask])
        neg_loss = -torch.pow(1.0 - target[neg_mask], self.beta) * torch.pow(pred[neg_mask], self.alpha) * torch.log(1.0 - pred[neg_mask])
        
        num_pos = pos_mask.float().sum()
        if num_pos > 0:
            total_loss = (pos_loss.sum() + neg_loss.sum()) / num_pos
        else:
            B, K = pred.shape[0], pred.shape[1]
            total_loss = neg_loss.sum() / max(float(B * K), 1.0)
            
        return total_loss


class HeatmapSteeredLoss(nn.Module):
    def __init__(self, lambda_m2f=1.0, lambda_heatmap=1.0):
        super().__init__()
        self.lambda_m2f = lambda_m2f
        self.lambda_heatmap = lambda_heatmap
        self.heatmap_criterion = GaussianFocalLoss(alpha=2.0, beta=4.0)

    def forward(self, model_outputs, gt_heatmaps):
        """
        Args:
            model_outputs (dict): Contains 'm2f_loss', 'pred_heatmaps'.
            gt_heatmaps (Tensor): (B, 4, H, W) ground-truth heatmaps in [0, 1].
        """
        m2f_loss = model_outputs.get('m2f_loss', torch.tensor(0.0, device=gt_heatmaps.device))
        pred_heatmaps = model_outputs['pred_heatmaps']
        
        # Compute continuous Gaussian focal loss
        heatmap_loss = self.heatmap_criterion(pred_heatmaps, gt_heatmaps)
        
        total_loss = self.lambda_m2f * m2f_loss + self.lambda_heatmap * heatmap_loss
        
        return total_loss, {
            'loss': float(total_loss.item()),
            'm2f_loss': float(m2f_loss.item()) if torch.is_tensor(m2f_loss) else float(m2f_loss),
            'heatmap_loss': float(heatmap_loss.item())
        }
