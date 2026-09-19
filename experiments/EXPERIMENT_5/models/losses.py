"""
Multi-Task Loss for Junction-Steered Mask2Former (EXPERIMENT_5).
Combines:
  1. Mask2Former Hungarian Matching Loss (Classification Focal + Mask BCE + Mask Dice)
  2. Visibility-Masked Coordinate Smooth-L1 Loss on 4 Biological Junctions
  3. Binary Cross-Entropy on Junction Visibility Flags
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class JunctionSteeredLoss(nn.Module):
    def __init__(self, lambda_m2f=1.0, lambda_coord=5.0, lambda_vis=1.0):
        super().__init__()
        self.lambda_m2f = lambda_m2f
        self.lambda_coord = lambda_coord
        self.lambda_vis = lambda_vis
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, model_outputs, gt_junction_coords, gt_junction_vis):
        """
        Args:
            model_outputs (dict): Output dict from JunctionSteeredMask2Former containing:
                - 'm2f_loss': scalar Hungarian loss
                - 'pred_junction_coords': (B, 4, 2) in [0, 1]^2
                - 'pred_junction_vis': (B, 4) visibility logits
            gt_junction_coords (Tensor): (B, 4, 2) normalized GT coordinates
            gt_junction_vis (Tensor): (B, 4) binary GT visibility flags {0, 1}
        Returns:
            total_loss (Tensor): scalar
            loss_dict (dict): component breakdowns
        """
        device = gt_junction_coords.device
        pred_coords = model_outputs['pred_junction_coords'] # (B, 4, 2)
        pred_vis_logits = model_outputs['pred_junction_vis'] # (B, 4)
        
        # 1. Mask2Former Base Loss
        m2f_loss = model_outputs.get('m2f_loss', torch.tensor(0.0, device=device))
        
        # 2. Junction Visibility Loss (BCE)
        vis_loss = self.bce_loss(pred_vis_logits, gt_junction_vis)
        
        # 3. Visibility-Masked Coordinate Loss (Smooth L1)
        # Coordinate loss is only computed for junctions visible in GT
        vis_mask = (gt_junction_vis > 0.5) # (B, 4)
        if vis_mask.sum() > 0:
            # Expand mask for 2D coords: (B, 4, 2)
            coord_mask = vis_mask.unsqueeze(-1).expand_as(pred_coords)
            diff = F.smooth_l1_loss(
                pred_coords[coord_mask],
                gt_junction_coords[coord_mask],
                beta=0.02,
                reduction='mean'
            )
            coord_loss = diff
        else:
            coord_loss = torch.tensor(0.0, device=device)
            
        # Total Weighted Multi-Task Loss
        total_loss = (self.lambda_m2f * m2f_loss +
                      self.lambda_coord * coord_loss +
                      self.lambda_vis * vis_loss)
                      
        loss_dict = {
            'loss': total_loss.item(),
            'm2f_loss': m2f_loss.item() if isinstance(m2f_loss, torch.Tensor) else m2f_loss,
            'coord_loss': coord_loss.item() if isinstance(coord_loss, torch.Tensor) else coord_loss,
            'vis_loss': vis_loss.item() if isinstance(vis_loss, torch.Tensor) else vis_loss
        }
        
        return total_loss, loss_dict
