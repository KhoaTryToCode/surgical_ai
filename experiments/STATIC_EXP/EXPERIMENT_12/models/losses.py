"""
Multi-Task Loss for Dual-Decoder Mask2Former (EXPERIMENT_12).
Combines:
  1. Landmark Objective (L3D):
     - Mask2Former Hungarian Matching Loss (Focal + BCE + Dice on Ridge, Silhouette, Falciform)
     - Visibility-Masked Coordinate Smooth-L1 Loss on 4 Continuous Junctions
     - Binary Cross-Entropy on Junction Visibility Flags
  2. Scene Objective (CholecSeg8k):
     - Mask2Former Hungarian Matching Loss on 13 Surgical Scene Classes (Liver, Gallbladder, Tools, etc.)
  3. Joint Total Loss:
     L_total = L_L3D + lambda_cholec * L_cholec
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskDualLoss(nn.Module):
    def __init__(self, lambda_m2f=1.0, lambda_coord=5.0, lambda_vis=1.0, lambda_cholec=0.5):
        super().__init__()
        self.lambda_m2f = lambda_m2f
        self.lambda_coord = lambda_coord
        self.lambda_vis = lambda_vis
        self.lambda_cholec = lambda_cholec
        self.bce_loss = nn.BCEWithLogitsLoss()

    def compute_landmark_loss(self, model_outputs, gt_junction_coords, gt_junction_vis):
        """
        Computes the complete loss for an L3D landmark batch.
        """
        device = gt_junction_coords.device
        pred_coords = model_outputs['pred_junction_coords']
        pred_vis_logits = model_outputs['pred_junction_vis']
        
        # 1. Base Hungarian loss
        m2f_loss = model_outputs.get('m2f_loss', torch.tensor(0.0, device=device))
        
        # 2. Junction visibility loss (BCE)
        vis_loss = self.bce_loss(pred_vis_logits, gt_junction_vis)
        
        # 3. Visibility-masked coordinate loss (Smooth L1)
        vis_mask = (gt_junction_vis > 0.5)
        if vis_mask.sum() > 0:
            coord_mask = vis_mask.unsqueeze(-1).expand_as(pred_coords)
            coord_loss = F.smooth_l1_loss(
                pred_coords[coord_mask],
                gt_junction_coords[coord_mask],
                beta=0.02,
                reduction='mean'
            )
        else:
            coord_loss = torch.tensor(0.0, device=device)
            
        total_landmark_loss = (self.lambda_m2f * m2f_loss +
                               self.lambda_coord * coord_loss +
                               self.lambda_vis * vis_loss)
                               
        loss_dict = {
            'landmark_loss': total_landmark_loss.item(),
            'landmark_m2f_loss': m2f_loss.item() if isinstance(m2f_loss, torch.Tensor) else m2f_loss,
            'coord_loss': coord_loss.item() if isinstance(coord_loss, torch.Tensor) else coord_loss,
            'vis_loss': vis_loss.item() if isinstance(vis_loss, torch.Tensor) else vis_loss
        }
        return total_landmark_loss, loss_dict

    def compute_scene_loss(self, model_outputs):
        """
        Computes the Hungarian matching loss for a CholecSeg8k scene batch.
        """
        device = next(iter(model_outputs.values())).device
        scene_loss = model_outputs.get('scene_loss', torch.tensor(0.0, device=device))
        
        loss_dict = {
            'scene_loss': scene_loss.item() if isinstance(scene_loss, torch.Tensor) else scene_loss
        }
        return scene_loss, loss_dict

    def combine_losses(self, landmark_loss, scene_loss):
        """
        Combines landmark and scene loss with balancing scalar lambda_cholec.
        L_total = L_L3D + lambda_cholec * L_cholec
        """
        total = landmark_loss + self.lambda_cholec * scene_loss
        return total
