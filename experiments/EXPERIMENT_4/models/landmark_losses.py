#!/usr/bin/env python3
"""
EXPERIMENT_4: Loss Functions for Landmark Master Tokens + Patch Bézier Decoder
Combines:
  - Multi-class Focal Loss (Classification across 64 patches)
  - Smooth L1 Control Point Loss (Active patches)
  - Bernstein Curve Sampling Loss (Active patches)
  - Phase 2 C0 Continuity & C1 Tangent Alignment (Epoch >= 31)
  - Auxiliary Landmark BCE Presence & Masked Centroid Smooth L1 Loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from experiments.EXPERIMENT_4.models.bezier_utils import bernstein_eval_torch

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, ignore_index=-1):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        return torch.mean(focal_weight * ce_loss)

class LandmarkBezierLoss(nn.Module):
    """
    Unified loss combining local patch Bézier losses with macroscopic landmark centroid & presence grounding.
    """
    def __init__(self, lambda_cls=2.0, lambda_ctrl=5.0, lambda_sample=2.0,
                 lambda_cont=1.0, lambda_tan=0.5, lambda_lm=1.0,
                 continuity_phase_epoch=31, num_sample_pts=10, ctrl_beta=0.02):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_ctrl = lambda_ctrl
        self.lambda_sample = lambda_sample
        self.lambda_cont = lambda_cont
        self.lambda_tan = lambda_tan
        self.lambda_lm = lambda_lm
        self.continuity_phase_epoch = continuity_phase_epoch
        self.num_sample_pts = num_sample_pts
        self.ctrl_beta = ctrl_beta
        self.focal_loss = FocalLoss(gamma=2.0, alpha=0.25)

    def forward(self, pred_class, pred_bezier, pred_presence, pred_centroid,
                target_class, target_bezier, active_mask,
                target_lm_presence, target_lm_centroid, epoch=1):
        """
        Args:
            pred_class: (B, 64, 4)
            pred_bezier: (B, 64, 4, 2)
            pred_presence: (B, 3) logits
            pred_centroid: (B, 3, 2) in [0, 1]^2
            target_class: (B, 64)
            target_bezier: (B, 64, 4, 2)
            active_mask: (B, 64) bool
            target_lm_presence: (B, 3) float in {0, 1}
            target_lm_centroid: (B, 3, 2) float in [0, 1]^2
            epoch: current training epoch
        """
        B, G = pred_class.shape[:2]
        
        # 1. Patch Classification Focal Loss
        L_cls = self.focal_loss(pred_class.view(B * G, 4), target_class.view(B * G))
        
        # 2. Control Point Smooth L1 & Bernstein Sampled L1
        if active_mask.any():
            L_ctrl = F.smooth_l1_loss(
                pred_bezier[active_mask], target_bezier[active_mask], beta=self.ctrl_beta
            )
            pred_sampled = bernstein_eval_torch(pred_bezier[active_mask], num_samples=self.num_sample_pts)
            gt_sampled   = bernstein_eval_torch(target_bezier[active_mask], num_samples=self.num_sample_pts)
            L_sample     = F.l1_loss(pred_sampled, gt_sampled)
        else:
            L_ctrl   = pred_bezier.sum() * 0.0
            L_sample = pred_bezier.sum() * 0.0
            
        total_loss = self.lambda_cls * L_cls + self.lambda_ctrl * L_ctrl + self.lambda_sample * L_sample
        
        # 3. Phase 2: Patch Continuity & Tangent Losses (Epoch >= continuity_phase_epoch)
        L_cont_val = 0.0
        L_tan_val  = 0.0
        
        if epoch >= self.continuity_phase_epoch:
            grid_size = int(G ** 0.5)
            cont_losses = []
            tan_losses  = []
            
            for b in range(B):
                active_b = active_mask[b]
                bezier_b = pred_bezier[b]
                class_b  = target_class[b]
                
                # Horizontal neighbors (left-right)
                for r in range(grid_size):
                    for c in range(grid_size - 1):
                        idx_left  = r * grid_size + c
                        idx_right = r * grid_size + c + 1
                        if active_b[idx_left] and active_b[idx_right] and (class_b[idx_left] == class_b[idx_right]) and (class_b[idx_left] > 0):
                            P0_left = torch.stack([c + bezier_b[idx_left, 0, 0], r + bezier_b[idx_left, 0, 1]])
                            P3_left = torch.stack([c + bezier_b[idx_left, 3, 0], r + bezier_b[idx_left, 3, 1]])
                            P0_right = torch.stack([(c + 1) + bezier_b[idx_right, 0, 0], r + bezier_b[idx_right, 0, 1]])
                            P3_right = torch.stack([(c + 1) + bezier_b[idx_right, 3, 0], r + bezier_b[idx_right, 3, 1]])
                            
                            gap_fwd = (P3_left - P0_right).pow(2).sum()
                            gap_rev = (P0_left - P3_right).pow(2).sum()
                            cont_losses.append(torch.minimum(gap_fwd, gap_rev))
                            
                            if gap_fwd <= gap_rev:
                                exit_tan  = bezier_b[idx_left, 3]  - bezier_b[idx_left, 2]
                                entry_tan = bezier_b[idx_right, 1] - bezier_b[idx_right, 0]
                            else:
                                exit_tan  = bezier_b[idx_right, 3]  - bezier_b[idx_right, 2]
                                entry_tan = bezier_b[idx_left, 1] - bezier_b[idx_left, 0]
                            cos_sim = F.cosine_similarity(exit_tan.unsqueeze(0), entry_tan.unsqueeze(0)).squeeze(0)
                            tan_losses.append(1.0 - cos_sim)
                            
                # Vertical neighbors (top-bottom)
                for r in range(grid_size - 1):
                    for c in range(grid_size):
                        idx_top    = r * grid_size + c
                        idx_bottom = (r + 1) * grid_size + c
                        if active_b[idx_top] and active_b[idx_bottom] and (class_b[idx_top] == class_b[idx_bottom]) and (class_b[idx_top] > 0):
                            P0_top = torch.stack([c + bezier_b[idx_top, 0, 0], r + bezier_b[idx_top, 0, 1]])
                            P3_top = torch.stack([c + bezier_b[idx_top, 3, 0], r + bezier_b[idx_top, 3, 1]])
                            P0_bot = torch.stack([c + bezier_b[idx_bottom, 0, 0], (r + 1) + bezier_b[idx_bottom, 0, 1]])
                            P3_bot = torch.stack([c + bezier_b[idx_bottom, 3, 0], (r + 1) + bezier_b[idx_bottom, 3, 1]])
                            
                            gap_v_fwd = (P3_top - P0_bot).pow(2).sum()
                            gap_v_rev = (P0_top - P3_bot).pow(2).sum()
                            cont_losses.append(torch.minimum(gap_v_fwd, gap_v_rev))
                            
                            if gap_v_fwd <= gap_v_rev:
                                exit_tan  = bezier_b[idx_top, 3]  - bezier_b[idx_top, 2]
                                entry_tan = bezier_b[idx_bottom, 1] - bezier_b[idx_bottom, 0]
                            else:
                                exit_tan  = bezier_b[idx_bottom, 3]  - bezier_b[idx_bottom, 2]
                                entry_tan = bezier_b[idx_top, 1] - bezier_b[idx_top, 0]
                            cos_sim = F.cosine_similarity(exit_tan.unsqueeze(0), entry_tan.unsqueeze(0)).squeeze(0)
                            tan_losses.append(1.0 - cos_sim)
                            
            if cont_losses:
                L_cont = torch.stack(cont_losses).mean()
                L_cont_val = L_cont.item()
                total_loss = total_loss + self.lambda_cont * L_cont
                
            if tan_losses:
                L_tan = torch.stack(tan_losses).mean()
                L_tan_val = L_tan.item()
                total_loss = total_loss + self.lambda_tan * L_tan

        # 4. Auxiliary Landmark Loss (Presence BCE + Masked Centroid Smooth L1)
        L_lm_bce = F.binary_cross_entropy_with_logits(pred_presence, target_lm_presence)
        
        lm_mask = target_lm_presence > 0.5 # (B, 3)
        if lm_mask.any():
            L_lm_com = F.smooth_l1_loss(
                pred_centroid[lm_mask], target_lm_centroid[lm_mask], beta=0.02
            )
        else:
            L_lm_com = pred_centroid.sum() * 0.0
            
        L_landmark = L_lm_bce + L_lm_com
        total_loss = total_loss + self.lambda_lm * L_landmark

        loss_dict = {
            'loss':        total_loss.item(),
            'cls_loss':    L_cls.item(),
            'ctrl_loss':   L_ctrl.item() if active_mask.any() else 0.0,
            'sample_loss': L_sample.item() if active_mask.any() else 0.0,
            'cont_loss':   L_cont_val,
            'tan_loss':    L_tan_val,
            'lm_bce':      L_lm_bce.item(),
            'lm_com':      L_lm_com.item() if lm_mask.any() else 0.0,
        }
        
        return total_loss, loss_dict
