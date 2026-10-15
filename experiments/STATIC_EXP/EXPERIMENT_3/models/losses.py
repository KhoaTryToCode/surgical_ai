import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Multi-class focal loss.
    FL = -alpha * (1-pt)^gamma * log(pt)
    """
    def __init__(self, gamma=2.0, alpha=0.25, ignore_index=-1):
        """
        Initializes the FocalLoss.
        Args:
            gamma (float): Gamma parameter.
            alpha (float): Alpha parameter.
            ignore_index (int): Index to ignore in loss calculation.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        """
        Forward pass.
        Args:
            pred: (N, C) logits
            target: (N,) int class labels
        Returns:
            Scalar loss
        """
        ce = F.cross_entropy(pred, target, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        return torch.mean(focal_weight * ce)


class BezierPatchLoss(nn.Module):
    """
    Loss function for Bezier patch model.
    """
    def __init__(self,
                 lambda_cls=2.0, lambda_ctrl=5.0, lambda_sample=2.0,
                 lambda_cont=1.0, lambda_tan=0.5,
                 continuity_phase_epoch=31,
                 num_sample_pts=10, ctrl_beta=0.02):
        """
        Initializes the BezierPatchLoss.
        Args:
            lambda_cls: Weight for classification loss.
            lambda_ctrl: Weight for control point loss.
            lambda_sample: Weight for sampled point loss.
            lambda_cont: Weight for continuity loss.
            lambda_tan: Weight for tangent loss.
            continuity_phase_epoch: Epoch to start continuity loss.
            num_sample_pts: Number of points to sample for curve evaluation.
            ctrl_beta: Beta parameter for smooth L1 loss.
        """
        super(BezierPatchLoss, self).__init__()
        self.lambda_cls = lambda_cls
        self.lambda_ctrl = lambda_ctrl
        self.lambda_sample = lambda_sample
        self.lambda_cont = lambda_cont
        self.lambda_tan = lambda_tan
        self.continuity_phase_epoch = continuity_phase_epoch
        self.num_sample_pts = num_sample_pts
        self.ctrl_beta = ctrl_beta
        self.focal_loss = FocalLoss(gamma=2.0, alpha=0.25)

    def forward(self, pred_class, pred_bezier, target_class, target_bezier, active_mask, epoch=1):
        """
        Compute total loss.
        Args:
            pred_class: (B, 64, 4) logits
            pred_bezier: (B, 64, 4, 2) control points in [0,1]^2
            target_class: (B, 64) int labels [0..3]
            target_bezier: (B, 64, 4, 2) GT control points in [0,1]^2
            active_mask: (B, 64) bool True where target_class > 0
            epoch: int current epoch
        Returns:
            Tuple (total_loss, loss_dict)
        """
        B, G = pred_class.shape[:2]  # B=batch, G=64
        
        # === L_cls: Focal Loss on all 64 patches ===
        L_cls = self.focal_loss(pred_class.view(B*G, 4), target_class.view(B*G))
        
        # === L_ctrl: Smooth L1 on active patches ===
        if active_mask.any():
            L_ctrl = F.smooth_l1_loss(
                pred_bezier[active_mask],    # (N_active, 4, 2)
                target_bezier[active_mask],  # (N_active, 4, 2)
                beta=self.ctrl_beta
            )
        else:
            L_ctrl = pred_bezier.sum() * 0.0  # zero gradient
            
        # === L_sample: L1 on Bernstein-sampled curve points ===
        if active_mask.any():
            # Import bernstein_eval_torch inline to avoid circular import
            from experiments.EXPERIMENT_3.models.bezier_utils import bernstein_eval_torch
            pred_pts = bernstein_eval_torch(pred_bezier[active_mask], self.num_sample_pts)   # (N_active, T, 2)
            gt_pts   = bernstein_eval_torch(target_bezier[active_mask], self.num_sample_pts) # (N_active, T, 2)
            L_sample = F.l1_loss(pred_pts, gt_pts)
        else:
            L_sample = pred_bezier.sum() * 0.0
            
        total_loss = self.lambda_cls * L_cls + self.lambda_ctrl * L_ctrl + self.lambda_sample * L_sample
        
        L_cont_val = 0.0
        L_tan_val  = 0.0
        
        # === Phase 2: Continuity losses (epoch >= continuity_phase_epoch) ===
        if epoch >= self.continuity_phase_epoch:
            # For each batch item, find adjacent active patch pairs and penalize boundary gaps
            grid_size = int(G ** 0.5)  # 8
            cont_losses = []
            tan_losses  = []
            
            for b in range(B):
                active_b = active_mask[b]   # (64,)
                bezier_b = pred_bezier[b]   # (64, 4, 2)
                class_b  = target_class[b]  # (64,)
                
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
                            
                            # Tangent direction matching
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
        
        loss_dict = {
            'loss':        total_loss.item(),
            'cls_loss':    L_cls.item(),
            'ctrl_loss':   L_ctrl.item() if active_mask.any() else 0.0,
            'sample_loss': L_sample.item() if active_mask.any() else 0.0,
            'cont_loss':   L_cont_val,
            'tan_loss':    L_tan_val,
        }
        return total_loss, loss_dict
