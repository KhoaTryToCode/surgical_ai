"""
EXP_13: Joint Continuous-Discrete Loss Suite
============================================
Unifies:
1. Mask2Former Multi-Class Pixel Loss (Cross-Entropy + Soft Dice)
2. TopoNet Centerline Constraint Loss (soft_dice_cldice on semantic map)
3. BCRNet Bidirectional Bézier Curve Loss (Chamfer L1 + Cosine Tangent)
4. Mask-Curve Mutual Containment Loss (Curve inside predicted mask)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from .toponet_cldice import SoftDiceCLDiceLoss, AnalyticalContinuousclDice
try:
    from ..utils.bezier_ops import evaluate_bezier_tangents
except (ImportError, ValueError):
    from utils.bezier_ops import evaluate_bezier_tangents


class JointUnifiedLoss(nn.Module):
    """
    Master continuous-discrete multi-objective loss for EXP_13.
    """

    def __init__(
        self,
        num_classes: int = 3,
        lambda_m2f_ce: float = 1.0,
        lambda_m2f_mask: float = 2.0,
        lambda_toponet: float = 1.0,
        lambda_crv: float = 3.0,
        lambda_tangent: float = 0.5,
        lambda_contain: float = 0.5,
        skel_num_iter: int = 20,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_m2f_ce = lambda_m2f_ce
        self.lambda_m2f_mask = lambda_m2f_mask
        self.lambda_toponet = lambda_toponet
        self.lambda_crv = lambda_crv
        self.lambda_tangent = lambda_tangent
        self.lambda_contain = lambda_contain

        # 1. TopoNet Centerline Loss on 4-class semantic map
        self.cldice_loss_fn = SoftDiceCLDiceLoss(
            alpha=0.5, smooth=1e-5, num_iter=skel_num_iter, exclude_background=True
        )

        # 2. Analytical Continuous clDice for sub-pixel curves
        self.ac_cldice = AnalyticalContinuousclDice()

        # 3. Cross Entropy Loss for semantic segmentation
        self.ce_loss_fn = nn.CrossEntropyLoss(
            weight=torch.tensor([0.2, 1.0, 1.0, 1.0], dtype=torch.float32)
        )

    def compute_curve_loss(
        self,
        pred_ctrl: torch.Tensor,       # (B, M, K, 6, 2)
        pred_curve: torch.Tensor,      # (B, M, K, 26, 2)
        pred_scores: torch.Tensor,     # (B, M, K)
        gt_ctrl: torch.Tensor,         # (B, M, 6, 2)
        gt_curve: torch.Tensor,        # (B, M, 26, 2)
        gt_exists: torch.Tensor,       # (B, M)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes bidirectional min-matching Chamfer distance, tangent error, and score loss.
        """
        B, M, K, N, _ = pred_curve.shape
        device = pred_curve.device

        loss_crv = torch.tensor(0.0, device=device)
        loss_tangent = torch.tensor(0.0, device=device)
        loss_score = torch.tensor(0.0, device=device)
        valid_count = 0

        # Evaluate GT tangents
        gt_ctrl_flat = gt_ctrl.view(B * M, -1, 2)
        gt_tangents = evaluate_bezier_tangents(gt_ctrl_flat, num_samples=N).view(B, M, N, 2)

        for b in range(B):
            for m in range(M):
                if gt_exists[b, m] < 0.5:
                    # Penalize false-positive proposals
                    loss_score = loss_score + F.binary_cross_entropy(pred_scores[b, m], torch.zeros_like(pred_scores[b, m]))
                    continue

                valid_count += 1
                gt_c = gt_curve[b, m]           # (N, 2)
                gt_c_rev = torch.flip(gt_c, [0]) # (N, 2)
                gt_t = gt_tangents[b, m]        # (N, 2)

                # Evaluate over all K proposals, find best matching proposal
                k_errors = []
                for k in range(K):
                    p_c = pred_curve[b, m, k]   # (N, 2)

                    # Bidirectional L1 Chamfer distance
                    d_fwd = torch.mean(torch.norm(p_c - gt_c, dim=-1))
                    d_rev = torch.mean(torch.norm(p_c - gt_c_rev, dim=-1))
                    d_min = torch.min(d_fwd, d_rev)
                    k_errors.append(d_min)

                k_errors_t = torch.stack(k_errors)  # (K,)
                best_k = torch.argmin(k_errors_t)

                # Loss on best proposal
                loss_crv = loss_crv + k_errors_t[best_k]

                # Tangent loss on best proposal
                p_ctrl_best = pred_ctrl[b, m, best_k:best_k+1]  # (1, 6, 2)
                p_tangents = evaluate_bezier_tangents(p_ctrl_best, num_samples=N).squeeze(0)  # (N, 2)
                cos_fwd = torch.sum(p_tangents * gt_t, dim=-1)
                cos_rev = torch.sum(p_tangents * (-gt_t), dim=-1)
                cos_sim = torch.max(cos_fwd, cos_rev)
                loss_tangent = loss_tangent + torch.mean(1.0 - cos_sim)

                # Proposal confidence supervision
                # Best proposal target is 1.0, others target 0.0
                score_target = torch.zeros(K, device=device)
                score_target[best_k] = 1.0
                loss_score = loss_score + F.binary_cross_entropy(pred_scores[b, m], score_target)

        if valid_count > 0:
            loss_crv = loss_crv / float(valid_count)
            loss_tangent = loss_tangent / float(valid_count)
            loss_score = loss_score / float(valid_count)

        return loss_crv, loss_tangent, loss_score

    def forward(
        self,
        model_outputs: dict,
        batch: dict,
    ) -> Dict[str, torch.Tensor]:
        device = model_outputs["semantic_logits"].device
        B = model_outputs["semantic_logits"].shape[0]

        # Targets
        gt_semantic = batch["gt_semantic"].to(device)  # (B, H, W) in {0, 1, 2, 3}
        gt_masks = batch["gt_masks"].to(device)        # (B, 3, H, W)
        gt_ctrl = batch["gt_ctrl_pts"].to(device)      # (B, 3, 6, 2)
        gt_curve = batch["gt_sample_pts"].to(device)   # (B, 3, 26, 2)
        gt_exists = batch["gt_exists"].to(device)      # (B, 3)

        # 1. Mask2Former Cross-Entropy Loss on Semantic Map
        sem_logits = model_outputs["semantic_logits"]  # (B, 4, H, W)
        self.ce_loss_fn.weight = self.ce_loss_fn.weight.to(device)
        loss_ce = self.ce_loss_fn(sem_logits, gt_semantic)

        # One-hot target for clDice
        gt_semantic_onehot = F.one_hot(gt_semantic, num_classes=self.num_classes + 1).permute(0, 3, 1, 2).float()

        # 2. TopoNet Soft Center-Line clDice Loss
        loss_toponet = self.cldice_loss_fn(sem_logits, gt_semantic_onehot)

        # 3. BCRNet Parametric Curve Losses
        pred_ctrl = model_outputs["final_ctrl_pts"]    # (B, M, K, 6, 2)
        pred_curve = model_outputs["final_curve_pts"]  # (B, M, K, 26, 2)
        pred_scores = model_outputs["final_scores"]    # (B, M, K)

        loss_crv, loss_tangent, loss_score = self.compute_curve_loss(
            pred_ctrl=pred_ctrl,
            pred_curve=pred_curve,
            pred_scores=pred_scores,
            gt_ctrl=gt_ctrl,
            gt_curve=gt_curve,
            gt_exists=gt_exists,
        )

        # 4. Mask-Curve Mutual Containment Loss
        # Samples predicted semantic logits at curve coordinates:
        # Curve must reside inside predicted landmark mask (channels 1..3)
        loss_contain = torch.tensor(0.0, device=device)
        contain_count = 0
        sem_probs = F.softmax(sem_logits, dim=1)  # (B, 4, H, W)

        for m in range(self.num_classes):
            # Select best proposal curve per class: (B, 26, 2)
            best_k = torch.argmax(pred_scores[:, m], dim=-1)  # (B,)
            pts_best = torch.stack([pred_curve[b, m, best_k[b]] for b in range(B)], dim=0) # (B, 26, 2)

            # Class mask probability channel (m + 1):
            prob_m = sem_probs[:, m+1:m+2]  # (B, 1, H, W)

            # Bilinear sample at curve points
            grid = pts_best.unsqueeze(2) * 2.0 - 1.0  # (B, 26, 1, 2)
            sampled_prob = F.grid_sample(prob_m, grid, align_corners=False, mode="bilinear").squeeze() # (B, 26)

            contain_penalty = (1.0 - sampled_prob).mean(dim=-1) * gt_exists[:, m]
            loss_contain = loss_contain + contain_penalty.sum()
            contain_count += gt_exists[:, m].sum().item()

        if contain_count > 0:
            loss_contain = loss_contain / float(contain_count)

        # Total Loss
        loss_total = (
            self.lambda_m2f_ce * loss_ce +
            self.lambda_toponet * loss_toponet +
            self.lambda_crv * loss_crv +
            self.lambda_tangent * loss_tangent +
            self.lambda_contain * loss_contain +
            0.5 * loss_score
        )

        return {
            "loss_total": loss_total,
            "loss_ce": loss_ce,
            "loss_toponet": loss_toponet,
            "loss_crv": loss_crv,
            "loss_tangent": loss_tangent,
            "loss_contain": loss_contain,
            "loss_score": loss_score,
        }
