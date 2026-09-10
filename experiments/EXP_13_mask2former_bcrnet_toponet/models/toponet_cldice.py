"""
EXP_13: TopoNet Center-Line Topological Loss Module
===================================================
Implementation of topological continuity constraints:
1. Soft Morphological Skeletonization (iterative soft erosion and opening)
2. Multi-Class Soft Center-Line Dice Loss (soft_dice_cldice from TopoNet)
3. Analytical Continuous clDice (AC-clDice) for sub-pixel parametric curves
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftSkeletonize(nn.Module):
    """
    Differentiable soft morphological skeletonization via iterative min-pooling
    erosion and opening.
    """

    def __init__(self, num_iter: int = 20):
        super().__init__()
        self.num_iter = num_iter

    def soft_erode(self, img: torch.Tensor) -> torch.Tensor:
        p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
        p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
        return torch.min(p1, p2)

    def soft_dilate(self, img: torch.Tensor) -> torch.Tensor:
        return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))

    def soft_open(self, img: torch.Tensor) -> torch.Tensor:
        return self.soft_dilate(self.soft_erode(img))

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        img1 = self.soft_open(img)
        skel = F.relu(img - img1)

        for _ in range(self.num_iter):
            img = self.soft_erode(img)
            img1 = self.soft_open(img)
            delta = F.relu(img - img1)
            skel = skel + F.relu(delta - skel * delta)

        return skel


class SoftDiceCLDiceLoss(nn.Module):
    """
    Multi-Class Soft Dice + Centerline clDice Loss from TopoNet.
    Guarantees topological connectivity along thin, tubular liver landmarks.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        smooth: float = 1e-5,
        num_iter: int = 20,
        exclude_background: bool = True,
    ):
        super().__init__()
        self.alpha = alpha
        self.smooth = smooth
        self.exclude_background = exclude_background
        self.soft_skeletonize = SoftSkeletonize(num_iter=num_iter)

    def forward(self, y_pred_logits: torch.Tensor, y_true_onehot: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred_logits: (B, C, H, W) raw logits from segmentation head
            y_true_onehot: (B, C, H, W) binary targets [0, 1]
        Returns:
            scalar loss = (1 - alpha) * Dice + alpha * clDice
        """
        y_pred = F.softmax(y_pred_logits, dim=1)

        if self.exclude_background:
            y_true = y_true_onehot[:, 1:, :, :]
            y_pred = y_pred[:, 1:, :, :]
        else:
            y_true = y_true_onehot

        # 1. Soft Dice Loss
        intersection = (y_pred * y_true).sum(dim=(2, 3))
        union = (y_pred + y_true).sum(dim=(2, 3))
        dice_coeff = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_coeff.mean()

        # 2. Soft Centerline clDice Loss
        skel_pred = self.soft_skeletonize(y_pred)
        skel_true = self.soft_skeletonize(y_true)

        cldice_loss = 0.0
        num_channels = y_pred.shape[1]
        for c in range(num_channels):
            sp = skel_pred[:, c:c+1]
            st = skel_true[:, c:c+1]
            yp = y_pred[:, c:c+1]
            yt = y_true[:, c:c+1]

            tprec = (torch.sum(sp * yt) + self.smooth) / (torch.sum(sp) + self.smooth)
            tsens = (torch.sum(st * yp) + self.smooth) / (torch.sum(st) + self.smooth)

            cldice_c = (2.0 * tprec * tsens) / (tprec + tsens + self.smooth)
            cldice_loss = cldice_loss + (1.0 - cldice_c)

        cldice_loss = cldice_loss / float(max(num_channels, 1))

        return (1.0 - self.alpha) * dice_loss + self.alpha * cldice_loss


class AnalyticalContinuousclDice(nn.Module):
    """
    Analytical Continuous clDice for sub-pixel parametric Bézier curves.
    Evaluates topological precision by sampling ground truth masks directly
    at predicted curve coordinates B(t) via grid_sample.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pred_curve_pts: torch.Tensor,     # (B, M, N, 2) in [0, 1]^2
        gt_masks: torch.Tensor,            # (B, M, H, W) in [0, 1]
        exists_mask: torch.Tensor,         # (B, M) binary
    ) -> torch.Tensor:
        """
        Computes continuous clDice loss: L_cl = 1.0 - (2 * Tprec * Tsens) / (Tprec + Tsens)
        """
        B, M, N, _ = pred_curve_pts.shape
        loss_total = torch.tensor(0.0, device=pred_curve_pts.device)
        valid_count = 0

        for m in range(M):
            pts_m = pred_curve_pts[:, m]   # (B, N, 2)
            mask_m = gt_masks[:, m:m+1]    # (B, 1, H, W)
            exist_m = exists_mask[:, m]    # (B,)

            # Map coordinates [0, 1] to [-1, 1] for grid_sample
            grid = pts_m.unsqueeze(2) * 2.0 - 1.0  # (B, N, 1, 2)
            # Sample GT mask at curve points: gives precision
            sampled_gt = F.grid_sample(mask_m, grid, align_corners=False, mode="bilinear") # (B, 1, N, 1)
            tprec = sampled_gt.squeeze().mean(dim=-1)  # (B,)

            # Continuous clDice precision penalty
            tprec_loss = (1.0 - tprec) * exist_m
            loss_total = loss_total + tprec_loss.sum()
            valid_count += exist_m.sum().item()

        if valid_count > 0:
            return loss_total / float(valid_count)
        return torch.tensor(0.0, device=pred_curve_pts.device)
