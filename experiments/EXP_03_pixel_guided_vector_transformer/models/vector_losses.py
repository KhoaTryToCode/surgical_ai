"""
Dual-Representation Loss Functions for Surgical-GeMap v2.

Combines:
1. Pixel Supervision: BCE Loss + Dice Loss on dense (1024x1024) 4-channel pixel masks
2. Vector Supervision: Point L1 + Direction Cosine + Geometric + Focal Classification Loss via Hungarian Matching
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


# ──────────────────────────────────────────────
#  Hungarian Matcher (Float32 Precision)
# ──────────────────────────────────────────────

class HungarianMatcher(nn.Module):
    """
    Optimal bipartite matching between N predicted queries and M GT polylines.
    Evaluates both forward and reverse GT polyline orientations.
    """

    def __init__(self, cls_weight=2.0, pts_weight=5.0):
        super().__init__()
        self.cls_weight = cls_weight
        self.pts_weight = pts_weight

    @torch.no_grad()
    def forward(self, pred_logits, pred_pts, gt_labels, gt_pts, num_instances):
        B, N, num_classes = pred_logits.shape
        matches = []

        for b in range(B):
            M = num_instances[b].item()

            if M == 0:
                matches.append((
                    torch.tensor([], dtype=torch.long),
                    torch.tensor([], dtype=torch.long),
                    torch.tensor([], dtype=torch.bool),
                ))
                continue

            with torch.amp.autocast('cuda', enabled=False):
                pred_logits_f32 = pred_logits[b].float()
                pred_pts_f32 = pred_pts[b].float()

                gt_cls_b = gt_labels[b, :M]
                gt_pts_b_fwd = gt_pts[b, :M].float()
                gt_pts_b_rev = torch.flip(gt_pts_b_fwd, dims=[1])

                pred_prob = pred_logits_f32.softmax(-1)
                cls_cost = -pred_prob[:, gt_cls_b]

                pts_cost_fwd = torch.cdist(
                    pred_pts_f32.flatten(1), gt_pts_b_fwd.flatten(1), p=1
                ) / (pred_pts.shape[2] * 2)

                pts_cost_rev = torch.cdist(
                    pred_pts_f32.flatten(1), gt_pts_b_rev.flatten(1), p=1
                ) / (pred_pts.shape[2] * 2)

                pts_cost_stacked = torch.stack([pts_cost_fwd, pts_cost_rev], dim=-1)
                pts_cost, is_rev_matrix = torch.min(pts_cost_stacked, dim=-1)

                cost = self.cls_weight * cls_cost + self.pts_weight * pts_cost

            cost = cost.detach().cpu().numpy()
            if not np.isfinite(cost).all():
                cost = np.nan_to_num(cost, nan=1e6, posinf=1e6, neginf=-1e6)

            row_ind, col_ind = linear_sum_assignment(cost)
            is_rev_matched = is_rev_matrix[row_ind, col_ind].bool()

            matches.append((
                torch.tensor(row_ind, dtype=torch.long),
                torch.tensor(col_ind, dtype=torch.long),
                is_rev_matched.cpu(),
            ))

        return matches


# ──────────────────────────────────────────────
#  Focal Loss (Vector Classification)
# ──────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        return loss.mean()


# ──────────────────────────────────────────────
#  Pixel Segmentation Losses (BCE + Dice)
# ──────────────────────────────────────────────

def pixel_dice_loss(pred_logits, target_masks, smooth=1e-5):
    """
    Multiclass Dice loss for dense pixel segmentation masks.
    pred_logits: (B, C, H, W)
    target_masks: (B, C, H, W) in [0, 1]
    """
    probs = torch.sigmoid(pred_logits)
    intersection = (probs * target_masks).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + target_masks.sum(dim=(2, 3))
    dice = 1.0 - (2.0 * intersection + smooth) / (union + smooth)
    return dice.mean()


def pixel_bce_loss(pred_logits, target_masks):
    return F.binary_cross_entropy_with_logits(pred_logits, target_masks)


# ──────────────────────────────────────────────
#  Vector Losses
# ──────────────────────────────────────────────

def point_l1_loss(pred_pts, gt_pts):
    return F.l1_loss(pred_pts, gt_pts, reduction='mean')


def direction_cosine_loss(pred_pts, gt_pts):
    pred_dirs = pred_pts[:, 1:, :] - pred_pts[:, :-1, :]
    gt_dirs = gt_pts[:, 1:, :] - gt_pts[:, :-1, :]

    pred_dirs_norm = F.normalize(pred_dirs, p=2, dim=-1, eps=1e-6)
    gt_dirs_norm = F.normalize(gt_dirs, p=2, dim=-1, eps=1e-6)

    cos_sim = (pred_dirs_norm * gt_dirs_norm).sum(dim=-1)
    loss = (1.0 - cos_sim).mean()
    return loss


def geometric_regularization_loss(pred_pts):
    diffs = pred_pts[:, 1:, :] - pred_pts[:, :-1, :]
    seg_lengths = torch.norm(diffs, p=2, dim=-1)
    mean_length = seg_lengths.mean(dim=-1, keepdim=True)
    length_uniformity_loss = F.mse_loss(seg_lengths, mean_length.expand_as(seg_lengths))

    if pred_pts.shape[1] >= 3:
        v1 = pred_pts[:, 1:-1, :] - pred_pts[:, :-2, :]
        v2 = pred_pts[:, 2:, :] - pred_pts[:, 1:-1, :]
        v1_n = F.normalize(v1, p=2, dim=-1, eps=1e-6)
        v2_n = F.normalize(v2, p=2, dim=-1, eps=1e-6)
        curvature = 1.0 - (v1_n * v2_n).sum(dim=-1)
        curvature_loss = curvature.mean()
    else:
        curvature_loss = 0.0

    return length_uniformity_loss + 0.5 * curvature_loss


# ──────────────────────────────────────────────
#  Dual-Representation Criterion (Surgical-GeMap v2)
# ──────────────────────────────────────────────

class SurgicalGeMapV2Criterion(nn.Module):
    """
    Combined Pixel + Vector Criterion for Surgical-GeMap v2.
    """

    def __init__(self,
                 num_classes=4,
                 N=30,
                 K=20,
                 pixel_dice_weight=1.0,
                 pixel_bce_weight=1.0,
                 cls_weight=2.0,
                 pts_weight=5.0,
                 dir_weight=2.0,
                 geo_weight=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.N = N
        self.K = K

        self.pixel_dice_weight = pixel_dice_weight
        self.pixel_bce_weight = pixel_bce_weight
        self.cls_weight = cls_weight
        self.pts_weight = pts_weight
        self.dir_weight = dir_weight
        self.geo_weight = geo_weight

        self.matcher = HungarianMatcher(cls_weight=cls_weight, pts_weight=pts_weight)
        self.focal_loss = FocalLoss()

    def _compute_vector_layer_loss(self, pred_logits, pred_pts, gt_labels, gt_pts, num_instances):
        device = pred_logits.device
        B = pred_logits.shape[0]

        matches = self.matcher(pred_logits, pred_pts, gt_labels, gt_pts, num_instances)

        target_cls = torch.zeros_like(pred_logits)
        matched_pred_pts = []
        matched_gt_pts = []

        total_gt_instances = sum(num_instances).item()
        if total_gt_instances == 0:
            total_gt_instances = 1

        for b, (pred_idx, gt_idx, is_rev) in enumerate(matches):
            if len(pred_idx) > 0:
                gt_classes = gt_labels[b, gt_idx.to(device)]
                target_cls[b, pred_idx.to(device), gt_classes] = 1.0

                pred_pts_b = pred_pts[b, pred_idx.to(device)]
                gt_pts_b = gt_pts[b, gt_idx.to(device)].to(device)

                for i in range(len(is_rev)):
                    if is_rev[i]:
                        gt_pts_b[i] = torch.flip(gt_pts_b[i], dims=[0])

                matched_pred_pts.append(pred_pts_b)
                matched_gt_pts.append(gt_pts_b)

        loss_cls = self.focal_loss(pred_logits, target_cls)

        if len(matched_pred_pts) > 0:
            all_pred_pts = torch.cat(matched_pred_pts, dim=0)
            all_gt_pts = torch.cat(matched_gt_pts, dim=0)

            loss_pts = point_l1_loss(all_pred_pts, all_gt_pts)
            loss_dir = direction_cosine_loss(all_pred_pts, all_gt_pts)
            loss_geo = geometric_regularization_loss(all_pred_pts)
        else:
            loss_pts = torch.tensor(0.0, device=device)
            loss_dir = torch.tensor(0.0, device=device)
            loss_geo = torch.tensor(0.0, device=device)

        vec_loss = (
            self.cls_weight * loss_cls
            + self.pts_weight * loss_pts
            + self.dir_weight * loss_dir
            + self.geo_weight * loss_geo
        )

        dict_out = {
            'loss_cls': loss_cls.item(),
            'loss_pts': loss_pts.item(),
            'loss_dir': loss_dir.item(),
            'loss_geo': loss_geo.item(),
        }
        return dict_out, vec_loss

    def forward(self, pixel_logits, pred_logits, pred_pts, gt_labels, gt_pts, num_instances, pixel_masks):
        """
        Args:
            pixel_logits: (B, 4, 1024, 1024)
            pred_logits: list of 6 layer logits or single layer (B, N, 4)
            pred_pts: list of 6 layer points or single layer (B, N, K, 2)
            gt_labels: (B, N_padded)
            gt_pts: (B, N_padded, K, 2)
            num_instances: (B,)
            pixel_masks: (B, 4, 1024, 1024) ground truth dense mask in [0, 1]
        """
        device = pixel_logits.device

        # 1. Pixel Losses
        loss_pixel_bce = pixel_bce_loss(pixel_logits, pixel_masks.to(device))
        loss_pixel_dice = pixel_dice_loss(pixel_logits, pixel_masks.to(device))

        pixel_loss = (
            self.pixel_bce_weight * loss_pixel_bce
            + self.pixel_dice_weight * loss_pixel_dice
        )

        # 2. Vector Auxiliary Losses
        if isinstance(pred_logits, list):
            vec_loss_total = 0.0
            last_vec_dict = None
            for layer_logits, layer_pts in zip(pred_logits, pred_pts):
                layer_dict, layer_loss = self._compute_vector_layer_loss(
                    layer_logits, layer_pts, gt_labels, gt_pts, num_instances
                )
                vec_loss_total += layer_loss
                last_vec_dict = layer_dict
        else:
            last_vec_dict, vec_loss_total = self._compute_vector_layer_loss(
                pred_logits, pred_pts, gt_labels, gt_pts, num_instances
            )

        total_loss = pixel_loss + vec_loss_total

        out_dict = {
            'loss_total': total_loss.item(),
            'loss_pixel_bce': loss_pixel_bce.item(),
            'loss_pixel_dice': loss_pixel_dice.item(),
            **last_vec_dict,
        }
        return out_dict, total_loss
