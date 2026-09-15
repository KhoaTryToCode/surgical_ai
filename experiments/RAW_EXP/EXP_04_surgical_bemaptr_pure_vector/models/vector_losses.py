"""
Point-Curve-Region (PCR) Loss & Permutation-Equivalent Matcher for Surgical-BeMapTR (EXP_04).

Combines:
1. MapTRv2 Permutation-Equivalent Bipartite Matcher (forward & reverse orientation invariant)
2. BeMapNet 3-Level PCR Loss (Point L1 + Bernstein Curve L1 + Dilated Region Line Mask Loss + Edge Direction Loss + Focal Classification Loss)
3. Chamfer & Fréchet Distance Evaluation Metrics
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


# ──────────────────────────────────────────────
#  Permutation-Equivalent Hungarian Matcher
# ──────────────────────────────────────────────

class PermutationEquivalentMatcher(nn.Module):
    """
    Evaluates both forward (gamma^0) and reverse (gamma^1) polyline orientations.
    Prevents orientation confusion from penalizing valid line predictions.
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
#  Direction Cosine Loss
# ──────────────────────────────────────────────

def direction_cosine_loss(pred_pts, gt_pts, eps=1e-5):
    """
    Numerically stable Cosine Similarity Loss on edge vectors.
    Uses sqrt(sum(x^2) + eps) inside norm to avoid 0/0 NaN autograd derivatives.
    """
    pred_dirs = pred_pts[:, 1:, :] - pred_pts[:, :-1, :]
    gt_dirs = gt_pts[:, 1:, :] - gt_pts[:, :-1, :]

    pred_norm = torch.sqrt((pred_dirs ** 2).sum(dim=-1, keepdim=True) + eps)
    gt_norm = torch.sqrt((gt_dirs ** 2).sum(dim=-1, keepdim=True) + eps)

    pred_unit = pred_dirs / pred_norm
    gt_unit = gt_dirs / gt_norm

    cos_sim = (pred_unit * gt_unit).sum(dim=-1)
    return (1.0 - cos_sim).mean()


# ──────────────────────────────────────────────
#  Vector Metrics for Evaluation
# ──────────────────────────────────────────────

def chamfer_distance(pred_pts, gt_pts):
    if pred_pts.dim() == 3:
        pred_pts = pred_pts.reshape(-1, 2)
    if gt_pts.dim() == 3:
        gt_pts = gt_pts.reshape(-1, 2)

    if pred_pts.shape[0] == 0 or gt_pts.shape[0] == 0:
        return torch.tensor(0.0)

    dists = torch.cdist(pred_pts.float(), gt_pts.float(), p=2)
    fwd = dists.min(dim=1)[0].mean()
    bwd = dists.min(dim=0)[0].mean()
    return (fwd + bwd) / 2.0


def frechet_distance(P, Q):
    n, m = P.shape[0], Q.shape[0]
    if n == 0 or m == 0:
        return torch.tensor(0.0)

    D = torch.cdist(P.unsqueeze(0).float(), Q.unsqueeze(0).float()).squeeze(0)
    dp = torch.full((n, m), float('inf'), device=P.device)
    dp[0, 0] = D[0, 0]

    for i in range(1, n):
        dp[i, 0] = max(dp[i - 1, 0].item(), D[i, 0].item())
    for j in range(1, m):
        dp[0, j] = max(dp[0, j - 1].item(), D[0, j].item())
    for i in range(1, n):
        for j in range(1, m):
            dp[i, j] = max(
                min(dp[i - 1, j].item(), dp[i, j - 1].item(), dp[i - 1, j - 1].item()),
                D[i, j].item()
            )

    return dp[n - 1, m - 1]


# ──────────────────────────────────────────────
#  Surgical-BeMapTR Criterion (Point-Curve-Region Loss)
# ──────────────────────────────────────────────

class SurgicalBeMapTRCriterion(nn.Module):
    """
    3-Level Point-Curve-Region (PCR) Loss Criterion for Surgical-BeMapTR.
    Loss = cls_w * L_cls + pts_w * L_point + curve_w * L_curve + dir_w * L_dir
    """

    def __init__(self,
                 num_classes=4,
                 N=30,
                 K_dense=20,
                 cls_weight=2.0,
                 pts_weight=5.0,
                 curve_weight=5.0,
                 dir_weight=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.N = N
        self.K_dense = K_dense

        self.cls_weight = cls_weight
        self.pts_weight = pts_weight
        self.curve_weight = curve_weight
        self.dir_weight = dir_weight

        self.matcher = PermutationEquivalentMatcher(cls_weight=cls_weight, pts_weight=pts_weight)
        self.focal_loss = FocalLoss()

    def _compute_single_layer_loss(self, pred_logits, pred_ctrl_pts, pred_restored_pts, gt_labels, gt_pts, num_instances):
        device = pred_logits.device
        B = pred_logits.shape[0]

        # Hungarian matching evaluated on restored curve points
        matches = self.matcher(pred_logits, pred_restored_pts, gt_labels, gt_pts, num_instances)

        target_cls_labels = torch.zeros((B, self.N), dtype=torch.long, device=device)
        matched_pred_restored = []
        matched_gt_pts = []

        total_gt_instances = sum(num_instances).item()

        for b, (pred_idx, gt_idx, is_rev) in enumerate(matches):
            if len(pred_idx) > 0:
                gt_classes = gt_labels[b, gt_idx.to(device)].long()
                target_cls_labels[b, pred_idx.to(device)] = gt_classes

                pred_pts_b = pred_restored_pts[b, pred_idx.to(device)]
                gt_pts_b = gt_pts[b, gt_idx.to(device)].to(device)

                for i in range(len(is_rev)):
                    if is_rev[i]:
                        gt_pts_b[i] = torch.flip(gt_pts_b[i], dims=[0])

                matched_pred_restored.append(pred_pts_b)
                matched_gt_pts.append(gt_pts_b)

        # CrossEntropyLoss: dim 1 is num_classes=4 (0: Background, 1: Ridge, 2: Silhouette, 3: Ligament)
        loss_cls = F.cross_entropy(pred_logits.permute(0, 2, 1), target_cls_labels)

        if len(matched_pred_restored) > 0:
            all_pred_curve = torch.cat(matched_pred_restored, dim=0)
            all_gt_curve = torch.cat(matched_gt_pts, dim=0)

            # Curve-level L1 Loss
            loss_curve = F.l1_loss(all_pred_curve, all_gt_curve, reduction='mean')
            # Edge Direction Cosine Loss
            loss_dir = direction_cosine_loss(all_pred_curve, all_gt_curve)
            # Point-level Loss (evaluated on curve endpoints)
            loss_point = F.l1_loss(all_pred_curve[:, [0, -1], :], all_gt_curve[:, [0, -1], :], reduction='mean')
        else:
            loss_curve = torch.tensor(0.0, device=device)
            loss_dir = torch.tensor(0.0, device=device)
            loss_point = torch.tensor(0.0, device=device)

        total_layer_loss = (
            self.cls_weight * loss_cls
            + self.pts_weight * loss_point
            + self.curve_weight * loss_curve
            + self.dir_weight * loss_dir
        )

        dict_out = {
            'loss_cls': loss_cls.item(),
            'loss_point': loss_point.item(),
            'loss_curve': loss_curve.item(),
            'loss_dir': loss_dir.item(),
        }
        return dict_out, total_layer_loss

    def forward(self, pred_logits, pred_ctrl_pts, pred_restored_pts, gt_labels, gt_pts, num_instances):
        """
        Args:
            pred_logits: list of layer logits or single layer (B, N, num_classes)
            pred_ctrl_pts: list of layer Bézier control points (B, N, 10, 2)
            pred_restored_pts: list of layer restored curve points (B, N, 20, 2)
            gt_labels: (B, N_padded)
            gt_pts: (B, N_padded, 20, 2)
            num_instances: (B,)
        """
        if isinstance(pred_logits, list):
            vec_loss_total = 0.0
            last_dict = None
            for layer_logits, layer_ctrl, layer_restored in zip(pred_logits, pred_ctrl_pts, pred_restored_pts):
                layer_dict, layer_loss = self._compute_single_layer_loss(
                    layer_logits, layer_ctrl, layer_restored, gt_labels, gt_pts, num_instances
                )
                vec_loss_total += layer_loss
                last_dict = layer_dict
        else:
            last_dict, vec_loss_total = self._compute_single_layer_loss(
                pred_logits, pred_ctrl_pts, pred_restored_pts, gt_labels, gt_pts, num_instances
            )

        out_dict = {
            'loss_total': vec_loss_total.item(),
            **last_dict,
        }
        return out_dict, vec_loss_total
