"""
Vector Loss Functions for Surgical-GeMap.

Adapted from GeMap (ECCV 2024) losses without mmdet/mmcv dependencies.
All losses operate on normalized [0, 1] 2D polyline coordinates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


# ──────────────────────────────────────────────
#  Hungarian Matcher
# ──────────────────────────────────────────────

class HungarianMatcher(nn.Module):
    """
    Optimal bipartite matching between N predicted queries and M GT polylines.

    Cost = cls_weight * classification_cost + pts_weight * point_L1_cost.

    Returns per-sample matching indices.
    """

    def __init__(self, cls_weight=2.0, pts_weight=5.0, num_classes=4):
        super().__init__()
        self.cls_weight = cls_weight
        self.pts_weight = pts_weight
        self.num_classes = num_classes

    @torch.no_grad()
    def forward(self, pred_logits, pred_pts, gt_labels, gt_pts, num_instances):
        """
        Args:
            pred_logits: (B, N, num_classes) classification logits
            pred_pts: (B, N, K, 2) predicted point coordinates [0, 1]
            gt_labels: (B, N_padded) GT class labels (0 = no-object)
            gt_pts: (B, N_padded, K, 2) GT polylines [0, 1]
            num_instances: (B,) number of real GT instances per sample

        Returns:
            List of (pred_indices, gt_indices) tuples, one per batch element.
        """
        B = pred_logits.shape[0]
        matches = []

        for b in range(B):
            M = num_instances[b].item()

            if M == 0:
                matches.append((
                    torch.tensor([], dtype=torch.long),
                    torch.tensor([], dtype=torch.long),
                ))
                continue

            # Extract valid GT for this sample
            gt_cls_b = gt_labels[b, :M]     # (M,)
            gt_pts_b = gt_pts[b, :M]        # (M, K, 2)

            # Classification cost: negative probability of GT class
            pred_prob = pred_logits[b].softmax(-1)  # (N, num_classes)
            cls_cost = -pred_prob[:, gt_cls_b]       # (N, M)

            # Point L1 cost: mean L1 distance between pred and GT points
            # pred_pts[b]: (N, K, 2), gt_pts_b: (M, K, 2)
            pts_cost = torch.cdist(
                pred_pts[b].flatten(1),  # (N, K*2)
                gt_pts_b.flatten(1),     # (M, K*2)
                p=1
            ) / (pred_pts.shape[2] * 2)  # normalize by K*2

            # Total cost
            cost = self.cls_weight * cls_cost + self.pts_weight * pts_cost
            cost = cost.detach().cpu().numpy()

            row_ind, col_ind = linear_sum_assignment(cost)
            matches.append((
                torch.tensor(row_ind, dtype=torch.long),
                torch.tensor(col_ind, dtype=torch.long),
            ))

        return matches


# ──────────────────────────────────────────────
#  Focal Loss (for query classification)
# ──────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Sigmoid focal loss for multi-class classification."""

    def __init__(self, alpha=0.25, gamma=2.0, num_classes=4):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, pred_logits, targets):
        """
        Args:
            pred_logits: (N_total, num_classes) raw logits
            targets: (N_total,) integer class labels
        """
        # One-hot encode targets
        target_onehot = F.one_hot(targets, self.num_classes).float()

        pred_sigmoid = pred_logits.sigmoid()
        pt = (1 - pred_sigmoid) * target_onehot + pred_sigmoid * (1 - target_onehot)
        focal_weight = (self.alpha * target_onehot + (1 - self.alpha) * (1 - target_onehot)) * pt.pow(self.gamma)

        loss = F.binary_cross_entropy_with_logits(
            pred_logits, target_onehot, reduction='none'
        ) * focal_weight

        return loss.sum() / max(1, (targets > 0).sum().item())


# ──────────────────────────────────────────────
#  Point L1 Loss
# ──────────────────────────────────────────────

class PointL1Loss(nn.Module):
    """L1 loss on matched predicted vs GT polyline points."""

    def __init__(self):
        super().__init__()

    def forward(self, pred_pts, gt_pts):
        """
        Args:
            pred_pts: (M_matched, K, 2) predicted points
            gt_pts: (M_matched, K, 2) GT points

        Returns:
            Scalar loss.
        """
        if pred_pts.numel() == 0:
            return pred_pts.sum() * 0.0
        return F.l1_loss(pred_pts, gt_pts, reduction='mean')


# ──────────────────────────────────────────────
#  Direction Cosine Loss
# ──────────────────────────────────────────────

class DirectionCosineLoss(nn.Module):
    """
    Cosine similarity loss on consecutive-point direction vectors.
    Encourages predicted polylines to follow the same tangent direction as GT.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred_pts, gt_pts):
        """
        Args:
            pred_pts: (M, K, 2) predicted points
            gt_pts: (M, K, 2) GT points
        """
        if pred_pts.numel() == 0:
            return pred_pts.sum() * 0.0

        # Direction vectors: (M, K-1, 2)
        pred_dirs = pred_pts[:, 1:] - pred_pts[:, :-1]
        gt_dirs = gt_pts[:, 1:] - gt_pts[:, :-1]

        # Flatten for CosineEmbeddingLoss
        M, D, C = pred_dirs.shape
        pred_flat = pred_dirs.reshape(-1, C)  # (M*(K-1), 2)
        gt_flat = gt_dirs.reshape(-1, C)

        target = torch.ones(pred_flat.shape[0], device=pred_flat.device)
        loss = F.cosine_embedding_loss(pred_flat, gt_flat, target, reduction='mean')
        return loss


# ──────────────────────────────────────────────
#  Geometric Loss (Intra + Inter)
# ──────────────────────────────────────────────

class GeometricLoss(nn.Module):
    """
    Adapted from GeMap's GeometricLoss.
    - Intra: preserves segment lengths and curvature within each polyline.
    - Inter: preserves pairwise spatial relationships between polylines.
    """

    def __init__(self, intra_weight=1.0, inter_weight=0.5):
        super().__init__()
        self.intra_weight = intra_weight
        self.inter_weight = inter_weight

    def compute_intra(self, pts):
        """
        Compute intra-instance geometric features.

        Args:
            pts: (M, K, 2)

        Returns:
            lengths: (M*(K-1),) segment lengths
            dots: (M*(K-2),) consecutive direction dot products
        """
        # Consecutive offsets
        offsets = pts[:, 1:] - pts[:, :-1]  # (M, K-1, 2)
        lengths = torch.norm(offsets, p=2, dim=-1).flatten()  # (M*(K-1),)

        # Consecutive dot products (curvature proxy)
        if offsets.shape[1] >= 2:
            d1 = offsets[:, :-1]  # (M, K-2, 2)
            d2 = offsets[:, 1:]   # (M, K-2, 2)
            norms = torch.norm(d1, p=2, dim=-1) * torch.norm(d2, p=2, dim=-1) + 1e-6
            dots = (d1 * d2).sum(-1) / norms
            dots = dots.flatten()
        else:
            dots = torch.tensor([], device=pts.device)

        return lengths, dots

    def forward(self, pred_pts, gt_pts):
        """
        Args:
            pred_pts: (M, K, 2) matched predicted points
            gt_pts: (M, K, 2) matched GT points
        """
        if pred_pts.numel() == 0:
            return pred_pts.sum() * 0.0

        loss = 0.0

        # ── Intra-instance ──
        pred_lengths, pred_dots = self.compute_intra(pred_pts)
        gt_lengths, gt_dots = self.compute_intra(gt_pts)

        # Filter out NaN/Inf
        valid_l = torch.isfinite(gt_lengths)
        if valid_l.any():
            loss += self.intra_weight * F.l1_loss(
                pred_lengths[valid_l], gt_lengths[valid_l])

        if gt_dots.numel() > 0:
            valid_d = torch.isfinite(gt_dots)
            if valid_d.any():
                loss += self.intra_weight * F.l1_loss(
                    pred_dots[valid_d], gt_dots[valid_d])

        # ── Inter-instance ──
        M = pred_pts.shape[0]
        if M >= 2:
            # Pairwise centroid distances
            pred_centroids = pred_pts.mean(dim=1)  # (M, 2)
            gt_centroids = gt_pts.mean(dim=1)       # (M, 2)

            pred_dists = torch.cdist(pred_centroids.unsqueeze(0),
                                     pred_centroids.unsqueeze(0)).squeeze(0)
            gt_dists = torch.cdist(gt_centroids.unsqueeze(0),
                                   gt_centroids.unsqueeze(0)).squeeze(0)

            # Upper triangle only (avoid redundancy)
            triu_idx = torch.triu_indices(M, M, offset=1)
            pred_inter = pred_dists[triu_idx[0], triu_idx[1]]
            gt_inter = gt_dists[triu_idx[0], triu_idx[1]]

            loss += self.inter_weight * F.l1_loss(pred_inter, gt_inter)

        return loss


# ──────────────────────────────────────────────
#  Chamfer Distance (for evaluation)
# ──────────────────────────────────────────────

def chamfer_distance(pred_pts, gt_pts):
    """
    Chamfer distance between two sets of polyline points.

    Args:
        pred_pts: (M1, K, 2) or (M1*K, 2) predicted points
        gt_pts: (M2, K, 2) or (M2*K, 2) GT points

    Returns:
        Scalar chamfer distance.
    """
    if pred_pts.dim() == 3:
        pred_pts = pred_pts.reshape(-1, 2)
    if gt_pts.dim() == 3:
        gt_pts = gt_pts.reshape(-1, 2)

    if pred_pts.shape[0] == 0 or gt_pts.shape[0] == 0:
        return torch.tensor(0.0)

    # (M1*K, M2*K)
    dists = torch.cdist(pred_pts.float(), gt_pts.float(), p=2)

    # Forward: for each pred point, min distance to any GT point
    fwd = dists.min(dim=1)[0].mean()
    # Backward: for each GT point, min distance to any pred point
    bwd = dists.min(dim=0)[0].mean()

    return (fwd + bwd) / 2.0


# ──────────────────────────────────────────────
#  Fréchet Distance (discrete, for evaluation)
# ──────────────────────────────────────────────

def frechet_distance(P, Q):
    """
    Discrete Fréchet distance between two polylines.

    Args:
        P: (K, 2) first polyline
        Q: (K, 2) second polyline

    Returns:
        Scalar Fréchet distance.
    """
    n, m = P.shape[0], Q.shape[0]
    if n == 0 or m == 0:
        return torch.tensor(0.0)

    # Distance matrix
    D = torch.cdist(P.unsqueeze(0).float(), Q.unsqueeze(0).float()).squeeze(0)

    # DP table
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
#  Combined Criterion
# ──────────────────────────────────────────────

class SurgicalGeMapCriterion(nn.Module):
    """
    Combined loss for Surgical-GeMap training.

    Performs Hungarian matching, then computes:
    - Focal classification loss on all N queries
    - Point L1 loss on matched queries
    - Direction cosine loss on matched queries
    - Geometric loss on matched queries
    """

    def __init__(self, num_classes=4, N=30, K=20,
                 cls_weight=2.0, pts_weight=5.0, dir_weight=2.0,
                 geo_weight=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.N = N
        self.K = K

        self.matcher = HungarianMatcher(
            cls_weight=cls_weight, pts_weight=pts_weight,
            num_classes=num_classes
        )
        self.focal_loss = FocalLoss(num_classes=num_classes)
        self.pts_loss = PointL1Loss()
        self.dir_loss = DirectionCosineLoss()
        self.geo_loss = GeometricLoss()

        self.cls_weight = cls_weight
        self.pts_weight = pts_weight
        self.dir_weight = dir_weight
        self.geo_weight = geo_weight

    def forward(self, pred_logits, pred_pts, gt_labels, gt_pts, num_instances):
        """
        Args:
            pred_logits: (B, N, num_classes)
            pred_pts: (B, N, K, 2) predicted normalized coordinates
            gt_labels: (B, N) padded GT labels
            gt_pts: (B, N, K, 2) padded GT polylines
            num_instances: (B,) real instance counts

        Returns:
            loss_dict: dict of individual loss terms
            total_loss: weighted sum
        """
        device = pred_logits.device
        B = pred_logits.shape[0]

        # ── Hungarian matching ──
        matches = self.matcher(pred_logits, pred_pts, gt_labels, gt_pts, num_instances)

        # ── Classification loss (all queries) ──
        # Build target: matched queries get GT class, unmatched get 0 (background)
        target_classes = torch.zeros(B, self.N, dtype=torch.long, device=device)
        for b, (pred_idx, gt_idx) in enumerate(matches):
            if len(pred_idx) > 0:
                target_classes[b, pred_idx.to(device)] = gt_labels[b, gt_idx.to(device)]

        loss_cls = self.focal_loss(
            pred_logits.reshape(-1, self.num_classes),
            target_classes.reshape(-1)
        )

        # ── Gather matched predictions and GT ──
        matched_pred_pts = []
        matched_gt_pts = []
        for b, (pred_idx, gt_idx) in enumerate(matches):
            if len(pred_idx) > 0:
                matched_pred_pts.append(pred_pts[b, pred_idx.to(device)])
                matched_gt_pts.append(gt_pts[b, gt_idx.to(device)].to(device))

        if matched_pred_pts:
            matched_pred = torch.cat(matched_pred_pts, dim=0)  # (M_total, K, 2)
            matched_gt = torch.cat(matched_gt_pts, dim=0)
        else:
            matched_pred = torch.zeros(0, self.K, 2, device=device)
            matched_gt = torch.zeros(0, self.K, 2, device=device)

        # ── Point L1 loss ──
        loss_pts = self.pts_loss(matched_pred, matched_gt)

        # ── Direction cosine loss ──
        loss_dir = self.dir_loss(matched_pred, matched_gt)

        # ── Geometric loss ──
        loss_geo = self.geo_loss(matched_pred, matched_gt)

        # ── Total ──
        total = (
            self.cls_weight * loss_cls +
            self.pts_weight * loss_pts +
            self.dir_weight * loss_dir +
            self.geo_weight * loss_geo
        )

        loss_dict = {
            'loss_cls': loss_cls.item(),
            'loss_pts': loss_pts.item(),
            'loss_dir': loss_dir.item(),
            'loss_geo': loss_geo.item(),
            'loss_total': total.item(),
        }

        return loss_dict, total
