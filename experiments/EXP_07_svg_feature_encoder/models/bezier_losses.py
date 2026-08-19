"""
EXP_07 — Bézier Spline Loss Functions.

Four loss components:
  1. L_curve    — Bidirectional Chamfer distance between sampled predicted Bézier and GT polyline.
  2. L_cls      — Focal cross-entropy classification loss.
  3. L_endpoint — Extra L1 penalty on P0/P3 vs GT polyline start/end.
  4. L_smooth   — Curvature regularization via second derivative of B(t).

Uses Hungarian bipartite matching to assign predicted queries to GT landmarks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def evaluate_bezier_curve(control_points: torch.Tensor, num_samples: int = 50) -> torch.Tensor:
    """
    Evaluate cubic Bézier at num_samples equally spaced t-values.

    Args:
        control_points: (..., 4, 2) — P0, C1, C2, P3.
    Returns:
        curve_pts: (..., num_samples, 2).
    """
    t = torch.linspace(0.0, 1.0, num_samples, device=control_points.device, dtype=control_points.dtype)
    # Reshape t for broadcasting: add leading dims to match control_points batch dims
    shape = [1] * (control_points.dim() - 2) + [num_samples, 1]
    t = t.view(*shape)

    p0 = control_points[..., 0:1, :]  # (..., 1, 2)
    c1 = control_points[..., 1:2, :]
    c2 = control_points[..., 2:3, :]
    p3 = control_points[..., 3:4, :]

    omt = 1.0 - t
    pts = (omt ** 3) * p0 + 3.0 * (omt ** 2) * t * c1 + 3.0 * omt * (t ** 2) * c2 + (t ** 3) * p3
    return pts


def chamfer_distance_1d(pred_pts: torch.Tensor, gt_pts: torch.Tensor) -> torch.Tensor:
    """
    Bidirectional Chamfer distance between two sets of 2D points.

    Args:
        pred_pts: (N, 2) predicted curve samples.
        gt_pts:   (M, 2) ground-truth polyline points.
    Returns:
        Scalar Chamfer distance.
    """
    # (N, M)
    diff = pred_pts.unsqueeze(1) - gt_pts.unsqueeze(0)  # (N, M, 2)
    dist_sq = (diff ** 2).sum(dim=-1)  # (N, M)

    # Pred → GT: for each predicted point, distance to nearest GT point
    min_pred_to_gt = dist_sq.min(dim=1)[0].mean()

    # GT → Pred: for each GT point, distance to nearest predicted point
    min_gt_to_pred = dist_sq.min(dim=0)[0].mean()

    return (min_pred_to_gt + min_gt_to_pred) / 2.0


def bezier_second_derivative(control_points: torch.Tensor, num_samples: int = 20) -> torch.Tensor:
    """
    Compute ||d²B/dt²|| for curvature regularization.

    For cubic Bézier: d²B/dt² = 6(1-t)(C2 - 2*C1 + P0) + 6t(P3 - 2*C2 + C1)

    Args:
        control_points: (..., 4, 2).
    Returns:
        mean curvature magnitude (scalar).
    """
    t = torch.linspace(0.0, 1.0, num_samples, device=control_points.device, dtype=control_points.dtype)
    shape = [1] * (control_points.dim() - 2) + [num_samples, 1]
    t = t.view(*shape)

    p0 = control_points[..., 0:1, :]
    c1 = control_points[..., 1:2, :]
    c2 = control_points[..., 2:3, :]
    p3 = control_points[..., 3:4, :]

    d2 = 6.0 * (1.0 - t) * (c2 - 2.0 * c1 + p0) + 6.0 * t * (p3 - 2.0 * c2 + c1)
    curvature_mag = torch.sqrt((d2 ** 2).sum(dim=-1) + 1e-8)  # (..., num_samples)
    return curvature_mag.mean()


class BezierSplineLoss(nn.Module):
    """
    Combined loss for Bézier Spline Transformer.

    Components:
        L_curve:    Chamfer distance (predicted curve ↔ GT polyline)
        L_cls:      Focal cross-entropy for landmark classification
        L_endpoint: L1 on P0/P3 vs GT start/end
        L_smooth:   Mean curvature penalty
    """

    def __init__(self, config):
        super().__init__()
        self.lambda_curve = config.lambda_curve
        self.lambda_cls = config.lambda_cls
        self.lambda_endpoint = config.lambda_endpoint
        self.lambda_smooth = config.lambda_smooth
        self.num_classes = config.num_classes
        self.num_curve_samples = 50  # Points sampled along Bézier for Chamfer

        # Focal loss parameters
        self.focal_alpha = 0.25
        self.focal_gamma = 2.0

    def focal_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Focal cross-entropy loss.

        Args:
            logits: (N, C+1) raw class logits.
            targets: (N,) class indices (0 = no object, 1..C = landmark classes).
        """
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        focal = self.focal_alpha * ((1.0 - pt) ** self.focal_gamma) * ce
        return focal.mean()

    @torch.no_grad()
    def hungarian_match(
        self,
        pred_control_points: torch.Tensor,
        gt_polylines: torch.Tensor,
        gt_classes: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> list:
        """
        Bipartite Hungarian matching between predicted queries and GT landmarks.

        Args:
            pred_control_points: (Q, 4, 2) predicted Bézier control points.
            gt_polylines: (N, K, 2) GT polyline points.
            gt_classes: (N,) GT class IDs.
            valid_mask: (N,) boolean active indicators.
        Returns:
            List of (pred_idx, gt_idx) matched pairs.
        """
        active_gt = [i for i in range(valid_mask.shape[0]) if valid_mask[i] > 0]
        if len(active_gt) == 0:
            return []

        Q = pred_control_points.shape[0]
        N_gt = len(active_gt)

        # Compute cost matrix: Chamfer distance between each pred curve and each GT polyline
        pred_curves = evaluate_bezier_curve(pred_control_points, num_samples=self.num_curve_samples)  # (Q, 50, 2)

        cost_matrix = torch.zeros(Q, N_gt, device=pred_control_points.device)
        for j, gt_i in enumerate(active_gt):
            gt_pts = gt_polylines[gt_i]  # (K, 2)
            # Filter out zero-padded points
            gt_valid = gt_pts[gt_pts.sum(dim=-1) > 1e-6]
            if gt_valid.shape[0] < 2:
                gt_valid = gt_pts[:2]

            for q in range(Q):
                cost_matrix[q, j] = chamfer_distance_1d(pred_curves[q], gt_valid)

        # Run Hungarian algorithm
        cost_np = cost_matrix.cpu().numpy()
        pred_indices, gt_local_indices = linear_sum_assignment(cost_np)

        matches = []
        for p_idx, gt_local in zip(pred_indices, gt_local_indices):
            gt_global = active_gt[gt_local]
            matches.append((int(p_idx), int(gt_global)))

        return matches

    def forward(
        self,
        pred_control_points: torch.Tensor,
        pred_class_logits: torch.Tensor,
        gt_polylines: torch.Tensor,
        gt_classes: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> dict:
        """
        Compute all 4 loss components.

        Args:
            pred_control_points: (B, Q, 4, 2) predicted Bézier control points in [0, 1].
            pred_class_logits: (B, Q, C+1) class logits.
            gt_polylines: (B, N, K, 2) GT polyline coordinates in [0, 1].
            gt_classes: (B, N) GT class IDs (0=bg, 1-4=landmark classes).
            valid_mask: (B, N) boolean.
        Returns:
            dict with "loss", "loss_dict", and "matches" per batch item.
        """
        B = pred_control_points.shape[0]
        Q = pred_control_points.shape[1]
        device = pred_control_points.device

        total_curve = torch.tensor(0.0, device=device)
        total_cls = torch.tensor(0.0, device=device)
        total_endpoint = torch.tensor(0.0, device=device)
        total_smooth = torch.tensor(0.0, device=device)

        all_matches = []

        for b in range(B):
            # Hungarian matching for this batch item
            matches = self.hungarian_match(
                pred_control_points[b], gt_polylines[b], gt_classes[b], valid_mask[b]
            )
            all_matches.append(matches)

            # Sample predicted curves
            pred_curves = evaluate_bezier_curve(
                pred_control_points[b], num_samples=self.num_curve_samples
            )  # (Q, 50, 2)

            # ── L_curve: Chamfer Distance ──
            for pred_idx, gt_idx in matches:
                gt_pts = gt_polylines[b, gt_idx]
                gt_valid = gt_pts[gt_pts.sum(dim=-1) > 1e-6]
                if gt_valid.shape[0] < 2:
                    gt_valid = gt_pts[:2]
                total_curve += chamfer_distance_1d(pred_curves[pred_idx], gt_valid)

            # ── L_endpoint: P0/P3 vs GT start/end ──
            for pred_idx, gt_idx in matches:
                gt_pts = gt_polylines[b, gt_idx]
                gt_valid = gt_pts[gt_pts.sum(dim=-1) > 1e-6]
                if gt_valid.shape[0] < 2:
                    gt_valid = gt_pts[:2]

                p0 = pred_control_points[b, pred_idx, 0]  # (2,)
                p3 = pred_control_points[b, pred_idx, 3]  # (2,)
                gt_start = gt_valid[0]
                gt_end = gt_valid[-1]

                # Check both orientations (GT polyline could be reversed)
                dist_fwd = F.l1_loss(p0, gt_start) + F.l1_loss(p3, gt_end)
                dist_rev = F.l1_loss(p0, gt_end) + F.l1_loss(p3, gt_start)
                total_endpoint += torch.min(dist_fwd, dist_rev)

            # ── L_cls: Focal Classification Loss ──
            # Build target class vector: matched queries get GT class, unmatched get 0 (no-object)
            target_cls = torch.zeros(Q, dtype=torch.long, device=device)
            for pred_idx, gt_idx in matches:
                target_cls[pred_idx] = gt_classes[b, gt_idx]
            total_cls += self.focal_loss(pred_class_logits[b], target_cls)

            # ── L_smooth: Curvature Regularization ──
            total_smooth += bezier_second_derivative(pred_control_points[b])

        # Average over batch
        n_matches = max(sum(len(m) for m in all_matches), 1)
        l_curve = total_curve / n_matches
        l_endpoint = total_endpoint / n_matches
        l_cls = total_cls / B
        l_smooth = total_smooth / B

        loss = (
            self.lambda_curve * l_curve
            + self.lambda_cls * l_cls
            + self.lambda_endpoint * l_endpoint
            + self.lambda_smooth * l_smooth
        )

        return {
            "loss": loss,
            "loss_dict": {
                "l_curve": l_curve.item(),
                "l_cls": l_cls.item(),
                "l_endpoint": l_endpoint.item(),
                "l_smooth": l_smooth.item(),
            },
            "matches": all_matches,
        }
