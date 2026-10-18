"""
Visibility-Aware Multi-Task Loss Objectives for Vis2Reg Replication.
Follows MICCAI 2026 paper §2.3:
1. L_3D: Observation-to-model one-way Chamfer distance (partial-to-complete alignment).
2. L_vis: Mask-gated visible-domain symmetric Chamfer distance.
3. L_sil: 2D Differentiable silhouette BCE + Dice loss.
4. L_def: Deformation field magnitude penalty.
5. L_smooth: kNN displacement smoothness regularizer.
6. L_topo: Local distance / topology preservation regularizer.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def one_way_chamfer(source_pts: Tensor, target_pts: Tensor, squared: bool = True) -> Tensor:
    """
    Computes one-way Chamfer distance: (1 / |source|) * sum_{s in source} min_{t in target} ||s - t||^2.
    source_pts: (B, N, 3)
    target_pts: (B, M, 3)
    Returns: scalar tensor.
    """
    # Pairwise squared Euclidean distance: (B, N, M)
    dists_sq = torch.cdist(source_pts, target_pts, p=2.0).square()
    # Min distance for each point in source
    min_dists_sq = dists_sq.min(dim=-1).values  # (B, N)
    if squared:
        return min_dists_sq.mean()
    return torch.sqrt(torch.clamp(min_dists_sq, min=1e-8)).mean()


def symmetric_chamfer(set_a: Tensor, set_b: Tensor, squared: bool = True) -> Tensor:
    """
    Computes symmetric Chamfer distance: CD_oneway(A, B) + CD_oneway(B, A).
    """
    return one_way_chamfer(set_a, set_b, squared=squared) + one_way_chamfer(set_b, set_a, squared=squared)


def compute_knn_indices(points: Tensor, k: int = 16) -> Tensor:
    """
    Finds k-nearest neighbors in coordinate space for each point (skipping self).
    points: (B, N, 3)
    Returns: indices (B, N, k)
    """
    B, N, _ = points.shape
    k_actual = min(k + 1, N)
    dists_sq = torch.cdist(points, points, p=2.0).square()
    # rank 0 is self, rank 1..k are neighbors
    knn_idx = dists_sq.topk(k=k_actual, dim=-1, largest=False).indices[:, :, 1:]
    return knn_idx


def displacement_smoothness_loss(delta: Tensor, knn_idx: Tensor) -> Tensor:
    """
    L_smooth = (1 / N) * sum_i (1 / |N(i)|) * sum_{j in N(i)} ||Delta_i - Delta_j||^2.
    delta: (B, N, 3)
    knn_idx: (B, N, k)
    """
    B, N, C = delta.shape
    k = knn_idx.shape[-1]

    # Gather neighbor displacements: (B, N, k, C)
    batch_idx = torch.arange(B, device=delta.device)[:, None, None].expand(B, N, k)
    src_idx = torch.arange(N, device=delta.device)[None, :, None].expand(B, N, k)
    neighbor_delta = delta[batch_idx, knn_idx]  # (B, N, k, 3)

    delta_i = delta.unsqueeze(2).expand_as(neighbor_delta)  # (B, N, k, 3)
    diff_sq = (delta_i - neighbor_delta).square().sum(dim=-1)  # (B, N, k)
    return diff_sq.mean()


def local_distance_preservation_loss(source: Tensor, warped: Tensor, knn_idx: Tensor) -> Tensor:
    """
    L_topo = (1 / N) * sum_i (1 / |N(i)|) * sum_{j in N(i)} (||w_i - w_j|| - ||p_i - p_j||)^2.
    source: (B, N, 3) undeformed CT points
    warped: (B, N, 3) deformed and registered points
    knn_idx: (B, N, k)
    """
    B, N, C = source.shape
    k = knn_idx.shape[-1]
    batch_idx = torch.arange(B, device=source.device)[:, None, None].expand(B, N, k)

    # 1. Original pairwise edge lengths in source space
    src_neighbors = source[batch_idx, knn_idx]  # (B, N, k, 3)
    src_i = source.unsqueeze(2).expand_as(src_neighbors)
    d_orig = torch.linalg.norm(src_i - src_neighbors, dim=-1)  # (B, N, k)

    # 2. Deformed pairwise edge lengths in warped space
    warped_neighbors = warped[batch_idx, knn_idx]  # (B, N, k, 3)
    warped_i = warped.unsqueeze(2).expand_as(warped_neighbors)
    d_warped = torch.linalg.norm(warped_i - warped_neighbors, dim=-1)  # (B, N, k)

    return (d_warped - d_orig).square().mean()


def silhouette_overlap_loss(pred_sil: Tensor, target_mask: Tensor, eps: float = 1e-6) -> Tensor:
    """
    L_sil = BCE(S_pred, M_gt) + DiceLoss(S_pred, M_gt).
    pred_sil: (B, H, W) or (B, 1, H, W) soft probabilities in [0, 1]
    target_mask: (B, H, W) or (B, 1, H, W) binary mask in {0, 1}
    """
    if pred_sil.dim() == 4:
        pred_sil = pred_sil.squeeze(1)
    if target_mask.dim() == 5:
        target_mask = target_mask[:, 0, 0]
    elif target_mask.dim() == 4:
        target_mask = target_mask.squeeze(1)

    # Resize target_mask if shapes differ
    if pred_sil.shape[-2:] != target_mask.shape[-2:]:
        target_mask = F.interpolate(
            target_mask.unsqueeze(1).float(),
            size=pred_sil.shape[-2:],
            mode='nearest'
        ).squeeze(1)

    pred_clamped = torch.clamp(pred_sil, min=1e-6, max=1.0 - 1e-6)
    target_float = target_mask.float()

    # 1. Binary Cross Entropy
    bce = F.binary_cross_entropy(pred_clamped, target_float)

    # 2. Soft Dice Loss
    intersection = (pred_clamped * target_float).sum(dim=(-2, -1))
    union = pred_clamped.sum(dim=(-2, -1)) + target_float.sum(dim=(-2, -1))
    dice = 1.0 - (2.0 * intersection + eps) / (union + eps)

    return bce + dice.mean()


def extract_visible_subset(
    warped_points: Tensor,
    intrinsics: Tensor,
    target_mask: Tensor,
    max_visible: int = 5000,
) -> Tensor:
    """
    Constructs the mask-consistent visible point subset V_v^M.
    Projects 3D camera-frame points into image coordinates and filters by target_mask.
    warped_points: (B, N, 3)
    intrinsics: (B, 3, 3)
    target_mask: (B, 1, 1, H, W) or (B, H, W)
    Returns: visible points (B, K_vis, 3)
    """
    B, N, _ = warped_points.shape
    if target_mask.dim() == 5:
        mask_2d = target_mask[:, 0, 0]  # (B, H, W)
    elif target_mask.dim() == 4:
        mask_2d = target_mask[:, 0]
    else:
        mask_2d = target_mask

    H, W = mask_2d.shape[-2:]
    visible_list = []

    for b in range(B):
        pts = warped_points[b]  # (N, 3)
        K = intrinsics[b]       # (3, 3)
        m = mask_2d[b]          # (H, W)

        # Perspective projection: u = fx * (x/z) + cx, v = fy * (y/z) + cy
        z = pts[:, 2].clamp(min=1e-3)
        u = (K[0, 0] * (pts[:, 0] / z) + K[0, 2]).round().long()
        v = (K[1, 1] * (pts[:, 1] / z) + K[1, 2]).round().long()

        # In-bounds mask
        in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (pts[:, 2] > 0)
        u_valid = u[in_bounds]
        v_valid = v[in_bounds]
        pts_valid = pts[in_bounds]

        # Mask gate: M(u, v) == 1
        mask_hits = m[v_valid, u_valid] > 0.5
        vis_pts = pts_valid[mask_hits]

        if len(vis_pts) == 0:
            # Fallback if no points project inside mask
            vis_pts = pts[:max_visible]
        elif len(vis_pts) > max_visible:
            choice = torch.randperm(len(vis_pts), device=pts.device)[:max_visible]
            vis_pts = vis_pts[choice]

        visible_list.append(vis_pts)

    # Pad to equal length across batch if needed
    max_len = max(len(p) for p in visible_list)
    padded = torch.zeros(B, max_len, 3, device=warped_points.device, dtype=warped_points.dtype)
    for b in range(B):
        padded[b, :len(visible_list[b])] = visible_list[b]

    return padded


class Vis2RegLoss(nn.Module):
    """
    Combined multi-task loss module for Vis2Reg.
    Weights are strictly set to the reported paper values (§3.1):
      lambda_3d     = 0.5
      lambda_vis    = 1.2
      lambda_sil    = 1.0
      lambda_def    = 0.1
      lambda_smooth = 0.1
      lambda_topo   = 0.3
    """
    def __init__(
        self,
        lambda_3d: float = 0.5,
        lambda_vis: float = 1.2,
        lambda_sil: float = 1.0,
        lambda_def: float = 0.1,
        lambda_smooth: float = 0.1,
        lambda_topo: float = 0.3,
        knn_k: int = 16,
    ):
        super().__init__()
        self.lambda_3d = lambda_3d
        self.lambda_vis = lambda_vis
        self.lambda_sil = lambda_sil
        self.lambda_def = lambda_def
        self.lambda_smooth = lambda_smooth
        self.lambda_topo = lambda_topo
        self.knn_k = knn_k

    def forward(
        self,
        model_outputs: Dict[str, Tensor],
        batch: Dict[str, Tensor],
        *,
        include_deformation: bool = True,
    ) -> Dict[str, Tensor]:
        """
        Computes all active loss terms.
        When include_deformation is False (Stage 1 or Stage 2 rigid warm-up),
        L_def, L_smooth, and L_topo are zeroed out.
        """
        warped_pts = model_outputs['warped_pcd']       # (B, N, 3)
        delta = model_outputs['deformation']           # (B, N, 3)
        target_pts = batch['target_points']            # (B, M, 3)
        K = batch['cam_intrinsics']                    # (B, 3, 3)
        mask = batch.get('intra_masks')                # (B, 1, 1, H, W)
        source_pts = batch['pre_points']               # (B, N, 3)

        losses = {}
        device = warped_pts.device
        zero = torch.tensor(0.0, device=device)

        # 1. Observation-to-Model One-Way Chamfer (L_3D)
        losses['3d'] = one_way_chamfer(target_pts, warped_pts, squared=True)

        # 2. Mask-Gated Visible-Domain Symmetric Chamfer (L_vis)
        if mask is not None:
            vis_subset = extract_visible_subset(warped_pts, K, mask)
            losses['vis'] = symmetric_chamfer(vis_subset, target_pts, squared=True)
        else:
            losses['vis'] = zero

        # 3. 2D Silhouette Loss (L_sil)
        rendered_mask = model_outputs.get('render_mask')
        if rendered_mask is not None and mask is not None:
            losses['sil'] = silhouette_overlap_loss(rendered_mask, mask)
        else:
            losses['sil'] = zero

        # 4. Deformation Regularization (L_def, L_smooth, L_topo)
        if include_deformation and delta is not None and delta.abs().sum() > 0:
            # 4a. Magnitude penalty: (1/N) * sum ||Delta_i||^2
            losses['def'] = delta.square().sum(dim=-1).mean()

            # Find kNN neighbors on undeformed CT points
            knn_idx = compute_knn_indices(source_pts, k=self.knn_k)

            # 4b. Smoothness penalty
            losses['smooth'] = displacement_smoothness_loss(delta, knn_idx)

            # 4c. Topology / Edge length preservation
            losses['topo'] = local_distance_preservation_loss(source_pts, warped_pts, knn_idx)
        else:
            losses['def'] = zero
            losses['smooth'] = zero
            losses['topo'] = zero

        # Total Weighted Multi-Task Loss
        total = (
            self.lambda_3d * losses['3d']
            + self.lambda_vis * losses['vis']
            + self.lambda_sil * losses['sil']
            + self.lambda_def * losses['def']
            + self.lambda_smooth * losses['smooth']
            + self.lambda_topo * losses['topo']
        )
        losses['total'] = total

        return losses


def rotation_geodesic_distance(R_pred: Tensor, R_gt: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Computes rotation error between predicted and ground-truth rotation matrices.
    R_pred: (B, 3, 3)
    R_gt: (B, 3, 3)
    Returns:
        rre_degrees: (B,) Relative Rotation Error in degrees
        loss_rot: (B,) 1 - cos(theta) = (3 - Trace(R_pred @ R_gt^T)) / 2
    """
    R_rel = torch.bmm(R_pred, R_gt.transpose(1, 2))
    trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
    cos_clamped = torch.clamp((trace - 1.0) / 2.0, min=-1.0, max=1.0)
    rre_deg = torch.acos(cos_clamped) * (180.0 / torch.pi)
    loss_rot = 1.0 - torch.clamp((trace - 1.0) / 2.0, min=-1.0 + 1e-7, max=1.0 - 1e-7)
    return rre_deg, loss_rot


def translation_error(t_pred: Tensor, t_gt: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Computes translation error between predicted and ground-truth translations.
    t_pred: (B, 3)
    t_gt: (B, 3)
    Returns:
        rte_mm: (B,) Relative Translation Error in mm (L2 norm)
        loss_trans: scalar Smooth L1 loss
    """
    diff = t_pred - t_gt
    rte_mm = torch.linalg.norm(diff, dim=-1)
    loss_trans = F.smooth_l1_loss(t_pred, t_gt)
    return rte_mm, loss_trans


class SyntheticPoseLoss(nn.Module):
    """
    Stage 1 Supervised Rigid Pose Loss on Synthetic Blender Simulations.
    L_pose = L_trans + lambda_rot * L_rot + lambda_pts * L_pts
    """
    def __init__(
        self,
        lambda_rot: float = 10.0,
        lambda_pts: float = 0.5,
    ):
        super().__init__()
        self.lambda_rot = lambda_rot
        self.lambda_pts = lambda_pts

    def forward(
        self,
        model_outputs: Dict[str, Tensor],
        batch: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        R_pred, t_pred, _ = model_outputs['rigid_pose']
        R_gt = batch['gt_R']
        t_gt = batch['gt_t']
        warped_pts = model_outputs['warped_pcd']
        target_pts = batch['target_points']

        # 1. Translation error & loss
        rte_mm, loss_trans = translation_error(t_pred, t_gt)

        # 2. Rotation error & loss
        rre_deg, loss_rot = rotation_geodesic_distance(R_pred, R_gt)

        # 3. Surface point alignment loss (One-Way Chamfer)
        loss_pts = one_way_chamfer(warped_pts, target_pts, squared=False)

        # Total multi-task pose loss
        total_loss = loss_trans + self.lambda_rot * loss_rot.mean() + self.lambda_pts * loss_pts

        return {
            'loss_total': total_loss,
            'loss_trans': loss_trans,
            'loss_rot': loss_rot.mean(),
            'loss_pts': loss_pts,
            'rre_deg': rre_deg.mean(),
            'rte_mm': rte_mm.mean(),
        }

