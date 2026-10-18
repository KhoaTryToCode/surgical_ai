from __future__ import annotations

from typing import Dict, Tuple

import torch


def _kabsch(src: torch.Tensor, tgt: torch.Tensor, weights: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    weights = torch.clamp(weights, min=0.0)
    weights = weights / (weights.sum() + eps)
    weights = weights.unsqueeze(-1)
    src_centroid = (src * weights).sum(dim=0, keepdim=True)
    tgt_centroid = (tgt * weights).sum(dim=0, keepdim=True)
    src_centered = src - src_centroid
    tgt_centered = tgt - tgt_centroid
    H = src_centered.t() @ (weights * tgt_centered)
    U, _, Vh = torch.linalg.svd(H)
    V = Vh.transpose(0, 1)
    Ut = U.transpose(0, 1)
    det = torch.sign(torch.det(V @ Ut))
    diag = torch.diag(torch.stack([src.new_tensor(1.0), src.new_tensor(1.0), det]))
    R = V @ diag @ Ut
    t = (tgt_centroid.t() - R @ src_centroid.t()).squeeze(-1)
    return R, t


def _score_pose(src: torch.Tensor, tgt: torch.Tensor, R: torch.Tensor, t: torch.Tensor, threshold: float) -> Tuple[torch.Tensor, torch.Tensor]:
    warped = (R @ src.t()).t() + t
    residual = torch.linalg.norm(warped - tgt, dim=-1)
    inliers = residual <= threshold
    return inliers, residual


def _icp_refine(
    src_points: torch.Tensor,
    tgt_points: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    max_iters: int,
    threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    used = 0
    for _ in range(max(0, int(max_iters))):
        warped = (R @ src_points.t()).t() + t
        dist = torch.cdist(warped.unsqueeze(0), tgt_points.unsqueeze(0)).squeeze(0)
        nn_dist, nn_idx = dist.min(dim=1)
        mask = nn_dist <= threshold
        if int(mask.sum().item()) < 3:
            break
        R_delta, t_delta = _kabsch(warped[mask], tgt_points[nn_idx[mask]], torch.ones_like(nn_dist[mask]))
        R = R_delta @ R
        t = R_delta @ t + t_delta
        used = int(mask.sum().item())
    return R, t, used


def robust_pose_from_soft_matches(
    src_points: torch.Tensor,
    tgt_points: torch.Tensor,
    soft_match: torch.Tensor,
    topk: int = 512,
    hypotheses: int = 128,
    inlier_threshold: float = 0.01,
    use_icp: bool = False,
    icp_iters: int = 10,
    icp_threshold: float = 0.01,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Estimate batched rigid poses from soft matches using RANSAC and optional ICP."""
    with torch.no_grad():
        B, N_src, _ = src_points.shape
        conf, nn_idx = soft_match.max(dim=2)
        R_out = []
        t_out = []
        inlier_counts = []
        mean_residuals = []
        icp_counts = []

        for b in range(B):
            k = min(max(3, int(topk)), N_src)
            src_sel_idx = torch.topk(conf[b], k=k, largest=True, sorted=False).indices
            tgt_sel_idx = nn_idx[b, src_sel_idx]
            src = src_points[b, src_sel_idx]
            tgt = tgt_points[b, tgt_sel_idx]
            weights = conf[b, src_sel_idx].clamp_min(1e-6)

            if src.shape[0] < 3:
                R_best = torch.eye(3, device=src_points.device, dtype=src_points.dtype)
                t_best = torch.zeros(3, device=src_points.device, dtype=src_points.dtype)
                R_out.append(R_best)
                t_out.append(t_best)
                inlier_counts.append(src.new_tensor(0.0))
                mean_residuals.append(src.new_tensor(float("inf")))
                icp_counts.append(src.new_tensor(0.0))
                continue

            best_inliers = None
            best_score = src.new_tensor(-1.0)
            for _ in range(max(1, int(hypotheses))):
                sample = torch.multinomial(weights, num_samples=3, replacement=False)
                R_h, t_h = _kabsch(src[sample], tgt[sample], torch.ones(3, device=src.device, dtype=src.dtype))
                inliers, residual = _score_pose(src, tgt, R_h, t_h, float(inlier_threshold))
                score = inliers.float().sum() - residual[inliers].mean().nan_to_num(posinf=1e6) * 0.01
                if score > best_score:
                    best_score = score
                    best_inliers = inliers

            if best_inliers is None or int(best_inliers.sum().item()) < 3:
                best_inliers = torch.topk(weights, k=min(3, weights.numel())).indices
                mask = torch.zeros_like(weights, dtype=torch.bool)
                mask[best_inliers] = True
                best_inliers = mask
            R_best, t_best = _kabsch(src[best_inliers], tgt[best_inliers], weights[best_inliers])
            icp_used = 0
            if use_icp:
                R_best, t_best, icp_used = _icp_refine(
                    src_points[b],
                    tgt_points[b],
                    R_best,
                    t_best,
                    max_iters=icp_iters,
                    threshold=float(icp_threshold),
                )
            _, final_residual = _score_pose(src, tgt, R_best, t_best, float(inlier_threshold))
            R_out.append(R_best)
            t_out.append(t_best)
            inlier_counts.append(best_inliers.float().sum())
            mean_residuals.append(final_residual.mean())
            icp_counts.append(src.new_tensor(float(icp_used)))

        stats = {
            "ransac_inliers": torch.stack(inlier_counts),
            "ransac_residual": torch.stack(mean_residuals),
            "icp_inliers": torch.stack(icp_counts),
        }
        return torch.stack(R_out, dim=0), torch.stack(t_out, dim=0), stats
