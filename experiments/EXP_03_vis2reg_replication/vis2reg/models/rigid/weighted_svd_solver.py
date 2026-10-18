import torch
from torch import nn


class WeightedSVDSolver(nn.Module):
    """Differentiable weighted SVD rigid solver using soft correspondences."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pre_points: torch.Tensor, tgt_points: torch.Tensor, soft_match: torch.Tensor):
        """
        Args:
            pre_points: (B, N_pre, 3)
            tgt_points: (B, N_tgt, 3)
            soft_match: (B, N_pre, N_tgt) row-softmax probabilities
        Returns:
            R_pred: (B, 3, 3)
            t_pred: (B, 3)
        """
        # sanitize inputs
        soft_match = torch.nan_to_num(soft_match, nan=0.0, posinf=0.0, neginf=0.0)
        pre_points = torch.nan_to_num(pre_points, nan=0.0, posinf=0.0, neginf=0.0)
        tgt_points = torch.nan_to_num(tgt_points, nan=0.0, posinf=0.0, neginf=0.0)
        # soft correspondence target points: y_hat_i = sum_j P_ij y_j
        y_hat = torch.bmm(soft_match, tgt_points)  # (B, N_pre, 3)

        weights = soft_match.sum(dim=2)  # (B, N_pre)
        weights = torch.clamp(weights, min=0.0)
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        denom = weights.sum(dim=1, keepdim=True) + self.eps
        weights = weights / denom
        weights = weights.unsqueeze(2)  # (B, N_pre, 1)

        x_centroid = (pre_points * weights).sum(dim=1, keepdim=True)
        y_centroid = (y_hat * weights).sum(dim=1, keepdim=True)

        x_centered = pre_points - x_centroid
        y_centered = y_hat - y_centroid

        H = torch.matmul(x_centered.transpose(1, 2), weights * y_centered)  # (B, 3, 3)

        # SVD on CPU can be more stable for small mats; stay on same device to keep it simple
        U, S, Vh = torch.linalg.svd(H, full_matrices=False)
        V = Vh.transpose(1, 2)
        Ut = U.transpose(1, 2)
        det = torch.sign(torch.linalg.det(V @ Ut))
        diag_vals = torch.stack([torch.ones_like(det), torch.ones_like(det), det], dim=1)  # (B, 3)
        eye = torch.diag_embed(diag_vals)  # (B, 3, 3)
        R = V @ eye @ Ut

        t = (y_centroid.transpose(1, 2) - torch.matmul(R, x_centroid.transpose(1, 2))).squeeze(2)

        return R, t

    def weighted_svd(self, src_points: torch.Tensor, ref_points: torch.Tensor, weights: torch.Tensor):
        """Single-cloud weighted SVD helper."""
        return self.forward_single(src_points, ref_points, weights)

    def forward_single(self, src_points: torch.Tensor, ref_points: torch.Tensor, weights: torch.Tensor):
        src_points = torch.nan_to_num(src_points, nan=0.0, posinf=0.0, neginf=0.0)
        ref_points = torch.nan_to_num(ref_points, nan=0.0, posinf=0.0, neginf=0.0)
        weights = torch.clamp(weights, min=0.0)
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        weights = weights / (weights.sum(dim=0, keepdim=True) + self.eps)
        weights = weights.unsqueeze(1)  # (N,1)

        src_centroid = (src_points * weights).sum(dim=0, keepdim=True)
        ref_centroid = (ref_points * weights).sum(dim=0, keepdim=True)
        src_centered = src_points - src_centroid
        ref_centered = ref_points - ref_centroid

        H = src_centered.t() @ (weights * ref_centered)
        U, _, Vh = torch.linalg.svd(H)
        V = Vh.transpose(0, 1)
        Ut = U.transpose(0, 1)
        det = torch.sign(torch.det(V @ Ut))
        diag_vals = torch.stack([torch.ones_like(det), torch.ones_like(det), det])
        eye = torch.diag(diag_vals)
        R = V @ eye @ Ut
        t = (ref_centroid.t() - R @ src_centroid.t()).squeeze(1)
        return R, t
