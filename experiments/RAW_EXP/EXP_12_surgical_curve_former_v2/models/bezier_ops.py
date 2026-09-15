"""
EXP_12: Bézier Math Utilities
=============================
Mathematically precise implementation of 5th-order Bézier curve operations.
All formulas grounded in classical Bernstein-Bézier theory.
"""
import math
import numpy as np
import torch
from scipy.special import comb


def bernstein_basis_matrix(num_samples: int, degree: int) -> np.ndarray:
    """
    Compute Bernstein polynomial basis matrix B in R^{num_samples x (degree+1)}.
    """
    t = np.linspace(0.0, 1.0, num_samples, dtype=np.float64)
    K = degree + 1
    B = np.zeros((num_samples, K), dtype=np.float64)
    for j in range(K):
        c = float(comb(degree, j, exact=True))
        B[:, j] = c * np.power(1.0 - t, degree - j) * np.power(t, j)
    row_sums = B.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6), (
        f"Bernstein partition-of-unity violated: max deviation = {np.abs(row_sums - 1.0).max():.2e}"
    )
    return B.astype(np.float32)


def evaluate_bezier_torch(ctrl_points: torch.Tensor, num_samples: int = 25) -> torch.Tensor:
    """
    Evaluate Bézier curve at N uniform parameter values t in [0, 1].
    """
    device = ctrl_points.device
    dtype = ctrl_points.dtype
    K = ctrl_points.shape[-2]
    degree = K - 1
    B_np = bernstein_basis_matrix(num_samples=num_samples, degree=degree)
    B = torch.from_numpy(B_np).to(device=device, dtype=dtype)
    return torch.matmul(B, ctrl_points)


def fit_bezier_least_squares(points: np.ndarray, degree: int = 5) -> np.ndarray:
    """
    Fit a Bézier curve of given degree to ordered 2D polyline points via chord-length
    parameterization and least squares (with pinned endpoints).
    """
    K = degree + 1
    M = len(points)

    if M < 2:
        pt = points[0] if M == 1 else np.array([0.5, 0.5], dtype=np.float32)
        return np.tile(pt, (K, 1)).astype(np.float32)

    if M < K:
        t_orig = np.linspace(0.0, 1.0, M)
        t_target = np.linspace(0.0, 1.0, K)
        xi = np.interp(t_target, t_orig, points[:, 0])
        yi = np.interp(t_target, t_orig, points[:, 1])
        return np.column_stack([xi, yi]).astype(np.float32)

    diffs = np.diff(points, axis=0)
    chord_lens = np.hypot(diffs[:, 0], diffs[:, 1])
    cum_len = np.insert(np.cumsum(chord_lens), 0, 0.0)
    total_len = cum_len[-1]

    if total_len < 1e-6:
        return np.tile(points[0], (K, 1)).astype(np.float32)

    t_vals = (cum_len / total_len).astype(np.float32)

    A = np.zeros((M, K), dtype=np.float64)
    for j in range(K):
        c = float(comb(degree, j, exact=True))
        A[:, j] = c * ((1.0 - t_vals) ** (degree - j)) * (t_vals ** j)

    try:
        ctrl, _, _, _ = np.linalg.lstsq(A, points.astype(np.float64), rcond=None)
    except np.linalg.LinAlgError:
        ctrl = np.linalg.pinv(A) @ points.astype(np.float64)

    ctrl[0] = points[0]
    ctrl[-1] = points[-1]
    return np.clip(ctrl, 0.0, 1.0).astype(np.float32)


def build_coord_grid(height: int, width: int, dtype=torch.float32) -> torch.Tensor:
    """
    Builds a (1, 1, 2, H, W) normalized coordinate grid in (0, 1)^2.
    """
    cx = (torch.arange(width, dtype=dtype) + 0.5) / float(width)
    cy = (torch.arange(height, dtype=dtype) + 0.5) / float(height)
    grid_y, grid_x = torch.meshgrid(cy, cx, indexing="ij")
    grid = torch.stack([grid_x, grid_y], dim=0)
    return grid.unsqueeze(0).unsqueeze(0)
