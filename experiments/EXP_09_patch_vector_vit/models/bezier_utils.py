import numpy as np
import torch
from scipy.interpolate import CubicSpline


def resample_polyline_by_arclength(points, step_size_px=8.0):
    """
    Resamples a sparse polyline (N, 2) into dense equidistant points
    using natural cubic spline interpolation parameterized by arc-length.
    
    Preserves exact curve morphology while guaranteeing uniform sample density.
    """
    points = np.array(points, dtype=np.float32)

    # Remove duplicate consecutive points if any
    if len(points) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    dist_between = np.linalg.norm(np.diff(points, axis=0), axis=1)
    valid_idx = np.insert(dist_between > 1e-4, 0, True)
    points = points[valid_idx]

    if len(points) < 2:
        return points

    # Compute cumulative distance along the curve (chord length)
    seg_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.insert(np.cumsum(seg_lengths), 0, 0.0)
    total_length = s[-1]

    if total_length < step_size_px:
        # If curve is very short, keep at least 3 points
        s_dense = np.linspace(0.0, total_length, 3)
    else:
        num_samples = max(int(np.ceil(total_length / step_size_px)) + 1, 3)
        s_dense = np.linspace(0.0, total_length, num_samples)

    # If only 2 points, use linear interpolation; for >= 3 points, use natural cubic spline
    if len(points) == 2:
        dense_x = np.interp(s_dense, s, points[:, 0])
        dense_y = np.interp(s_dense, s, points[:, 1])
    else:
        # bc_type='natural' keeps curvature minimal at open endpoints
        cs_x = CubicSpline(s, points[:, 0], bc_type="natural")
        cs_y = CubicSpline(s, points[:, 1], bc_type="natural")
        dense_x = cs_x(s_dense)
        dense_y = cs_y(s_dense)

    dense_points = np.stack([dense_x, dense_y], axis=1)
    return dense_points.astype(np.float32)


def fit_cubic_bezier_least_squares(points_in_patch: np.ndarray, reg: float = 1e-4) -> np.ndarray:
    """
    Fits a parametric Cubic Bézier Curve (4 control points P0, P1, P2, P3)
    to a sequence of 2D points inside a local patch using closed-form linear least-squares.
    
    Args:
        points_in_patch: (M, 2) array of coordinates in local patch space [0, 1]^2, ordered by arc-length.
        reg: Tikhonov regularization factor for (A^T A + reg * I).
        
    Returns:
        ctrl_points: (4, 2) array of control points [P0, P1, P2, P3] in [0, 1]^2.
    """
    pts = np.asarray(points_in_patch, dtype=np.float32)
    M = len(pts)

    if M == 0:
        return np.zeros((4, 2), dtype=np.float32)
    if M == 1:
        # Point degenerate
        p = np.clip(pts[0], 0.0, 1.0)
        return np.repeat(p[np.newaxis, :], 4, axis=0)

    # Fixed endpoints
    P0 = np.clip(pts[0], 0.0, 1.0)
    P3 = np.clip(pts[-1], 0.0, 1.0)

    if M == 2:
        # Linear degenerate inside patch
        P1 = (2.0 * P0 + P3) / 3.0
        P2 = (P0 + 2.0 * P3) / 3.0
        return np.stack([P0, P1, P2, P3], axis=0).astype(np.float32)

    # Compute chord lengths to parameterize t_k in [0, 1]
    diffs = np.diff(pts, axis=0)
    dists = np.sqrt((diffs ** 2).sum(axis=-1))
    cum_d = np.concatenate(([0.0], np.cumsum(dists)))
    total_d = cum_d[-1]

    if total_d < 1e-6:
        P1 = (2.0 * P0 + P3) / 3.0
        P2 = (P0 + 2.0 * P3) / 3.0
        return np.stack([P0, P1, P2, P3], axis=0).astype(np.float32)

    t = cum_d / total_d  # (M,)

    # Cubic Bernstein basis weights for intermediate control points P1, P2
    # B(t) = (1-t)^3 * P0 + 3(1-t)^2*t * P1 + 3(1-t)*t^2 * P2 + t^3 * P3
    b0 = (1.0 - t) ** 3
    b1 = 3.0 * (1.0 - t) ** 2 * t
    b2 = 3.0 * (1.0 - t) * (t ** 2)
    b3 = t ** 3

    # Residual rhs = pts - b0 * P0 - b3 * P3
    rhs = pts - b0[:, np.newaxis] * P0 - b3[:, np.newaxis] * P3  # (M, 2)

    # Design matrix A = [b1, b2] of shape (M, 2)
    A = np.stack([b1, b2], axis=1)  # (M, 2)

    # Normal equations: (A^T A + reg * I) [P1; P2] = A^T rhs
    ATA = A.T @ A + reg * np.eye(2, dtype=np.float32)
    AT_rhs = A.T @ rhs  # (2, 2)

    try:
        sol = np.linalg.solve(ATA, AT_rhs)  # (2, 2) -> [P1; P2]
        P1 = np.clip(sol[0], 0.0, 1.0)
        P2 = np.clip(sol[1], 0.0, 1.0)
    except np.linalg.LinAlgError:
        P1 = (2.0 * P0 + P3) / 3.0
        P2 = (P0 + 2.0 * P3) / 3.0

    return np.stack([P0, P1, P2, P3], axis=0).astype(np.float32)


def get_bernstein_matrix_numpy(num_samples: int = 10) -> np.ndarray:
    """
    Computes Bernstein polynomial basis matrix of shape (num_samples, 4).
    """
    t = np.linspace(0.0, 1.0, num_samples, dtype=np.float32)
    b0 = (1.0 - t) ** 3
    b1 = 3.0 * (1.0 - t) ** 2 * t
    b2 = 3.0 * (1.0 - t) * (t ** 2)
    b3 = t ** 3
    return np.stack([b0, b1, b2, b3], axis=1)  # (num_samples, 4)


def sample_cubic_bezier_numpy(control_points: np.ndarray, num_samples: int = 10) -> np.ndarray:
    """
    Samples points along cubic Bézier curves.
    
    Args:
        control_points: (..., 4, 2) control points.
        num_samples: Number of equidistant samples along t in [0, 1].
        
    Returns:
        sampled: (..., num_samples, 2) coordinates along the curve.
    """
    M = get_bernstein_matrix_numpy(num_samples)  # (S, 4)
    # Tensor dot along control point axis: (..., 4, 2) with (S, 4) -> (..., S, 2)
    return np.einsum('sa,...ac->...sc', M, control_points)


def sample_cubic_bezier_torch(control_points: torch.Tensor, num_samples: int = 10) -> torch.Tensor:
    """
    Differentiable batch sampling along cubic Bézier curves in PyTorch.
    
    Args:
        control_points: (B, N, 4, 2) or (N, 4, 2) control point tensor in [0, 1]^2.
        num_samples: Number of equidistant samples along t in [0, 1].
        
    Returns:
        sampled: (B, N, num_samples, 2) sampled coordinates along the curves.
    """
    device = control_points.device
    dtype = control_points.dtype
    t = torch.linspace(0.0, 1.0, num_samples, device=device, dtype=dtype)
    b0 = (1.0 - t) ** 3
    b1 = 3.0 * (1.0 - t) ** 2 * t
    b2 = 3.0 * (1.0 - t) * (t ** 2)
    b3 = t ** 3
    M = torch.stack([b0, b1, b2, b3], dim=1)  # (S, 4)

    # Einsum: (S, 4) x (..., 4, 2) -> (..., S, 2)
    return torch.einsum('sa,...ac->...sc', M, control_points)
