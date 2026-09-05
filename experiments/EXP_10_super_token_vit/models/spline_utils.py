import numpy as np
import torch
from scipy.special import comb


def compute_bezier_basis_matrix(num_samples: int = 64, degree: int = 5) -> np.ndarray:
    """
    Computes Bernstein polynomial basis matrix of size (num_samples, degree + 1).
    For degree = 5, K = 6 control points (P0..P5).
    
    B(t) = sum_{i=0}^n comb(n, i) * (1 - t)^(n - i) * t^i * P_i
    """
    t = np.linspace(0.0, 1.0, num_samples, dtype=np.float32)
    n = degree
    basis = np.zeros((num_samples, n + 1), dtype=np.float32)
    for i in range(n + 1):
        c = comb(n, i)
        basis[:, i] = c * ((1.0 - t) ** (n - i)) * (t ** i)
    return basis


def evaluate_bezier_curve_torch(ctrl_points: torch.Tensor, num_samples: int = 64) -> torch.Tensor:
    """
    Evaluates Bézier curves along t in [0, 1] using batched matrix multiplication.
    
    Args:
        ctrl_points: Tensor of shape (..., K, 2) with K control points (typically K=6, degree=5)
        num_samples: Number of sample points along the curve (e.g. 64)
        
    Returns:
        sampled_points: Tensor of shape (..., num_samples, 2)
    """
    device = ctrl_points.device
    dtype = ctrl_points.dtype
    K = ctrl_points.shape[-2]
    degree = K - 1
    
    basis_np = compute_bezier_basis_matrix(num_samples=num_samples, degree=degree)
    basis_tensor = torch.from_numpy(basis_np).to(device=device, dtype=dtype)  # (N, K)
    
    # sampled_points = basis @ ctrl_points -> (..., N, 2)
    sampled = torch.matmul(basis_tensor, ctrl_points)
    return sampled


def resample_polyline_by_arclength(points: np.ndarray, step_size_px: float = 8.0) -> np.ndarray:
    """
    Resamples an arbitrary ordered polyline to uniform arc-length spacing.
    """
    if len(points) < 2:
        return points
        
    diffs = np.diff(points, axis=0)
    dists = np.hypot(diffs[:, 0], diffs[:, 1])
    cum_dist = np.insert(np.cumsum(dists), 0, 0.0)
    total_length = cum_dist[-1]
    
    if total_length <= 1e-4:
        return np.repeat(points[:1], 2, axis=0)
        
    num_samples = max(int(np.ceil(total_length / step_size_px)) + 1, 4)
    target_dists = np.linspace(0.0, total_length, num_samples)
    
    resampled_x = np.interp(target_dists, cum_dist, points[:, 0])
    resampled_y = np.interp(target_dists, cum_dist, points[:, 1])
    return np.column_stack([resampled_x, resampled_y]).astype(np.float32)


def fit_bezier_least_squares_np(points: np.ndarray, degree: int = 5) -> np.ndarray:
    """
    Fits degree-n Bézier curve (K = degree + 1 control points) to ordered polyline points
    using chord-length parameterization and linear least squares.
    
    Args:
        points: (M, 2) ordered polyline points in [0, 1] or pixel coordinates
        degree: Bézier degree (default 5 for K=6 control points)
        
    Returns:
        ctrl_points: (K, 2) fitted control points
    """
    K = degree + 1
    if len(points) < K:
        # Interpolate points if fewer than K
        t_orig = np.linspace(0.0, 1.0, len(points))
        t_target = np.linspace(0.0, 1.0, K)
        x_interp = np.interp(t_target, t_orig, points[:, 0])
        y_interp = np.interp(t_target, t_orig, points[:, 1])
        return np.column_stack([x_interp, y_interp]).astype(np.float32)
        
    # Chord-length parameterization
    diffs = np.diff(points, axis=0)
    chord_dists = np.hypot(diffs[:, 0], diffs[:, 1])
    cum_dists = np.insert(np.cumsum(chord_dists), 0, 0.0)
    total_len = cum_dists[-1]
    
    if total_len <= 1e-6:
        return np.repeat(points[:1], K, axis=0).astype(np.float32)
        
    t_vals = (cum_dists / total_len).astype(np.float32)
    
    # Construct Bernstein design matrix A: (M, K)
    M = len(points)
    A = np.zeros((M, K), dtype=np.float32)
    for i in range(K):
        c = comb(degree, i)
        A[:, i] = c * ((1.0 - t_vals) ** (degree - i)) * (t_vals ** i)
        
    # Pin endpoints: P0 = points[0], PK-1 = points[-1]
    # Solve unconstrained or regularized least-squares: ctrl = pinv(A) @ points
    try:
        ctrl_points = np.linalg.lstsq(A, points, rcond=None)[0]
    except Exception:
        # Fallback to pseudo-inverse
        ctrl_points = np.linalg.pinv(A) @ points
        
    # Enforce strict endpoint anchor
    ctrl_points[0] = points[0]
    ctrl_points[-1] = points[-1]
    return ctrl_points.astype(np.float32)
