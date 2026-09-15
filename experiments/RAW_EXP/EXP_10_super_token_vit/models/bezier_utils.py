import numpy as np
import torch


def sample_cubic_bezier_batch(ctrl_pts: torch.Tensor, num_samples: int = 10) -> torch.Tensor:
    """
    Evaluates cubic Bézier curves at uniform parameter values t in [0, 1].
    
    Args:
        ctrl_pts: (..., 4, 2) tensor of control points [P0, P1, P2, P3] in [0, 1]^2
        num_samples: Number of sample points along the curve (default 10)
        
    Returns:
        sampled_pts: (..., num_samples, 2) evaluated coordinates
    """
    device = ctrl_pts.device
    dtype = ctrl_pts.dtype
    t = torch.linspace(0.0, 1.0, num_samples, device=device, dtype=dtype)  # (N,)
    
    b0 = (1.0 - t) ** 3
    b1 = 3.0 * (1.0 - t) ** 2 * t
    b2 = 3.0 * (1.0 - t) * (t ** 2)
    b3 = t ** 3
    
    # Basis shape: (1, ..., 1, N, 4)
    basis = torch.stack([b0, b1, b2, b3], dim=-1)  # (N, 4)
    
    # Expand basis for broadcasting with arbitrary batch dimensions
    lead_dims = len(ctrl_pts.shape) - 2
    for _ in range(lead_dims):
        basis = basis.unsqueeze(0)
    # basis: (1, ..., 1, N, 4), ctrl_pts: (B, ..., 4, 2)
    # sampled = basis @ ctrl_pts -> (B, ..., N, 2)
    sampled = torch.matmul(basis, ctrl_pts)
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


def fit_cubic_bezier_least_squares(points: np.ndarray) -> np.ndarray:
    """
    Fits a cubic Bézier curve (4 control points P0, P1, P2, P3) to ordered 2D points
    inside a patch using chord-length parameterization and least-squares.
    
    Enforces strict endpoint anchoring: P0 = points[0], P3 = points[-1].
    """
    if len(points) < 2:
        pt = points[0] if len(points) == 1 else np.array([0.5, 0.5], dtype=np.float32)
        return np.tile(pt, (4, 1)).astype(np.float32)
        
    if len(points) == 2:
        p0 = points[0]
        p3 = points[1]
        p1 = p0 + (p3 - p0) / 3.0
        p2 = p0 + 2.0 * (p3 - p0) / 3.0
        return np.stack([p0, p1, p2, p3], axis=0).astype(np.float32)
        
    # Chord-length parameterization
    diffs = np.diff(points, axis=0)
    dists = np.hypot(diffs[:, 0], diffs[:, 1])
    cum_dist = np.insert(np.cumsum(dists), 0, 0.0)
    total_len = cum_dist[-1]
    
    if total_len <= 1e-6:
        return np.tile(points[0], (4, 1)).astype(np.float32)
        
    t = cum_dist / total_len
    
    # Known fixed endpoints
    p0 = points[0]
    p3 = points[-1]
    
    # Internal basis functions for intermediate control points P1, P2
    b0 = (1.0 - t) ** 3
    b1 = 3.0 * (1.0 - t) ** 2 * t
    b2 = 3.0 * (1.0 - t) * (t ** 2)
    b3 = t ** 3
    
    # Target residuals: points - b0*P0 - b3*P3 = b1*P1 + b2*P2
    rhs_x = points[:, 0] - b0 * p0[0] - b3 * p3[0]
    rhs_y = points[:, 1] - b0 * p0[1] - b3 * p3[1]
    
    A = np.column_stack([b1, b2])  # (M, 2)
    
    try:
        p12_x = np.linalg.lstsq(A, rhs_x, rcond=None)[0]
        p12_y = np.linalg.lstsq(A, rhs_y, rcond=None)[0]
        p1 = np.array([p12_x[0], p12_y[0]], dtype=np.float32)
        p2 = np.array([p12_x[1], p12_y[1]], dtype=np.float32)
    except Exception:
        p1 = p0 + (p3 - p0) / 3.0
        p2 = p0 + 2.0 * (p3 - p0) / 3.0
        
    ctrl_points = np.stack([p0, p1, p2, p3], axis=0)
    return np.clip(ctrl_points, 0.0, 1.0).astype(np.float32)
