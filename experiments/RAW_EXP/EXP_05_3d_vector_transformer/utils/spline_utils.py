import numpy as np
import cv2
from scipy.interpolate import CubicSpline
import torch

def resample_polyline_arc_length(points: np.ndarray, K: int = 20) -> np.ndarray:
    """
    Uniformly resamples an (M, D) polyline to exactly K vertices using
    arc-length cubic spline interpolation (D=2 for 2D, D=3 for 3D).
    Includes robust handling for duplicate consecutive points and short polylines.
    
    Args:
        points: (M, D) float array of vertex coordinates.
        K: Target number of output points (default: 20).
    Returns:
        resampled_points: (K, D) float array of uniformly spaced points.
    """
    points = np.array(points, dtype=np.float32)
    M, D = points.shape
    if M < 2:
        # Repeat single point K times
        return np.repeat(points[:1], K, axis=0)

    # 1. Deduplicate consecutive duplicate/near-duplicate points
    keep_mask = np.ones(M, dtype=bool)
    keep_mask[1:] = np.linalg.norm(points[1:] - points[:-1], axis=-1) > 1e-4
    points = points[keep_mask]
    M, D = points.shape

    if M < 2:
        return np.repeat(points[:1], K, axis=0)

    # 2. Compute cumulative arc-length distances
    diffs = np.diff(points, axis=0)
    step_dists = np.sqrt(np.sum(diffs**2, axis=-1))
    cum_dists = np.insert(np.cumsum(step_dists), 0, 0.0)
    total_len = cum_dists[-1]

    if total_len < 1e-5:
        # Degenerate zero-length segment: return linear interpolation
        t_orig = np.linspace(0.0, 1.0, M)
        t_new = np.linspace(0.0, 1.0, K)
        resampled = np.zeros((K, D), dtype=np.float32)
        for d in range(D):
            resampled[:, d] = np.interp(t_new, t_orig, points[:, d])
        return resampled

    # 3. Normalized arc-length parameter t in [0.0, 1.0] with strict monotonicity safeguard
    t_orig = cum_dists / total_len
    # Guarantee strictly increasing values for CubicSpline even if precision issues occur
    t_orig = t_orig + np.arange(len(t_orig), dtype=np.float32) * 1e-7
    t_orig = t_orig / t_orig[-1]

    t_new = np.linspace(0.0, 1.0, K)

    resampled = np.zeros((K, D), dtype=np.float32)
    for d in range(D):
        if M >= 4:
            try:
                cs = CubicSpline(t_orig, points[:, d], bc_type='natural')
                resampled[:, d] = cs(t_new)
            except Exception:
                resampled[:, d] = np.interp(t_new, t_orig, points[:, d])
        else:
            # Fallback to linear interpolation for short polylines (2 or 3 points)
            resampled[:, d] = np.interp(t_new, t_orig, points[:, d])

    return resampled

def draw_polyline_mask(points: np.ndarray, height: int = 1024, width: int = 1024, stroke_thickness: int = 35) -> np.ndarray:
    """
    Rasterizes a 2D polyline into a binary segmentation mask (1024x1024 uint8)
    using cv2.line with thick stroke width, matching TopoNet's protocol.
    
    Args:
        points: (K, 2) array of 2D pixel coordinates (x, y).
        height: Image height.
        width: Image width.
        stroke_thickness: Line stroke width in pixels (default: 35).
    Returns:
        mask: (height, width) uint8 binary mask array in {0, 1}.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    if len(points) < 2:
        return mask

    pts_int = np.round(points).astype(np.int32)
    for i in range(len(pts_int) - 1):
        pt1 = tuple(pts_int[i])
        pt2 = tuple(pts_int[i + 1])
        cv2.line(mask, pt1, pt2, color=1, thickness=stroke_thickness)

    return mask
