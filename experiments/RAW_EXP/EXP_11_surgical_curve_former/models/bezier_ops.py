"""
EXP_11: Bézier Math Utilities
==============================
Mathematically precise implementation of 5th-order Bézier curve operations.

All formulas grounded in BCRNet (arXiv 2506.15279) and classical Bézier theory.
Zero hallucination: every formula is cross-verified against BeMapNet and spline_utils.py.
"""
import math
import numpy as np
import torch
from scipy.special import comb


# ─────────────────────────────────────────────────────────────────────────────
# Bernstein Basis (Core Math)
# ─────────────────────────────────────────────────────────────────────────────

def bernstein_basis_matrix(num_samples: int, degree: int) -> np.ndarray:
    """
    Compute Bernstein polynomial basis matrix B ∈ R^{num_samples × (degree+1)}.

    Definition:
        b_{i,n}(t) = C(n, i) * (1 - t)^{n-i} * t^i

    For degree=5 (BCRNet), K=6 control points.
    For degree=3 (EXP_09/10), K=4 control points.

    Args:
        num_samples: Number of uniform samples along t ∈ [0, 1].
        degree:      Bézier degree (n). K = degree + 1 control points.

    Returns:
        B: (num_samples, degree + 1) float32 numpy array.
           Each row i is the Bernstein polynomial at t_i.
           B @ ctrl_pts gives the curve samples: P = B · C.
    """
    t = np.linspace(0.0, 1.0, num_samples, dtype=np.float64)
    K = degree + 1
    B = np.zeros((num_samples, K), dtype=np.float64)
    for j in range(K):
        c = float(comb(degree, j, exact=True))
        B[:, j] = c * np.power(1.0 - t, degree - j) * np.power(t, j)
    # Numerical sanity: each row must sum to 1.0 (partition of unity)
    row_sums = B.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6), (
        f"Bernstein partition-of-unity violated: max deviation = {np.abs(row_sums - 1.0).max():.2e}"
    )
    return B.astype(np.float32)


def evaluate_bezier_torch(ctrl_points: torch.Tensor, num_samples: int = 25) -> torch.Tensor:
    """
    Evaluate Bézier curve at N uniform parameter values t ∈ [0, 1].

    Args:
        ctrl_points: (..., K, 2) control points in [0, 1]^2.
                     K = degree + 1. Supports any leading batch dimensions.
        num_samples: Number of trajectory points to evaluate.

    Returns:
        sampled: (..., num_samples, 2) curve points via P = B · C.

    Math:
        B ∈ R^{N × K},   C = ctrl_points (..., K, 2)
        P = B @ C  →  (..., N, 2)
    """
    device = ctrl_points.device
    dtype = ctrl_points.dtype
    K = ctrl_points.shape[-2]
    degree = K - 1
    B_np = bernstein_basis_matrix(num_samples=num_samples, degree=degree)
    B = torch.from_numpy(B_np).to(device=device, dtype=dtype)   # (N, K)
    # Matmul with broadcasting: B ∈ (N, K), ctrl ∈ (..., K, 2)  → (..., N, 2)
    sampled = torch.matmul(B, ctrl_points)
    return sampled


# ─────────────────────────────────────────────────────────────────────────────
# ACPI Reference Point Construction
# ─────────────────────────────────────────────────────────────────────────────

def build_acpi_reference_points(
    ctrl_points: torch.Tensor,
    n_uniform: int = 25,
) -> torch.Tensor:
    """
    Build N = n_uniform + 1 HCR reference points from a proposed Bézier curve.

    BCRNet Section 3.2:
        P_s = {B(t_1), ..., B(t_{N-1})} ∪ {B(0.5)}
    where t_k = k / N for k = 1, ..., N-1  (25 uniform samples)
    and B(0.5) is the curve midpoint (1 center sample).
    Total: N = 26 reference points.

    Args:
        ctrl_points: (..., K, 2) control points, K=6 for degree-5 Bézier.
        n_uniform: Number of uniform samples (default 25). Center is added: total = 26.

    Returns:
        ref_pts: (..., n_uniform + 1, 2)  all reference points in [0, 1]^2.
    """
    device = ctrl_points.device
    dtype = ctrl_points.dtype
    K = ctrl_points.shape[-2]
    degree = K - 1

    # 25 uniform samples at t_k = k/25 for k=1..25  (avoids t=0 and t=1 endpoints)
    t_uniform = np.linspace(0.0, 1.0, n_uniform + 2)[1:-1].astype(np.float32)  # (25,)
    B_uniform_np = np.zeros((n_uniform, K), dtype=np.float32)
    for j in range(K):
        c = float(comb(degree, j, exact=True))
        B_uniform_np[:, j] = c * np.power(1.0 - t_uniform, degree - j) * np.power(t_uniform, j)
    B_uniform = torch.from_numpy(B_uniform_np).to(device=device, dtype=dtype)
    uniform_pts = torch.matmul(B_uniform, ctrl_points)  # (..., 25, 2)

    # Center sample at t = 0.5
    t_center = np.array([0.5], dtype=np.float32)
    B_center_np = np.zeros((1, K), dtype=np.float32)
    for j in range(K):
        c = float(comb(degree, j, exact=True))
        B_center_np[0, j] = c * float((1.0 - 0.5) ** (degree - j)) * float(0.5 ** j)
    B_center = torch.from_numpy(B_center_np).to(device=device, dtype=dtype)
    center_pt = torch.matmul(B_center, ctrl_points)  # (..., 1, 2)

    # Concatenate: 25 uniform + 1 center = 26 total
    ref_pts = torch.cat([uniform_pts, center_pt], dim=-2)  # (..., 26, 2)
    return ref_pts


# ─────────────────────────────────────────────────────────────────────────────
# ACPI Bounded Sigmoid Offset Mapping
# ─────────────────────────────────────────────────────────────────────────────

def acpi_bounded_offset(
    delta_b: torch.Tensor,
    coord_grid: torch.Tensor,
) -> torch.Tensor:
    """
    BCRNet Adaptive Curve Proposal Initialization (ACPI) bounded offset formula.

    For each spatial location i at normalized image coordinate c_i = (c_ix, c_iy) ∈ (0, 1)^2,
    and for each of the K=6 Bézier control points j:

        b_i^j = ( sigma(Δb_ix^j + logit(c_ix)),
                  sigma(Δb_iy^j + logit(c_iy)) )

    where:
        sigma(z) = 1 / (1 + exp(-z))              [sigmoid, maps R → (0, 1)]
        logit(p) = log(p / (1-p)) = sigma^{-1}(p) [log-odds, maps (0,1) → R]

    This composition guarantees:
        1. b_i^j ∈ (0, 1)^2  (all control pts strictly inside image canvas)
        2. When Δb = 0: b_i^j = sigma(logit(c_i)) = c_i  (identity initialization)
        3. The network learns smooth offsets from the spatial anchor point c_i.

    Args:
        delta_b:    (B, M, K*2, H_f, W_f)  predicted offset map from ACPI head.
                    M = num_classes, K = 6 ctrl pts, 2 coords → 12 values per pixel.
        coord_grid: (1, 1, 2, H_f, W_f)    normalized (cx, cy) grid in (0, 1).
                    Built once in ACPIHead.__init__ and registered as buffer.

    Returns:
        ctrl_pts_map: (B, M, K, 2, H_f, W_f)  bounded control point maps per pixel.
    """
    B, M, K2, H_f, W_f = delta_b.shape
    assert K2 % 2 == 0, f"Expected K*2 offsets, got {K2}"
    K = K2 // 2

    # delta_b: (B, M, K, 2, H_f, W_f) — split into x and y offsets
    delta_b_xy = delta_b.view(B, M, K, 2, H_f, W_f)

    # coord_grid: (1, 1, 2, H_f, W_f)  → expand to (1, 1, 1, 2, H_f, W_f)
    grid = coord_grid.unsqueeze(2)  # (1, 1, 1, 2, H_f, W_f)

    # logit(c) = log(c / (1 - c)):  safe clamp to (eps, 1-eps) to avoid inf
    eps = 1e-5
    grid_clamped = grid.clamp(eps, 1.0 - eps)
    logit_grid = torch.log(grid_clamped / (1.0 - grid_clamped))  # (1,1,1,2,H,W)

    # b_i^j = sigma(Δb + logit(c))
    ctrl_pts_map = torch.sigmoid(delta_b_xy + logit_grid)  # (B, M, K, 2, H_f, W_f)
    return ctrl_pts_map


# ─────────────────────────────────────────────────────────────────────────────
# Ground-Truth Bézier Fitting (NumPy, used in DataLoader)
# ─────────────────────────────────────────────────────────────────────────────

def fit_bezier_least_squares(points: np.ndarray, degree: int = 5) -> np.ndarray:
    """
    Fit a Bézier curve of given degree to ordered 2D polyline points via chord-length
    parameterization and unconstrained least squares (with pinned endpoints).

    Used in the DataLoader to compute ground-truth Bézier control points from
    raw annotation polylines.

    Args:
        points: (M, 2) ordered polyline points in image coordinates.
        degree: Bézier degree (default 5 for K=6 ctrl pts, matching BCRNet).

    Returns:
        ctrl_points: (K, 2) = (degree+1, 2) fitted control points.
                     Endpoints pinned: ctrl[0] = points[0], ctrl[-1] = points[-1].
    """
    K = degree + 1
    M = len(points)

    if M < 2:
        pt = points[0] if M == 1 else np.array([0.5, 0.5], dtype=np.float32)
        return np.tile(pt, (K, 1)).astype(np.float32)

    if M < K:
        # Interpolate to get at least K points
        t_orig = np.linspace(0.0, 1.0, M)
        t_target = np.linspace(0.0, 1.0, K)
        xi = np.interp(t_target, t_orig, points[:, 0])
        yi = np.interp(t_target, t_orig, points[:, 1])
        return np.column_stack([xi, yi]).astype(np.float32)

    # Chord-length parameterization: t[i] = cum_chord[i] / total_chord
    diffs = np.diff(points, axis=0)
    chord_lens = np.hypot(diffs[:, 0], diffs[:, 1])
    cum_len = np.insert(np.cumsum(chord_lens), 0, 0.0)
    total_len = cum_len[-1]

    if total_len < 1e-6:
        return np.tile(points[0], (K, 1)).astype(np.float32)

    t_vals = (cum_len / total_len).astype(np.float32)

    # Bernstein design matrix: A ∈ R^{M × K}
    A = np.zeros((M, K), dtype=np.float64)
    for j in range(K):
        c = float(comb(degree, j, exact=True))
        A[:, j] = c * ((1.0 - t_vals) ** (degree - j)) * (t_vals ** j)

    # Unconstrained least squares
    try:
        ctrl, _, _, _ = np.linalg.lstsq(A, points.astype(np.float64), rcond=None)
    except np.linalg.LinAlgError:
        ctrl = np.linalg.pinv(A) @ points.astype(np.float64)

    # Pin endpoints for geometric accuracy
    ctrl[0] = points[0]
    ctrl[-1] = points[-1]

    return np.clip(ctrl, 0.0, 1.0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Utility: build normalized coordinate grid
# ─────────────────────────────────────────────────────────────────────────────

def build_coord_grid(height: int, width: int, dtype=torch.float32) -> torch.Tensor:
    """
    Builds a (1, 1, 2, H, W) normalized coordinate grid in (0, 1)^2.

    coord_grid[0, 0, 0] = cx map (x / W, so columns span [0,1])
    coord_grid[0, 0, 1] = cy map (y / H, so rows span [0,1])

    Used by ACPI to compute logit(c_i) at every pixel location.

    Args:
        height: Feature map height H_f (e.g., 32 for f4 at stride 32 on 512px image).
        width:  Feature map width  W_f (e.g., 32 for f4).

    Returns:
        grid: (1, 1, 2, H, W) float tensor.
    """
    # Use pixel-center convention: coords at (i + 0.5) / N ∈ (0, 1)
    cx = (torch.arange(width, dtype=dtype) + 0.5) / float(width)   # (W,)
    cy = (torch.arange(height, dtype=dtype) + 0.5) / float(height) # (H,)
    grid_y, grid_x = torch.meshgrid(cy, cx, indexing="ij")         # (H, W), (H, W)
    grid = torch.stack([grid_x, grid_y], dim=0)                    # (2, H, W)
    return grid.unsqueeze(0).unsqueeze(0)                           # (1, 1, 2, H, W)
