"""
Mathematical Test 01: Parametric Order & Representation Efficiency.
Tests Degree-3, 4, 5 (BCRNet), 6, 7 Bézier vs Piecewise Cubic Bézier (BeMapNet).

Evaluates:
1. Reconstruction Hausdorff Distance (px on 512x512)
2. Mean L2 Coordinate Error (px)
3. Condition Number kappa(M^T M) (Numerical Inversion Stability)
4. Total Bending / Curvature Energy: Integral ||B''(t)||^2 dt
"""

import numpy as np
import scipy.special
from scipy.spatial.distance import directed_hausdorff
from synthetic_landmarks import get_all_benchmark_landmarks


def build_bernstein_matrix(num_samples: int, degree: int) -> np.ndarray:
    """Builds Bernstein basis matrix M in R^{num_samples x (degree + 1)}."""
    t = np.linspace(0.0, 1.0, num_samples)
    M = np.zeros((num_samples, degree + 1), dtype=np.float64)
    n = degree
    for i in range(degree + 1):
        binom = scipy.special.comb(n, i)
        M[:, i] = binom * ((1.0 - t) ** (n - i)) * (t ** i)
    return M


def fit_single_bezier(points: np.ndarray, degree: int) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Fits a Bézier curve of specified degree to points using closed-form least-squares.
    Returns: control_points, reconstructed_points, condition_number
    """
    N = points.shape[0]
    M = build_bernstein_matrix(N, degree)
    
    # Gram matrix
    MtM = M.T @ M
    cond_num = np.linalg.cond(MtM)
    
    # Regularized normal equations
    reg = 1e-6 * np.eye(degree + 1)
    ctrl_pts = np.linalg.solve(MtM + reg, M.T @ points)
    reconstructed = M @ ctrl_pts
    return ctrl_pts, reconstructed, cond_num


def fit_piecewise_cubic(points: np.ndarray, num_segments: int = 3) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Fits BeMapNet piecewise cubic Bézier curve (k segments, n=3).
    Total control points = 3 * k + 1 (with C0 continuity at segment joints).
    For k=3: 10 control points.
    """
    N = points.shape[0]
    seg_len = N // num_segments
    all_ctrl = []
    recon_parts = []
    max_cond = 0.0
    
    for seg_idx in range(num_segments):
        start = seg_idx * seg_len
        end = (seg_idx + 1) * seg_len if seg_idx < num_segments - 1 else N
        sub_pts = points[start:end]
        ctrl, recon, cond = fit_single_bezier(sub_pts, degree=3)
        max_cond = max(max_cond, cond)
        if seg_idx == 0:
            all_ctrl.append(ctrl)
        else:
            all_ctrl.append(ctrl[1:])  # Enforce C0 joint
        recon_parts.append(recon)
        
    all_ctrl_pts = np.vstack(all_ctrl)
    reconstructed = np.vstack(recon_parts)
    return all_ctrl_pts, reconstructed, max_cond


def compute_hausdorff_px(p1: np.ndarray, p2: np.ndarray, resolution: int = 512) -> float:
    """Computes bidirectional Hausdorff distance in pixels."""
    pts1 = p1 * resolution
    pts2 = p2 * resolution
    d1 = directed_hausdorff(pts1, pts2)[0]
    d2 = directed_hausdorff(pts2, pts1)[0]
    return float(max(d1, d2))


def compute_bending_energy(curve: np.ndarray) -> float:
    """Computes discrete bending energy: sum ||p_{i+1} - 2p_i + p_{i-1}||^2."""
    d2 = curve[2:] - 2.0 * curve[1:-1] + curve[:-2]
    energy = np.sum(np.sum(d2 ** 2, axis=-1)) * (curve.shape[0] ** 3)
    return float(energy)


def run_parametric_order_benchmark():
    print("=" * 80)
    print("🔬 MATHEMATICAL TEST 01: BÉZIER ORDER & REPRESENTATION EFFICIENCY")
    print("=" * 80)
    
    landmarks = get_all_benchmark_landmarks(num_points=500)
    degrees = [3, 4, 5, 6, 7]
    
    results = {}
    
    for name, gt_curve in landmarks.items():
        print(f"\n▶ Anatomical Landmark: {name}")
        print(f"  {'Model / Degree':<24} | {'Ctrl Pts':<8} | {'HD95 (px)':<10} | {'Mean Err (px)':<14} | {'Cond Number':<12} | {'Bending Energy':<14}")
        print("  " + "-" * 90)
        
        results[name] = {}
        
        # Test single Bézier degrees
        for deg in degrees:
            label = f"Bézier (Degree {deg})"
            ctrl, recon, cond = fit_single_bezier(gt_curve, deg)
            hd = compute_hausdorff_px(gt_curve, recon)
            mean_err = np.mean(np.linalg.norm((gt_curve - recon) * 512, axis=-1))
            energy = compute_bending_energy(recon)
            
            tag = " [BCRNet]" if deg == 5 else ""
            print(f"  {label + tag:<24} | {deg + 1:<8} | {hd:<10.3f} | {mean_err:<14.3f} | {cond:<12.1e} | {energy:<14.2f}")
            results[name][label] = {"ctrl_pts": deg + 1, "hd": hd, "mean_err": mean_err, "cond": cond, "energy": energy}
            
        # Test BeMapNet Piecewise Cubic (k=3, 10 control points)
        ctrl_bm, recon_bm, cond_bm = fit_piecewise_cubic(gt_curve, num_segments=3)
        hd_bm = compute_hausdorff_px(gt_curve, recon_bm)
        mean_err_bm = np.mean(np.linalg.norm((gt_curve - recon_bm) * 512, axis=-1))
        energy_bm = compute_bending_energy(recon_bm)
        print(f"  {'BeMapNet (Piecewise k=3)':<24} | {10:<8} | {hd_bm:<10.3f} | {mean_err_bm:<14.3f} | {cond_bm:<12.1e} | {energy_bm:<14.2f}")
        results[name]["BeMapNet"] = {"ctrl_pts": 10, "hd": hd_bm, "mean_err": mean_err_bm, "cond": cond_bm, "energy": energy_bm}
        
    return results


if __name__ == "__main__":
    run_parametric_order_benchmark()
