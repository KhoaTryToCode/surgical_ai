"""
Mathematical Test 05: Reference Point Sampling Strategies in HCR.
Compares 4 sampling distributions along 5th-order Bézier curves for HCR deformable attention:
1. Uniform Parameter Sampling: t_i = i / (N - 1)  [BCRNet Baseline]
2. Chebyshev-Lobatto Sampling: t_i = 0.5 * (1 - cos(pi * i / (N - 1))) [Endpoint clustering]
3. Arc-Length Equidistant Sampling: equal physical Euclidean distance along curve
4. Curvature-Adaptive Sampling: density proportional to sqrt(|kappa(t)|) + eps

Evaluates:
- Least-squares refitting condition number kappa(M^T M)
- Max Hausdorff Reconstruction Error on high-curvature landmarks (Anterior Ridge, Falciform)
- Apex local reconstruction error (px)
"""

import numpy as np
import scipy.special
from synthetic_landmarks import get_all_benchmark_landmarks
from test_01_parametric_order import fit_single_bezier, compute_hausdorff_px


def evaluate_bezier_at_t(b: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Evaluates Bézier curve b in R^{6x2} at parameter values t in [0, 1]."""
    degree = len(b) - 1
    N = len(t)
    M = np.zeros((N, degree + 1), dtype=np.float64)
    for i in range(degree + 1):
        c = scipy.special.comb(degree, i)
        M[:, i] = c * ((1.0 - t) ** (degree - i)) * (t ** i)
    return M @ b, M


def compute_curvature_profile(b: np.ndarray, num_eval: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """Computes analytical curvature kappa(t) = |x' y'' - y' x''| / (x'^2 + y'^2)^(3/2)."""
    t = np.linspace(0.0, 1.0, num_eval)
    degree = len(b) - 1
    
    # 1st derivative control points: Delta b = degree * (b_{j+1} - b_j)
    db = degree * (b[1:] - b[:-1])
    d1_pts, _ = evaluate_bezier_at_t(db, t)
    xp, yp = d1_pts[:, 0], d1_pts[:, 1]
    
    # 2nd derivative control points: Delta^2 b = degree * (degree - 1) * (b_{j+2} - 2 b_{j+1} + b_j)
    ddb = degree * (degree - 1) * (b[2:] - 2.0 * b[1:-1] + b[:-2])
    d2_pts, _ = evaluate_bezier_at_t(ddb, t)
    xpp, ypp = d2_pts[:, 0], d2_pts[:, 1]
    
    speed_sq = xp ** 2 + yp ** 2
    speed = np.sqrt(speed_sq) + 1e-8
    kappa = np.abs(xp * ypp - yp * xpp) / (speed_sq * speed)
    return t, kappa, speed


def sample_strategies(b: np.ndarray, N: int = 26) -> dict[str, np.ndarray]:
    t_fine, kappa, speed = compute_curvature_profile(b, num_eval=2000)
    
    # 1. Uniform parameter
    t_uniform = np.linspace(0.0, 1.0, N)
    
    # 2. Chebyshev-Lobatto
    k = np.arange(N)
    t_chebyshev = 0.5 * (1.0 - np.cos(np.pi * k / (N - 1)))
    
    # 3. Arc-Length Equidistant
    arc_length = np.cumsum(speed) * (1.0 / len(t_fine))
    arc_length = arc_length / arc_length[-1]  # normalize to [0, 1]
    s_targets = np.linspace(0.0, 1.0, N)
    t_arclength = np.interp(s_targets, arc_length, t_fine)
    
    # 4. Curvature-Adaptive
    weight = np.sqrt(kappa) + 0.1
    cum_weight = np.cumsum(weight)
    cum_weight = cum_weight / cum_weight[-1]
    w_targets = np.linspace(0.0, 1.0, N)
    t_curvature = np.interp(w_targets, cum_weight, t_fine)
    
    return {
        "1: Uniform Parameter (BCRNet)": t_uniform,
        "2: Chebyshev-Lobatto": t_chebyshev,
        "3: Arc-Length Equidistant": t_arclength,
        "4: Curvature-Adaptive": t_curvature,
    }


def run_sampling_benchmark():
    print("=" * 80)
    print("🔬 MATHEMATICAL TEST 05: REFERENCE POINT SAMPLING STRATEGIES FOR HCR")
    print("=" * 80)
    
    landmarks = get_all_benchmark_landmarks(num_points=500)
    degree = 5
    N_queries = 26  # Same as BCRNet
    
    for name, gt_raw in landmarks.items():
        b_gt, _, _ = fit_single_bezier(gt_raw, degree=degree)
        
        # Dense ground truth curve for testing
        t_dense = np.linspace(0.0, 1.0, 200)
        pts_gt_dense, _ = evaluate_bezier_at_t(b_gt, t_dense)
        
        strat_dict = sample_strategies(b_gt, N=N_queries)
        
        print(f"\n▶ Anatomical Landmark: {name}")
        print(f"  {'Sampling Strategy':<30} | {'Max HD (px)':<12} | {'Mean Err (px)':<14} | {'Condition kappa(M^T M)':<22}")
        print("  " + "-" * 84)
        
        for s_name, t_samples in strat_dict.items():
            # Sample points at t_samples
            sampled_pts, M_sample = evaluate_bezier_at_t(b_gt, t_samples)
            
            # Least-squares refit from sampled points (simulating HCR curve refitting step)
            MtM = M_sample.T @ M_sample
            cond = float(np.linalg.cond(MtM))
            
            # Refitted control points
            reg = 1e-7 * np.eye(degree + 1)
            b_refit = np.linalg.solve(MtM + reg, M_sample.T @ sampled_pts)
            
            # Evaluate on dense evaluation grid
            pts_refit_dense, _ = evaluate_bezier_at_t(b_refit, t_dense)
            
            hd_px = compute_hausdorff_px(pts_gt_dense, pts_refit_dense)
            mean_err = float(np.mean(np.linalg.norm((pts_gt_dense - pts_refit_dense) * 512.0, axis=-1)))
            
            print(f"  {s_name:<30} | {hd_px:<12.4f} | {mean_err:<14.4f} | {cond:<22.1f}")


if __name__ == "__main__":
    run_sampling_benchmark()
