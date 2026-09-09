"""
Mathematical Test 10: Reference Point Query Count (N in {12, 18, 26, 36, 50}).
Analyzes the trade-off between query token count N in HCR deformable cross-attention:
- Theoretical reconstruction fidelity of refitted 5th-order Bézier curve
- Gram matrix condition number kappa(M^T M)
- FLOPs and computational complexity of 3-way structured self-attention:
  O(M * K * N^2 * C) for Intra-Curve Self-Attention
"""

import numpy as np
import scipy.special
from synthetic_landmarks import generate_anterior_ridge, generate_falciform_ligament
from test_01_parametric_order import fit_single_bezier, compute_hausdorff_px


def run_point_count_test():
    print("=" * 80)
    print("🔬 MATHEMATICAL TEST 10: HCR REFERENCE POINT COUNT SCALING")
    print("=" * 80)
    
    # Candidate query counts
    N_candidates = [8, 12, 18, 26, 36, 50]
    degree = 5
    
    landmarks = {
        "Falciform_S_Curve": generate_falciform_ligament(num_points=500),
        "Anterior_Ridge_Apex": generate_anterior_ridge(num_points=500),
    }
    
    for lm_name, gt_raw in landmarks.items():
        b_gt, _, _ = fit_single_bezier(gt_raw, degree=degree)
        
        # Dense ground truth for evaluation
        t_eval = np.linspace(0.0, 1.0, 500)
        M_eval = np.zeros((len(t_eval), degree + 1), dtype=np.float64)
        for i in range(degree + 1):
            M_eval[:, i] = scipy.special.comb(degree, i) * ((1.0 - t_eval) ** (degree - i)) * (t_eval ** i)
        pts_gt_eval = M_eval @ b_gt
        
        print(f"\n▶ Landmark: {lm_name}")
        print(f"  {'Points N':<10} | {'Max HD (px)':<14} | {'Mean Err (px)':<14} | {'Condition kappa':<16} | {'Intra-Attn FLOPS Rel':<20}")
        print("  " + "-" * 82)
        
        for N in N_candidates:
            t_sample = np.linspace(0.0, 1.0, N)
            M_sample = np.zeros((N, degree + 1), dtype=np.float64)
            for i in range(degree + 1):
                M_sample[:, i] = scipy.special.comb(degree, i) * ((1.0 - t_sample) ** (degree - i)) * (t_sample ** i)
                
            pts_sample = M_sample @ b_gt
            
            # Add small regression noise simulating deformable attention output (+/- 2px noise)
            np.random.seed(42)
            noise = np.random.normal(0, 2.0 / 512.0, pts_sample.shape)
            pts_noisy = pts_sample + noise
            
            # Refit Bézier from N noisy query points
            MtM = M_sample.T @ M_sample
            cond = float(np.linalg.cond(MtM))
            b_refit = np.linalg.solve(MtM + 1e-6 * np.eye(degree + 1), M_sample.T @ pts_noisy)
            
            pts_refit = M_eval @ b_refit
            hd_px = compute_hausdorff_px(pts_gt_eval, pts_refit)
            mean_err_px = float(np.mean(np.linalg.norm((pts_gt_eval - pts_refit) * 512.0, axis=-1)))
            
            # Intra-curve attention matrix complexity relative to N=26 (26^2 = 676)
            rel_flops = (N ** 2) / (26.0 ** 2)
            
            tag = " [BCRNet Standard]" if N == 26 else ""
            print(f"  {N:<10} | {hd_px:<14.3f} | {mean_err_px:<14.3f} | {cond:<16.1f} | {rel_flops:<20.2f}x{tag}")


if __name__ == "__main__":
    run_point_count_test()
