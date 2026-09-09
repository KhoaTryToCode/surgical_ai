"""
Mathematical Test 03: Orientation Ambiguity & Permutation-Equivalent Matching.
Analyzes the mathematical penalty inflicted on a geometrically perfect curve prediction
that has reversed direction indexing (t -> 1-t).

Compares:
1. Standard Unidirectional Control Point L1 (BCRNet)
2. Standard Unidirectional Interpolated Point L1
3. MapTRv2 Bidirectional Min-Matching (Forward vs Reverse)
4. Order-Free Chamfer / Sinkhorn Distance
"""

import numpy as np
import torch
import scipy.special
from synthetic_landmarks import get_all_benchmark_landmarks
from test_01_parametric_order import fit_single_bezier, build_bernstein_matrix


def run_orientation_test():
    print("=" * 80)
    print("🔬 MATHEMATICAL TEST 03: ORIENTATION AMBIGUITY & PERMUTATION MATCHING")
    print("=" * 80)
    
    landmarks = get_all_benchmark_landmarks(num_points=100)
    degree = 5
    num_eval_pts = 26  # Same as BCRNet N=26
    M = build_bernstein_matrix(num_eval_pts, degree)
    
    for name, gt_raw in landmarks.items():
        # Fit exact GT Bézier control points
        b_gt, _, _ = fit_single_bezier(gt_raw, degree=degree)
        
        # Inverted prediction: geometrically identical curve, reversed indexing
        b_reversed = b_gt[::-1].copy()
        
        # Sample points along both parameterizations
        pts_gt = M @ b_gt
        pts_reversed = M @ b_reversed
        
        # 1. Unidirectional Control Point L1 (BCRNet L_crv component)
        l1_ctrl_uni = np.mean(np.abs(b_reversed - b_gt)) * 512.0  # in px
        
        # 2. Unidirectional Point Distance
        l1_pts_uni = np.mean(np.linalg.norm((pts_reversed - pts_gt) * 512.0, axis=-1))
        
        # 3. MapTRv2 Bidirectional Distance
        # Compare forward vs backward order
        dist_forward = np.mean(np.linalg.norm((pts_reversed - pts_gt) * 512.0, axis=-1))
        dist_backward = np.mean(np.linalg.norm((pts_reversed - pts_gt[::-1]) * 512.0, axis=-1))
        dist_bidirectional = min(dist_forward, dist_backward)
        
        # 4. Chamfer Distance (Set-to-Set geometry)
        # For each point in P1, min distance to P2, and vice versa
        diff = pts_reversed[:, None, :] - pts_gt[None, :, :]  # (N, N, 2)
        dists = np.linalg.norm(diff * 512.0, axis=-1)  # (N, N)
        chamfer = 0.5 * (np.mean(np.min(dists, axis=1)) + np.mean(np.min(dists, axis=0)))
        
        print(f"\n▶ Anatomical Landmark: {name}")
        print(f"  Geometrically Identical (Inverted Direction):")
        print(f"  • Unidirectional Control Point L1 Error: {l1_ctrl_uni:>8.2f} px  <-- FATAL FALSE PENALTY!")
        print(f"  • Unidirectional Sampled Point L1 Error: {l1_pts_uni:>8.2f} px  <-- FATAL FALSE PENALTY!")
        print(f"  • MapTRv2 Bidirectional Min-Match Error: {dist_bidirectional:>8.2f} px  <-- EXACT 0.0 px RECOVERY")
        print(f"  • Set Chamfer Distance:                 {chamfer:>8.2f} px  <-- EXACT 0.0 px RECOVERY")


if __name__ == "__main__":
    run_orientation_test()
