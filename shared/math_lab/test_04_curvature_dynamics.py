"""
Mathematical Test 04: Curvature & Tangent Regularization in Curve Optimization.
Simulates gradient descent trajectory optimization of 5th-order Bézier control points
from noisy initialization to anatomical target curves.

Compares 4 Loss Formulations:
1. Loss A: Pure Control Point L1
2. Loss B: Control Point L1 + Dense Sampled Point L1
3. Loss C: Control Point L1 + Dense Point L1 + 2nd-Order Curvature Regularizer
4. Loss D (Proposed): Triplet (L1 + Dense + Curvature + Directional Tangent Cosine)

Evaluates:
- Convergence Speed (Iterations to MAE < 2 px)
- Peak Optimization Curvature / Loop Occurrence
- Final Smoothness (Bending Energy)
"""

import numpy as np
import torch
import torch.optim as optim
import scipy.special
from synthetic_landmarks import generate_anterior_ridge
from test_01_parametric_order import fit_single_bezier


def get_bernstein_torch(num_samples: int = 50, degree: int = 5) -> torch.Tensor:
    t = np.linspace(0.0, 1.0, num_samples)
    M = np.zeros((num_samples, degree + 1), dtype=np.float32)
    for i in range(degree + 1):
        c = scipy.special.comb(degree, i)
        M[:, i] = c * ((1.0 - t) ** (degree - i)) * (t ** i)
    return torch.tensor(M, dtype=torch.float32)


def check_self_intersection(points: np.ndarray) -> bool:
    """
    Checks if a 2D polyline self-intersects using line segment cross-products.
    """
    N = len(points)
    for i in range(N - 1):
        p1, p2 = points[i], points[i + 1]
        for j in range(i + 2, N - 1):
            if i == 0 and j == N - 2:
                continue
            p3, p4 = points[j], points[j + 1]
            
            # Line intersection test
            def ccw(A, B, C):
                return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
            
            if (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4)):
                return True
    return False


def run_curvature_dynamics_test():
    print("=" * 80)
    print("🔬 MATHEMATICAL TEST 04: CURVATURE & TANGENT OPTIMIZATION DYNAMICS")
    print("=" * 80)
    
    device = torch.device("cpu")
    num_eval = 50
    M = get_bernstein_torch(num_samples=num_eval, degree=5).to(device)
    
    # Ground truth: Anterior ridge (sharp apex, high curvature)
    gt_curve_np = generate_anterior_ridge(num_points=100)
    b_gt_np, _, _ = fit_single_bezier(gt_curve_np, degree=5)
    b_gt = torch.tensor(b_gt_np, dtype=torch.float32, device=device)
    pts_gt = M @ b_gt
    
    # Repeat across 10 random noisy initializations
    num_trials = 10
    torch.manual_seed(42)
    
    loss_configs = {
        "A: Pure Control L1": {"w_ctrl": 1.0, "w_pts": 0.0, "w_curv": 0.0, "w_tan": 0.0},
        "B: Ctrl + Dense Pts": {"w_ctrl": 1.0, "w_pts": 1.0, "w_curv": 0.0, "w_tan": 0.0},
        "C: B + 2nd-Order Curv": {"w_ctrl": 1.0, "w_pts": 1.0, "w_curv": 5.0, "w_tan": 0.0},
        "D: Triplet (+ Tangent)": {"w_ctrl": 1.0, "w_pts": 1.0, "w_curv": 5.0, "w_tan": 2.0},
    }
    
    results = {k: {"steps": [], "loops": 0, "final_err": [], "energy": []} for k in loss_configs}
    
    for trial in range(num_trials):
        # Add significant random displacement to control points (+/- 40 px)
        noise = torch.randn_like(b_gt) * (40.0 / 512.0)
        b_init = torch.clamp(b_gt + noise, 0.0, 1.0)
        
        for name, cfg in loss_configs.items():
            b_opt = b_init.clone().detach().requires_grad_(True)
            optimizer = optim.Adam([b_opt], lr=0.01)
            
            steps_to_converge = 300
            had_loop = False
            
            for step in range(300):
                optimizer.zero_grad()
                pts_pred = M @ b_opt
                
                # Check for loop
                if not had_loop and step % 10 == 0:
                    if check_self_intersection(pts_pred.detach().numpy()):
                        had_loop = True
                        
                # 1. Control L1
                l_ctrl = torch.mean(torch.abs(b_opt - b_gt))
                
                # 2. Dense Point L1
                l_pts = torch.mean(torch.norm(pts_pred - pts_gt, dim=-1))
                
                # 3. 2nd-Order Curvature Regularizer
                d2_b = b_opt[2:] - 2.0 * b_opt[1:-1] + b_opt[:-2]
                l_curv = torch.mean(torch.sum(d2_b ** 2, dim=-1))
                
                # 4. Tangent Direction Cosine Loss
                t_pred = pts_pred[1:] - pts_pred[:-1]
                t_gt = pts_gt[1:] - pts_gt[:-1]
                dot = torch.sum(t_pred * t_gt, dim=-1)
                norm_p = torch.norm(t_pred, dim=-1) + 1e-6
                norm_g = torch.norm(t_gt, dim=-1) + 1e-6
                cos_sim = dot / (norm_p * norm_g)
                l_tan = torch.mean(1.0 - cos_sim)
                
                loss = (
                    cfg["w_ctrl"] * l_ctrl +
                    cfg["w_pts"] * l_pts +
                    cfg["w_curv"] * l_curv +
                    cfg["w_tan"] * l_tan
                )
                
                loss.backward()
                optimizer.step()
                
                # Check convergence (Mean L2 error < 2 px on 512x512)
                cur_err_px = float(torch.mean(torch.norm((pts_pred.detach() - pts_gt) * 512.0, dim=-1)).item())
                if cur_err_px < 2.0 and steps_to_converge == 300:
                    steps_to_converge = step
                    
            final_err_px = float(torch.mean(torch.norm((pts_pred.detach() - pts_gt) * 512.0, dim=-1)).item())
            d2_final = pts_pred[2:] - 2.0 * pts_pred[1:-1] + pts_pred[:-2]
            energy = float(torch.sum(torch.sum(d2_final ** 2, dim=-1)).item()) * (num_eval ** 3)
            
            results[name]["steps"].append(steps_to_converge)
            results[name]["final_err"].append(final_err_px)
            results[name]["energy"].append(energy)
            if had_loop:
                results[name]["loops"] += 1
                
    print(f"\nOptimization Benchmark Results across {num_trials} Random Noisy Trials:")
    print(f"  {'Configuration':<26} | {'Avg Steps to <2px':<18} | {'Loop/Knot Rate':<16} | {'Final Err (px)':<16} | {'Smoothness Energy':<16}")
    print("  " + "-" * 96)
    
    for name, res in results.items():
        avg_steps = np.mean(res["steps"])
        loop_pct = (res["loops"] / num_trials) * 100.0
        avg_err = np.mean(res["final_err"])
        avg_energy = np.mean(res["energy"])
        print(f"  {name:<26} | {avg_steps:<18.1f} | {loop_pct:>13.0f}%   | {avg_err:<16.3f} | {avg_energy:<16.2f}")


if __name__ == "__main__":
    run_curvature_dynamics_test()
