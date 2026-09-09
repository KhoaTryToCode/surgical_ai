"""
Mathematical Test 04b: Absolute Curvature vs Residual Curvature Regularization.
Tests the hypothesis:
Absolute curvature penalty (sum ||b''||^2) flattens real anatomical sharp curves.
Does Residual Curvature Loss (sum ||b'' - b''_gt||^2) preserve sharp apexes while accelerating convergence?
"""

import numpy as np
import torch
import torch.optim as optim
from synthetic_landmarks import generate_anterior_ridge
from test_01_parametric_order import fit_single_bezier
from test_04_curvature_dynamics import get_bernstein_torch


def run_residual_curvature_test():
    print("=" * 80)
    print("🔬 MATHEMATICAL TEST 04b: ABSOLUTE VS RESIDUAL CURVATURE REGULARIZATION")
    print("=" * 80)
    
    device = torch.device("cpu")
    num_eval = 50
    M = get_bernstein_torch(num_samples=num_eval, degree=5).to(device)
    
    gt_curve_np = generate_anterior_ridge(num_points=100)
    b_gt_np, _, _ = fit_single_bezier(gt_curve_np, degree=5)
    b_gt = torch.tensor(b_gt_np, dtype=torch.float32, device=device)
    pts_gt = M @ b_gt
    
    d2_gt = b_gt[2:] - 2.0 * b_gt[1:-1] + b_gt[:-2]
    
    loss_configs = {
        "B: Ctrl + Dense Pts (Baseline)": {"w_ctrl": 1.0, "w_pts": 1.0, "type": "none", "w_c": 0.0},
        "C1: Absolute Curv (w=0.01)": {"w_ctrl": 1.0, "w_pts": 1.0, "type": "absolute", "w_c": 0.01},
        "C2: Absolute Curv (w=0.10)": {"w_ctrl": 1.0, "w_pts": 1.0, "type": "absolute", "w_c": 0.10},
        "R1: Residual Curv (w=0.10)": {"w_ctrl": 1.0, "w_pts": 1.0, "type": "residual", "w_c": 0.10},
        "R2: Residual Curv (w=1.00)": {"w_ctrl": 1.0, "w_pts": 1.0, "type": "residual", "w_c": 1.00},
        "R3: Residual Curv (w=5.00)": {"w_ctrl": 1.0, "w_pts": 1.0, "type": "residual", "w_c": 5.00},
    }
    
    num_trials = 10
    results = {k: {"steps": [], "final_err": [], "energy": []} for k in loss_configs}
    
    torch.manual_seed(42)
    for trial in range(num_trials):
        noise = torch.randn_like(b_gt) * (40.0 / 512.0)
        b_init = torch.clamp(b_gt + noise, 0.0, 1.0)
        
        for name, cfg in loss_configs.items():
            b_opt = b_init.clone().detach().requires_grad_(True)
            optimizer = optim.Adam([b_opt], lr=0.01)
            steps_to_converge = 300
            
            for step in range(300):
                optimizer.zero_grad()
                pts_pred = M @ b_opt
                
                l_ctrl = torch.mean(torch.abs(b_opt - b_gt))
                l_pts = torch.mean(torch.norm(pts_pred - pts_gt, dim=-1))
                
                d2_pred = b_opt[2:] - 2.0 * b_opt[1:-1] + b_opt[:-2]
                if cfg["type"] == "absolute":
                    l_c = torch.mean(torch.sum(d2_pred ** 2, dim=-1))
                elif cfg["type"] == "residual":
                    l_c = torch.mean(torch.sum((d2_pred - d2_gt) ** 2, dim=-1))
                else:
                    l_c = 0.0
                    
                loss = cfg["w_ctrl"] * l_ctrl + cfg["w_pts"] * l_pts + cfg["w_c"] * l_c
                loss.backward()
                optimizer.step()
                
                cur_err_px = float(torch.mean(torch.norm((pts_pred.detach() - pts_gt) * 512.0, dim=-1)).item())
                if cur_err_px < 2.0 and steps_to_converge == 300:
                    steps_to_converge = step
                    
            final_err_px = float(torch.mean(torch.norm((pts_pred.detach() - pts_gt) * 512.0, dim=-1)).item())
            d2_final = pts_pred[2:] - 2.0 * pts_pred[1:-1] + pts_pred[:-2]
            energy = float(torch.sum(torch.sum(d2_final ** 2, dim=-1)).item()) * (num_eval ** 3)
            
            results[name]["steps"].append(steps_to_converge)
            results[name]["final_err"].append(final_err_px)
            results[name]["energy"].append(energy)
            
    print(f"\nBenchmark Results across {num_trials} Random Noisy Trials:")
    print(f"  {'Configuration':<32} | {'Avg Steps to <2px':<18} | {'Final Err (px)':<16} | {'Apex Energy':<14}")
    print("  " + "-" * 88)
    
    for name, res in results.items():
        avg_steps = np.mean(res["steps"])
        avg_err = np.mean(res["final_err"])
        avg_energy = np.mean(res["energy"])
        print(f"  {name:<32} | {avg_steps:<18.1f} | {avg_err:<16.3f} | {avg_energy:<14.2f}")


if __name__ == "__main__":
    run_residual_curvature_test()
