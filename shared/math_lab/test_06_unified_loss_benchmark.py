"""
Mathematical Test 06 (Corrected Metric): Unified Loss Benchmark.
Measures TRUE geometric error: min(error_forward, error_reverse).
Pits:
1. Baseline Loss (Gaussian Rasterizer + Unidirectional Matching)
2. SurgicalCurveLoss_v2 (Cauchy Rasterizer + Bidirectional Min-Matching)
"""

import numpy as np
import torch
import torch.optim as optim
from synthetic_landmarks import generate_anterior_ridge, generate_falciform_ligament
from test_01_parametric_order import fit_single_bezier
from test_02_gradient_basins import soft_rasterize_kernel, soft_dice_loss
from test_04_curvature_dynamics import get_bernstein_torch


def run_unified_loss_benchmark_corrected():
    print("=" * 80)
    print("🔬 MATHEMATICAL TEST 06: UNIFIED LOSS BENCHMARK (TRUE GEOMETRIC ERROR)")
    print("=" * 80)
    
    device = torch.device("cpu")
    num_samples = 50
    M = get_bernstein_torch(num_samples=num_samples, degree=5).to(device)
    
    landmarks = {
        "Falciform_S_Curve": generate_falciform_ligament(num_points=100),
        "Anterior_Ridge_Apex": generate_anterior_ridge(num_points=100),
    }
    
    num_trials = 10
    torch.manual_seed(42)
    
    for lm_name, gt_raw in landmarks.items():
        b_gt_np, _, _ = fit_single_bezier(gt_raw, degree=5)
        b_gt = torch.tensor(b_gt_np, dtype=torch.float32, device=device)
        b_gt_rev = torch.flip(b_gt, dims=[0])
        
        pts_gt = M @ b_gt
        pts_gt_rev = torch.flip(pts_gt, dims=[0])
        mask_gt = soft_rasterize_kernel(pts_gt, grid_size=128, sigma=0.00586, kernel_type="gaussian").detach()
        
        results = {
            "Baseline Loss (Gaussian + Unidirectional)": {"steps": [], "final_err": [], "diverged": 0},
            "SurgicalCurveLoss_v2 (Cauchy + Bidirectional)": {"steps": [], "final_err": [], "diverged": 0},
        }
        
        for trial in range(num_trials):
            # 50% inverted direction trials, 50px random displacement
            is_reversed = (trial % 2 == 1)
            base_init = b_gt_rev.clone() if is_reversed else b_gt.clone()
            noise = torch.randn_like(b_gt) * (50.0 / 512.0)
            b_init = torch.clamp(base_init + noise, 0.0, 1.0)
            
            # --- 1. Evaluate Baseline Loss ---
            b_base = b_init.clone().detach().requires_grad_(True)
            opt_base = optim.Adam([b_base], lr=0.01)
            conv_base = 300
            for step in range(300):
                opt_base.zero_grad()
                pts = M @ b_base
                
                l_ctrl = torch.mean(torch.abs(b_base - b_gt))
                l_pts = torch.mean(torch.norm(pts - pts_gt, dim=-1))
                mask_pred = soft_rasterize_kernel(pts, grid_size=128, sigma=0.0195, kernel_type="gaussian")
                l_dice = soft_dice_loss(mask_pred, mask_gt)
                
                loss = l_ctrl + l_pts + 0.5 * l_dice
                loss.backward()
                opt_base.step()
                
                err_fwd = float(torch.mean(torch.norm((pts.detach() - pts_gt) * 512.0, dim=-1)).item())
                err_rev = float(torch.mean(torch.norm((pts.detach() - pts_gt_rev) * 512.0, dim=-1)).item())
                geom_err = min(err_fwd, err_rev)
                if geom_err < 2.0 and conv_base == 300:
                    conv_base = step
                    
            final_err_base = min(
                float(torch.mean(torch.norm((pts.detach() - pts_gt) * 512.0, dim=-1)).item()),
                float(torch.mean(torch.norm((pts.detach() - pts_gt_rev) * 512.0, dim=-1)).item())
            )
            results["Baseline Loss (Gaussian + Unidirectional)"]["steps"].append(conv_base)
            results["Baseline Loss (Gaussian + Unidirectional)"]["final_err"].append(final_err_base)
            if final_err_base > 5.0:
                results["Baseline Loss (Gaussian + Unidirectional)"]["diverged"] += 1
                
            # --- 2. Evaluate SurgicalCurveLoss_v2 ---
            b_v2 = b_init.clone().detach().requires_grad_(True)
            opt_v2 = optim.Adam([b_v2], lr=0.01)
            conv_v2 = 300
            for step in range(300):
                opt_v2.zero_grad()
                pts = M @ b_v2
                
                l_fwd = torch.mean(torch.abs(b_v2 - b_gt)) + torch.mean(torch.norm(pts - pts_gt, dim=-1))
                l_rev = torch.mean(torch.abs(b_v2 - b_gt_rev)) + torch.mean(torch.norm(pts - pts_gt_rev, dim=-1))
                l_bidir = torch.min(l_fwd, l_rev)
                
                mask_pred = soft_rasterize_kernel(pts, grid_size=128, sigma=0.0195, kernel_type="cauchy")
                l_dice = soft_dice_loss(mask_pred, mask_gt)
                
                loss = l_bidir + 0.5 * l_dice
                loss.backward()
                opt_v2.step()
                
                err_fwd = float(torch.mean(torch.norm((pts.detach() - pts_gt) * 512.0, dim=-1)).item())
                err_rev = float(torch.mean(torch.norm((pts.detach() - pts_gt_rev) * 512.0, dim=-1)).item())
                geom_err = min(err_fwd, err_rev)
                if geom_err < 2.0 and conv_v2 == 300:
                    conv_v2 = step
                    
            final_err_v2 = min(
                float(torch.mean(torch.norm((pts.detach() - pts_gt) * 512.0, dim=-1)).item()),
                float(torch.mean(torch.norm((pts.detach() - pts_gt_rev) * 512.0, dim=-1)).item())
            )
            results["SurgicalCurveLoss_v2 (Cauchy + Bidirectional)"]["steps"].append(conv_v2)
            results["SurgicalCurveLoss_v2 (Cauchy + Bidirectional)"]["final_err"].append(final_err_v2)
            if final_err_v2 > 5.0:
                results["SurgicalCurveLoss_v2 (Cauchy + Bidirectional)"]["diverged"] += 1
                
        print(f"\n▶ Anatomical Landmark: {lm_name} (50% Inverted, 50px Noise, True Geometric Metric):")
        print(f"  {'Configuration':<44} | {'Avg Steps to <2px':<18} | {'Divergence Rate':<16} | {'Final Geom Err (px)':<20}")
        print("  " + "-" * 102)
        for cfg_name, res in results.items():
            avg_steps = np.mean(res["steps"])
            div_rate = (res["diverged"] / num_trials) * 100.0
            avg_err = np.mean(res["final_err"])
            print(f"  {cfg_name:<44} | {avg_steps:<18.1f} | {div_rate:>13.0f}%   | {avg_err:<20.3f}")


if __name__ == "__main__":
    run_unified_loss_benchmark_corrected()
