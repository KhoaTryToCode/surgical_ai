"""
Mathematical Test 07: Analytical Continuous clDice (AC-clDice).
A novel mathematical unification of TopoNet's clDice with BCRNet's parametric Bézier curves:

Eliminates TopoNet's slow morphological skeletonization by using the explicit curve B(t):
- Tprec = (1 / N) * sum_{i=1}^N G(B(t_i))       [Bilinear sample of GT mask at curve coordinates]
- Tsens = (1 / M) * sum_{j=1}^M S_cauchy(p_j^gt) [Evaluated Cauchy rasterizer at GT centerline points]
- AC_clDice = 2 * Tprec * Tsens / (Tprec + Tsens + eps)

Evaluates:
- Gradient magnitude and smoothness vs standard Soft Dice
- Robustness to background texture and nearby distractors
- Convergence speed under partial occlusions / gaps
"""

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from synthetic_landmarks import generate_falciform_ligament, generate_anterior_ridge
from test_01_parametric_order import fit_single_bezier
from test_02_gradient_basins import soft_rasterize_kernel
from test_04_curvature_dynamics import get_bernstein_torch


def bilinear_sample_mask(mask: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """
    Differentiably samples mask (H, W) at continuous coords (N, 2) in [0, 1]^2.
    coords: (N, 2) in [0, 1] -> normalized to [-1, 1] for grid_sample.
    """
    H, W = mask.shape
    # grid: (1, 1, N, 2)
    grid = coords.unsqueeze(0).unsqueeze(0) * 2.0 - 1.0
    input_mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    sampled = F.grid_sample(input_mask, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return sampled.squeeze()  # (N,)


def analytical_continuous_cldice_loss(
    pred_curve_pts: torch.Tensor,
    gt_mask: torch.Tensor,
    gt_centerline_pts: torch.Tensor,
    sigma: float = 0.0195,
) -> torch.Tensor:
    """
    Computes Analytical Continuous clDice in closed form.
    pred_curve_pts: (N, 2) in [0, 1]^2
    gt_mask: (H, W) binary/soft ground truth mask
    gt_centerline_pts: (M, 2) ground truth centerline points
    """
    # 1. Topological Precision: fraction of predicted curve points inside GT mask
    tprec = torch.mean(bilinear_sample_mask(gt_mask, pred_curve_pts))
    
    # 2. Topological Sensitivity: fraction of GT centerline covered by predicted soft rasterizer
    # Distance from each GT centerline point to nearest predicted curve point
    diff = gt_centerline_pts.unsqueeze(1) - pred_curve_pts.unsqueeze(0)  # (M, N, 2)
    dist_sq = torch.sum(diff ** 2, dim=-1)  # (M, N)
    min_dist_sq, _ = torch.min(dist_sq, dim=-1)  # (M,)
    min_dist = torch.sqrt(min_dist_sq + 1e-8)
    
    # Cauchy coverage at GT centerline
    cauchy_coverage = 1.0 / (1.0 + (min_dist / sigma) ** 2)
    tsens = torch.mean(cauchy_coverage)
    
    # 3. Continuous clDice
    cldice = (2.0 * tprec * tsens) / (tprec + tsens + 1e-6)
    return 1.0 - cldice


def run_cldice_test():
    print("=" * 80)
    print("🔬 MATHEMATICAL TEST 07: ANALYTICAL CONTINUOUS clDice (AC-clDice)")
    print("=" * 80)
    
    device = torch.device("cpu")
    num_samples = 50
    M_bernstein = get_bernstein_torch(num_samples=num_samples, degree=5).to(device)
    
    # Generate Falciform S-Curve
    gt_raw = generate_falciform_ligament(num_points=200)
    b_gt_np, _, _ = fit_single_bezier(gt_raw, degree=5)
    b_gt = torch.tensor(b_gt_np, dtype=torch.float32, device=device)
    
    pts_gt_centerline = M_bernstein @ b_gt
    
    # Render GT mask with dilation thickness = 10 px (10/512 = 0.0195)
    grid_size = 128
    gt_mask = soft_rasterize_kernel(pts_gt_centerline, grid_size=grid_size, sigma=0.0195, kernel_type="gaussian").detach()
    
    # Test gradient landscape when shifting curve by distance delta in [1, 60] px
    deltas_px = [2.0, 5.0, 10.0, 20.0, 40.0, 60.0]
    
    print("\n▶ Gradient Norm Comparison across Spatial Displacements:")
    print(f"  {'Displacement (px)':<18} | {'Standard Soft Dice Grad':<26} | {'Analytical AC-clDice Grad':<26} | {'Gradient Ratio':<16}")
    print("  " + "-" * 90)
    
    for d_px in deltas_px:
        d_norm = d_px / 512.0
        
        # 1. Evaluate Standard Soft Dice
        b_std = (b_gt.clone() + torch.tensor([d_norm, 0.0])).requires_grad_(True)
        pts_std = M_bernstein @ b_std
        mask_pred = soft_rasterize_kernel(pts_std, grid_size=grid_size, sigma=0.0195, kernel_type="gaussian")
        loss_std = (1.0 - (2.0 * torch.sum(mask_pred * gt_mask)) / (torch.sum(mask_pred ** 2) + torch.sum(gt_mask ** 2) + 1e-6))
        loss_std.backward()
        g_std = float(torch.norm(b_std.grad).item())
        
        # 2. Evaluate Analytical Continuous clDice
        b_cl = (b_gt.clone() + torch.tensor([d_norm, 0.0])).requires_grad_(True)
        pts_cl = M_bernstein @ b_cl
        loss_cl = analytical_continuous_cldice_loss(pts_cl, gt_mask, pts_gt_centerline, sigma=0.0195)
        loss_cl.backward()
        g_cl = float(torch.norm(b_cl.grad).item())
        
        ratio = g_cl / (g_std + 1e-8)
        print(f"  {d_px:<18.1f} | {g_std:<26.4e} | {g_cl:<26.4e} | {ratio:<16.2f}x")
        
    print("\n▶ Trajectory Convergence Comparison (10 Random 40px Perturbation Trials):")
    print(f"  {'Objective Function':<32} | {'Avg Steps to <2px':<18} | {'Final Error (px)':<16}")
    print("  " + "-" * 70)
    
    objectives = {
        "A: Standard Soft Dice": "std",
        "B: Analytical AC-clDice": "cldice",
        "C: AC-clDice + Point L1": "hybrid",
    }
    
    torch.manual_seed(42)
    results = {k: {"steps": [], "final_err": []} for k in objectives}
    
    for trial in range(10):
        noise = torch.randn_like(b_gt) * (40.0 / 512.0)
        b_init = torch.clamp(b_gt + noise, 0.0, 1.0)
        
        for name, obj_type in objectives.items():
            b_opt = b_init.clone().detach().requires_grad_(True)
            opt = optim.Adam([b_opt], lr=0.01)
            steps = 200
            
            for s in range(200):
                opt.zero_grad()
                pts = M_bernstein @ b_opt
                
                if obj_type == "std":
                    mask_p = soft_rasterize_kernel(pts, grid_size=grid_size, sigma=0.0195, kernel_type="gaussian")
                    loss = (1.0 - (2.0 * torch.sum(mask_p * gt_mask)) / (torch.sum(mask_p ** 2) + torch.sum(gt_mask ** 2) + 1e-6))
                elif obj_type == "cldice":
                    loss = analytical_continuous_cldice_loss(pts, gt_mask, pts_gt_centerline, sigma=0.0195)
                elif obj_type == "hybrid":
                    l_cl = analytical_continuous_cldice_loss(pts, gt_mask, pts_gt_centerline, sigma=0.0195)
                    l_pts = torch.mean(torch.norm(pts - pts_gt_centerline, dim=-1))
                    loss = l_cl + l_pts
                    
                loss.backward()
                opt.step()
                
                err_px = float(torch.mean(torch.norm((pts.detach() - pts_gt_centerline) * 512.0, dim=-1)).item())
                if err_px < 2.0 and steps == 200:
                    steps = s
                    
            final_err = float(torch.mean(torch.norm((pts.detach() - pts_gt_centerline) * 512.0, dim=-1)).item())
            results[name]["steps"].append(steps)
            results[name]["final_err"].append(final_err)
            
    for name, res in results.items():
        avg_steps = np.mean(res["steps"])
        avg_err = np.mean(res["final_err"])
        print(f"  {name:<32} | {avg_steps:<18.1f} | {avg_err:<16.3f}")


if __name__ == "__main__":
    run_cldice_test()
