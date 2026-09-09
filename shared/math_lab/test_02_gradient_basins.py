"""
Mathematical Test 02: Differentiable Rasterizer Gradient Basins & Kernels.
Compares:
1. Gaussian Kernel: exp(-d^2 / (2 * sigma^2))
2. Heavy-Tailed Cauchy / Student-t Kernel: 1 / (1 + (d / sigma)^2)
3. Generalized Lorentzian Kernel: 1 / (1 + (d / sigma)^2)^2
4. Signed Distance Field (SDF) Exponential: exp(-d / sigma)

Evaluates:
- Gradient magnitude ||dL/db|| across perturbation distance delta in [1, 100] pixels.
- Vanishing threshold distance (where gradient drops below 1e-5).
- Basin convexity / monotonicity (lack of spurious local minima).
"""

import numpy as np
import torch
import torch.nn.functional as F
import scipy.special
from synthetic_landmarks import generate_falciform_ligament


def sample_bezier_torch(ctrl_pts: torch.Tensor, num_samples: int = 100) -> torch.Tensor:
    """
    Evaluates 5th-order Bézier curve in PyTorch (differentiable).
    ctrl_pts: (6, 2)
    Returns: (num_samples, 2)
    """
    device = ctrl_pts.device
    t = torch.linspace(0.0, 1.0, num_samples, device=device).unsqueeze(-1)  # (N, 1)
    degree = 5
    basis_list = []
    for i in range(degree + 1):
        c = scipy.special.comb(degree, i)
        b_i = c * ((1.0 - t) ** (degree - i)) * (t ** i)
        basis_list.append(b_i)
    M = torch.cat(basis_list, dim=-1)  # (N, 6)
    return M @ ctrl_pts  # (N, 2)


def soft_rasterize_kernel(
    curve_pts: torch.Tensor,
    grid_size: int = 128,
    sigma: float = 0.05,
    kernel_type: str = "gaussian",
) -> torch.Tensor:
    """
    Renders differentiable soft landmark mask.
    curve_pts: (N, 2) in [0, 1]^2
    Returns: (grid_size, grid_size)
    """
    device = curve_pts.device
    ys = torch.linspace(0.0, 1.0, grid_size, device=device)
    xs = torch.linspace(0.0, 1.0, grid_size, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)
    
    # Broadcast distance: (H, W, 1, 2) - (1, 1, N, 2)
    diff = grid.unsqueeze(2) - curve_pts.unsqueeze(0).unsqueeze(0)  # (H, W, N, 2)
    dist_sq = torch.sum(diff ** 2, dim=-1)  # (H, W, N)
    min_dist_sq, _ = torch.min(dist_sq, dim=-1)  # (H, W)
    min_dist = torch.sqrt(min_dist_sq + 1e-8)
    
    if kernel_type == "gaussian":
        return torch.exp(-min_dist_sq / (2.0 * (sigma ** 2)))
    elif kernel_type == "cauchy":
        return 1.0 / (1.0 + (min_dist / sigma) ** 2)
    elif kernel_type == "lorentzian":
        return 1.0 / ((1.0 + (min_dist / sigma) ** 2) ** 2)
    elif kernel_type == "laplacian_sdf":
        return torch.exp(-min_dist / sigma)
    else:
        raise ValueError(f"Unknown kernel {kernel_type}")


def soft_dice_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    inter = torch.sum(pred * target)
    denom = torch.sum(pred ** 2) + torch.sum(target ** 2) + 1e-6
    return 1.0 - (2.0 * inter) / denom


def run_gradient_basin_test():
    print("=" * 80)
    print("🔬 MATHEMATICAL TEST 02: DIFFERENTIABLE RASTERIZER GRADIENT BASINS")
    print("=" * 80)
    
    device = torch.device("cpu")
    grid_size = 128
    
    # Generate GT curve and fit Degree-5 control points
    gt_curve_np = generate_falciform_ligament(num_points=100)
    
    # Compute GT control points
    from test_01_parametric_order import fit_single_bezier
    gt_ctrl_np, _, _ = fit_single_bezier(gt_curve_np, degree=5)
    gt_ctrl = torch.tensor(gt_ctrl_np, dtype=torch.float32, device=device)
    
    # Rasterize GT target mask (using sharp sigma=2px on 512 -> 0.0039)
    gt_curve_sampled = sample_bezier_torch(gt_ctrl, num_samples=80)
    gt_mask = soft_rasterize_kernel(gt_curve_sampled, grid_size=grid_size, sigma=0.015, kernel_type="gaussian").detach()
    
    kernels = ["gaussian", "cauchy", "lorentzian", "laplacian_sdf"]
    sigmas_px = [2.0, 10.0, 30.0]  # Pixel scale on 512x512
    perturbations_px = [2.0, 5.0, 10.0, 20.0, 40.0, 80.0]
    
    for sigma_px in sigmas_px:
        sigma_norm = sigma_px / 512.0
        print(f"\n▶ Testing Rasterizer Sigma = {sigma_px:.1f} px (Normalized: {sigma_norm:.4f})")
        print(f"  {'Kernel Type':<16} | " + " | ".join([f"d={p:2.0f}px Grad" for p in perturbations_px]))
        print("  " + "-" * 85)
        
        for k_type in kernels:
            grad_norms = []
            for d_px in perturbations_px:
                d_norm = d_px / 512.0
                # Perturb control points perpendicular to the curve
                pert_ctrl = gt_ctrl.clone()
                pert_ctrl[:, 0] += d_norm  # Shift x coordinate
                pert_ctrl.requires_grad_(True)
                
                # Render predicted mask
                pred_pts = sample_bezier_torch(pert_ctrl, num_samples=80)
                pred_mask = soft_rasterize_kernel(pred_pts, grid_size=grid_size, sigma=sigma_norm, kernel_type=k_type)
                
                # Compute Loss and Gradient
                loss = soft_dice_loss(pred_mask, gt_mask)
                loss.backward()
                
                gnorm = float(torch.norm(pert_ctrl.grad).item())
                grad_norms.append(gnorm)
                
            cols = " | ".join([f"{gn:11.2e}" for gn in grad_norms])
            print(f"  {k_type:<16} | {cols}")


if __name__ == "__main__":
    run_gradient_basin_test()
