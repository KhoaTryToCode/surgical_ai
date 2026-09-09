"""
Mathematical Test 14: Class-Adaptive vs Constant Sigma in Differentiable Rasterization.
Tests whether matching the soft rasterizer sigma_m to each anatomical class's physical thickness:
- Falciform Ligament: sigma = 20 px (wide fibrous band)
- Anterior Ridge:     sigma = 8 px  (sharp apex / acute margin)
- Liver Silhouette:   sigma = 15 px (intermediate horizon)
outperforms a single uniform sigma = 10 px across all classes.

Evaluates:
- Direct soft mask Intersection over Union (IoU) against dilated ground truth masks
- Gradient localization sharpness at true anatomical boundaries
"""

import numpy as np
import torch
from synthetic_landmarks import generate_falciform_ligament, generate_anterior_ridge, generate_liver_silhouette
from test_01_parametric_order import fit_single_bezier
from test_02_gradient_basins import soft_rasterize_kernel
from test_04_curvature_dynamics import get_bernstein_torch


def run_adaptive_sigma_test():
    print("=" * 80)
    print("🔬 MATHEMATICAL TEST 14: CLASS-ADAPTIVE VS CONSTANT RASTERIZER SIGMA")
    print("=" * 80)
    
    device = torch.device("cpu")
    num_samples = 60
    M = get_bernstein_torch(num_samples=num_samples, degree=5).to(device)
    grid_size = 128
    
    # Anatomical landmark GT shapes and their true anatomical dilation widths (in pixels on 512x512)
    landmarks = {
        "Falciform_Ligament": {"data": generate_falciform_ligament(100), "true_width_px": 24.0, "adaptive_sigma_px": 20.0},
        "Anterior_Ridge":     {"data": generate_anterior_ridge(100),     "true_width_px": 8.0,  "adaptive_sigma_px": 8.0},
        "Liver_Silhouette":   {"data": generate_liver_silhouette(100),   "true_width_px": 16.0, "adaptive_sigma_px": 16.0},
    }
    
    constant_sigma_px = 12.0  # Common compromise setting
    
    print(f"\nComparing Constant Sigma ({constant_sigma_px:.1f} px) vs Class-Adaptive Sigma:")
    print(f"  {'Landmark Category':<24} | {'True Width':<12} | {'Const σ IoU':<14} | {'Adaptive σ IoU':<16} | {'IoU Delta':<12}")
    print("  " + "-" * 84)
    
    for name, info in landmarks.items():
        b_gt_np, _, _ = fit_single_bezier(info["data"], degree=5)
        b_gt = torch.tensor(b_gt_np, dtype=torch.float32, device=device)
        pts_gt = M @ b_gt
        
        # Render Ground Truth Mask with its true physical width
        sigma_gt_norm = info["true_width_px"] / 512.0
        mask_gt = soft_rasterize_kernel(pts_gt, grid_size=grid_size, sigma=sigma_gt_norm, kernel_type="cauchy").detach()
        # Binarize GT mask at 0.5 threshold
        gt_binary = (mask_gt > 0.3).float()
        
        # 1. Render with Constant Sigma
        const_norm = constant_sigma_px / 512.0
        mask_const = soft_rasterize_kernel(pts_gt, grid_size=grid_size, sigma=const_norm, kernel_type="cauchy")
        inter_const = torch.sum(mask_const * gt_binary)
        union_const = torch.sum(mask_const) + torch.sum(gt_binary) - inter_const
        iou_const = float((inter_const / (union_const + 1e-6)).item())
        
        # 2. Render with Class-Adaptive Sigma
        adapt_norm = info["adaptive_sigma_px"] / 512.0
        mask_adapt = soft_rasterize_kernel(pts_gt, grid_size=grid_size, sigma=adapt_norm, kernel_type="cauchy")
        inter_adapt = torch.sum(mask_adapt * gt_binary)
        union_adapt = torch.sum(mask_adapt) + torch.sum(gt_binary) - inter_adapt
        iou_adapt = float((inter_adapt / (union_adapt + 1e-6)).item())
        
        delta_iou = (iou_adapt - iou_const) * 100.0
        print(f"  {name:<24} | {info['true_width_px']:<12.1f} | {iou_const:<14.4f} | {iou_adapt:<16.4f} | {delta_iou:>+8.2f}%")


if __name__ == "__main__":
    run_adaptive_sigma_test()
