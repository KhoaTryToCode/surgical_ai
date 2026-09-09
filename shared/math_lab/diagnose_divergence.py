"""
Divergence Diagnosis Script.
Traces step-by-step loss and gradient values for a single inverted trial
to determine why bidirectional matching + Cauchy rasterizer diverged.
"""

import numpy as np
import torch
import torch.optim as optim
from synthetic_landmarks import generate_falciform_ligament
from test_01_parametric_order import fit_single_bezier
from test_02_gradient_basins import soft_rasterize_kernel, soft_dice_loss
from test_04_curvature_dynamics import get_bernstein_torch


def diagnose():
    device = torch.device("cpu")
    num_samples = 50
    M = get_bernstein_torch(num_samples=num_samples, degree=5).to(device)
    
    gt_raw = generate_falciform_ligament(num_points=100)
    b_gt_np, _, _ = fit_single_bezier(gt_raw, degree=5)
    b_gt = torch.tensor(b_gt_np, dtype=torch.float32, device=device)
    b_gt_rev = torch.flip(b_gt, dims=[0])
    pts_gt = M @ b_gt
    mask_gt = soft_rasterize_kernel(pts_gt, grid_size=128, sigma=0.00586, kernel_type="gaussian").detach()
    
    # Inverted initialization + 50px noise
    torch.manual_seed(42)
    noise = torch.randn_like(b_gt) * (50.0 / 512.0)
    b_init = torch.clamp(b_gt_rev + noise, 0.0, 1.0)
    
    print("--- Tracing Baseline (Unidirectional) ---")
    b_base = b_init.clone().detach().requires_grad_(True)
    opt = optim.Adam([b_base], lr=0.01)
    for s in range(50):
        opt.zero_grad()
        pts = M @ b_base
        l_ctrl = torch.mean(torch.abs(b_base - b_gt))
        l_pts = torch.mean(torch.norm(pts - pts_gt, dim=-1))
        mask_pred = soft_rasterize_kernel(pts, grid_size=128, sigma=0.0195, kernel_type="gaussian")
        l_dice = soft_dice_loss(mask_pred, mask_gt)
        loss = l_ctrl + l_pts + 0.5 * l_dice
        loss.backward()
        opt.step()
        if s % 10 == 0:
            err = float(torch.mean(torch.norm((pts.detach() - pts_gt) * 512.0, dim=-1)).item())
            print(f"Step {s:2d}: Loss={loss.item():.4f}, L_ctrl={l_ctrl.item():.4f}, L_dice={l_dice.item():.4f}, Err={err:.1f}px")
            
    print("\n--- Tracing SurgicalCurveLoss_v2 (Bidirectional + Cauchy) ---")
    b_v2 = b_init.clone().detach().requires_grad_(True)
    opt_v2 = optim.Adam([b_v2], lr=0.01)
    for s in range(50):
        opt_v2.zero_grad()
        pts = M @ b_v2
        
        l_fwd = torch.mean(torch.abs(b_v2 - b_gt)) + torch.mean(torch.norm(pts - pts_gt, dim=-1))
        l_rev = torch.mean(torch.abs(b_v2 - b_gt_rev)) + torch.mean(torch.norm(pts - torch.flip(pts_gt, dims=[0]), dim=-1))
        
        # Which one is selected?
        selected = "REV" if l_rev < l_fwd else "FWD"
        l_bidir = torch.min(l_fwd, l_rev)
        
        mask_pred = soft_rasterize_kernel(pts, grid_size=128, sigma=0.0195, kernel_type="cauchy")
        l_dice = soft_dice_loss(mask_pred, mask_gt)
        
        loss = l_bidir + 0.5 * l_dice
        loss.backward()
        opt_v2.step()
        if s % 10 == 0:
            err_fwd = float(torch.mean(torch.norm((pts.detach() - pts_gt) * 512.0, dim=-1)).item())
            err_rev = float(torch.mean(torch.norm((pts.detach() - torch.flip(pts_gt, dims=[0])) * 512.0, dim=-1)).item())
            print(f"Step {s:2d}: Sel={selected}, L_fwd={l_fwd.item():.4f}, L_rev={l_rev.item():.4f}, L_dice={l_dice.item():.4f}, Err_fwd={err_fwd:.1f}px, Err_rev={err_rev:.1f}px")


if __name__ == "__main__":
    diagnose()
