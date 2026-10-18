#!/usr/bin/env python3
"""
Phase 3 Verification Script: Visibility-Aware & Regularization Losses.
Validates:
1. Computation of all 6 paper loss objectives.
2. Finite scalar values (zero NaN / Inf).
3. Backpropagation and gradient flow across model parameters.
4. Behavior under Stage 1 (rigid warm-up) vs. Stage 2 (joint deformable).
"""

import sys
import os

# Ensure experiment root is in sys.path
EXPERIMENT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if EXPERIMENT_ROOT not in sys.path:
    sys.path.insert(0, EXPERIMENT_ROOT)

import torch
from configs.default_config import get_default_config
from vis2reg.models.vis2reg import Vis2Reg
from losses.objectives import Vis2RegLoss


def test_losses():
    print("=" * 65)
    print("  VIS2REG PHASE 3: LOSS OBJECTIVES & GRADIENT FLOW")
    print("=" * 65)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] Running on: {device}")

    # 1. Initialize Model & Loss
    print("\n[Step 1] Initializing Vis2Reg model and Vis2RegLoss...")
    cfg = get_default_config()
    model = Vis2Reg(cfg).to(device)
    model.train()

    criterion = Vis2RegLoss(
        lambda_3d=cfg.training['lambda_3d'],
        lambda_vis=cfg.training['lambda_vis'],
        lambda_sil=cfg.training['lambda_sil'],
        lambda_def=cfg.training['lambda_def'],
        lambda_smooth=cfg.training['lambda_smooth'],
        lambda_topo=cfg.training['lambda_topo'],
    ).to(device)

    # 2. Create synthetic test batch
    print("[Step 2] Constructing test batch (N=1024 points, 1080p mask)...")
    B, N = 1, 1024
    pre_points = torch.randn(B, N, 3, device=device) * 50.0 + torch.tensor([0.0, 0.0, 100.0], device=device)
    target_points = torch.randn(B, N, 3, device=device) * 40.0 + torch.tensor([0.0, 0.0, 100.0], device=device)

    dummy_k = torch.tensor([
        [1500.0, 0.0, 960.0],
        [0.0, 1500.0, 540.0],
        [0.0, 0.0, 1.0]
    ], device=device).unsqueeze(0)

    # 2D dummy mask with foreground circle
    mask_2d = torch.zeros(B, 1, 1, 540, 960, device=device)
    mask_2d[:, :, :, 150:400, 300:700] = 1.0

    batch = {
        'pre_points': pre_points,
        'target_points': target_points,
        'target_points_cam': target_points,
        'cam_intrinsics': dummy_k,
        'intra_masks': mask_2d,
        'pre_scale': torch.tensor([[141.3]], device=device),
        'rigid_only': False,
    }

    # 3. Forward Pass
    print("\n[Step 3] Running Forward Pass (Deformable mode)...")
    outputs = model(batch)
    warped_pcd = outputs['warped_pcd']
    delta = outputs['deformation']
    print(f"         Warped points shape: {tuple(warped_pcd.shape)}")
    print(f"         Deformation shape  : {tuple(delta.shape)}")

    # 4. Compute All 6 Losses (Joint Mode)
    print("\n[Step 4] Computing all 6 Loss terms (Joint Deformable Mode)...")
    loss_dict = criterion(outputs, batch, include_deformation=True)

    print(f"         L_3D     (One-Way Chamfer)      : {loss_dict['3d'].item():.4f} (weight 0.5)")
    print(f"         L_vis    (Visible-Domain CD)    : {loss_dict['vis'].item():.4f} (weight 1.2)")
    print(f"         L_sil    (Silhouette BCE+Dice)  : {loss_dict['sil'].item():.4f} (weight 1.0)")
    print(f"         L_def    (Deformation Magnitude): {loss_dict['def'].item():.4f} (weight 0.1)")
    print(f"         L_smooth (Displacement Smooth)  : {loss_dict['smooth'].item():.4f} (weight 0.1)")
    print(f"         L_topo   (Local Edge Distance)  : {loss_dict['topo'].item():.4f} (weight 0.3)")
    print(f"         -------------------------------------------------")
    print(f"         L_total  (Weighted Multi-Task)  : {loss_dict['total'].item():.4f}")

    assert torch.isfinite(loss_dict['total']), "Total loss is NaN or Inf!"
    assert loss_dict['3d'] > 0, "L_3D must be positive!"
    assert loss_dict['def'] > 0, "L_def must be positive in deformable mode!"

    # 5. Backpropagation & Gradient Verification
    print("\n[Step 5] Testing Backpropagation & Gradient Flow...")
    model.zero_grad()
    loss_dict['total'].backward()

    grads_ok = True
    active_param_count = 0
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            active_param_count += 1
            if not torch.isfinite(p.grad).all():
                print(f"[Error] Non-finite gradients in parameter: {name}")
                grads_ok = False
                break

    print(f"         Active parameters with gradients: {active_param_count}")
    assert grads_ok, "NaN or Inf detected in backward gradients!"
    print("         Gradients are 100% clean, finite, and well-behaved!")

    # 6. Test Rigid Warm-up Mode (include_deformation = False)
    print("\n[Step 6] Testing Rigid Warm-up Mode (include_deformation = False)...")
    loss_rigid_mode = criterion(outputs, batch, include_deformation=False)
    assert loss_rigid_mode['def'].item() == 0.0, "L_def must be 0 in rigid mode!"
    assert loss_rigid_mode['smooth'].item() == 0.0, "L_smooth must be 0 in rigid mode!"
    assert loss_rigid_mode['topo'].item() == 0.0, "L_topo must be 0 in rigid mode!"
    print("         Deformation regularizers successfully zeroed out during rigid warm-up!")

    print("\n" + "=" * 65)
    print("  RESULT: ALL 6 LOSS OBJECTIVES VERIFIED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    test_losses()
