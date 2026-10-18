#!/usr/bin/env python3
"""
Phase 1 Smoke Test for Vis2Reg Replication.
Verifies module imports, model instantiation, parameter initialization,
and forward execution on dummy point cloud tensors (CPU or GPU).
"""
import sys
import os

# Ensure this experiment folder is in Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import torch
from configs.default_config import get_default_config
from vis2reg.models.vis2reg import Vis2Reg


def run_smoke_test():
    print("=" * 60)
    print("  VIS2REG PHASE 1: SMOKE TEST & MODEL VERIFICATION")
    print("=" * 60)

    # 1. Device detection
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[Device] Using Apple Silicon MPS")
    else:
        device = torch.device("cpu")
        print("[Device] Using CPU")

    # 2. Configuration & Model Instantiation
    print("\n[Step 1] Loading default paper configuration...")
    cfg = get_default_config()
    print("[Step 2] Instantiating Vis2Reg model...")
    model = Vis2Reg(cfg).to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"         Total Parameters     : {total_params:,}")
    print(f"         Trainable Parameters : {trainable_params:,}")

    # 3. Create dummy batch matching Vis2Reg contracts
    print("\n[Step 3] Constructing dummy test batch (N=1024 points)...")
    batch_size = 1
    num_pts = 1024
    
    # Pre-op CT point cloud & Intra-op point cloud
    pre_points = torch.randn(batch_size, num_pts, 3, device=device)
    target_points = torch.randn(batch_size, num_pts, 3, device=device)
    
    dummy_k = torch.tensor([
        [1500.0, 0.0, 960.0],
        [0.0, 1500.0, 540.0],
        [0.0, 0.0, 1.0]
    ], device=device).unsqueeze(0)

    batch_rigid = {
        'pre_points': pre_points,
        'target_points': target_points,
        'target_points_cam': target_points,
        'cam_intrinsics': dummy_k,
        'pre_scale': torch.tensor([[141.3]], device=device),
        'rigid_only': True,
    }

    # 4. Test Forward Pass: Stage 1 Rigid Mode
    print("\n[Step 4] Testing Forward Pass: Rigid-Only Mode (Stage 1)...")
    with torch.no_grad():
        out_rigid = model(batch_rigid)

    R, t, scale = out_rigid['rigid_pose']
    warped_pcd = out_rigid['warped_pcd']
    print(f"         Output 'warped_pcd' shape : {tuple(warped_pcd.shape)}")
    print(f"         Estimated Rotation R      : {tuple(R.shape)}")
    print(f"         Estimated Translation t   : {tuple(t.shape)}")
    print(f"         Scale factor              : {scale.item():.4f}")
    assert warped_pcd.shape == (batch_size, num_pts, 3), "Shape mismatch on warped_pcd!"

    # 5. Test Forward Pass: Stage 2 Joint Deformable Mode
    print("\n[Step 5] Testing Forward Pass: Deformable Mode (Stage 2)...")
    batch_deform = {
        'pre_points': pre_points,
        'target_points': target_points,
        'target_points_cam': target_points,
        'cam_intrinsics': dummy_k,
        'pre_scale': torch.tensor([[141.3]], device=device),
        'rigid_only': False,
    }
    with torch.no_grad():
        out_deform = model(batch_deform)

    delta = out_deform['deformation']
    print(f"         Output 'warped_pcd' shape : {tuple(out_deform['warped_pcd'].shape)}")
    print(f"         Output 'deformation' delta: {tuple(delta.shape)}")
    assert delta.shape == (batch_size, num_pts, 3), "Shape mismatch on deformation delta!"

    print("\n" + "=" * 60)
    print("  RESULT: ALL SMOKE TESTS PASSED SUCCESSFULLY! (PHASE 1 COMPLETE)")
    print("=" * 60)


if __name__ == "__main__":
    run_smoke_test()
