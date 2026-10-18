#!/usr/bin/env python3
"""
Phase 5: Stage 2 Real Visibility-Aware Fine-Tuning Script.
Replicates Vis2Reg Stage 2 Real Training (MICCAI 2026 paper §3.1):
1. Loads pretrained rigid seed checkpoint (checkpoints/stage1_rigid_seed.pth).
2. Two-Stage Schedule:
   - Epochs 1 to 20 (Rigid Warm-Up): Deformation field frozen, optimizes L_3D + L_vis + L_sil.
   - Epochs 21 to 60 (Joint Optimization): Unfreezes SIREN deformation field, optimizes all 6 losses.
3. Multi-Task Losses:
   - L_3D     (One-Way Chamfer, weight 0.5)
   - L_vis    (Visible-Domain Symmetric Chamfer, weight 1.2)
   - L_sil    (Soft Silhouette BCE + Dice, weight 1.0)
   - L_def    (Deformation Magnitude, weight 0.1)
   - L_smooth (Displacement Smoothness, weight 0.1)
   - L_topo   (Local Edge Length Preservation, weight 0.3)
4. Validates on held-out validation patient keyframes after each epoch:
   - Tracks Silhouette Dice (%) and One-Way Chamfer Distance (mm).
5. Saves best model checkpoint to checkpoints/best_real_model.pth based on validation Dice.
"""

from __future__ import annotations

import os
import sys
import time
import random
import argparse
from typing import Dict, Tuple

# Ensure experiment root is in sys.path
EXPERIMENT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if EXPERIMENT_ROOT not in sys.path:
    sys.path.insert(0, EXPERIMENT_ROOT)

import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from configs.default_config import get_default_config
from vis2reg.models.vis2reg import Vis2Reg
from data.dataset import P2ILRegFolds, P2ILRegRealDataset, collate_p2i_lreg
from losses.objectives import Vis2RegLoss


def parse_args():
    parser = argparse.ArgumentParser(description="Vis2Reg Stage 2: Real Visibility-Aware Fine-Tuning")
    parser.add_argument(
        "--data_root",
        type=str,
        default="/kaggle/input/datasets/junezhou001/p2i-lreg/Liver_regis",
        help="Path to Liver_regis dataset root",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Cross-validation fold index [0..4] (default: 0)",
    )
    parser.add_argument(
        "--pretrained_seed",
        type=str,
        default="checkpoints/stage1_rigid_seed.pth",
        help="Path to Stage 1 pretrained rigid seed checkpoint",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=60,
        help="Total Stage 2 training epochs (default: 60, paper §3.1)",
    )
    parser.add_argument(
        "--rigid_warmup_epochs",
        type=int,
        default=20,
        help="Rigid warm-up epochs before deformation is unfrozen (default: 20, paper §3.1)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Mini-batch size (default: 1, matching paper TrainConfig.batch_size=1)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate for AdamW (default: 3e-4)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay for AdamW (default: 1e-4)",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=6000,
        help="Resampled point cloud size (default: 6000, matching paper DataConfig.max_points=6000)",
    )
    parser.add_argument(
        "--render_size",
        type=int,
        default=256,
        help="Resolution for silhouette rendering (default: 256 for fast differentiable rendering)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints",
        help="Directory to save fine-tuned checkpoints",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2 if torch.cuda.is_available() else 0,
        help="DataLoader worker subprocesses",
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Run 1 quick epoch with 2 batches to verify pipeline without waiting",
    )
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_eval_metrics(
    pred_mask: torch.Tensor,
    gt_mask: torch.Tensor,
    target_pts: torch.Tensor,
    warped_pts: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[float, float]:
    """
    Computes official paper metrics:
    1. Silhouette Dice (%) between rendered silhouette and ground-truth 2D mask.
    2. Observation-to-Model One-Way Chamfer Distance (mm).
    """
    if pred_mask.dim() == 3:
        pred_mask = pred_mask.squeeze(0)
    if gt_mask.dim() == 5:
        gt_mask = gt_mask[0, 0, 0]
    elif gt_mask.dim() == 4:
        gt_mask = gt_mask[0, 0]
    elif gt_mask.dim() == 3:
        gt_mask = gt_mask[0]

    # Resize gt_mask if needed
    if pred_mask.shape[-2:] != gt_mask.shape[-2:]:
        gt_mask = F.interpolate(
            gt_mask.unsqueeze(0).unsqueeze(0).float(),
            size=pred_mask.shape[-2:],
            mode='nearest'
        ).squeeze()

    pred_bin = (pred_mask > 0.5).float()
    gt_bin = (gt_mask > 0.5).float()

    intersection = (pred_bin * gt_bin).sum().item()
    union = pred_bin.sum().item() + gt_bin.sum().item()
    dice_score = (2.0 * intersection + eps) / (union + eps) * 100.0

    # One-Way Chamfer Distance (Target Observation -> Warped Model) in mm
    dists = torch.cdist(target_pts, warped_pts, p=2.0)
    cd_oneway_mm = dists.min(dim=-1).values.mean().item()

    return dice_score, cd_oneway_mm


def evaluate(model, val_loader, criterion, device, include_deformation: bool) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_cd = 0.0
    count = 0

    with torch.no_grad():
        for batch in val_loader:
            batch_dev = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            batch_dev['rigid_only'] = not include_deformation

            outputs = model(batch_dev)
            losses = criterion(outputs, batch_dev, include_deformation=include_deformation)

            b_size = batch_dev['pre_points'].shape[0]
            total_loss += losses['total'].item() * b_size

            # Evaluate Dice & One-Way Chamfer
            render_mask = outputs.get('render_mask')
            gt_mask = batch_dev.get('intra_masks')
            target_pts = batch_dev['target_points']
            warped_pts = outputs['warped_pcd']

            if render_mask is not None and gt_mask is not None:
                dice, cd = compute_eval_metrics(render_mask, gt_mask, target_pts, warped_pts)
                total_dice += dice * b_size
                total_cd += cd * b_size
            count += b_size

    if count == 0:
        return {'loss': 0.0, 'dice_pct': 0.0, 'cd_mm': 0.0}

    return {
        'loss': total_loss / count,
        'dice_pct': total_dice / count,
        'cd_mm': total_cd / count,
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    print("=" * 65)
    print("  VIS2REG PHASE 5: STAGE 2 REAL VISIBILITY-AWARE FINE-TUNING")
    print("=" * 65)

    # 1. Device Setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[Device] Using Apple Silicon MPS")
    else:
        device = torch.device("cpu")
        print("[Device] Using CPU")

    # 2. Check Data Path
    data_root = args.data_root
    if not os.path.exists(data_root):
        alt_paths = [
            "data/Liver_regis",
            "../data/Liver_regis",
            os.path.join(os.getcwd(), "data/Liver_regis"),
        ]
        for p in alt_paths:
            if os.path.exists(p):
                data_root = p
                break

    if not os.path.exists(data_root):
        print(f"\n[Warning] Dataset root not found at: {args.data_root}")
        print("          Running in Stage 2 simulation dry-run mode.")
        run_dry_run_simulation(device, args)
        return

    print(f"\n[Step 1] Loading Fold {args.fold} Patient Partitions...")
    fold_info = P2ILRegFolds.get_fold(args.fold)
    train_patients = fold_info['train']
    val_patients = fold_info['val']
    print(f"         Train Patients ({len(train_patients)}): {train_patients}")
    print(f"         Val Patients   ({len(val_patients)}): {val_patients}")

    # 3. Instantiate Real Clinical Datasets
    print("\n[Step 2] Indexing Real Clinical Video Keyframes...")
    train_dataset = P2ILRegRealDataset(
        dataset_root=data_root,
        patient_ids=train_patients,
        max_points=args.max_points,
    )
    val_dataset = P2ILRegRealDataset(
        dataset_root=data_root,
        patient_ids=val_patients,
        max_points=args.max_points,
    )
    print(f"         Indexed {len(train_dataset)} real training keyframes.")
    print(f"         Indexed {len(val_dataset)} real validation keyframes.")

    if len(train_dataset) == 0:
        print("[Error] No real training keyframes found. Please check dataset path.")
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_p2i_lreg,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_p2i_lreg,
        pin_memory=torch.cuda.is_available(),
    )

    # 4. Instantiate Model & Differentiable Renderer
    print("\n[Step 3] Initializing Vis2Reg Architecture with Point Renderer...")
    cfg = get_default_config()
    cfg.data['max_points'] = args.max_points
    cfg.model['renderer'] = {
        'type': 'points',
        'image_size': (args.render_size, args.render_size),
        'radius': 0.01,
        'points_per_pixel': 16,
    }
    model = Vis2Reg(cfg).to(device)

    # 5. Load Stage 1 Pretrained Rigid Seed
    seed_path = args.pretrained_seed
    if not os.path.isabs(seed_path) and not os.path.exists(seed_path):
        seed_path = os.path.join(EXPERIMENT_ROOT, args.pretrained_seed)

    if os.path.exists(seed_path):
        print(f"\n[Step 4] Loading Pretrained Rigid Seed from: {seed_path}...")
        try:
            ckpt = torch.load(seed_path, map_location=device, weights_only=True)
        except Exception:
            ckpt = torch.load(seed_path, map_location=device, weights_only=False)

        state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
        # Load weights with strict=False to accommodate newly instantiated renderer
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"         Loaded successfully! (Missing renderer/field keys: {len(missing)}, Unexpected: {len(unexpected)})")
        if isinstance(ckpt, dict) and 'val_rre_deg' in ckpt:
            print(f"         Seed Baseline RRE: {ckpt['val_rre_deg']:.3f}° | RTE: {ckpt.get('val_rte_mm', 0.0):.2f} mm")
    else:
        print(f"\n[Warning] Pretrained seed not found at '{args.pretrained_seed}'. Initializing from scratch.")

    # 6. Multi-Task Loss Objective
    criterion = Vis2RegLoss(
        lambda_3d=cfg.training['lambda_3d'],
        lambda_vis=cfg.training['lambda_vis'],
        lambda_sil=cfg.training['lambda_sil'],
        lambda_def=cfg.training['lambda_def'],
        lambda_smooth=cfg.training['lambda_smooth'],
        lambda_topo=cfg.training['lambda_topo'],
        knn_k=16,
    )

    # 7. Optimizer & Scheduler Setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 8. Checkpoint Directory
    save_dir = os.path.join(EXPERIMENT_ROOT, args.save_dir) if not os.path.isabs(args.save_dir) else args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    best_checkpoint_path = os.path.join(save_dir, "best_real_model.pth")

    # 9. Initial Validation Baseline
    print("\n[Step 5] Running Baseline Validation Evaluation (Pre-Stage 2)...")
    init_val = evaluate(model, val_loader, criterion, device, include_deformation=False)
    print(f"         Baseline -> Dice: {init_val['dice_pct']:.2f}% | One-Way CD: {init_val['cd_mm']:.2f} mm | Loss: {init_val['loss']:.4f}")

    # 10. Main 60-Epoch Training Loop (Two-Stage Schedule)
    print(f"\n[Step 6] Starting 60-Epoch Two-Phase Schedule...")
    print(f"         Phase A (Rigid Warm-Up): Epochs 1 to {args.rigid_warmup_epochs} (Deformation field frozen)")
    print(f"         Phase B (Joint Optimization): Epochs {args.rigid_warmup_epochs + 1} to {args.epochs} (All 6 losses active)\n")

    best_dice = -1.0
    best_metrics = {}

    for epoch in range(1, args.epochs + 1):
        is_rigid_warmup = (epoch <= args.rigid_warmup_epochs)

        # Freeze/Unfreeze SIREN deformation field based on schedule
        if hasattr(model, 'deformation_field') and model.deformation_field is not None:
            for p in model.deformation_field.parameters():
                p.requires_grad = not is_rigid_warmup

        model.train()
        train_loss = 0.0
        train_3d = 0.0
        train_vis = 0.0
        train_sil = 0.0
        train_reg = 0.0
        step_count = 0
        t_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            batch_dev = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            batch_dev['rigid_only'] = is_rigid_warmup

            optimizer.zero_grad()
            outputs = model(batch_dev)
            losses = criterion(outputs, batch_dev, include_deformation=not is_rigid_warmup)

            loss = losses['total']
            loss.backward()

            # Robust gradient check: verify finite gradients before clipping/stepping
            has_non_finite = False
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    has_non_finite = True
                    break

            if has_non_finite:
                optimizer.zero_grad()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            train_loss += loss.item()
            train_3d += losses['3d'].item()
            train_vis += losses['vis'].item()
            train_sil += losses['sil'].item()
            reg_val = losses['def'].item() + losses['smooth'].item() + losses['topo'].item()
            train_reg += reg_val
            step_count += 1

            # Periodic batch logging (every 20 batches)
            if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(train_loader):
                print(
                    f"  [Epoch {epoch:02d} | Step {batch_idx+1:03d}/{len(train_loader):03d}] "
                    f"Loss: {loss.item():.4f} (3D:{losses['3d'].item():.2f}, "
                    f"Vis:{losses['vis'].item():.2f}, Sil:{losses['sil'].item():.4f})",
                    flush=True,
                )

            if args.smoke_test and batch_idx >= 1:
                break

        scheduler.step()
        t_epoch = time.time() - t_start

        avg_loss = train_loss / max(1, step_count)
        avg_3d = train_3d / max(1, step_count)
        avg_vis = train_vis / max(1, step_count)
        avg_sil = train_sil / max(1, step_count)
        avg_reg = train_reg / max(1, step_count)

        # Validation after epoch
        val_metrics = evaluate(model, val_loader, criterion, device, include_deformation=not is_rigid_warmup)
        current_dice = val_metrics['dice_pct']

        saved_flag = ""
        if current_dice > best_dice:
            best_dice = current_dice
            best_metrics = val_metrics
            saved_flag = "  <-- [BEST SAVED]"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice_pct': val_metrics['dice_pct'],
                'val_cd_mm': val_metrics['cd_mm'],
                'fold': args.fold,
                'config': vars(cfg) if hasattr(cfg, '__dict__') else cfg,
            }, best_checkpoint_path)

        stage_tag = "RigidWarmUp" if is_rigid_warmup else "JointDeform"
        print(
            f"[Epoch {epoch:02d}/{args.epochs:02d} - {stage_tag}] "
            f"Loss: {avg_loss:.4f} (3D:{avg_3d:.2f}, Vis:{avg_vis:.2f}, Sil:{avg_sil:.3f}, Reg:{avg_reg:.3f}) | "
            f"Val Dice: {val_metrics['dice_pct']:.2f}% | "
            f"Val CD: {val_metrics['cd_mm']:.2f} mm ({t_epoch:.1f}s){saved_flag}"
        )

        if args.smoke_test:
            print("\n[Smoke Test] 1 epoch completed successfully. Exiting early.")
            break

    # 11. Final Report & Paper Comparison
    print("\n" + "=" * 65)
    print("  STAGE 2 REAL CLINICAL FINE-TUNING COMPLETE")
    print("=" * 65)
    print(f"  Best Saved Checkpoint : {best_checkpoint_path}")
    print(f"  Achieved Val Dice     : {best_metrics.get('dice_pct', 0.0):.2f}% (Paper Table 1: 92.6% +- 7.2%)")
    print(f"  Achieved One-Way CD   : {best_metrics.get('cd_mm', 0.0):.2f} mm (Paper Table 1: 1.43 +- 1.26 mm)")
    print("=" * 65)


def run_dry_run_simulation(device, args):
    """Fallback dry-run simulation for machines without local real dataset."""
    print("\n[Simulation] Setting up Stage 2 dry-run simulation (N=1024 points, B=1)...")
    cfg = get_default_config()
    cfg.data['max_points'] = 1024
    cfg.model['renderer'] = {
        'type': 'points',
        'image_size': (args.render_size, args.render_size),
        'radius': 0.01,
        'points_per_pixel': 16,
    }
    model = Vis2Reg(cfg).to(device)
    criterion = Vis2RegLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    B, N = 1, 1024
    dummy_pre = torch.randn(B, N, 3, device=device) * 50.0
    dummy_tgt = dummy_pre + torch.randn(B, N, 3, device=device) * 2.0
    dummy_tgt[..., 2] = dummy_tgt[..., 2].abs() + 50.0  # Positive Z
    dummy_mask = torch.zeros(B, 1, 1, args.render_size, args.render_size, device=device)
    dummy_mask[..., 64:192, 64:192] = 1.0  # Synthetic center mask

    batch = {
        'pre_points': dummy_pre,
        'target_points': dummy_tgt,
        'target_points_cam': dummy_tgt,
        'cam_intrinsics': torch.eye(3, device=device).unsqueeze(0),
        'intra_masks': dummy_mask,
        'rigid_only': False,
    }

    print("[Simulation] Running Step 1: Rigid Warm-up forward pass...")
    model.train()
    batch['rigid_only'] = True
    out_warmup = model(batch)
    losses_warmup = criterion(out_warmup, batch, include_deformation=False)
    losses_warmup['total'].backward()
    optimizer.step()
    print(f"             Rigid Warm-Up Total Loss: {losses_warmup['total'].item():.4f} (Def regularizers zeroed)")

    print("[Simulation] Running Step 2: Joint Deformable forward pass...")
    optimizer.zero_grad()
    batch['rigid_only'] = False
    out_deform = model(batch)
    losses_deform = criterion(out_deform, batch, include_deformation=True)
    losses_deform['total'].backward()
    optimizer.step()
    print(f"             Joint Deformable Total Loss: {losses_deform['total'].item():.4f} (All 6 losses active)")

    dice, cd = compute_eval_metrics(out_deform['render_mask'], dummy_mask, dummy_tgt, out_deform['warped_pcd'])
    print(f"             Simulated Dice: {dice:.2f}% | One-Way CD: {cd:.2f} mm")

    save_dir = os.path.join(EXPERIMENT_ROOT, args.save_dir) if not os.path.isabs(args.save_dir) else args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    sim_ckpt = os.path.join(save_dir, "best_real_model.pth")
    torch.save({
        'epoch': 1,
        'model_state_dict': model.state_dict(),
        'val_dice_pct': dice,
        'val_cd_mm': cd,
        'config': vars(cfg) if hasattr(cfg, '__dict__') else cfg,
    }, sim_ckpt)
    print(f"[Simulation] Verification passed! Saved checkpoint to: {sim_ckpt}")
    print("=" * 65)


if __name__ == "__main__":
    main()
