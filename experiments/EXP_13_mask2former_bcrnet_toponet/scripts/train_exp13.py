#!/usr/bin/env python3
"""
EXP_13: Training Script — Mask2Former-BCRNet with TopoNet Loss
=============================================================
Unified training pipeline combining:
- Mask2Former 4-channel RGB-D Pixel Decoder & Masked Attention Engine
- TopoNet Multi-Class Centerline Constraint Loss (soft clDice)
- BCRNet 5th-Order Parametric Bézier Curve Refinement
- AMP mixed-precision & CosineAnnealingLR
- WandB live dashboard logging
"""
import os
import sys
import argparse
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure EXP_13 modules can be imported
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP13_DIR = os.path.dirname(SCRIPT_DIR)
WORKSPACE_DIR = os.path.dirname(os.path.dirname(EXP13_DIR))

for p in [EXP13_DIR, WORKSPACE_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from configs.exp13_config import EXP13Config, resolve_dataset_dir, resolve_checkpoint_dir
from utils.dataset import Mask2FormerBCRNetDataset
from models.mask2former_bcrnet import Mask2FormerBCRNet
from models.joint_losses import JointUnifiedLoss

# Try WandB
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def compute_eval_metrics(pred_sem: np.ndarray, gt_sem: np.ndarray, num_classes: int = 3):
    """
    Computes per-class Dice on 2D semantic maps (1: Ridge, 2: Silhouette, 3: Ligament).
    """
    dices = {}
    smooth = 1e-5
    for c in range(1, num_classes + 1):
        p_c = (pred_sem == c).astype(np.float32)
        g_c = (gt_sem == c).astype(np.float32)
        intersection = np.sum(p_c * g_c)
        dice = (2.0 * intersection + smooth) / (np.sum(p_c) + np.sum(g_c) + smooth)
        dices[c] = float(dice)
    return dices


def main():
    parser = argparse.ArgumentParser(description="Train EXP_13 Mask2Former-BCRNet")
    parser.add_argument("--dataset_dir", type=str, default=None, help="Dataset directory")
    parser.add_argument("--save_dir", type=str, default=None, help="Checkpoint save directory")
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=6e-5, help="Learning rate")
    parser.add_argument("--no_amp", action="store_true", help="Disable AMP mixed precision")
    parser.add_argument("--no_wandb", action="store_true", help="Disable WandB logging")
    args = parser.parse_args()

    cfg = EXP13Config()
    if args.dataset_dir:
        cfg.dataset_dir = resolve_dataset_dir(args.dataset_dir)
    if args.save_dir:
        cfg.save_dir = resolve_checkpoint_dir(args.save_dir)
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    cfg.use_amp = not args.no_amp

    os.makedirs(cfg.save_dir, exist_ok=True)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("🚀 [EXP_13] Master Synthesis: Mask2Former + TopoNet Loss + BCRNet")
    print(f"   Device:         {device}")
    print(f"   Dataset:        {cfg.dataset_dir}")
    print(f"   Epochs:         {cfg.epochs}  |  Batch: {cfg.batch_size}  |  LR: {cfg.lr}")
    print(f"   AMP:            {cfg.use_amp}  |  Save Dir: {cfg.save_dir}")
    print("=" * 80)

    # 1. Initialize WandB
    use_wandb = WANDB_AVAILABLE and (not args.no_wandb)
    if use_wandb:
        try:
            if cfg.wandb_key:
                wandb.login(key=cfg.wandb_key)
            wandb.init(
                project=cfg.wandb_project,
                name="EXP_13_Mask2Former_BCRNet_TopoNet",
                id=cfg.wandb_id,
                resume="allow",
                config=vars(cfg),
            )
            print(" WandB initialized successfully.")
        except Exception as e:
            print(f"⚠️  WandB init failed ({e}). Proceeding offline.")
            use_wandb = False

    # 2. Datasets & DataLoaders
    train_dataset = Mask2FormerBCRNetDataset(
        dataset_dir=cfg.dataset_dir,
        mode="train",
        image_size=cfg.image_size,
        use_depth=cfg.use_depth,
        bezier_degree=cfg.bezier_degree,
        num_sample_pts=cfg.num_sample_pts,
    )
    val_dataset = Mask2FormerBCRNetDataset(
        dataset_dir=cfg.dataset_dir,
        mode="val",
        image_size=cfg.image_size,
        use_depth=cfg.use_depth,
        bezier_degree=cfg.bezier_degree,
        num_sample_pts=cfg.num_sample_pts,
    )

    num_workers = min(cfg.num_workers, os.cpu_count() or 1)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # 3. Model & Loss Function
    model = Mask2FormerBCRNet(cfg).to(device)
    loss_fn = JointUnifiedLoss(
        num_classes=cfg.num_classes,
        lambda_m2f_ce=cfg.lambda_m2f_ce,
        lambda_m2f_mask=cfg.lambda_m2f_mask,
        lambda_toponet=cfg.lambda_toponet,
        lambda_crv=cfg.lambda_crv,
        lambda_tangent=cfg.lambda_tangent,
        lambda_contain=cfg.lambda_contain,
        skel_num_iter=cfg.skel_num_iter,
    ).to(device)

    # 4. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.min_lr
    )
    scaler = torch.amp.GradScaler('cuda', enabled=cfg.use_amp and device.type == "cuda")

    best_val_dice = 0.0
    best_ckpt_path = os.path.join(cfg.save_dir, "best_model.pth")
    latest_ckpt_path = os.path.join(cfg.save_dir, "latest_model.pth")

    # 5. Training Loop
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss_accum = 0.0
        loss_components_accum = {
            "loss_ce": 0.0,
            "loss_toponet": 0.0,
            "loss_crv": 0.0,
            "loss_tangent": 0.0,
            "loss_contain": 0.0,
            "loss_score": 0.0,
        }

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{cfg.epochs:02d} [Train]")
        for batch in pbar:
            optimizer.zero_grad()
            images = batch["image"].to(device)

            with torch.amp.autocast('cuda', enabled=cfg.use_amp and device.type == "cuda"):
                outputs = model(images, target_size=(cfg.image_size, cfg.image_size))
                loss_dict = loss_fn(outputs, batch)
                loss = loss_dict["loss_total"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            total_loss_accum += loss.item()
            for k in loss_components_accum:
                loss_components_accum[k] += loss_dict[k].item()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "topo": f"{loss_dict['loss_toponet'].item():.3f}",
                "crv": f"{loss_dict['loss_crv'].item():.3f}",
            })

        num_batches = len(train_loader)
        avg_train_loss = total_loss_accum / num_batches
        avg_components = {k: v / num_batches for k, v in loss_components_accum.items()}
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        # 6. Validation Loop
        model.eval()
        val_dices = {1: [], 2: [], 3: []}
        with torch.no_grad():
            for val_batch in tqdm(val_loader, desc=f"Epoch {epoch:02d}/{cfg.epochs:02d} [Val]"):
                v_images = val_batch["image"].to(device)
                v_gt_sem = val_batch["gt_semantic"].cpu().numpy()[0]  # (H, W)

                pred_dict = model.predict_best_curves(v_images, target_size=(cfg.image_size, cfg.image_size))
                pred_sem = pred_dict["semantic_pred"].cpu().numpy()[0]  # (H, W)

                step_dices = compute_eval_metrics(pred_sem, v_gt_sem, cfg.num_classes)
                for c in val_dices:
                    val_dices[c].append(step_dices[c])

        mean_ridge_dice = float(np.mean(val_dices[1]))
        mean_silh_dice  = float(np.mean(val_dices[2]))
        mean_lig_dice   = float(np.mean(val_dices[3]))
        macro_mean_dice = (mean_ridge_dice + mean_silh_dice + mean_lig_dice) / 3.0

        print(
            f"📊 [Epoch {epoch:02d}] Train Loss: {avg_train_loss:.4f}  |  "
            f"Val Dice -> Ridge: {mean_ridge_dice*100:.2f}%  |  "
            f"Silh: {mean_silh_dice*100:.2f}%  |  "
            f"Lig: {mean_lig_dice*100:.2f}%  |  "
            f"Macro Mean: {macro_mean_dice*100:.2f}%"
        )

        # 7. Logging & Checkpointing
        log_payload = {
            "epoch": epoch,
            "train/loss_total": avg_train_loss,
            "train/loss_ce": avg_components["loss_ce"],
            "train/loss_toponet": avg_components["loss_toponet"],
            "train/loss_crv": avg_components["loss_crv"],
            "train/loss_tangent": avg_components["loss_tangent"],
            "train/loss_contain": avg_components["loss_contain"],
            "val/dice_ridge": mean_ridge_dice,
            "val/dice_silhouette": mean_silh_dice,
            "val/dice_ligament": mean_lig_dice,
            "val/dice_macro_mean": macro_mean_dice,
            "lr": current_lr,
        }
        if use_wandb:
            wandb.log(log_payload)

        # Save latest
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "cfg": vars(cfg),
            "macro_mean_dice": macro_mean_dice,
        }, latest_ckpt_path)

        # Save best
        if macro_mean_dice > best_val_dice:
            best_val_dice = macro_mean_dice
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "cfg": vars(cfg),
                "macro_mean_dice": macro_mean_dice,
            }, best_ckpt_path)
            print(f"  🏆 New Best Model! Val Dice: {best_val_dice*100:.2f}% -> '{best_ckpt_path}'")

    if use_wandb:
        wandb.finish()
    print(f"\n✅ Training Complete! Best Validation Dice: {best_val_dice*100:.2f}%")


if __name__ == "__main__":
    main()
