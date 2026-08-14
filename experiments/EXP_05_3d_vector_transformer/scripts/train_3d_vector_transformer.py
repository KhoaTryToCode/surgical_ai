import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add experiment root to path
EXP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if EXP_DIR not in sys.path:
    sys.path.append(EXP_DIR)

from configs.exp05_config import config
from utils.dataset_3d import Surgical3DVectorDataset
from models.surgical_3d_vector_transformer import Surgical3DVectorTransformer

def parse_args():
    parser = argparse.ArgumentParser(description="Train EXP_05 3D Vector Space Transformer")
    parser.add_argument("--dataset_dir", type=str, default=config.dataset_dir, help="Path to surgical dataset")
    parser.add_argument("--epochs", type=int, default=config.num_epochs, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=config.batch_size, help="Batch size")
    parser.add_argument("--lr", type=float, default=config.learning_rate, help="Learning rate")
    parser.add_argument("--save_dir", type=str, default="checkpoints/EXP_05", help="Checkpoint save directory")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases experiment tracking")
    parser.add_argument("--wandb_project", type=str, default="Surgical_AI_3D_Vector", help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default="EXP_05_Swin_3D_Vector_Transformer", help="W&B run name")
    return parser.parse_args()

def compute_mask_metrics(pred_masks: torch.Tensor, target_masks: torch.Tensor, eps: float = 1e-5):
    """
    Computes BOTH Hard (thresholded > 0.5) and Soft (continuous sigmoid) 2D Mask IoU & Dice metrics.
    pred_masks: (B, N, H, W) raw logits
    target_masks: (B, N, H, W) binary GT masks
    """
    with torch.no_grad():
        probs = torch.sigmoid(pred_masks.float())
        hard_pred = (probs > 0.5).float()
        gt_bin = (target_masks.float() > 0.5).float()

        active = (gt_bin.sum(dim=(-2, -1)) > 0)

        # 1. Hard Metrics (Standard Paper Benchmark)
        hard_inter = (hard_pred * gt_bin).sum(dim=(-2, -1))
        hard_union = hard_pred.sum(dim=(-2, -1)) + gt_bin.sum(dim=(-2, -1)) - hard_inter
        hard_iou = (hard_inter + eps) / (hard_union + eps)
        hard_dice = (2.0 * hard_inter + eps) / (hard_pred.sum(dim=(-2, -1)) + gt_bin.sum(dim=(-2, -1)) + eps)

        # 2. Soft Metrics (Continuous Optimization Tracking)
        soft_inter = (probs * gt_bin).sum(dim=(-2, -1))
        soft_union = probs.sum(dim=(-2, -1)) + gt_bin.sum(dim=(-2, -1)) - soft_inter
        soft_iou = (soft_inter + eps) / (soft_union + eps)
        soft_dice = (2.0 * soft_inter + eps) / (probs.sum(dim=(-2, -1)) + gt_bin.sum(dim=(-2, -1)) + eps)

        if active.sum() > 0:
            m_hard_iou = hard_iou[active].mean().item()
            m_hard_dice = hard_dice[active].mean().item()
            m_soft_iou = soft_iou[active].mean().item()
            m_soft_dice = soft_dice[active].mean().item()
        else:
            m_hard_iou = hard_iou.mean().item()
            m_hard_dice = hard_dice.mean().item()
            m_soft_iou = soft_iou.mean().item()
            m_soft_dice = soft_dice.mean().item()

    return {
        "hard_iou": m_hard_iou, "hard_dice": m_hard_dice,
        "soft_iou": m_soft_iou, "soft_dice": m_soft_dice
    }

def main():
    args = parse_args()
    print("=" * 70)
    print("🚀 Starting Training: EXP_05 Monocular 3D Vector Space Transformer")
    print("=" * 70)
    print(f"📁 Dataset Directory: {args.dataset_dir}")
    print(f"⚙️ Hyperparameters: Batch Size={args.batch_size}, LR={args.lr}, Epochs={args.epochs}")

    # Initialize Weights & Biases if requested
    use_wandb = args.wandb or ("WANDB_API_KEY" in os.environ)
    if use_wandb:
        try:
            import wandb
            api_key = os.environ.get("WANDB_API_KEY", "83f4544a22543e319c6009abceaac90b634c68a3")
            if api_key:
                wandb.login(key=api_key)
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=vars(args)
            )
            print(f"📊 Weights & Biases initialized: Project='{args.wandb_project}', Run='{args.wandb_run_name}'")
        except Exception as e:
            print(f"⚠️ Could not initialize wandb: {e}")
            use_wandb = False

    # Device selection (CUDA -> MPS -> CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"💻 Execution Device: CUDA ({torch.cuda.get_device_name(0)})")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("💻 Execution Device: Apple Silicon MPS")
    else:
        device = torch.device("cpu")
        print("💻 Execution Device: CPU")

    os.makedirs(args.save_dir, exist_ok=True)

    # 1. Dataset & DataLoader (Train & Val)
    if not os.path.exists(args.dataset_dir):
        print(f"⚠️ Warning: Dataset path '{args.dataset_dir}' does not exist yet.")
        print("💡 Make sure to run dataset preparation script first: python shared/utils/prepare_dataset.py --target_dir /kaggle/working/L3D")
        return

    train_dataset = Surgical3DVectorDataset(
        dataset_dir=args.dataset_dir,
        num_instances=config.num_instances,
        num_points=config.num_points,
        mode="train"
    )
    val_dataset = Surgical3DVectorDataset(
        dataset_dir=args.dataset_dir,
        num_instances=config.num_instances,
        num_points=config.num_points,
        mode="val"
    )
    print(f"📊 Training Dataset Size: {len(train_dataset)} images | Validation Dataset Size: {len(val_dataset)} images")

    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=True if len(train_dataset) > args.batch_size else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=False
    ) if len(val_dataset) > 0 else None

    # 2. Instantiate Model & Optimizer
    model = Surgical3DVectorTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    use_cuda_amp = (device.type == "cuda")
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_cuda_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_cuda_amp)

    # 3. Training & Validation Loop
    best_val_loss = float("inf")
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        train_metrics = {"hard_iou": 0.0, "hard_dice": 0.0, "soft_iou": 0.0, "soft_dice": 0.0}
        epoch_loss_dict = {"l_cls": 0.0, "l_pos": 0.0, "l_tan": 0.0, "l_curv": 0.0, "l_mask": 0.0}
        start_time = time.time()

        pbar = tqdm(loader, desc=f"Epoch {epoch:2d}/{args.epochs:2d} [Train]", leave=False)
        for batch_idx, batch in enumerate(pbar):
            images = batch["image"].to(device)
            depth = batch["depth"].to(device)
            targets = {
                "target_classes": batch["target_classes"].to(device),
                "target_polylines": batch["target_polylines"].to(device),
                "target_masks": batch["target_masks"].to(device),
                "valid_mask": batch["valid_mask"].to(device)
            }

            optimizer.zero_grad()
            if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                autocast_ctx = torch.amp.autocast("cuda", enabled=use_cuda_amp)
            else:
                autocast_ctx = torch.cuda.amp.autocast(enabled=use_cuda_amp)

            with autocast_ctx:
                outputs = model(images, depth, targets=targets)
                loss = outputs["loss"]
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            for k, v in outputs["loss_dict"].items():
                epoch_loss_dict[k] += v

            # Compute batch Hard & Soft metrics
            b_m = compute_mask_metrics(outputs["pred_masks"], targets["target_masks"])
            for mk in train_metrics:
                train_metrics[mk] += b_m[mk]

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "iou": f"{b_m['hard_iou']*100:.1f}%"})

        scheduler.step()
        elapsed = time.time() - start_time
        avg_train_loss = epoch_loss / max(len(loader), 1)
        for mk in train_metrics:
            train_metrics[mk] /= max(len(loader), 1)

        for k in epoch_loss_dict:
            epoch_loss_dict[k] /= max(len(loader), 1)

        # 4. Validation Loop
        avg_val_loss = avg_train_loss
        val_metrics = dict(train_metrics)
        val_loss_dict = dict(epoch_loss_dict)

        if val_loader is not None and len(val_loader) > 0:
            model.eval()
            val_epoch_loss = 0.0
            val_metrics = {"hard_iou": 0.0, "hard_dice": 0.0, "soft_iou": 0.0, "soft_dice": 0.0}
            val_epoch_loss_dict = {"l_cls": 0.0, "l_pos": 0.0, "l_tan": 0.0, "l_curv": 0.0, "l_mask": 0.0}
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch:2d}/{args.epochs:2d} [Val]", leave=False)
                for batch in val_pbar:
                    images = batch["image"].to(device)
                    depth = batch["depth"].to(device)
                    targets = {
                        "target_classes": batch["target_classes"].to(device),
                        "target_polylines": batch["target_polylines"].to(device),
                        "target_masks": batch["target_masks"].to(device),
                        "valid_mask": batch["valid_mask"].to(device)
                    }
                    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                        autocast_ctx = torch.amp.autocast("cuda", enabled=use_cuda_amp)
                    else:
                        autocast_ctx = torch.cuda.amp.autocast(enabled=use_cuda_amp)

                    with autocast_ctx:
                        val_outputs = model(images, depth, targets=targets)
                        v_loss = val_outputs["loss"]

                    val_epoch_loss += v_loss.item()
                    for k, v in val_outputs["loss_dict"].items():
                        val_epoch_loss_dict[k] += v

                    v_m = compute_mask_metrics(val_outputs["pred_masks"], targets["target_masks"])
                    for mk in val_metrics:
                        val_metrics[mk] += v_m[mk]

            avg_val_loss = val_epoch_loss / max(len(val_loader), 1)
            for mk in val_metrics:
                val_metrics[mk] /= max(len(val_loader), 1)

            for k in val_epoch_loss_dict:
                val_loss_dict[k] = val_epoch_loss_dict[k] / max(len(val_loader), 1)

        print(f"Epoch [{epoch:2d}/{args.epochs:2d}] ({elapsed:.1f}s) | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"Val Hard IoU: {val_metrics['hard_iou']*100:.1f}% | Val Soft IoU: {val_metrics['soft_iou']*100:.1f}% | "
              f"Val Hard Dice: {val_metrics['hard_dice']*100:.1f}% | Val Soft Dice: {val_metrics['soft_dice']*100:.1f}%", flush=True)

        if use_wandb:
            try:
                import wandb
                wandb.log({
                    "epoch": epoch,
                    "train/total_loss": avg_train_loss,
                    "train/hard_iou_pct": train_metrics["hard_iou"] * 100.0,
                    "train/hard_dice_pct": train_metrics["hard_dice"] * 100.0,
                    "train/soft_iou_pct": train_metrics["soft_iou"] * 100.0,
                    "train/soft_dice_pct": train_metrics["soft_dice"] * 100.0,
                    "train/l_cls": epoch_loss_dict['l_cls'],
                    "train/l_pos": epoch_loss_dict['l_pos'],
                    "train/l_tan": epoch_loss_dict['l_tan'],
                    "train/l_curv": epoch_loss_dict['l_curv'],
                    "train/l_mask": epoch_loss_dict['l_mask'],
                    "val/total_loss": avg_val_loss,
                    "val/hard_iou_pct": val_metrics["hard_iou"] * 100.0,
                    "val/hard_dice_pct": val_metrics["hard_dice"] * 100.0,
                    "val/soft_iou_pct": val_metrics["soft_iou"] * 100.0,
                    "val/soft_dice_pct": val_metrics["soft_dice"] * 100.0,
                    "val/l_cls": val_loss_dict['l_cls'],
                    "val/l_pos": val_loss_dict['l_pos'],
                    "val/l_tan": val_loss_dict['l_tan'],
                    "val/l_curv": val_loss_dict['l_curv'],
                    "val/l_mask": val_loss_dict['l_mask'],
                    "learning_rate": optimizer.param_groups[0]['lr']
                })
            except Exception:
                pass

        # Save Best Checkpoint on Validation Loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_path = os.path.join(args.save_dir, "best_surgical_3d_vector.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "config": config
            }, ckpt_path)
            print(f"  💾 Best validation checkpoint saved to '{ckpt_path}'")

    print("=" * 70)
    print(f"✅ EXP_05 Training Finished cleanly! Best Val Loss = {best_val_loss:.4f}")
    print("=" * 70)

if __name__ == "__main__":
    main()
