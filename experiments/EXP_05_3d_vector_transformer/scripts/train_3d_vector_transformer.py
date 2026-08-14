import os
import sys
import time
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

def compute_mask_iou_dice(pred_masks: torch.Tensor, target_masks: torch.Tensor, threshold: float = 0.0, eps: float = 1e-6):
    """
    Computes 2D Mask Intersection over Union (IoU) and Dice Coefficient (0.0 to 1.0).
    pred_masks: (B, N, H, W) logits
    target_masks: (B, N, H, W) binary GT masks
    """
    with torch.no_grad():
        pred_bin = (pred_masks > threshold).float()
        gt_bin = (target_masks > 0.5).float()

        intersection = (pred_bin * gt_bin).sum(dim=(-2, -1)) # (B, N)
        union = pred_bin.sum(dim=(-2, -1)) + gt_bin.sum(dim=(-2, -1)) - intersection

        iou = (intersection + eps) / (union + eps)
        dice = (2.0 * intersection + eps) / (pred_bin.sum(dim=(-2, -1)) + gt_bin.sum(dim=(-2, -1)) + eps)

        # Average over active masks
        active = (gt_bin.sum(dim=(-2, -1)) > 0)
        if active.sum() > 0:
            mean_iou = iou[active].mean().item()
            mean_dice = dice[active].mean().item()
        else:
            mean_iou = iou.mean().item()
            mean_dice = dice.mean().item()

    return mean_iou, mean_dice

def plot_training_metrics(history: dict, save_dir: str):
    """
    Plots training and validation curves for Total Loss, 2D Mask IoU & Dice, 3D Position Loss, and Tangent Loss.
    Saves plot to save_dir/training_metrics_plot.png.
    """
    epochs = history["epoch"]
    if len(epochs) == 0:
        return

    plt.figure(figsize=(16, 12))
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # Subplot 1: Total Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, history["train_loss"], 'o-', label="Train Loss", color="#1f77b4", linewidth=2)
    plt.plot(epochs, history["val_loss"], 's--', label="Val Loss", color="#ff7f0e", linewidth=2)
    plt.title("Total Deep Supervision Loss", fontsize=14, fontweight='bold')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Subplot 2: 2D Mask IoU & Dice Coefficient (%)
    plt.subplot(2, 2, 2)
    plt.plot(epochs, [v * 100 for v in history["train_iou"]], 'o-', label="Train IoU (%)", color="#2ca02c", linewidth=2)
    plt.plot(epochs, [v * 100 for v in history["val_iou"]], 's--', label="Val IoU (%)", color="#8c564b", linewidth=2)
    plt.plot(epochs, [v * 100 for v in history["train_dice"]], '^:', label="Train Dice (%)", color="#9467bd", linewidth=2)
    plt.plot(epochs, [v * 100 for v in history["val_dice"]], 'd-.', label="Val Dice (%)", color="#d62728", linewidth=2)
    plt.title("2D Mask Segmentation Quality (IoU & Dice %)", fontsize=14, fontweight='bold')
    plt.xlabel("Epoch")
    plt.ylabel("Percentage (%)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Subplot 3: 3D Position Loss (Pos3D)
    plt.subplot(2, 2, 3)
    plt.plot(epochs, history["train_l_pos"], 'o-', label="Train 3D Pos Loss", color="#17becf", linewidth=2)
    plt.plot(epochs, history["val_l_pos"], 's--', label="Val 3D Pos Loss", color="#bcbd22", linewidth=2)
    plt.title("3D Polyline Positional Loss (Smooth L1)", fontsize=14, fontweight='bold')
    plt.xlabel("Epoch")
    plt.ylabel("Positional Loss")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Subplot 4: Tangent Edge Alignment Loss (Tan)
    plt.subplot(2, 2, 4)
    plt.plot(epochs, history["train_l_tan"], 'o-', label="Train Tangent Loss", color="#e377c2", linewidth=2)
    plt.plot(epochs, history["val_l_tan"], 's--', label="Val Tangent Loss", color="#7f7f7f", linewidth=2)
    plt.title("3D Cosine Tangent Orientation Loss", fontsize=14, fontweight='bold')
    plt.xlabel("Epoch")
    plt.ylabel("Tangent Loss")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "training_metrics_plot.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"📊 Training metrics plot saved to '{plot_path}'")

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

    # History dictionary for metric plotting
    history = {
        "epoch": [],
        "train_loss": [], "val_loss": [],
        "train_iou": [], "val_iou": [],
        "train_dice": [], "val_dice": [],
        "train_l_pos": [], "val_l_pos": [],
        "train_l_tan": [], "val_l_tan": []
    }

    # 3. Training & Validation Loop
    best_val_loss = float("inf")
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_iou = 0.0
        epoch_dice = 0.0
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

            # Compute batch IoU & Dice metrics
            b_iou, b_dice = compute_mask_iou_dice(outputs["pred_masks"], targets["target_masks"])
            epoch_iou += b_iou
            epoch_dice += b_dice

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "iou": f"{b_iou*100:.1f}%"})

        scheduler.step()
        elapsed = time.time() - start_time
        avg_train_loss = epoch_loss / max(len(loader), 1)
        avg_train_iou = epoch_iou / max(len(loader), 1)
        avg_train_dice = epoch_dice / max(len(loader), 1)

        for k in epoch_loss_dict:
            epoch_loss_dict[k] /= max(len(loader), 1)

        # 4. Validation Loop
        avg_val_loss = avg_train_loss
        avg_val_iou = avg_train_iou
        avg_val_dice = avg_train_dice
        val_loss_dict = dict(epoch_loss_dict)

        if val_loader is not None and len(val_loader) > 0:
            model.eval()
            val_epoch_loss = 0.0
            val_epoch_iou = 0.0
            val_epoch_dice = 0.0
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

                    v_iou, v_dice = compute_mask_iou_dice(val_outputs["pred_masks"], targets["target_masks"])
                    val_epoch_iou += v_iou
                    val_epoch_dice += v_dice

            avg_val_loss = val_epoch_loss / max(len(val_loader), 1)
            avg_val_iou = val_epoch_iou / max(len(val_loader), 1)
            avg_val_dice = val_epoch_dice / max(len(val_loader), 1)
            for k in val_epoch_loss_dict:
                val_loss_dict[k] = val_epoch_loss_dict[k] / max(len(val_loader), 1)

        print(f"Epoch [{epoch:2d}/{args.epochs:2d}] ({elapsed:.1f}s) | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"Val IoU: {avg_val_iou*100:.1f}% | Val Dice: {avg_val_dice*100:.1f}% | "
              f"Val Pos3D: {val_loss_dict['l_pos']:.3f} | Val Tan: {val_loss_dict['l_tan']:.3f}", flush=True)

        # Record metrics history
        history["epoch"].append(epoch)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_iou"].append(avg_train_iou)
        history["val_iou"].append(avg_val_iou)
        history["train_dice"].append(avg_train_dice)
        history["val_dice"].append(avg_val_dice)
        history["train_l_pos"].append(epoch_loss_dict['l_pos'])
        history["val_l_pos"].append(val_loss_dict['l_pos'])
        history["train_l_tan"].append(epoch_loss_dict['l_tan'])
        history["val_l_tan"].append(val_loss_dict['l_tan'])

        if use_wandb:
            try:
                import wandb
                wandb.log({
                    "epoch": epoch,
                    "train/total_loss": avg_train_loss,
                    "train/mask_iou_pct": avg_train_iou * 100.0,
                    "train/mask_dice_pct": avg_train_dice * 100.0,
                    "train/l_cls": epoch_loss_dict['l_cls'],
                    "train/l_pos": epoch_loss_dict['l_pos'],
                    "train/l_tan": epoch_loss_dict['l_tan'],
                    "train/l_curv": epoch_loss_dict['l_curv'],
                    "train/l_mask": epoch_loss_dict['l_mask'],
                    "val/total_loss": avg_val_loss,
                    "val/mask_iou_pct": avg_val_iou * 100.0,
                    "val/mask_dice_pct": avg_val_dice * 100.0,
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
                "history": history,
                "config": config
            }, ckpt_path)
            print(f"  💾 Best validation checkpoint saved to '{ckpt_path}'")

        # Save metrics plot image after each epoch
        plot_training_metrics(history, args.save_dir)

    print("=" * 70)
    print(f"✅ EXP_05 Training Finished cleanly! Best Val Loss = {best_val_loss:.4f}")
    print(f"📊 Training curves plot saved to: '{os.path.join(args.save_dir, 'training_metrics_plot.png')}'")
    print("=" * 70)

if __name__ == "__main__":
    main()
