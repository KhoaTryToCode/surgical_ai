import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import sys
import argparse
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Ensure experiment root is in sys.path
exp_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if exp_root not in sys.path:
    sys.path.insert(0, exp_root)

from configs.exp09_config import config
from models.patch_vector_vit import PatchBezierViT
from models.patch_losses import PatchBezierLoss
from models.patch_merger import merge_patch_beziers_to_image
from utils.dataset_patch_vit import PatchBezierLandmarkDataset


def parse_args():
    parser = argparse.ArgumentParser(description="EXP_09: Train Patch-Bézier ViT for Surgical Landmark Detection")
    parser.add_argument("--dataset_dir", type=str, default=config.dataset_dir, help="Root path to surgical dataset")
    parser.add_argument("--epochs", type=int, default=config.num_epochs, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=config.batch_size, help="Batch size")
    parser.add_argument("--lr", type=float, default=config.learning_rate, help="Head learning rate")
    parser.add_argument("--backbone_lr_mult", type=float, default=config.backbone_lr_mult, help="Backbone lr multiplier")
    parser.add_argument("--save_dir", type=str, default="checkpoints/EXP_09", help="Directory to save model checkpoints")
    parser.add_argument("--pretrained", action="store_true", default=config.pretrained, help="Use ImageNet pretrained ViT")
    parser.add_argument("--use_depth", action="store_true", default=config.use_depth, help="Ingest Depth Anything V2 as 4th channel")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_key", type=str, default=config.wandb_key, help="W&B API key")
    parser.add_argument("--device", type=str, default="", help="Force specific device (cuda, mps, cpu)")
    return parser.parse_args()


def main():
    args = parse_args()
    in_chans = 4 if args.use_depth else 3
    print("=" * 75)
    print(f"🔥 [EXP_09] Training Patch-Level Bézier Vision Transformer (RGB-D: {args.use_depth}, Channels: {in_chans})")
    print(f"📂 Dataset:    {args.dataset_dir}")
    print(f"⚙️ Epochs:     {args.epochs} | Batch Size: {args.batch_size} | Base LR: {args.lr}")
    print(f"📐 Resolution: {config.image_size}×{config.image_size} | Patch: {config.patch_size}×{config.patch_size} (Grid: {config.grid_size}×{config.grid_size})")
    print("=" * 75)

    # Device selection
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"🚀 Execution Device: {device}")

    # Initialize WandB if requested
    use_wandb = args.wandb
    if use_wandb:
        try:
            import wandb
            if args.wandb_key:
                wandb.login(key=args.wandb_key)
            wandb.init(
                project=config.wandb_project,
                name=f"{config.wandb_run_name}_{time.strftime('%Y%m%d_%H%M%S')}",
                config=vars(args)
            )
            print("📈 WandB logging initialized.")
        except Exception as e:
            print(f"⚠️ Failed to initialize WandB ({e}). Continuing with local logging.")
            use_wandb = False

    # Datasets and Loaders
    train_dataset = PatchBezierLandmarkDataset(
        dataset_dir=args.dataset_dir,
        mode="train",
        image_size=config.image_size,
        patch_size=config.patch_size,
        spline_step_px=config.spline_step_px,
        stroke_thickness=config.stroke_thickness,
        use_depth=args.use_depth
    )
    val_dataset = PatchBezierLandmarkDataset(
        dataset_dir=args.dataset_dir,
        mode="val",
        image_size=config.image_size,
        patch_size=config.patch_size,
        spline_step_px=config.spline_step_px,
        stroke_thickness=config.stroke_thickness,
        use_depth=args.use_depth
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=config.num_workers if device.type != "mps" else 0,
        pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config.num_workers if device.type != "mps" else 0
    )

    # Model
    model = PatchBezierViT(
        backbone_name=config.backbone_name,
        in_chans=in_chans,
        pretrained=args.pretrained,
        image_size=config.image_size,
        patch_size=config.patch_size,
        num_classes=config.num_classes,
        embed_dim=config.embed_dim,
        dropout=config.dropout
    ).to(device)

    # Criterion
    criterion = PatchBezierLoss(
        lambda_cls=config.lambda_cls,
        lambda_ctrl=config.lambda_ctrl,
        lambda_sample=config.lambda_sample,
        lambda_tan=config.lambda_tan,
        lambda_cont=config.lambda_cont,
        focal_alpha=config.focal_alpha,
        focal_gamma=config.focal_gamma,
        num_samples=config.num_sampled_points
    )

    # Optimizer with differential learning rate
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = AdamW([
        {"params": backbone_params, "lr": args.lr * args.backbone_lr_mult},
        {"params": head_params, "lr": args.lr}
    ], weight_decay=config.weight_decay)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=config.min_lr)

    # Checkpoint directory
    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float("inf")

    # Training Loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_accum = 0.0
        train_cls_accum = 0.0
        train_ctrl_accum = 0.0
        train_sample_accum = 0.0
        train_tan_accum = 0.0
        train_cont_accum = 0.0
        num_batches = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            target_classes = batch["target_classes"].to(device)
            target_beziers = batch["target_beziers"].to(device)
            active_mask = batch["active_mask"].to(device)

            optimizer.zero_grad()
            pred_dict = model(images)
            loss_dict = criterion(pred_dict, target_classes, target_beziers, active_mask)

            loss = loss_dict["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss_accum += loss.item()
            train_cls_accum += loss_dict["loss_cls"].item()
            train_ctrl_accum += loss_dict["loss_ctrl"].item()
            train_sample_accum += loss_dict["loss_sample"].item()
            train_tan_accum += loss_dict["loss_tan"].item()
            train_cont_accum += loss_dict["loss_cont"].item()
            num_batches += 1

        scheduler.step()

        # Epoch Metrics
        avg_train_loss = train_loss_accum / max(num_batches, 1)
        avg_cls_loss = train_cls_accum / max(num_batches, 1)
        avg_ctrl_loss = train_ctrl_accum / max(num_batches, 1)
        avg_sample_loss = train_sample_accum / max(num_batches, 1)

        # Validation Step
        model.eval()
        val_loss_accum = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                target_classes = batch["target_classes"].to(device)
                target_beziers = batch["target_beziers"].to(device)
                active_mask = batch["active_mask"].to(device)

                pred_dict = model(images)
                loss_dict = criterion(pred_dict, target_classes, target_beziers, active_mask)
                val_loss_accum += loss_dict["loss"].item()
                val_batches += 1

        avg_val_loss = val_loss_accum / max(val_batches, 1)

        print(
            f"Epoch [{epoch:03d}/{args.epochs:03d}] | "
            f"Train Loss: {avg_train_loss:.4f} (Cls: {avg_cls_loss:.4f}, Ctrl: {avg_ctrl_loss:.4f}, Sample: {avg_sample_loss:.4f}) | "
            f"Val Loss: {avg_val_loss:.4f} | LR: {optimizer.param_groups[1]['lr']:.2e}"
        )

        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train/loss": avg_train_loss,
                "train/loss_cls": avg_cls_loss,
                "train/loss_ctrl": avg_ctrl_loss,
                "train/loss_sample": avg_sample_loss,
                "val/loss": avg_val_loss,
                "lr": optimizer.param_groups[1]['lr']
            })

        # Save Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = os.path.join(args.save_dir, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "config": vars(config)
            }, best_path)
            print(f"   🌟 New best model saved to: {best_path} (Val Loss: {best_val_loss:.4f})")

        # Periodic checkpoint
        if epoch % 20 == 0:
            ckpt_path = os.path.join(args.save_dir, f"epoch_{epoch:03d}.pth")
            torch.save(model.state_dict(), ckpt_path)

    print("=" * 75)
    print(f"🏁 Training complete! Best Val Loss: {best_val_loss:.4f}")
    print("=" * 75)


if __name__ == "__main__":
    main()
