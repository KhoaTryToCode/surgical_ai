import os
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="torch.amp.*")
warnings.filterwarnings("ignore", module="torch.cuda.amp.*")
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import time
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

exp_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if exp_root not in sys.path:
    sys.path.insert(0, exp_root)

from configs.exp10_config import EXP10Config, resolve_dataset_dir
from models.macro_patch_vit import MacroPatchViT
from models.macro_losses import MacroPatchLoss
from models.macro_merger import merge_macro_beziers_to_image, compute_batch_dice
from utils.dataset_macro_vit import MacroPatchLandmarkDataset


def parse_args():
    default_cfg = EXP10Config()
    parser = argparse.ArgumentParser(description="EXP_10: Train Macro-Patch Geometric ViT (Way A)")
    parser.add_argument("--dataset_dir", type=str, default=resolve_dataset_dir(), help="Root path to surgical dataset")
    parser.add_argument("--epochs", type=int, default=default_cfg.num_epochs, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=default_cfg.batch_size, help="Batch size")
    parser.add_argument("--lr", type=float, default=default_cfg.learning_rate, help="Head learning rate")
    parser.add_argument("--backbone_lr_mult", type=float, default=default_cfg.backbone_lr_mult, help="Backbone lr multiplier")
    parser.add_argument("--save_dir", type=str, default="checkpoints/EXP_10", help="Directory to save checkpoints")
    parser.add_argument("--backbone", type=str, default=default_cfg.backbone_name, help="ViT backbone name")
    parser.add_argument("--macro_patch_size", type=int, default=default_cfg.macro_patch_size, help="Macro patch size (64 px)")
    parser.add_argument("--amp", action="store_true", default=default_cfg.use_amp, help="Enable AMP")
    parser.add_argument("--use_depth", action="store_true", default=default_cfg.use_depth, help="Ingest Depth Anything V2")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases")
    parser.add_argument("--wandb_key", type=str, default=default_cfg.wandb_key, help="W&B API key")
    parser.add_argument("--device", type=str, default="", help="Force device (cuda, mps, cpu)")
    return parser.parse_args()


def main():
    args = parse_args()
    in_chans = 4 if args.use_depth else 3
    print("=" * 75)
    print("🚀 [EXP_10] Training Macro-Patch Geometric Vision Transformer (Way A)")
    print(f"🏛️ Backbone:          {args.backbone} (RGB-D in_chans={in_chans})")
    print(f"📐 Macro-Patch Grid:   {args.macro_patch_size}×{args.macro_patch_size} px (Grid: 8×8 = 64 Macro-Tokens)")
    print(f"📂 Dataset Path:       {args.dataset_dir}")
    print(f"⚙️ Epochs:            {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}")
    print(f"⚡ Mixed Precision:    {args.amp}")
    print("=" * 75)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Execution Device: {device}")

    # WandB setup
    use_wandb = args.wandb
    if use_wandb and args.wandb_key:
        try:
            import wandb
            wandb.login(key=args.wandb_key)
            wandb.init(
                project="Surgical_AI_EXP10_Macro",
                name=f"EXP10_Macro_{args.backbone}_{args.macro_patch_size}px",
                config=vars(args)
            )
            print("✅ Weights & Biases initialized.")
        except Exception as e:
            print(f"⚠️ Could not initialize WandB ({e}). Continuing without online logging.")
            use_wandb = False

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs("outputs/EXP_10", exist_ok=True)

    # 1. Dataset & DataLoaders
    train_dataset = MacroPatchLandmarkDataset(
        dataset_dir=args.dataset_dir,
        mode="train",
        image_size=512,
        macro_patch_size=args.macro_patch_size,
        use_depth=args.use_depth
    )
    val_dataset = MacroPatchLandmarkDataset(
        dataset_dir=args.dataset_dir,
        mode="val",
        image_size=512,
        macro_patch_size=args.macro_patch_size,
        use_depth=args.use_depth
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
        drop_last=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda")
    )

    # 2. Model Initialization
    model = MacroPatchViT(
        backbone_name=args.backbone,
        in_chans=in_chans,
        pretrained=True,
        image_size=512,
        micro_patch_size=16,
        macro_patch_size=args.macro_patch_size,
        num_classes=4,
        embed_dim=768,
        hidden_dim=256,
        macro_depth=2,
        macro_heads=8
    ).to(device)

    # 3. Optimizer with Differential Learning Rate
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
    ], weight_decay=1e-4)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = MacroPatchLoss(
        lambda_cls=2.0,
        lambda_ctrl=5.0,
        lambda_sample=5.0,
        lambda_tan=1.0,
        lambda_cont=1.5,
        lambda_tan_cont=1.0,
        macro_grid_size=512 // args.macro_patch_size,
        macro_patch_size=args.macro_patch_size
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type == "cuda"))

    best_val_dice = 0.0
    print("\n🏁 Starting Training Loop...")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        
        train_loss_accum = 0.0
        train_cls_accum = 0.0
        train_ctrl_accum = 0.0
        train_sample_accum = 0.0
        train_cont_accum = 0.0
        train_tan_cont_accum = 0.0

        for batch_idx, batch in enumerate(train_loader):
            img = batch["image"].to(device, non_blocking=True)
            target_dict = {
                "target_classes": batch["target_classes"].to(device, non_blocking=True),
                "target_beziers": batch["target_beziers"].to(device, non_blocking=True),
                "active_mask": batch["active_mask"].to(device, non_blocking=True)
            }

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(args.amp and device.type == "cuda")):
                pred_dict = model(img)
                loss_dict = criterion(pred_dict, target_dict)
                loss = loss_dict["loss"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss_accum += loss.item()
            train_cls_accum += loss_dict["loss_cls"].item()
            train_ctrl_accum += loss_dict["loss_ctrl"].item()
            train_sample_accum += loss_dict["loss_sample"].item()
            train_cont_accum += loss_dict["loss_cont"].item()
            train_tan_cont_accum += loss_dict["loss_tan_cont"].item()

        scheduler.step()
        num_batches = max(len(train_loader), 1)
        t_loss = train_loss_accum / num_batches
        t_cls = train_cls_accum / num_batches
        t_ctrl = train_ctrl_accum / num_batches
        t_samp = train_sample_accum / num_batches
        t_cont = train_cont_accum / num_batches
        t_tcont = train_tan_cont_accum / num_batches

        # -------------------------------------------------------------
        # Validation Evaluation
        # -------------------------------------------------------------
        model.eval()
        val_dice_scores = []
        val_ctrl_errors_px = []

        with torch.no_grad():
            for val_batch in val_loader:
                v_img = val_batch["image"].to(device)
                v_target_classes = val_batch["target_classes"].cpu().numpy()
                v_target_beziers = val_batch["target_beziers"].cpu().numpy()
                v_active_mask = val_batch["active_mask"].cpu().numpy()
                v_eval_masks = val_batch["target_masks"].cpu().numpy()

                v_preds = model(v_img)
                macro_logits_np = v_preds["macro_logits"].cpu().numpy()
                macro_beziers_np = v_preds["macro_beziers"].cpu().numpy()

                B_val = v_img.shape[0]
                for b in range(B_val):
                    render_res = merge_macro_beziers_to_image(
                        macro_logits_np[b],
                        macro_beziers_np[b],
                        macro_patch_size=args.macro_patch_size,
                        image_size=512,
                        confidence_thresh=0.30
                    )
                    pred_pixel_masks = render_res["pixel_masks"]
                    
                    # Compute Dice per class
                    for c in range(4):
                        d = compute_batch_dice(pred_pixel_masks[c], v_eval_masks[b, c])
                        val_dice_scores.append(d)
                        
                    # Compute control point error on active macro-patches
                    active_b = v_active_mask[b]
                    if active_b.sum() > 0:
                        err = np.linalg.norm(
                            (macro_beziers_np[b, active_b] - v_target_beziers[b, active_b]) * float(args.macro_patch_size),
                            axis=-1
                        ).mean()
                        val_ctrl_errors_px.append(err)

        mean_val_dice = float(np.mean(val_dice_scores)) if len(val_dice_scores) > 0 else 0.0
        mean_ctrl_err = float(np.mean(val_ctrl_errors_px)) if len(val_ctrl_errors_px) > 0 else 0.0
        epoch_dur = time.time() - epoch_start

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] ({epoch_dur:.1f}s) | "
            f"Train Loss: {t_loss:.4f} (Cls: {t_cls:.3f}, Ctrl: {t_ctrl:.3f}, Samp: {t_samp:.3f}, Cont: {t_cont:.3f}) | "
            f"Val Dice: {mean_val_dice * 100:.2f}% | Ctrl Err: {mean_ctrl_err:.2f} px"
        )

        if use_wandb:
            import wandb
            wandb.log({
                "epoch": epoch,
                "train/total_loss": t_loss,
                "train/loss_cls": t_cls,
                "train/loss_ctrl": t_ctrl,
                "train/loss_sample": t_samp,
                "train/loss_cont": t_cont,
                "val/dice_score": mean_val_dice,
                "val/ctrl_point_error_px": mean_ctrl_err,
                "lr/backbone": optimizer.param_groups[0]["lr"],
                "lr/head": optimizer.param_groups[1]["lr"]
            })

        # Checkpointing
        save_dict = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_dice": mean_val_dice,
            "config": vars(args)
        }
        torch.save(save_dict, os.path.join(args.save_dir, "latest_model.pth"))
        if mean_val_dice > best_val_dice:
            best_val_dice = mean_val_dice
            torch.save(save_dict, os.path.join(args.save_dir, "best_model.pth"))
            print(f"  ⭐ New Best Validation Dice: {best_val_dice * 100:.2f}%! Saved best_model.pth")

    print("\n" + "=" * 75)
    print(f"✅ Training Complete! Best Validation Dice: {best_val_dice * 100:.2f}%")
    print(f"📦 Checkpoint saved at: {args.save_dir}/best_model.pth")
    print("=" * 75)


if __name__ == "__main__":
    main()
