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

# Ensure experiment root is in sys.path
exp_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if exp_root not in sys.path:
    sys.path.insert(0, exp_root)

from configs.exp10_config import EXP10Config, resolve_dataset_dir
from models.super_token_vit import SuperTokenGeometricViT
from models.dual_domain_loss import DualDomainGeometricLoss
from models.spline_utils import evaluate_bezier_curve_torch
from utils.dataset_super_token import SuperTokenLandmarkDataset


def parse_args():
    default_cfg = EXP10Config()
    parser = argparse.ArgumentParser(description="EXP_10: Train Super-Token Geometric ViT")
    parser.add_argument("--dataset_dir", type=str, default=resolve_dataset_dir(), help="Root path to surgical dataset")
    parser.add_argument("--epochs", type=int, default=default_cfg.num_epochs, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=default_cfg.batch_size, help="Batch size")
    parser.add_argument("--lr", type=float, default=default_cfg.learning_rate, help="Head learning rate")
    parser.add_argument("--backbone_lr_mult", type=float, default=default_cfg.backbone_lr_mult, help="Backbone lr multiplier")
    parser.add_argument("--save_dir", type=str, default="checkpoints/EXP_10", help="Directory to save checkpoints")
    parser.add_argument("--backbone", type=str, default=default_cfg.backbone_name, help="ViT backbone name")
    parser.add_argument("--num_ctrl_points", type=int, default=default_cfg.num_ctrl_points, help="Number of control points (K=6)")
    parser.add_argument("--amp", action="store_true", default=default_cfg.use_amp, help="Enable AMP")
    parser.add_argument("--use_depth", action="store_true", default=default_cfg.use_depth, help="Ingest Depth Anything V2")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases")
    parser.add_argument("--wandb_key", type=str, default=default_cfg.wandb_key, help="W&B API key")
    parser.add_argument("--device", type=str, default="", help="Force device (cuda, mps, cpu)")
    return parser.parse_args()


def render_curves_to_eval_mask(ctrl_points_np, exist_probs_np, exist_thresh=0.35, canvas_size=512, stroke_px=2):
    """
    Renders predicted control points to (C, canvas_size, canvas_size) binary evaluation masks.
    """
    C, K, _ = ctrl_points_np.shape
    masks = np.zeros((C, canvas_size, canvas_size), dtype=np.float32)
    
    for c in range(C):
        if exist_probs_np[c] < exist_thresh:
            continue
        # Evaluate 64 dense points
        pts = ctrl_points_np[c]  # (K, 2) in [0, 1]
        t_vals = np.linspace(0.0, 1.0, 64)
        from scipy.special import comb
        deg = K - 1
        dense_curve = np.zeros((64, 2), dtype=np.float32)
        for i in range(K):
            c_val = comb(deg, i) * ((1.0 - t_vals) ** (deg - i)) * (t_vals ** i)
            dense_curve += np.outer(c_val, pts[i])
            
        pts_pix = np.clip(np.round(dense_curve * canvas_size), 0, canvas_size - 1).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(masks[c], [pts_pix], isClosed=False, color=1.0, thickness=stroke_px)
        
    return masks


def compute_dice_score(pred_mask, gt_mask, eps=1e-6):
    intersection = (pred_mask * gt_mask).sum()
    cardinality = pred_mask.sum() + gt_mask.sum()
    if cardinality == 0:
        return 1.0  # True negative
    return float((2.0 * intersection + eps) / (cardinality + eps))


def main():
    args = parse_args()
    in_chans = 4 if args.use_depth else 3
    print("=" * 75)
    print("🚀 [EXP_10] Training Super-Token Geometric Vision Transformer")
    print(f"🏛️ Backbone:          {args.backbone} (RGB-D in_chans={in_chans})")
    print(f"📐 Curve Degrees:      K={args.num_ctrl_points} Control Points (Continuous Spline)")
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
                project="Surgical_AI_EXP10",
                name=f"EXP10_SuperToken_{args.backbone}_K{args.num_ctrl_points}",
                config=vars(args)
            )
            print("✅ Weights & Biases initialized.")
        except Exception as e:
            print(f"⚠️ Could not initialize WandB ({e}). Continuing without online logging.")
            use_wandb = False

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs("outputs/EXP_10", exist_ok=True)

    # 1. Dataset & DataLoaders
    train_dataset = SuperTokenLandmarkDataset(
        dataset_dir=args.dataset_dir,
        mode="train",
        image_size=512,
        patch_size=16,
        num_ctrl_points=args.num_ctrl_points,
        render_size=128,
        use_depth=args.use_depth
    )
    val_dataset = SuperTokenLandmarkDataset(
        dataset_dir=args.dataset_dir,
        mode="val",
        image_size=512,
        patch_size=16,
        num_ctrl_points=args.num_ctrl_points,
        render_size=128,
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
    model = SuperTokenGeometricViT(
        backbone_name=args.backbone,
        in_chans=in_chans,
        pretrained=True,
        image_size=512,
        patch_size=16,
        num_classes=4,
        num_ctrl_points=args.num_ctrl_points,
        embed_dim=768,
        hidden_dim=512,
        render_size=128,
        cross_attn_heads=8
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
    criterion = DualDomainGeometricLoss(
        lambda_attn=2.0,
        lambda_vector=5.0,
        lambda_dice=5.0,
        lambda_exist=1.5
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type == "cuda"))

    best_val_dice = 0.0
    print("\n🏁 Starting Training Loop...")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        
        train_loss_accum = 0.0
        train_attn_accum = 0.0
        train_vec_accum = 0.0
        train_dice_accum = 0.0
        train_exist_accum = 0.0

        for batch_idx, batch in enumerate(train_loader):
            img = batch["image"].to(device, non_blocking=True)
            target_dict = {
                "target_exists": batch["target_exists"].to(device, non_blocking=True),
                "target_ctrl_points": batch["target_ctrl_points"].to(device, non_blocking=True),
                "target_attn_masks": batch["target_attn_masks"].to(device, non_blocking=True),
                "target_render_masks": batch["target_render_masks"].to(device, non_blocking=True)
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
            train_attn_accum += loss_dict["loss_attn"].item()
            train_vec_accum += loss_dict["loss_vector"].item()
            train_dice_accum += loss_dict["loss_dice"].item()
            train_exist_accum += loss_dict["loss_exist"].item()

        scheduler.step()
        num_batches = max(len(train_loader), 1)
        t_loss = train_loss_accum / num_batches
        t_attn = train_attn_accum / num_batches
        t_vec = train_vec_accum / num_batches
        t_dice = train_dice_accum / num_batches
        t_exist = train_exist_accum / num_batches

        # -------------------------------------------------------------
        # Validation Evaluation
        # -------------------------------------------------------------
        model.eval()
        val_dice_scores = []
        val_vec_errors_px = []

        with torch.no_grad():
            for val_batch in val_loader:
                v_img = val_batch["image"].to(device)
                v_targets = val_batch["target_exists"].cpu().numpy()       # (B, C)
                v_gt_ctrls = val_batch["target_ctrl_points"].cpu().numpy()  # (B, C, K, 2)
                v_eval_masks = val_batch["target_eval_masks"].cpu().numpy() # (B, C, 512, 512)

                v_preds = model(v_img)
                pred_ctrls_np = v_preds["ctrl_points"].cpu().numpy()       # (B, C, K, 2)
                pred_probs_np = v_preds["exist_probs"].cpu().numpy()       # (B, C)

                B_val = v_img.shape[0]
                for b in range(B_val):
                    pred_mask_512 = render_curves_to_eval_mask(
                        pred_ctrls_np[b],
                        pred_probs_np[b],
                        exist_thresh=0.35,
                        canvas_size=512,
                        stroke_px=2
                    )
                    for c in range(4):
                        if v_targets[b, c] > 0.5:
                            # Landmark is active
                            d = compute_dice_score(pred_mask_512[c], v_eval_masks[b, c])
                            val_dice_scores.append(d)
                            # Euclidean error on control points in pixels
                            err_px = np.linalg.norm((pred_ctrls_np[b, c] - v_gt_ctrls[b, c]) * 512.0, axis=-1).mean()
                            val_vec_errors_px.append(err_px)
                        else:
                            # Empty class validation
                            d = compute_dice_score(pred_mask_512[c], v_eval_masks[b, c])
                            val_dice_scores.append(d)

        mean_val_dice = float(np.mean(val_dice_scores)) if len(val_dice_scores) > 0 else 0.0
        mean_vec_err = float(np.mean(val_vec_errors_px)) if len(val_vec_errors_px) > 0 else 0.0
        epoch_dur = time.time() - epoch_start

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] ({epoch_dur:.1f}s) | "
            f"Train Loss: {t_loss:.4f} (Attn: {t_attn:.3f}, Vec: {t_vec:.3f}, Dice: {t_dice:.3f}, Exist: {t_exist:.3f}) | "
            f"Val Dice: {mean_val_dice * 100:.2f}% | Ctrl Err: {mean_vec_err:.2f} px"
        )

        # WandB logging
        if use_wandb:
            import wandb
            wandb.log({
                "epoch": epoch,
                "train/total_loss": t_loss,
                "train/loss_attn": t_attn,
                "train/loss_vector": t_vec,
                "train/loss_dice": t_dice,
                "train/loss_exist": t_exist,
                "val/dice_score": mean_val_dice,
                "val/ctrl_point_error_px": mean_vec_err,
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
    print(f"📦 Checkpoints saved at: {args.save_dir}/best_model.pth")
    print("=" * 75)


if __name__ == "__main__":
    main()
