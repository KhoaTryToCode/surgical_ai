import os
import sys
import time
import argparse
import warnings

# Suppress harmless warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", message=".*Glyph.*")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add experiment root to path
EXP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if EXP_DIR not in sys.path:
    sys.path.append(EXP_DIR)

from configs.exp08_config import config
from utils.dataset_sequential import SequentialLandmarkDataset
from models.surgical_lstm_mdn import SurgicalLSTM_MDN
from models.mdn_losses import MDNLossSuite


def parse_args():
    parser = argparse.ArgumentParser(description="Train EXP_08 CNN-LSTM-MDN Sequential Landmark Detection")
    parser.add_argument("--dataset_dir", type=str, default=config.dataset_dir, help="Path to surgical dataset")
    parser.add_argument("--epochs", type=int, default=config.num_epochs, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=config.batch_size, help="Batch size")
    parser.add_argument("--lr", type=float, default=config.learning_rate, help="Learning rate")
    parser.add_argument("--save_dir", type=str, default="checkpoints/EXP_08", help="Checkpoint save directory")
    parser.add_argument("--viz_interval", type=int, default=20, help="Diagnostic overlay visualization every N steps")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases experiment tracking")
    parser.add_argument("--wandb_project", type=str, default="Surgical_AI_LSTM_MDN", help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default="EXP_08_ResNet18_LSTM_MDN", help="W&B run name")
    return parser.parse_args()


def compute_validation_metrics(model, val_loader, device, config, max_batches=50):
    """
    Compute validation metrics using autoregressive inference.
    
    Metrics:
        - hard_dice:   Binary mask Dice (threshold > 0.5) 
        - poly_err_px: Mean polyline coordinate error in pixel space
    """
    model.eval()
    
    all_hard_dices = []
    all_poly_errs = []
    
    K = config.num_points
    S = config.image_size
    stroke_t = config.mask_stroke_thickness
    
    import cv2
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= max_batches:
                break
            
            images = batch["image"].to(device)
            target_polylines = batch["target_polylines"].to(device)  # (B, N, K, 2)
            target_masks = batch["target_masks"].to(device)          # (B, N, S, S)
            valid_mask = batch["valid_mask"].to(device)              # (B, N)
            target_classes = batch["target_classes"].to(device)      # (B, N)
            
            B = images.shape[0]
            N = target_polylines.shape[1]
            
            # Autoregressive inference
            outputs = model(images)
            pred_polylines = outputs["predicted_polylines"]  # (B, num_classes, K, 2)
            
            for b in range(B):
                active_gt = [i for i in range(N) if valid_mask[b, i]]
                if len(active_gt) == 0:
                    continue
                
                for gi in active_gt:
                    gt_cls = target_classes[b, gi].item()
                    gt_poly = target_polylines[b, gi].cpu().numpy()   # (K, 2)
                    gt_mask = target_masks[b, gi].cpu().numpy()       # (S, S)
                    
                    if gt_cls < 1 or gt_cls > config.num_classes:
                        continue
                    
                    pred_poly = pred_polylines[b, gt_cls - 1].cpu().numpy()  # (K, 2) same class
                    
                    # 1. Polyline Error (pixels at image_size)
                    err_fwd = np.mean(np.abs(pred_poly - gt_poly)) * S
                    err_rev = np.mean(np.abs(pred_poly - gt_poly[::-1])) * S
                    all_poly_errs.append(min(err_fwd, err_rev))
                    
                    # 2. Rasterize predicted polyline → binary mask for Dice
                    pred_mask = np.zeros((S, S), dtype=np.uint8)
                    pts_pix = (pred_poly * float(S)).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(pred_mask, [pts_pix], isClosed=False, color=1, thickness=stroke_t)
                    pred_mask_f = pred_mask.astype(np.float32)
                    gt_mask_bin = (gt_mask > 0.5).astype(np.float32)
                    
                    # Hard Dice
                    eps = 1e-5
                    inter = (pred_mask_f * gt_mask_bin).sum()
                    dice = (2.0 * inter + eps) / (pred_mask_f.sum() + gt_mask_bin.sum() + eps)
                    all_hard_dices.append(dice)
    
    model.train()
    
    return {
        "hard_dice": float(np.mean(all_hard_dices)) if all_hard_dices else 0.0,
        "poly_err_px": float(np.mean(all_poly_errs)) if all_poly_errs else 999.0
    }


def render_step_diagnostic(image_tensor, gt_polylines, valid_mask, target_classes,
                           instance_outputs, step, epoch, save_path="vis.png"):
    """
    Renders a 2-panel diagnostic visualization:
        Panel 1: GT polylines (cyan) vs predicted expected polylines (magenta) with error springs
        Panel 2: Loss breakdown dashboard
    """
    try:
        import cv2
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
        img_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        img_rgb = np.clip(img_np * std + mean, 0.0, 1.0)
        S = img_rgb.shape[0]
        
        class_names = ["BG", "Ridge", "Silhouette", "Falciform", "Gallbladder"]
        colors_gt = ['#00ffcc', '#00ff88', '#00e1ff', '#33ffaa']
        colors_pred = ['#ff00ff', '#ff3366', '#ffaa00', '#ff00aa']
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 8), facecolor='#0d1117')
        ax.set_facecolor('#161b22')
        ax.imshow(img_rgb)
        
        out_idx = 0
        for i in range(valid_mask.shape[0]):
            if not valid_mask[i]:
                continue
            if out_idx >= len(instance_outputs):
                break
            
            gt_poly = gt_polylines[i].cpu().numpy()  # (K, 2)
            expected_pts = instance_outputs[out_idx]["expected_points"].detach().cpu().numpy()  # (K, 2)
            cls_id = int(target_classes[i].item())
            c_name = class_names[cls_id] if cls_id < len(class_names) else f"Cls{cls_id}"
            
            c_gt = colors_gt[out_idx % len(colors_gt)]
            c_pred = colors_pred[out_idx % len(colors_pred)]
            
            # Draw in pixel space
            u_gt = gt_poly[:, 0] * S
            v_gt = gt_poly[:, 1] * S
            u_pred = np.clip(expected_pts[:, 0] * S, 0, S - 1)
            v_pred = np.clip(expected_pts[:, 1] * S, 0, S - 1)
            
            ax.plot(u_gt, v_gt, color=c_gt, linewidth=2.5, linestyle='-', marker='o',
                    markersize=3, label=f"GT: {c_name}")
            ax.plot(u_pred, v_pred, color=c_pred, linewidth=2.0, linestyle='--', marker='s',
                    markersize=3, label=f"Pred: {c_name}")
            
            # Error springs
            for k in range(len(u_gt)):
                ax.plot([u_pred[k], u_gt[k]], [v_pred[k], v_gt[k]],
                        color='#ffff00', linestyle=':', linewidth=0.8, alpha=0.6)
            
            out_idx += 1
        
        ax.set_title(f"Step {step:05d} [Ep {epoch:02d}] GT=Cyan, Pred=Magenta",
                     color='white', fontsize=12, fontweight='bold')
        ax.axis('off')
        ax.legend(loc='lower left', facecolor='#161b22', labelcolor='white', fontsize=8)
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches='tight', facecolor='#0d1117')
        plt.close(fig)
    except Exception as e:
        print(f"⚠️ Visualization error: {e}")


def main():
    args = parse_args()
    
    print("=" * 70)
    print("🧬 EXP_08: CNN-LSTM-MDN Sequential Surgical Landmark Detection")
    print("=" * 70)
    print(f"📁 Dataset: {args.dataset_dir}")
    print(f"🔧 Batch Size: {args.batch_size} | LR: {args.lr} | Epochs: {args.epochs}")
    print(f"📐 Image Size: {config.image_size}×{config.image_size}")
    print(f"🧠 LSTM Hidden: {config.lstm_hidden_dim} | MDN Components: {config.mdn_num_components}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Device: {device}")
    
    # W&B initialization
    wandb_run = None
    if args.wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config={
                    "backbone": config.backbone_name,
                    "lstm_hidden": config.lstm_hidden_dim,
                    "mdn_components": config.mdn_num_components,
                    "num_points": config.num_points,
                    "image_size": config.image_size,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "lambda_mdn": config.lambda_mdn,
                    "lambda_point": config.lambda_point,
                    "lambda_dir": config.lambda_dir,
                    "lambda_mask": config.lambda_mask,
                    "lambda_eos": config.lambda_eos
                }
            )
            print("📊 W&B tracking enabled.")
        except ImportError:
            print("⚠️ wandb not installed. Logging to stdout only.")
    
    # 1. Datasets
    train_dataset = SequentialLandmarkDataset(
        args.dataset_dir, 
        num_instances=config.num_instances,
        num_points=config.num_points,
        mode="train",
        image_size=config.image_size,
        stroke_thickness=config.mask_stroke_thickness
    )
    val_dataset = SequentialLandmarkDataset(
        args.dataset_dir,
        num_instances=config.num_instances,
        num_points=config.num_points,
        mode="val",
        image_size=config.image_size,
        stroke_thickness=config.mask_stroke_thickness
    )
    
    if len(val_dataset) == 0:
        print("⚠️ No val split found, trying 'test'...")
        val_dataset = SequentialLandmarkDataset(
            args.dataset_dir,
            num_instances=config.num_instances,
            num_points=config.num_points,
            mode="test",
            image_size=config.image_size,
            stroke_thickness=config.mask_stroke_thickness
        )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True
    )
    
    print(f"📊 Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    
    # 2. Model
    model = SurgicalLSTM_MDN(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🧠 Model Parameters: {total_params:,} total | {trainable_params:,} trainable")
    
    # 3. Optimizer with differential LR
    param_groups = model.get_param_groups(args.lr)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=config.weight_decay)
    
    # 4. LR Scheduler: Linear warmup + Cosine annealing
    total_steps = args.epochs * len(train_loader)
    warmup_steps = config.warmup_epochs * len(train_loader)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(float(warmup_steps), 1.0)
        progress = float(step - warmup_steps) / max(float(total_steps - warmup_steps), 1.0)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # 5. Loss
    loss_fn = MDNLossSuite(config).to(device)
    
    # 6. Checkpoint directory
    os.makedirs(args.save_dir, exist_ok=True)
    vis_dir = os.path.join(args.save_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # 7. Training Loop
    best_val_dice = 0.0
    global_step = 0
    
    print(f"\n{'=' * 70}")
    print(f"🚀 Starting Training: {args.epochs} epochs, {len(train_loader)} steps/epoch")
    print(f"{'=' * 70}\n")
    
    for epoch in range(args.epochs):
        model.train()
        epoch_losses = {"l_mdn": [], "l_point": [], "l_dir": [], "l_mask": [], "l_eos": [], "total": []}
        epoch_start = time.time()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:03d}/{args.epochs:03d}", ncols=120)
        
        for step, batch in enumerate(pbar):
            images = batch["image"].to(device)
            gt_polylines = batch["target_polylines"].to(device)
            gt_masks = batch["target_masks"].to(device)
            gt_classes = batch["target_classes"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            
            # Forward pass (teacher forced)
            outputs = model(images, gt_polylines, gt_classes, valid_mask)
            
            instance_outputs = outputs["instance_outputs"]
            gt_polys_matched = outputs["gt_polylines_matched"]
            
            # Collect GT masks aligned with matched instances
            gt_masks_matched = []
            inst_idx = 0
            B, N = valid_mask.shape
            for b in range(B):
                for i in range(N):
                    if valid_mask[b, i]:
                        gt_masks_matched.append(gt_masks[b, i])
                        inst_idx += 1
            
            # Compute loss
            if len(instance_outputs) > 0:
                total_loss, loss_dict = loss_fn(instance_outputs, gt_polys_matched, gt_masks_matched)
            else:
                total_loss = torch.tensor(0.0, device=device, requires_grad=True)
                loss_dict = {"l_mdn": 0.0, "l_point": 0.0, "l_dir": 0.0, "l_mask": 0.0, "l_eos": 0.0, "total": 0.0}
            
            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping (prevents LSTM gradient explosion)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            scheduler.step()
            
            # Logging
            for k in epoch_losses:
                epoch_losses[k].append(loss_dict.get(k, 0.0) if isinstance(loss_dict.get(k), float) else loss_dict.get(k, 0.0))
            
            pbar.set_postfix({
                "loss": f"{loss_dict.get('total', 0.0):.4f}",
                "pt": f"{loss_dict.get('l_point', 0.0):.4f}",
                "mdn": f"{loss_dict.get('l_mdn', 0.0):.4f}",
                "lr": f"{optimizer.param_groups[-1]['lr']:.2e}"
            })
            
            # W&B step logging
            if wandb_run is not None:
                log_data = {f"train/{k}": v for k, v in loss_dict.items()}
                log_data["train/lr"] = optimizer.param_groups[-1]['lr']
                wandb_run.log(log_data, step=global_step)
            
            # Diagnostic visualization
            if global_step % args.viz_interval == 0 and len(instance_outputs) > 0:
                viz_path = os.path.join(vis_dir, f"step_{global_step:06d}.png")
                render_step_diagnostic(
                    images[0], gt_polylines[0], valid_mask[0], gt_classes[0],
                    instance_outputs, global_step, epoch + 1, viz_path
                )
            
            global_step += 1
        
        epoch_time = time.time() - epoch_start
        
        # Epoch summary
        avg_losses = {k: float(np.mean(v)) if len(v) > 0 else 0.0 for k, v in epoch_losses.items()}
        print(f"\n📊 Epoch {epoch+1:03d} Summary ({epoch_time:.1f}s):")
        print(f"   Total: {avg_losses['total']:.4f} | Point: {avg_losses['l_point']:.4f} | "
              f"MDN: {avg_losses['l_mdn']:.4f} | Dir: {avg_losses['l_dir']:.4f} | "
              f"Mask: {avg_losses['l_mask']:.4f} | EOS: {avg_losses['l_eos']:.4f}")
        
        # Validation (every 5 epochs)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"🔍 Running validation (autoregressive inference)...")
            val_metrics = compute_validation_metrics(model, val_loader, device, config)
            
            print(f"   Val Hard Dice: {val_metrics['hard_dice']:.4f} | "
                  f"Val Poly Error: {val_metrics['poly_err_px']:.1f}px")
            
            if wandb_run is not None:
                wandb_run.log({
                    "val/hard_dice": val_metrics["hard_dice"],
                    "val/poly_err_px": val_metrics["poly_err_px"],
                    "epoch": epoch + 1
                }, step=global_step)
            
            # Save best model
            if val_metrics["hard_dice"] > best_val_dice:
                best_val_dice = val_metrics["hard_dice"]
                ckpt_path = os.path.join(args.save_dir, "best_model.pth")
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_hard_dice": best_val_dice,
                    "val_poly_err_px": val_metrics["poly_err_px"],
                    "config": config
                }, ckpt_path)
                print(f"   💾 Best model saved (Dice: {best_val_dice:.4f}) → {ckpt_path}")
        
        # Periodic checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(args.save_dir, f"epoch_{epoch+1:03d}.pth")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config
            }, ckpt_path)
            print(f"   💾 Checkpoint saved → {ckpt_path}")
    
    print(f"\n{'=' * 70}")
    print(f"✅ Training Complete! Best Val Dice: {best_val_dice:.4f}")
    print(f"{'=' * 70}")
    
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
