import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

def compute_mask_metrics(pred_masks: torch.Tensor, target_masks: torch.Tensor, valid_mask: torch.Tensor = None, eps: float = 1e-5):
    """
    Computes BOTH Hard (thresholded > 0.5) and Soft (continuous sigmoid) 2D Mask IoU & Dice metrics
    using Optimal Instance Matching for each active GT landmark.
    pred_masks: (B, N, H, W) raw logits
    target_masks: (B, N, H, W) binary GT masks
    valid_mask: (B, N) active GT indicators
    """
    with torch.no_grad():
        B, N, H, W = pred_masks.shape
        probs = torch.sigmoid(pred_masks.float())
        hard_pred = (probs > 0.5).float()
        gt_bin = (target_masks.float() > 0.5).float()

        hard_ious, hard_dices = [], []
        soft_ious, soft_dices = [], []

        for b in range(B):
            if valid_mask is not None:
                active_gt = [g for g in range(N) if valid_mask[b, g] > 0 and gt_bin[b, g].sum() > 0]
            else:
                active_gt = [g for g in range(N) if gt_bin[b, g].sum() > 0]

            if len(active_gt) == 0:
                continue

            for g in active_gt:
                target_m = gt_bin[b, g] # (H, W)
                
                best_h_iou = 0.0
                best_h_dice = 0.0
                best_s_iou = 0.0
                best_s_dice = 0.0
                
                for q in range(N):
                    # Hard Metrics
                    h_m = hard_pred[b, q]
                    h_inter = (h_m * target_m).sum().item()
                    h_union = h_m.sum().item() + target_m.sum().item() - h_inter
                    h_iou = (h_inter + eps) / (h_union + eps)
                    h_dice = (2.0 * h_inter + eps) / (h_m.sum().item() + target_m.sum().item() + eps)

                    # Soft Metrics
                    s_m = probs[b, q]
                    s_inter = (s_m * target_m).sum().item()
                    s_union = s_m.sum().item() + target_m.sum().item() - s_inter
                    s_iou = (s_inter + eps) / (s_union + eps)
                    s_dice = (2.0 * s_inter + eps) / (s_m.sum().item() + target_m.sum().item() + eps)

                    if s_dice > best_s_dice or (best_s_dice == 0.0 and h_dice > best_h_dice):
                        best_h_iou = h_iou
                        best_h_dice = h_dice
                        best_s_iou = s_iou
                        best_s_dice = s_dice

                hard_ious.append(best_h_iou)
                hard_dices.append(best_h_dice)
                soft_ious.append(best_s_iou)
                soft_dices.append(best_s_dice)

        if len(hard_ious) > 0:
            return {
                "hard_iou": float(np.mean(hard_ious)),
                "hard_dice": float(np.mean(hard_dices)),
                "soft_iou": float(np.mean(soft_ious)),
                "soft_dice": float(np.mean(soft_dices))
            }
        else:
            return {"hard_iou": 0.0, "hard_dice": 0.0, "soft_iou": 0.0, "soft_dice": 0.0}

def render_prediction_overlay(image_tensor, gt_polylines, gt_masks, valid_mask, pred_polylines, pred_masks, epoch, split="Train", save_path="vis.png"):
    """
    Renders a side-by-side 2D & 3D visualization of Ground Truth vs Model Prediction for an epoch.
    Supports multi-instance landmark visualization across all active landmarks.
    """
    try:
        # Denormalize image
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
        img_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        img_rgb = np.clip(img_np * std + mean, 0.0, 1.0)

        active_gt_indices = [i for i in range(valid_mask.shape[0]) if valid_mask[i] > 0]
        if len(active_gt_indices) == 0:
            active_gt_indices = [0]

        colors_gt = ['#00ffcc', '#00ff88', '#00e1ff', '#33ffaa']
        colors_pred = ['#ff00ff', '#ff3366', '#ffaa00', '#ff00aa']

        fig = plt.figure(figsize=(14, 6), facecolor='#0d1117')
        
        # 1. 2D Visual Overlay (RGB + All GT Contours + Predicted Mask Heatmap + Projected 2D Polylines)
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.set_facecolor('#161b22')
        ax1.imshow(img_rgb)
        
        # Combined Predicted Mask Heatmap Overlay
        combined_pred_mask = torch.sigmoid(pred_masks).max(dim=0)[0].cpu().numpy()
        ax1.imshow(combined_pred_mask, cmap='magma', alpha=0.35)

        # 2. 3D Camera Space Setup
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        ax2.set_facecolor('#0d1117')

        f_canon = 1.732051
        for idx, gt_i in enumerate(active_gt_indices):
            gt_m = gt_masks[gt_i].cpu().numpy() if gt_masks is not None else np.zeros((1024, 1024))
            gt_p = gt_polylines[gt_i].cpu().numpy() if gt_polylines is not None else np.zeros((20, 3))
            c_gt = colors_gt[idx % len(colors_gt)]
            c_pred = colors_pred[idx % len(colors_pred)]

            # Draw GT Contour in 2D
            if gt_m.sum() > 0:
                ax1.contour(gt_m, levels=[0.5], colors=[c_gt], linewidths=2.5)

            # Find best-matched query for this specific GT landmark
            best_q = 0
            best_s = -1.0
            gt_m_t = torch.from_numpy(gt_m).float()
            for q in range(pred_masks.shape[0]):
                pm_t = torch.sigmoid(pred_masks[q]).cpu().float()
                inter = (pm_t * gt_m_t).sum().item()
                if inter > best_s:
                    best_s = inter
                    best_q = q

            pred_p = pred_polylines[best_q].cpu().numpy()
            
            # Project 2D Predicted line onto image
            z_safe = np.clip(pred_p[:, 2] * 0.45 + 0.55, 0.1, 2.0)
            u_norm = (pred_p[:, 0] / z_safe) / f_canon
            v_norm = (pred_p[:, 1] / z_safe) / f_canon
            u_pix = np.clip((u_norm * 0.5 + 0.5) * 1024, 0, 1023)
            v_pix = np.clip((v_norm * 0.5 + 0.5) * 1024, 0, 1023)
            
            ax1.plot(u_pix, v_pix, color=c_pred, linewidth=2.5, linestyle='--', marker='o', markersize=3, 
                     label=f"Pred #{idx+1}" if idx < 2 else None)

            # Draw 3D in Camera Metric Space
            if gt_p.sum() != 0:
                ax2.plot(gt_p[:, 0], gt_p[:, 1], gt_p[:, 2], color=c_gt, linewidth=3.5, label=f"GT 3D #{idx+1}")
                ax2.scatter(gt_p[:, 0], gt_p[:, 1], gt_p[:, 2], color=c_gt, s=25)

            ax2.plot(pred_p[:, 0], pred_p[:, 1], pred_p[:, 2], color=c_pred, linewidth=2.5, linestyle='--', 
                     label=f"Pred 3D #{idx+1}")
            ax2.scatter(pred_p[:, 0], pred_p[:, 1], pred_p[:, 2], color=c_pred, s=25)

        ax1.set_title(f"Epoch {epoch:02d} [{split}] 2D Projection & Mask (Cyan=GT, Magenta=Pred)", color='white', fontsize=12, fontweight='bold')
        ax1.axis('off')
        ax1.legend(loc="lower left", facecolor='#161b22', labelcolor='white')

        ax2.set_title(f"Epoch {epoch:02d} [{split}] 3D Polyline (Camera Metric Space)", color='white', fontsize=12, fontweight='bold')
        ax2.tick_params(colors='white')
        ax2.set_xlim([-1, 1]); ax2.set_ylim([-1, 1]); ax2.set_zlim([-1, 1])
        ax2.legend(loc="upper left", facecolor='#161b22', labelcolor='white')

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=140, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close()
        return save_path
    except Exception as e:
        print(f"⚠️ Could not render visual overlay: {e}")
        return None

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
    vis_dir = os.path.join(args.save_dir, "epoch_visualizations")
    os.makedirs(vis_dir, exist_ok=True)

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

    # 2. Instantiate Model & Optimizer (Backbone LR = 0.1x to prevent pre-trained weight drift)
    model = Surgical3DVectorTransformer(config).to(device)
    
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr * 0.1},
        {"params": head_params, "lr": args.lr}
    ], weight_decay=config.weight_decay)
    
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

        # Cache last training batch for epoch visualization
        last_train_batch = None
        last_train_outputs = None

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

            # Compute batch Hard & Soft metrics with Optimal Instance Matching
            b_m = compute_mask_metrics(outputs["pred_masks"], targets["target_masks"], targets["valid_mask"])
            for mk in train_metrics:
                train_metrics[mk] += b_m[mk]

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "iou": f"{b_m['hard_iou']*100:.1f}%"})
            
            if batch_idx == 0:
                last_train_batch = batch
                last_train_outputs = {
                    "pred_masks": outputs["pred_masks"].detach(),
                    "pred_polylines": outputs["pred_polylines"].detach()
                }

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

        last_val_batch = None
        last_val_outputs = None

        if val_loader is not None and len(val_loader) > 0:
            model.eval()
            val_epoch_loss = 0.0
            val_metrics = {"hard_iou": 0.0, "hard_dice": 0.0, "soft_iou": 0.0, "soft_dice": 0.0}
            val_epoch_loss_dict = {"l_cls": 0.0, "l_pos": 0.0, "l_tan": 0.0, "l_curv": 0.0, "l_mask": 0.0}
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch:2d}/{args.epochs:2d} [Val]", leave=False)
                for v_idx, batch in enumerate(val_pbar):
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

                    v_m = compute_mask_metrics(val_outputs["pred_masks"], targets["target_masks"], targets["valid_mask"])
                    for mk in val_metrics:
                        val_metrics[mk] += v_m[mk]

                    if v_idx == 0:
                        last_val_batch = batch
                        last_val_outputs = {
                            "pred_masks": val_outputs["pred_masks"].detach(),
                            "pred_polylines": val_outputs["pred_polylines"].detach()
                        }

            avg_val_loss = val_epoch_loss / max(len(val_loader), 1)
            for mk in val_metrics:
                val_metrics[mk] /= max(len(val_loader), 1)

            for k in val_epoch_loss_dict:
                val_loss_dict[k] = val_epoch_loss_dict[k] / max(len(val_loader), 1)

        print(f"Epoch [{epoch:2d}/{args.epochs:2d}] ({elapsed:.1f}s) | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"Val Hard IoU: {val_metrics['hard_iou']*100:.1f}% | Val Soft IoU: {val_metrics['soft_iou']*100:.1f}% | "
              f"Val Hard Dice: {val_metrics['hard_dice']*100:.1f}% | Val Soft Dice: {val_metrics['soft_dice']*100:.1f}%", flush=True)

        # 5. Render 1 Train & 1 Val Visual Diagnostic Image per Epoch
        train_img_path = os.path.join(vis_dir, f"epoch_{epoch:02d}_train.png")
        val_img_path = os.path.join(vis_dir, f"epoch_{epoch:02d}_val.png")

        if last_train_batch is not None and last_train_outputs is not None:
            render_prediction_overlay(
                last_train_batch["image"][0],
                last_train_batch["target_polylines"][0],
                last_train_batch["target_masks"][0],
                last_train_batch["valid_mask"][0],
                last_train_outputs["pred_polylines"][0],
                last_train_outputs["pred_masks"][0],
                epoch=epoch, split="Train", save_path=train_img_path
            )

        if last_val_batch is not None and last_val_outputs is not None:
            render_prediction_overlay(
                last_val_batch["image"][0],
                last_val_batch["target_polylines"][0],
                last_val_batch["target_masks"][0],
                last_val_batch["valid_mask"][0],
                last_val_outputs["pred_polylines"][0],
                last_val_outputs["pred_masks"][0],
                epoch=epoch, split="Val", save_path=val_img_path
            )

        if os.path.exists(train_img_path) or os.path.exists(val_img_path):
            print(f"  🖼️ Epoch {epoch:02d} visual overlays saved to: '{vis_dir}/'", flush=True)

        # 6. Log metrics and visual overlays to W&B
        if use_wandb:
            try:
                import wandb
                log_payload = {
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
                }

                if os.path.exists(train_img_path):
                    log_payload["train/epoch_prediction_visual"] = wandb.Image(train_img_path)
                if os.path.exists(val_img_path):
                    log_payload["val/epoch_prediction_visual"] = wandb.Image(val_img_path)

                wandb.log(log_payload)
            except Exception as e:
                print(f"⚠️ Wandb logging error: {e}")

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
