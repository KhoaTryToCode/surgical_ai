import os
import sys
import time
import argparse
import warnings

# Suppress harmless Matplotlib font & HF Hub warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", message=".*Glyph.*")
warnings.filterwarnings("ignore", message=".*unauthenticated requests to the HF Hub.*")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
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

from configs.exp06_config import config
from utils.dataset_2d import Surgical2DVectorDataset
from models.surgical_2d_vector_transformer import Surgical2DVectorTransformer

def parse_args():
    parser = argparse.ArgumentParser(description="Train EXP_06 Direct 2D Vector Space Transformer")
    parser.add_argument("--dataset_dir", type=str, default=config.dataset_dir, help="Path to surgical dataset")
    parser.add_argument("--epochs", type=int, default=config.num_epochs, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=config.batch_size, help="Batch size")
    parser.add_argument("--lr", type=float, default=config.learning_rate, help="Learning rate")
    parser.add_argument("--save_dir", type=str, default="checkpoints/EXP_06", help="Checkpoint save directory")
    parser.add_argument("--viz_interval", type=int, default=10, help="Log diagnostic overlay visualization every N steps")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases experiment tracking")
    parser.add_argument("--wandb_project", type=str, default="Surgical_AI_2D_Vector", help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default="EXP_06_Swin_2D_Vector_Transformer", help="W&B run name")
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

def render_step_diagnostic_overlay(image_tensor, gt_polylines, gt_masks, valid_mask, target_classes,
                                   pred_polylines, pred_masks, pred_cls, loss_dict, total_loss,
                                   step, epoch, save_path="vis.png"):
    """
    Renders an in-depth 3-panel step diagnostic visualization:
      Panel 1: Ground Truth (Cyan) vs Pred Polylines (Magenta) + Yellow Error Springs
      Panel 2: Dot-Product Attention Heatmap + White GT Contours
      Panel 3: Diagnostic Loss & Error Breakdown Dashboard
    """
    try:
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
        img_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        img_rgb = np.clip(img_np * std + mean, 0.0, 1.0)

        active_gt_indices = [i for i in range(valid_mask.shape[0]) if valid_mask[i] > 0]
        if len(active_gt_indices) == 0:
            active_gt_indices = [0]

        colors_gt = ['#00ffcc', '#00ff88', '#00e1ff', '#33ffaa']
        colors_pred = ['#ff00ff', '#ff3366', '#ffaa00', '#ff00aa']
        class_names = ["Background", "Falciform", "Anterior Ridge", "Silhouette", "Gallbladder"]

        fig = plt.figure(figsize=(20, 7), facecolor='#0d1117')
        gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.2, 1.0])

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])

        # 1. Panel 1: Vector Overlay with Error Springs
        ax1.set_facecolor('#161b22')
        ax1.imshow(img_rgb)

        # 2. Panel 2: Mask Heatmap
        ax2.set_facecolor('#161b22')
        ax2.imshow(img_rgb)
        combined_pred_mask = torch.sigmoid(pred_masks).max(dim=0)[0].cpu().numpy()
        ax2.imshow(combined_pred_mask, cmap='magma', alpha=0.60)

        mean_pixel_errors = []
        max_pixel_errors = []
        match_info = []

        # 1-to-1 Bipartite Hungarian Assignment for all active GT landmarks
        from scipy.optimize import linear_sum_assignment
        N_cand = pred_polylines.shape[0]
        N_gt = len(active_gt_indices)
        cost_matrix = np.zeros((N_cand, N_gt))

        for j, gt_i in enumerate(active_gt_indices):
            gt_p_t = torch.from_numpy(gt_polylines[gt_i].cpu().numpy()).float()
            for q in range(N_cand):
                p_cand = pred_polylines[q].cpu().float()
                d_fwd = torch.mean(torch.abs(p_cand - gt_p_t)).item()
                d_rev = torch.mean(torch.abs(p_cand - torch.flip(gt_p_t, dims=[0]))).item()
                cost_matrix[q, j] = min(d_fwd, d_rev)

        pred_assign, gt_assign = linear_sum_assignment(cost_matrix)
        gt_to_query = {j: q for q, j in zip(pred_assign, gt_assign)}

        for j, gt_i in enumerate(active_gt_indices):
            gt_m = gt_masks[gt_i].cpu().numpy() if gt_masks is not None else np.zeros((1024, 1024))
            gt_p = gt_polylines[gt_i].cpu().numpy() if gt_polylines is not None else np.zeros((20, 2))
            c_id = int(target_classes[gt_i].item()) if target_classes is not None else 0
            c_name = class_names[c_id] if c_id < len(class_names) else f"Class {c_id}"

            c_gt = colors_gt[j % len(colors_gt)]
            c_pred = colors_pred[j % len(colors_pred)]

            # Draw GT Contour in 2D
            if gt_m.sum() > 0:
                ax1.contour(gt_m, levels=[0.5], colors=[c_gt], linewidths=2.5)
                ax2.contour(gt_m, levels=[0.5], colors=['#ffffff'], linewidths=1.5, linestyles=':')

            # Unique Hungarian-assigned query for this GT landmark
            best_q = gt_to_query[j]
            pred_p = pred_polylines[best_q].cpu().numpy()

            # Coordinates in 1024px space
            u_gt = gt_p[:, 0] * 1024.0
            v_gt = gt_p[:, 1] * 1024.0
            u_pred = np.clip(pred_p[:, 0] * 1024.0, 0, 1023)
            v_pred = np.clip(pred_p[:, 1] * 1024.0, 0, 1023)

            # Draw GT Polyline Points
            ax1.plot(u_gt, v_gt, color=c_gt, linewidth=2.5, linestyle='-', marker='o', markersize=4, label=f"GT: {c_name}")
            # Draw Predicted Polyline Points
            ax1.plot(u_pred, v_pred, color=c_pred, linewidth=3.0, linestyle='--', marker='s', markersize=5, label=f"Pred Query #{best_q}")

            # Draw Yellow Error Springs connecting corresponding points
            # Check orientation alignment
            dist_fwd = np.mean(np.sqrt((u_pred - u_gt)**2 + (v_pred - v_gt)**2))
            dist_rev = np.mean(np.sqrt((u_pred - u_gt[::-1])**2 + (v_pred - v_gt[::-1])**2))
            if dist_rev < dist_fwd:
                u_gt_aligned = u_gt[::-1]
                v_gt_aligned = v_gt[::-1]
                cur_dist = dist_rev
            else:
                u_gt_aligned = u_gt
                v_gt_aligned = v_gt
                cur_dist = dist_fwd

            mean_pixel_errors.append(cur_dist)
            pt_dists = np.sqrt((u_pred - u_gt_aligned)**2 + (v_pred - v_gt_aligned)**2)
            max_pixel_errors.append(np.max(pt_dists))

            for k in range(len(u_pred)):
                ax1.plot([u_pred[k], u_gt_aligned[k]], [v_pred[k], v_gt_aligned[k]], color='#ffff00', linestyle=':', linewidth=1.0, alpha=0.7)

            match_info.append(f"* Query #{best_q} -> {c_name} (Avg Err: {cur_dist:.1f}px)")

        ax1.set_title(f"Step {step:05d} [Ep {epoch:02d}] Vectors (Cyan=GT, Magenta=Pred, Yellow=Error)", color='white', fontsize=11, fontweight='bold')
        ax1.axis('off')
        ax1.legend(loc="lower left", facecolor='#161b22', labelcolor='white', fontsize=9)

        ax2.set_title(f"Dot-Product 2D Mask Attention Heatmap", color='white', fontsize=11, fontweight='bold')
        ax2.axis('off')

        # 3. Panel 3: Diagnostic Metrics Card
        ax3.set_facecolor('#161b22')
        ax3.axis('off')

        avg_err = np.mean(mean_pixel_errors) if len(mean_pixel_errors) > 0 else 0.0
        max_err = np.max(max_pixel_errors) if len(max_pixel_errors) > 0 else 0.0
        l_pos = loss_dict.get("l_pos", 0.0)
        l_cls = loss_dict.get("l_cls", 0.0)
        l_mask = loss_dict.get("l_mask", 0.0)

        card_text = (
            f"[LIVE STEP DIAGNOSTICS] Step {step:05d} | Epoch {epoch:02d}\n"
            f"---------------------------------------------------\n"
            f"TOTAL BATCH LOSS:  {total_loss:.4f}\n\n"
            f"Coordinate Loss (L_pos):  {l_pos:.4f}\n"
            f"   * Mean Point Error:       {avg_err:.1f} px / 1024 px\n"
            f"   * Max Point Error:        {max_err:.1f} px\n"
            f"   * Normalized L1 Dist:     {avg_err/1024.0:.4f}\n\n"
            f"Classification Loss (L_cls): {l_cls:.4f}\n"
            f"Mask BCE+Dice Loss (L_mask): {l_mask:.4f}\n"
            f"---------------------------------------------------\n"
            f"Bipartite Hungarian Matches:\n" + "\n".join(match_info) + "\n\n"
            f"Loss Reflection Analysis:\n"
            f"{'* Prediction is FAR from GT -> High L_pos penalty!' if avg_err > 50 else '* Prediction is CLOSE to GT -> Low L_pos gradient!'}\n"
            f"* Yellow lines show the exact pull direction of backprop."
        )

        ax3.text(0.05, 0.95, card_text, transform=ax3.transAxes, color='#e6edf3',
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.6', facecolor='#0d1117', edgecolor='#30363d', alpha=0.9))

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=130, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close('all')
        return save_path, avg_err
    except Exception as e:
        print(f"⚠️ Could not render step diagnostic overlay: {e}")
        plt.close('all')
        return None, 0.0

def render_prediction_overlay(image_tensor, gt_polylines, gt_masks, valid_mask, pred_polylines, pred_masks, epoch, split="Train", save_path="vis.png"):
    """
    Renders a clean 2D visualization of Ground Truth Contours vs Model Predicted 2D Polylines and Masks.
    """
    try:
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
        img_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        img_rgb = np.clip(img_np * std + mean, 0.0, 1.0)

        active_gt_indices = [i for i in range(valid_mask.shape[0]) if valid_mask[i] > 0]
        if len(active_gt_indices) == 0:
            active_gt_indices = [0]

        colors_gt = ['#00ffcc', '#00ff88', '#00e1ff', '#33ffaa']
        colors_pred = ['#ff00ff', '#ff3366', '#ffaa00', '#ff00aa']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), facecolor='#0d1117')
        
        # 1. Left Panel: Ground Truth Contours + Predicted 2D Polylines
        ax1.set_facecolor('#161b22')
        ax1.imshow(img_rgb)
        
        # 2. Right Panel: Dot-Product Predicted 2D Mask Heatmap
        ax2.set_facecolor('#161b22')
        ax2.imshow(img_rgb)
        combined_pred_mask = torch.sigmoid(pred_masks).max(dim=0)[0].cpu().numpy()
        ax2.imshow(combined_pred_mask, cmap='magma', alpha=0.55)

        for idx, gt_i in enumerate(active_gt_indices):
            gt_m = gt_masks[gt_i].cpu().numpy() if gt_masks is not None else np.zeros((1024, 1024))
            gt_p = gt_polylines[gt_i].cpu().numpy() if gt_polylines is not None else np.zeros((20, 2))
            c_gt = colors_gt[idx % len(colors_gt)]
            c_pred = colors_pred[idx % len(colors_pred)]

            # Draw GT Contour in 2D
            if gt_m.sum() > 0:
                ax1.contour(gt_m, levels=[0.5], colors=[c_gt], linewidths=2.5)
                ax2.contour(gt_m, levels=[0.5], colors=['#ffffff'], linewidths=1.5, linestyles=':')

            # Find best-matched query for this GT landmark
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
            
            # Direct 2D coordinates in pixel space (u, v) * 1024
            u_pix = np.clip(pred_p[:, 0] * 1024.0, 0, 1023)
            v_pix = np.clip(pred_p[:, 1] * 1024.0, 0, 1023)
            
            ax1.plot(u_pix, v_pix, color=c_pred, linewidth=3.0, linestyle='--', marker='o', markersize=4, 
                     label=f"Pred Polyline #{idx+1}")

        ax1.set_title(f"Epoch {epoch:02d} [{split}] 2D Vector Polylines (Cyan=GT, Magenta=Pred)", color='white', fontsize=12, fontweight='bold')
        ax1.axis('off')
        ax1.legend(loc="lower left", facecolor='#161b22', labelcolor='white')

        ax2.set_title(f"Epoch {epoch:02d} [{split}] Dot-Product 2D Mask Attention Heatmap", color='white', fontsize=12, fontweight='bold')
        ax2.axis('off')

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=140, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close('all')
        return save_path
    except Exception as e:
        print(f"⚠️ Could not render visual overlay: {e}")
        plt.close('all')
        return None

def main():
    args = parse_args()
    print("=" * 70)
    print("🚀 Starting Training: EXP_06 Direct 2D Vector Space Transformer")
    print("=" * 70)
    print(f"📁 Dataset Directory: {args.dataset_dir}")
    print(f"⚙️ Hyperparameters: Batch Size={args.batch_size}, LR={args.lr}, Epochs={args.epochs}")

    # Initialize Weights & Biases
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
    diag_dir = os.path.join(args.save_dir, "live_diagnostics")
    os.makedirs(diag_dir, exist_ok=True)
    anchor_dir = os.path.join(args.save_dir, "anchor_progression")
    os.makedirs(anchor_dir, exist_ok=True)

    # 1. Dataset & DataLoader
    if not os.path.exists(args.dataset_dir):
        print(f"⚠️ Warning: Dataset path '{args.dataset_dir}' does not exist.")
        return

    train_dataset = Surgical2DVectorDataset(
        dataset_dir=args.dataset_dir,
        num_instances=config.num_instances,
        num_points=config.num_points,
        mode="train"
    )
    val_dataset = Surgical2DVectorDataset(
        dataset_dir=args.dataset_dir,
        num_instances=config.num_instances,
        num_points=config.num_points,
        mode="val"
    )
    print(f"📊 Training Dataset Size: {len(train_dataset)} images | Validation Dataset Size: {len(val_dataset)} images")

    # Select Fixed Anchor Frames for Time-Lapse Progression Tracking
    fixed_train_anchor = None
    if len(train_dataset) > 0:
        for i in range(min(len(train_dataset), 50)):
            s = train_dataset[i]
            if s["valid_mask"].sum() >= 2:
                fixed_train_anchor = {k: v.unsqueeze(0).to(device) for k, v in s.items()}
                break
        if fixed_train_anchor is None:
            fixed_train_anchor = {k: v.unsqueeze(0).to(device) for k, v in train_dataset[0].items()}

    fixed_val_anchor = None
    if len(val_dataset) > 0:
        for i in range(min(len(val_dataset), 50)):
            s = val_dataset[i]
            if s["valid_mask"].sum() >= 2:
                fixed_val_anchor = {k: v.unsqueeze(0).to(device) for k, v in s.items()}
                break
        if fixed_val_anchor is None:
            fixed_val_anchor = {k: v.unsqueeze(0).to(device) for k, v in val_dataset[0].items()}

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

    # 2. Instantiate Model & Optimizer (Backbone LR = 0.1x)
    model = Surgical2DVectorTransformer(config).to(device)
    
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
        {"params": backbone_params, "lr": args.lr * config.backbone_lr_mult},
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
        epoch_loss_dict = {"l_cls": 0.0, "l_pos": 0.0, "l_mask": 0.0}
        start_time = time.time()

        last_train_batch = None
        last_train_outputs = None

        pbar = tqdm(loader, desc=f"Epoch {epoch:2d}/{args.epochs:2d} [Train]", leave=False)
        for batch_idx, batch in enumerate(pbar):
            images = batch["image"].to(device)
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
                outputs = model(images, targets=targets)
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

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "dice": f"{b_m['hard_dice']*100:.1f}%"})
            
            global_step = (epoch - 1) * len(loader) + batch_idx + 1

            # Render Live Diagnostic Overlay every viz_interval steps (e.g. every 10 batches)
            if global_step % args.viz_interval == 0 or (epoch == 1 and batch_idx in [0, 2, 5]):
                diag_path = os.path.join(diag_dir, f"step_{global_step:05d}_loss_{loss.item():.2f}.png")
                res_path, avg_pixel_err = render_step_diagnostic_overlay(
                    image_tensor=batch["image"][0],
                    gt_polylines=batch["target_polylines"][0],
                    gt_masks=batch["target_masks"][0],
                    valid_mask=batch["valid_mask"][0],
                    target_classes=batch["target_classes"][0],
                    pred_polylines=outputs["pred_polylines"][0].detach(),
                    pred_masks=outputs["pred_masks"][0].detach(),
                    pred_cls=outputs["pred_cls"][0].detach() if "pred_cls" in outputs else None,
                    loss_dict=outputs["loss_dict"],
                    total_loss=loss.item(),
                    step=global_step,
                    epoch=epoch,
                    save_path=diag_path
                )
                if use_wandb and res_path and os.path.exists(res_path):
                    wandb.log({
                        "live_diagnostic_overlay": wandb.Image(res_path, caption=f"Step {global_step:05d} (Ep {epoch}) | Loss: {loss.item():.3f} | Pixel Err: {avg_pixel_err:.1f}px"),
                        "live_pixel_error_px": avg_pixel_err
                    }, step=global_step)

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
            val_epoch_loss_dict = {"l_cls": 0.0, "l_pos": 0.0, "l_mask": 0.0}
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch:2d}/{args.epochs:2d} [Val]", leave=False)
                for v_idx, batch in enumerate(val_pbar):
                    images = batch["image"].to(device)
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
                        val_outputs = model(images, targets=targets)
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

        # 5. Render Fixed Anchor Frames (Time-Lapse Progression Tracking)
        anchor_train_path = os.path.join(anchor_dir, f"epoch_{epoch:02d}_train_anchor.png")
        anchor_val_path = os.path.join(anchor_dir, f"epoch_{epoch:02d}_val_anchor.png")

        if fixed_train_anchor is not None:
            model.eval()
            with torch.no_grad():
                a_targets = {
                    "target_classes": fixed_train_anchor["target_classes"],
                    "target_polylines": fixed_train_anchor["target_polylines"],
                    "target_masks": fixed_train_anchor["target_masks"],
                    "valid_mask": fixed_train_anchor["valid_mask"]
                }
                with autocast_ctx:
                    a_out = model(fixed_train_anchor["image"], targets=a_targets)
                
                render_step_diagnostic_overlay(
                    image_tensor=fixed_train_anchor["image"][0],
                    gt_polylines=fixed_train_anchor["target_polylines"][0],
                    gt_masks=fixed_train_anchor["target_masks"][0],
                    valid_mask=fixed_train_anchor["valid_mask"][0],
                    target_classes=fixed_train_anchor["target_classes"][0],
                    pred_polylines=a_out["pred_polylines"][0],
                    pred_masks=a_out["pred_masks"][0],
                    pred_cls=a_out.get("pred_cls", None)[0] if "pred_cls" in a_out else None,
                    loss_dict=a_out["loss_dict"],
                    total_loss=a_out["loss"].item(),
                    step=epoch, epoch=epoch,
                    save_path=anchor_train_path
                )

        if fixed_val_anchor is not None:
            model.eval()
            with torch.no_grad():
                v_a_targets = {
                    "target_classes": fixed_val_anchor["target_classes"],
                    "target_polylines": fixed_val_anchor["target_polylines"],
                    "target_masks": fixed_val_anchor["target_masks"],
                    "valid_mask": fixed_val_anchor["valid_mask"]
                }
                with autocast_ctx:
                    v_a_out = model(fixed_val_anchor["image"], targets=v_a_targets)
                
                render_step_diagnostic_overlay(
                    image_tensor=fixed_val_anchor["image"][0],
                    gt_polylines=fixed_val_anchor["target_polylines"][0],
                    gt_masks=fixed_val_anchor["target_masks"][0],
                    valid_mask=fixed_val_anchor["valid_mask"][0],
                    target_classes=fixed_val_anchor["target_classes"][0],
                    pred_polylines=v_a_out["pred_polylines"][0],
                    pred_masks=v_a_out["pred_masks"][0],
                    pred_cls=v_a_out.get("pred_cls", None)[0] if "pred_cls" in v_a_out else None,
                    loss_dict=v_a_out["loss_dict"],
                    total_loss=v_a_out["loss"].item(),
                    step=epoch, epoch=epoch,
                    save_path=anchor_val_path
                )

        print(f"  [ANCHOR] Epoch {epoch:02d} fixed anchor progression saved to: '{anchor_dir}/'", flush=True)

        # 6. Render Latest Batch Overlays
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

        # 6. Log to Weights & Biases
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
                    "train/l_mask": epoch_loss_dict['l_mask'],
                    "val/total_loss": avg_val_loss,
                    "val/hard_iou_pct": val_metrics["hard_iou"] * 100.0,
                    "val/hard_dice_pct": val_metrics["hard_dice"] * 100.0,
                    "val/soft_iou_pct": val_metrics["soft_iou"] * 100.0,
                    "val/soft_dice_pct": val_metrics["soft_dice"] * 100.0,
                    "val/l_cls": val_loss_dict['l_cls'],
                    "val/l_pos": val_loss_dict['l_pos'],
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
            ckpt_path = os.path.join(args.save_dir, "best_surgical_2d_vector.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "config": config
            }, ckpt_path)
            print(f"  💾 Best validation checkpoint saved to '{ckpt_path}'")

    print("=" * 70)
    print(f"✅ EXP_06 Training Finished cleanly! Best Val Loss = {best_val_loss:.4f}")
    print("=" * 70)

if __name__ == "__main__":
    main()
