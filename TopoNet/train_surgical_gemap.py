"""
Training Script for Surgical-GeMap.

Self-contained Kaggle-compatible script that:
1. Trains Surgical-GeMap with vector losses (Point L1, Direction, Geometric, Focal)
2. Evaluates on validation set with BOTH pixel metrics (Dice/IoU/ASSD) and
   vector metrics (Chamfer/Fréchet Distance)
3. Logs everything to WandB
4. Saves best checkpoint by validation Dice

Usage on Kaggle:
    python train_surgical_gemap.py --data_path /kaggle/working/L3D --epochs 60
"""

import argparse
import os
import sys
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

# ── Setup paths ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from utils.vector_dataset import (
    VectorLandmarkDataset,
    rasterize_normalized_polylines,
)
from utils.metrics import evaluation
from utils import prepare_dataset
from models.surgical_gemap import SurgicalGeMap
from models.vector_losses import (
    SurgicalGeMapCriterion,
    chamfer_distance,
    frechet_distance,
)

try:
    from medpy.metric import assd
except ImportError:
    assd = None

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# ──────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='Surgical-GeMap Training')
    parser.add_argument('--data_path', type=str, default='/kaggle/working/L3D',
                        help='Path to L3D dataset root')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--img_size', type=int, default=1024)
    parser.add_argument('--N', type=int, default=30, help='Max polyline queries')
    parser.add_argument('--K', type=int, default=20, help='Points per polyline')
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--num_decoder_layers', type=int, default=6)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--save_dir', type=str, default='checkpoints_gemap')
    parser.add_argument('--wandb_key', type=str,
                        default='83f4544a22543e319c6009abceaac90b634c68a3')
    parser.add_argument('--wandb_entity', type=str, default='10423057')
    parser.add_argument('--wandb_project', type=str,
                        default='liver-landmark-segmentation-ablation')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


# ──────────────────────────────────────────────
#  Pixel Metrics (TopoNet-compatible)
# ──────────────────────────────────────────────

def compute_pixel_metrics(pred_polylines, pred_classes, gt_mask_4ch, img_size=1024):
    """
    Rasterize predicted polylines and compute pixel metrics against GT.

    Args:
        pred_polylines: list of (K, 2) normalized [0,1] polylines
        pred_classes: list of int class labels
        gt_mask_4ch: (4, H, W) binary GT mask tensor
        img_size: output resolution

    Returns:
        dice, iou, assd_val
    """
    # Rasterize prediction
    if len(pred_polylines) > 0:
        pred_mask = rasterize_normalized_polylines(
            pred_polylines, pred_classes, H=img_size, W=img_size, thickness=35
        )
    else:
        pred_mask = np.zeros((img_size, img_size), dtype=np.uint8)

    # Convert GT from 4-channel to class index
    gt_4ch = gt_mask_4ch.cpu().numpy()  # (4, H, W) in [0, 1]
    gt_class = np.argmax(gt_4ch, axis=0)  # (H, W)

    # Binary masks for classes 1, 2, 3
    pred_bin = np.array([pred_mask == i for i in range(4)]).astype(np.uint8)
    gt_bin = np.array([gt_class == i for i in range(4)]).astype(np.uint8)

    # Dice & IoU (classes 1-3 only)
    iou, dice = evaluation(pred_bin[1:].flatten(), gt_bin[1:].flatten())

    # ASSD
    assd_val = 80.0  # penalty for empty predictions
    if assd is not None:
        if np.count_nonzero(pred_bin[1:]) > 0 and np.count_nonzero(gt_bin[1:]) > 0:
            try:
                assd_val = assd(pred_bin[1:], gt_bin[1:])
            except Exception:
                assd_val = 80.0

    return dice, iou, assd_val


# ──────────────────────────────────────────────
#  Vector Metrics
# ──────────────────────────────────────────────

def compute_vector_metrics(pred_pts_list, pred_cls_list,
                           gt_pts, gt_labels, num_instances):
    """
    Compute Chamfer and Fréchet distances per class.

    Args:
        pred_pts_list: list of (K, 2) predicted polylines (normalized)
        pred_cls_list: list of int predicted classes
        gt_pts: (N_padded, K, 2) GT polylines tensor
        gt_labels: (N_padded,) GT labels tensor
        num_instances: int, number of real GT instances

    Returns:
        mean_chamfer, mean_frechet
    """
    M = num_instances
    chamfer_vals = []
    frechet_vals = []

    for cls in [1, 2, 3]:
        # Predicted polylines for this class
        pred_cls_pts = [
            torch.tensor(p, dtype=torch.float32)
            for p, c in zip(pred_pts_list, pred_cls_list)
            if c == cls
        ]
        # GT polylines for this class
        gt_cls_pts = [
            gt_pts[i].float()
            for i in range(M)
            if gt_labels[i].item() == cls
        ]

        if len(pred_cls_pts) == 0 or len(gt_cls_pts) == 0:
            if len(gt_cls_pts) > 0:
                chamfer_vals.append(1.0)   # max penalty
                frechet_vals.append(1.0)
            continue

        # Stack all points for Chamfer distance
        pred_all = torch.stack(pred_cls_pts)  # (Mp, K, 2)
        gt_all = torch.stack(gt_cls_pts)       # (Mg, K, 2)

        chamfer_vals.append(chamfer_distance(pred_all, gt_all).item())

        # Fréchet: average across all pred-gt pairs (matched by order)
        n_pairs = min(len(pred_cls_pts), len(gt_cls_pts))
        for i in range(n_pairs):
            frechet_vals.append(frechet_distance(pred_cls_pts[i], gt_cls_pts[i]).item())

    mean_chamfer = np.mean(chamfer_vals) if chamfer_vals else 1.0
    mean_frechet = np.mean(frechet_vals) if frechet_vals else 1.0

    return mean_chamfer, mean_frechet


# ──────────────────────────────────────────────
#  Extract predictions from model output
# ──────────────────────────────────────────────

def extract_predictions(pred_logits, pred_pts, conf_threshold=0.3):
    """
    Convert raw model output to lists of polylines and classes.

    Args:
        pred_logits: (N, num_classes) logits for one sample
        pred_pts: (N, K, 2) predicted points for one sample
        conf_threshold: minimum confidence to keep a prediction

    Returns:
        polylines: list of (K, 2) numpy arrays
        classes: list of int class labels
    """
    probs = pred_logits.softmax(-1)  # (N, 4)
    # For each query: class = argmax, confidence = max prob
    max_probs, max_cls = probs.max(-1)  # (N,), (N,)

    polylines = []
    classes = []

    for i in range(len(max_cls)):
        cls = max_cls[i].item()
        conf = max_probs[i].item()

        if cls == 0:  # background
            continue
        if conf < conf_threshold:
            continue

        pts = pred_pts[i].detach().cpu().numpy()  # (K, 2)
        polylines.append(pts)
        classes.append(cls)

    return polylines, classes


# ──────────────────────────────────────────────
#  Training Loop
# ──────────────────────────────────────────────

def train_one_epoch(model, criterion, dataloader, optimizer, scheduler,
                    device, epoch):
    model.train()
    epoch_losses = {
        'loss_total': 0, 'loss_cls': 0, 'loss_pts': 0,
        'loss_dir': 0, 'loss_geo': 0,
    }
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [TRAIN]")
    for batch in pbar:
        images, gt_pts, gt_labels, num_instances, pixel_masks, paths = batch
        images = images.to(device)
        gt_pts = gt_pts.to(device)
        gt_labels = gt_labels.to(device)
        num_instances = num_instances.to(device)

        # Forward
        pred_logits, pred_pts = model(images)

        # Loss
        loss_dict, total_loss = criterion(
            pred_logits, pred_pts, gt_labels, gt_pts, num_instances
        )

        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Accumulate
        for k, v in loss_dict.items():
            epoch_losses[k] += v
        num_batches += 1

        pbar.set_postfix({
            'loss': f"{loss_dict['loss_total']:.4f}",
            'pts': f"{loss_dict['loss_pts']:.4f}",
            'cls': f"{loss_dict['loss_cls']:.4f}",
        })

    if scheduler is not None:
        scheduler.step()

    # Average
    for k in epoch_losses:
        epoch_losses[k] /= max(num_batches, 1)

    return epoch_losses


# ──────────────────────────────────────────────
#  Validation Loop
# ──────────────────────────────────────────────

@torch.no_grad()
def validate(model, dataloader, device, img_size=1024):
    model.eval()

    all_dice, all_iou, all_assd = [], [], []
    all_chamfer, all_frechet = [], []

    pbar = tqdm(dataloader, desc="[VAL]")
    for batch in pbar:
        images, gt_pts, gt_labels, num_instances, pixel_masks, paths = batch
        images = images.to(device)

        pred_logits, pred_pts = model(images)

        B = images.shape[0]
        for b in range(B):
            # Extract predictions for this sample
            polylines, classes = extract_predictions(
                pred_logits[b], pred_pts[b], conf_threshold=0.3
            )

            # ── Pixel metrics ──
            dice, iou, assd_val = compute_pixel_metrics(
                polylines, classes, pixel_masks[b], img_size
            )
            all_dice.append(dice)
            all_iou.append(iou)
            all_assd.append(assd_val)

            # ── Vector metrics ──
            M = num_instances[b].item()
            chamfer, frechet = compute_vector_metrics(
                polylines, classes,
                gt_pts[b].cpu(), gt_labels[b].cpu(), M
            )
            all_chamfer.append(chamfer)
            all_frechet.append(frechet)

    metrics = {
        'val_dice': np.mean(all_dice),
        'val_iou': np.mean(all_iou),
        'val_assd': np.mean(all_assd),
        'val_chamfer': np.mean(all_chamfer),
        'val_frechet': np.mean(all_frechet),
    }
    return metrics


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────

def main():
    args = parse_args()

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── WandB ──
    if HAS_WANDB and args.wandb_key:
        wandb.login(key=args.wandb_key)
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name='SurgicalGeMap_SwinTiny',
            config=vars(args),
        )

    # ── Dataset ──
    train_files, test_files, val_files = prepare_dataset.get_split(args.data_path)
    print(f"Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")

    train_dataset = VectorLandmarkDataset(
        train_files, N=args.N, K=args.K, img_size=args.img_size, mode='train'
    )
    val_dataset = VectorLandmarkDataset(
        val_files, N=args.N, K=args.K, img_size=args.img_size, mode='val'
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=2, pin_memory=True
    )

    # ── Model ──
    model = SurgicalGeMap(
        img_size=args.img_size,
        num_classes=args.num_classes,
        N=args.N,
        K=args.K,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_decoder_layers=args.num_decoder_layers,
        pretrained_backbone=True,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count:,}")

    # ── Loss ──
    criterion = SurgicalGeMapCriterion(
        num_classes=args.num_classes, N=args.N, K=args.K,
        cls_weight=2.0, pts_weight=5.0, dir_weight=2.0, geo_weight=0.5
    ).to(device)

    # ── Optimizer ──
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # ── Training ──
    os.makedirs(args.save_dir, exist_ok=True)
    best_dice = 0.0

    for epoch in range(args.epochs):
        t0 = time.time()

        # Train
        train_losses = train_one_epoch(
            model, criterion, train_loader, optimizer, scheduler,
            device, epoch
        )

        # Validate
        val_metrics = validate(model, val_loader, device, args.img_size)

        epoch_time = time.time() - t0

        # Print
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{args.epochs} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_losses['loss_total']:.4f} "
              f"(pts={train_losses['loss_pts']:.4f}, "
              f"cls={train_losses['loss_cls']:.4f}, "
              f"dir={train_losses['loss_dir']:.4f}, "
              f"geo={train_losses['loss_geo']:.4f})")
        print(f"  Val Dice: {val_metrics['val_dice']:.4f} | "
              f"Val IoU: {val_metrics['val_iou']:.4f} | "
              f"Val ASSD: {val_metrics['val_assd']:.2f}")
        print(f"  Val Chamfer: {val_metrics['val_chamfer']:.4f} | "
              f"Val Fréchet: {val_metrics['val_frechet']:.4f}")
        print(f"{'='*70}\n")

        # WandB
        if HAS_WANDB and wandb.run is not None:
            log_dict = {}
            for k, v in train_losses.items():
                log_dict[f'train/{k}'] = v
            for k, v in val_metrics.items():
                log_dict[k] = v
            log_dict['lr'] = optimizer.param_groups[0]['lr']
            wandb.log(log_dict, step=epoch + 1)

        # Save best
        if val_metrics['val_dice'] > best_dice:
            best_dice = val_metrics['val_dice']
            save_path = os.path.join(args.save_dir, 'best_surgical_gemap.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
                'val_metrics': val_metrics,
                'args': vars(args),
            }, save_path)
            print(f"  ★ New best Dice: {best_dice:.4f} — saved to {save_path}")

    # ── Final checkpoint ──
    final_path = os.path.join(args.save_dir, 'final_surgical_gemap.pth')
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'best_dice': best_dice,
        'args': vars(args),
    }, final_path)
    print(f"\nTraining complete. Best Dice: {best_dice:.4f}")

    if HAS_WANDB and wandb.run is not None:
        wandb.finish()


if __name__ == '__main__':
    main()
