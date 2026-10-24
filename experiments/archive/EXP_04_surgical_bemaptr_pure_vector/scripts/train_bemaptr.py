"""
Training Script for Surgical-BeMapTR (EXP_04: Pure Vectorized Landmark Segmentation).

Features:
1. Pure Vector Architecture with Piecewise Bézier Curves <k=3, n=3> (BeMapNet)
2. Permutation-Equivalent Matching & Hierarchical Queries (MapTRv2)
3. Evaluates on validation set with BOTH pixel metrics (Dice/IoU/ASSD via rasterized curves)
   and vector metrics (Chamfer/Fréchet Distance)
4. Logs metrics to WandB and saves best checkpoint by validation Dice
"""

import argparse
import os
import sys
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

# ── Setup paths ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(os.path.dirname(EXP_DIR))

for p in [SCRIPT_DIR, os.path.join(EXP_DIR, 'models'), os.path.join(REPO_ROOT, 'shared'), REPO_ROOT]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

try:
    from utils.vector_dataset import VectorLandmarkDataset, rasterize_normalized_polylines
    from utils.metrics import evaluation
    from utils import prepare_dataset
except ImportError:
    from shared.utils.vector_dataset import VectorLandmarkDataset, rasterize_normalized_polylines
    from shared.utils.metrics import evaluation
    from shared.utils import prepare_dataset

try:
    from surgical_bemaptr import SurgicalBeMapTR
    from vector_losses import (
        SurgicalBeMapTRCriterion,
        chamfer_distance,
        frechet_distance,
    )
except ImportError:
    models_dir = os.path.join(EXP_DIR, 'models')
    if models_dir not in sys.path:
        sys.path.insert(0, models_dir)
    from surgical_bemaptr import SurgicalBeMapTR
    from vector_losses import (
        SurgicalBeMapTRCriterion,
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
#  CLI Arguments
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train Surgical-BeMapTR (Pure Vectorized Landmark Segmentation)")
    parser.add_argument('--data_path', type=str, default='/kaggle/working/L3D',
                        help='Path to L3D dataset root containing train/val/test')
    parser.add_argument('--epochs', type=int, default=60, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--num_queries', type=int, default=30, help='Number of polyline queries (N)')
    parser.add_argument('--num_points', type=int, default=20, help='Dense points per polyline (K)')
    parser.add_argument('--embed_dim', type=int, default=256, help='Transformer embedding dimension')
    parser.add_argument('--num_decoder_layers', type=int, default=6, help='Number of decoder layers')
    parser.add_argument('--coord_feat_size', type=int, default=64, help='Spatial centroid grid size (64, 128, or 256)')
    parser.add_argument('--wandb_project', type=str, default='liver-landmark-segmentation-ablation')
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--wandb_key', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='checkpoints_bemaptr')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training from')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


# ──────────────────────────────────────────────
#  Prediction Extraction & Evaluation Helpers
# ──────────────────────────────────────────────

def extract_predictions(pred_logits, pred_restored_pts, conf_threshold=0.05):
    """
    Extracts candidate polylines for evaluation.
    Selects all non-background queries (class 1, 2, or 3) where top predicted class != 0 (Background)
    and probability exceeds conf_threshold (default 0.05).
    """
    probs = F.softmax(pred_logits, dim=-1)
    pred_cls = torch.argmax(probs, dim=-1)
    top_prob = probs.max(dim=-1)[0]

    keep = (pred_cls > 0) & (top_prob > conf_threshold)
    polylines = pred_restored_pts[keep].detach().cpu().numpy()
    cls_list = pred_cls[keep].detach().cpu().numpy().tolist()

    return polylines, cls_list


def compute_pixel_metrics(polylines_list, cls_list, gt_mask_4ch, img_size=1024):
    """Rasterizes predicted restored polylines to compute Dice, IoU, and ASSD."""
    pred_mask = rasterize_normalized_polylines(
        polylines_list, cls_list, H=img_size, W=img_size, thickness=19
    )

    gt_4ch = gt_mask_4ch.cpu().numpy()
    gt_class = np.argmax(gt_4ch, axis=0)

    pred_bin = np.array([pred_mask == i for i in range(4)]).astype(np.uint8)
    gt_bin = np.array([gt_class == i for i in range(4)]).astype(np.uint8)

    iou, dice = evaluation(pred_bin[1:].flatten(), gt_bin[1:].flatten())

    assd_val = 80.0
    if assd is not None:
        if np.count_nonzero(pred_bin[1:]) > 0 and np.count_nonzero(gt_bin[1:]) > 0:
            try:
                assd_val = assd(pred_bin[1:], gt_bin[1:])
            except Exception:
                assd_val = 80.0

    return dice, iou, assd_val


def compute_vector_metrics(pred_pts_list, pred_cls_list, gt_pts, gt_labels, num_instances):
    M = num_instances
    chamfer_vals = []
    frechet_vals = []

    for cls in [1, 2, 3]:
        pred_cls_pts = [
            torch.tensor(p, dtype=torch.float32)
            for p, c in zip(pred_pts_list, pred_cls_list)
            if c == cls
        ]
        gt_cls_pts = [
            gt_pts[i].float()
            for i in range(M)
            if gt_labels[i].item() == cls
        ]

        if len(pred_cls_pts) == 0 or len(gt_cls_pts) == 0:
            if len(gt_cls_pts) > 0:
                chamfer_vals.append(1.0)
                frechet_vals.append(1.0)
            continue

        pred_all = torch.stack(pred_cls_pts)
        gt_all = torch.stack(gt_cls_pts)

        chamfer_vals.append(chamfer_distance(pred_all, gt_all).item())

        n_pairs = min(len(pred_cls_pts), len(gt_cls_pts))
        for i in range(n_pairs):
            frechet_vals.append(frechet_distance(pred_cls_pts[i], gt_cls_pts[i]).item())

    mean_chamfer = np.mean(chamfer_vals) if chamfer_vals else 1.0
    mean_frechet = np.mean(frechet_vals) if frechet_vals else 1.0

    return mean_chamfer, mean_frechet


# ──────────────────────────────────────────────
#  Training & Validation Loops
# ──────────────────────────────────────────────

def train_one_epoch(model, criterion, dataloader, optimizer, scheduler, scaler, device, epoch):
    model.train()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    epoch_losses = {
        'loss_total': 0, 'loss_cls': 0, 'loss_point': 0,
        'loss_curve': 0, 'loss_dir': 0,
    }
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [TRAIN]")
    for batch in pbar:
        images, gt_pts, gt_labels, num_instances, pixel_masks, paths = batch
        images = images.to(device)
        gt_pts = gt_pts.to(device)
        gt_labels = gt_labels.to(device)
        num_instances = num_instances.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
            pred_logits, pred_ctrl_pts, pred_restored_pts = model(images)
            loss_dict, total_loss = criterion(
                pred_logits, pred_ctrl_pts, pred_restored_pts, gt_labels, gt_pts, num_instances
            )

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        for k, v in loss_dict.items():
            if k in epoch_losses:
                epoch_losses[k] += v
        num_batches += 1

        pbar.set_postfix({
            'loss': f"{loss_dict['loss_total']:.4f}",
            'curve': f"{loss_dict['loss_curve']:.4f}",
            'dir': f"{loss_dict['loss_dir']:.4f}",
        })

    if scheduler is not None:
        scheduler.step()

    for k in epoch_losses:
        epoch_losses[k] /= max(num_batches, 1)

    return epoch_losses


@torch.no_grad()
def validate(model, dataloader, device, img_size=1024):
    model.eval()

    all_dice, all_iou, all_assd = [], [], []
    all_chamfer, all_frechet = [], []

    pbar = tqdm(dataloader, desc="[VAL]")
    for batch in pbar:
        images, gt_pts, gt_labels, num_instances, pixel_masks, paths = batch
        images = images.to(device)

        pred_logits, pred_ctrl_pts, pred_restored_pts = model(images)

        B = images.shape[0]
        for b in range(B):
            polylines, classes = extract_predictions(
                pred_logits[b], pred_restored_pts[b], conf_threshold=0.1
            )

            dice, iou, assd_val = compute_pixel_metrics(
                polylines, classes, pixel_masks[b], img_size
            )
            all_dice.append(dice)
            all_iou.append(iou)
            all_assd.append(assd_val)

            M = num_instances[b].item()
            chamfer, frechet = compute_vector_metrics(
                polylines, classes, gt_pts[b].cpu(), gt_labels[b].cpu(), M
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
#  Main Execution
# ──────────────────────────────────────────────

def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if HAS_WANDB and args.wandb_key:
        wandb.login(key=args.wandb_key)
        wandb_kwargs = {
            'project': args.wandb_project,
            'name': 'Surgical_BeMapTR_PureVector',
            'config': vars(args),
        }
        if args.wandb_entity:
            wandb_kwargs['entity'] = args.wandb_entity
        wandb.init(**wandb_kwargs)

    train_files, test_files, val_files = prepare_dataset.get_split(args.data_path)
    print(f"Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")

    train_dataset = VectorLandmarkDataset(train_files, N=args.num_queries, K=args.num_points)
    val_dataset = VectorLandmarkDataset(val_files, N=args.num_queries, K=args.num_points)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = SurgicalBeMapTR(
        img_size=1024, num_classes=4, N=args.num_queries, K_dense=args.num_points,
        bezier_k=3, bezier_n=3, embed_dim=args.embed_dim,
        num_decoder_layers=args.num_decoder_layers, coord_feat_size=args.coord_feat_size,
        pretrained_backbone=True
    ).to(device)

    criterion = SurgicalBeMapTRCriterion(
        num_classes=4, N=args.num_queries, K_dense=args.num_points,
        cls_weight=2.0, pts_weight=5.0, curve_weight=5.0, dir_weight=2.0
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')

    start_epoch = 1
    best_dice = 0.0

    if args.resume and os.path.exists(args.resume):
        print(f"Resuming training from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_dice = checkpoint.get('best_dice', 0.0)
        print(f"Resumed from epoch {start_epoch-1} with best Dice: {best_dice:.4f}")

    print("\nStarting Surgical-BeMapTR Training (Pure Vector Architecture)...")
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_losses = train_one_epoch(model, criterion, train_loader, optimizer, scheduler, scaler, device, epoch)
        val_metrics = validate(model, val_loader, device)
        elapsed = time.time() - t0

        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{args.epochs} ({elapsed:.1f}s)")
        print(f"  Train Loss: {train_losses['loss_total']:.4f} (curve={train_losses['loss_curve']:.4f}, point={train_losses['loss_point']:.4f}, dir={train_losses['loss_dir']:.4f})")
        print(f"  Val Dice: {val_metrics['val_dice']:.4f} | Val IoU: {val_metrics['val_iou']:.4f} | Val ASSD: {val_metrics['val_assd']:.2f}")
        print(f"  Val Chamfer: {val_metrics['val_chamfer']:.4f} | Val Fréchet: {val_metrics['val_frechet']:.4f}")
        print(f"{'='*70}\n")

        if HAS_WANDB and wandb.run is not None:
            wandb.log({'epoch': epoch, **train_losses, **val_metrics})

        # Save latest checkpoint after every epoch for seamless recovery
        latest_ckpt_path = os.path.join(args.output_dir, 'latest_surgical_bemaptr.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_dice': best_dice,
            'val_metrics': val_metrics,
        }, latest_ckpt_path)

        if val_metrics['val_dice'] > best_dice:
            best_dice = val_metrics['val_dice']
            ckpt_path = os.path.join(args.output_dir, 'best_surgical_bemaptr.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
                'val_metrics': val_metrics,
            }, ckpt_path)
            print(f"  ★ New best Dice: {best_dice:.4f} — saved to {ckpt_path}\n")

    print(f"\nTraining Complete! Best Val Dice: {best_dice:.4f}")


if __name__ == '__main__':
    main()
