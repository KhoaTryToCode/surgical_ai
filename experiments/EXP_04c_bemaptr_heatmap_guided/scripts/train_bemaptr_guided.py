"""
Training Script for Surgical-BeMapTR v4 (EXP_04c: Heatmap-Guided Vector Prompt Learning).

Features:
1. Feature Map Modulation: Passes Branch B physical edge heatmap directly into Branch A forward pass.
2. Relative Anchoring: Refines reference points using relative offsets.
3. Evaluates both Vector Metrics (Chamfer/Fréchet) and Rasterized Mask Metrics (Dice/IoU/ASSD).
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

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(os.path.dirname(EXP_DIR))

# Search paths for imports
for p in [
    SCRIPT_DIR,
    os.path.join(EXP_DIR, 'models'),
    os.path.join(REPO_ROOT, 'experiments', 'EXP_04b_bemaptr_aux_edge', 'models'),
    os.path.join(REPO_ROOT, 'experiments', 'EXP_04_surgical_bemaptr_pure_vector', 'models'),
    os.path.join(REPO_ROOT, 'shared'),
    REPO_ROOT,
]:
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

from surgical_bemaptr_guided import SurgicalBeMapTRGuided
from vector_losses import SurgicalBeMapTRCriterion, chamfer_distance, frechet_distance

try:
    from medpy.metric import assd
except ImportError:
    assd = None

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def compute_aux_edge_loss(aux_edge_logits, pixel_masks):
    target = F.interpolate(pixel_masks.float(), size=aux_edge_logits.shape[-2:], mode='nearest')
    bce_loss = F.binary_cross_entropy_with_logits(aux_edge_logits, target)
    probs = torch.sigmoid(aux_edge_logits)
    intersection = (probs * target).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice_loss = 1.0 - (2.0 * intersection + 1.0) / (union + 1.0)
    return bce_loss + dice_loss.mean()


def parse_args():
    parser = argparse.ArgumentParser(description="Train Surgical-BeMapTR v4 (EXP_04c Guided)")
    parser.add_argument('--data_path', type=str, default='/kaggle/working/L3D')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--save_dir', type=str, default='checkpoints_bemaptr_guided')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='surgical_bemaptr_guided')
    return parser.parse_args()


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss_epoch = 0.0
    total_vec_loss_epoch = 0.0
    total_aux_loss_epoch = 0.0

    pbar = tqdm(dataloader, desc="[TRAIN]")
    for batch in pbar:
        images, gt_pts, gt_labels, num_instances, pixel_masks, _ = batch
        images = images.to(device)
        gt_pts = gt_pts.to(device)
        gt_labels = gt_labels.to(device)
        num_instances = num_instances.to(device)
        pixel_masks = pixel_masks.to(device)

        optimizer.zero_grad()
        intermediate_logits, intermediate_ctrl_pts, intermediate_restored_pts, aux_edge_logits = model(images)

        targets = []
        for b in range(images.shape[0]):
            M = num_instances[b].item()
            targets.append({
                'pts': gt_pts[b, :M],
                'labels': gt_labels[b, :M],
            })

        loss_dict = criterion(
            intermediate_logits,
            intermediate_ctrl_pts,
            intermediate_restored_pts,
            targets
        )
        vec_loss = loss_dict['loss_total']
        aux_loss = compute_aux_edge_loss(aux_edge_logits, pixel_masks)
        loss = vec_loss + 1.0 * aux_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss_epoch += loss.item()
        total_vec_loss_epoch += vec_loss.item()
        total_aux_loss_epoch += aux_loss.item()

        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'vec': f"{vec_loss.item():.4f}",
            'aux': f"{aux_loss.item():.4f}",
        })

    num_batches = len(dataloader)
    return total_loss_epoch / num_batches, total_vec_loss_epoch / num_batches, total_aux_loss_epoch / num_batches


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()

    total_loss_epoch = 0.0
    all_dice, all_iou, all_assd = [], [], []
    all_chamfer, all_frechet = [], []

    for batch in dataloader:
        images, gt_pts, gt_labels, num_instances, pixel_masks, _ = batch
        images = images.to(device)
        gt_pts = gt_pts.to(device)
        gt_labels = gt_labels.to(device)
        num_instances = num_instances.to(device)
        pixel_masks = pixel_masks.to(device)

        pred_logits, pred_ctrl_pts, pred_restored_pts, aux_edge_logits = model(images)

        targets = []
        for b in range(images.shape[0]):
            M = num_instances[b].item()
            targets.append({
                'pts': gt_pts[b, :M],
                'labels': gt_labels[b, :M],
            })

        loss_dict = criterion(
            [pred_logits], [pred_ctrl_pts], [pred_restored_pts], targets
        )
        vec_loss = loss_dict['loss_total']
        aux_loss = compute_aux_edge_loss(aux_edge_logits, pixel_masks)
        loss = vec_loss + 1.0 * aux_loss
        total_loss_epoch += loss.item()

        probs = F.softmax(pred_logits[0], dim=-1)
        pred_cls = torch.argmax(probs, dim=-1)
        top_prob = probs.max(dim=-1)[0]
        keep = (pred_cls > 0) & (top_prob > 0.05)

        pred_polylines = pred_restored_pts[0][keep].cpu().numpy()
        pred_classes = pred_cls[keep].cpu().numpy().tolist()

        pred_raster = rasterize_normalized_polylines(
            pred_polylines, pred_classes, H=1024, W=1024, thickness=19
        )

        gt_4ch = pixel_masks[0].cpu().numpy()
        gt_class = np.argmax(gt_4ch, axis=0)

        pred_bin = np.array([pred_raster == c for c in range(4)]).astype(np.uint8)
        gt_bin = np.array([gt_class == c for c in range(4)]).astype(np.uint8)

        iou, dice = evaluation(pred_bin[1:].flatten(), gt_bin[1:].flatten())
        all_dice.append(dice)
        all_iou.append(iou)

        if assd is not None and pred_bin[1:].sum() > 0 and gt_bin[1:].sum() > 0:
            try:
                assd_val = assd(pred_bin[1:], gt_bin[1:])
                all_assd.append(assd_val)
            except Exception:
                pass

        gt_poly_list = gt_pts[0, :num_instances[0]].cpu().numpy()
        if len(pred_polylines) > 0 and len(gt_poly_list) > 0:
            for p_pred in pred_polylines:
                min_cd = min(chamfer_distance(p_pred, p_gt) for p_gt in gt_poly_list)
                min_fd = min(frechet_distance(p_pred, p_gt) for p_gt in gt_poly_list)
                all_chamfer.append(min_cd)
                all_frechet.append(min_fd)

    mean_loss = total_loss_epoch / len(dataloader)
    mean_dice = np.mean(all_dice) if all_dice else 0.0
    mean_iou = np.mean(all_iou) if all_iou else 0.0
    mean_assd = np.mean(all_assd) if all_assd else 999.0
    mean_chamfer = np.mean(all_chamfer) if all_chamfer else 1.0
    mean_frechet = np.mean(all_frechet) if all_frechet else 1.0

    return {
        'val_loss': mean_loss,
        'val_dice': mean_dice,
        'val_iou': mean_iou,
        'val_assd': mean_assd,
        'val_chamfer': mean_chamfer,
        'val_frechet': mean_frechet,
    }


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print("=" * 80)
    print("Starting Surgical-BeMapTR v4 Training (EXP_04c Heatmap-Guided)...")
    print(f"Device: {device} | Epochs: {args.epochs} | Batch Size: {args.batch_size}")
    print("=" * 80)

    train_files, test_files, val_files = prepare_dataset.get_split(args.data_path)
    print(f"Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")

    train_dataset = VectorLandmarkDataset(train_files, N=30, K=20)
    val_dataset = VectorLandmarkDataset(val_files, N=30, K=20)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    model = SurgicalBeMapTRGuided(
        img_size=1024, num_classes=4, N=30, K_dense=20,
        bezier_k=3, bezier_n=3, embed_dim=256, num_heads=8,
        num_decoder_layers=6, coord_feat_dim=64, coord_feat_size=64,
        pretrained_backbone=True
    ).to(device)

    criterion = SurgicalBeMapTRCriterion(
        num_classes=4,
        cls_weight=2.0,
        pts_weight=5.0,
        curve_weight=5.0,
        dir_weight=2.0,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    os.makedirs(args.save_dir, exist_ok=True)
    best_dice = 0.0

    if args.wandb and HAS_WANDB:
        wandb.init(project=args.wandb_project, config=vars(args))

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, vec_loss, aux_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"\nEpoch {epoch}/{args.epochs} ({elapsed:.1f}s)")
        print(f"  Train Loss: {train_loss:.4f} (vec={vec_loss:.4f}, aux={aux_loss:.4f})")
        print(f"  Val Dice  : {val_metrics['val_dice']:.4f} | IoU: {val_metrics['val_iou']:.4f} | ASSD: {val_metrics['val_assd']:.2f}px")
        print(f"  Val Chamfer: {val_metrics['val_chamfer']:.4f} | Fréchet: {val_metrics['val_frechet']:.4f}\n")

        if val_metrics['val_dice'] > best_dice:
            best_dice = val_metrics['val_dice']
            ckpt_path = os.path.join(args.save_dir, 'best_surgical_bemaptr_guided.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
                'val_metrics': val_metrics,
            }, ckpt_path)
            print(f" Saved New Best Checkpoint: {ckpt_path} (Val Dice: {best_dice:.4f})")

        latest_path = os.path.join(args.save_dir, 'latest_surgical_bemaptr_guided.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_metrics': val_metrics,
        }, latest_path)

        if args.wandb and HAS_WANDB:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_vec_loss': vec_loss,
                'train_aux_loss': aux_loss,
                **val_metrics,
            })

    print(f"\nTraining Complete! Best Val Dice: {best_dice:.4f}")


if __name__ == '__main__':
    main()
