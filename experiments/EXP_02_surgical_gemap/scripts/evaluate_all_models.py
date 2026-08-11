"""
Unified Evaluation Script for All Models.

Evaluates TopoNet, Mask2Former variants, and Surgical-GeMap on the
validation set using BOTH pixel metrics and vector metrics.

Produces a single comparison table.

Usage:
    python evaluate_all_models.py \
        --data_path /kaggle/working/L3D \
        --toponet_ckpt checkpoints/best_toponet.pth \
        --gemap_ckpt checkpoints_gemap/best_surgical_gemap.pth \
        --mask2former_swin_ckpt checkpoints/best_mask2former_swin.pth
"""

import argparse
import os
import sys
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from utils.vector_dataset import VectorLandmarkDataset, rasterize_normalized_polylines
from utils.pixel_to_vector import prediction_mask_to_vectors
from utils.metrics import evaluation
from utils import prepare_dataset
from models.vector_losses import chamfer_distance, frechet_distance

try:
    from medpy.metric import assd
except ImportError:
    assd = None


# ──────────────────────────────────────────────
#  Shared Evaluation Functions
# ──────────────────────────────────────────────

def compute_pixel_metrics_from_mask(pred_mask, gt_mask_4ch, img_size=1024):
    """Compute Dice, IoU, ASSD from pixel masks."""
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


def compute_vector_metrics_from_polylines(pred_polylines, pred_classes,
                                          gt_pts, gt_labels, num_instances, K=20):
    """Compute Chamfer and Fréchet distances between polyline sets."""
    M = num_instances
    chamfer_vals = []
    frechet_vals = []

    for cls in [1, 2, 3]:
        pred_cls_pts = [
            torch.tensor(p, dtype=torch.float32)
            for p, c in zip(pred_polylines, pred_classes)
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
#  Evaluate Surgical-GeMap
# ──────────────────────────────────────────────

def evaluate_surgical_gemap(ckpt_path, val_loader, device, img_size=1024):
    """Evaluate Surgical-GeMap model."""
    from models.surgical_gemap import SurgicalGeMap
    from train_surgical_gemap import extract_predictions

    checkpoint = torch.load(ckpt_path, map_location=device)
    model_args = checkpoint.get('args', {})

    model = SurgicalGeMap(
        img_size=model_args.get('img_size', img_size),
        num_classes=model_args.get('num_classes', 4),
        N=model_args.get('N', 30),
        K=model_args.get('K', 20),
        embed_dim=model_args.get('embed_dim', 256),
        num_heads=model_args.get('num_heads', 8),
        num_decoder_layers=model_args.get('num_decoder_layers', 6),
        pretrained_backbone=False,
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_dice, all_iou, all_assd = [], [], []
    all_chamfer, all_frechet = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Surgical-GeMap"):
            images, gt_pts, gt_labels, num_instances, pixel_masks, paths = batch
            images = images.to(device)

            pred_logits, pred_pts = model(images)

            B = images.shape[0]
            for b in range(B):
                polylines, classes = extract_predictions(
                    pred_logits[b], pred_pts[b], conf_threshold=0.3
                )

                # Pixel metrics
                if len(polylines) > 0:
                    pred_mask = rasterize_normalized_polylines(
                        polylines, classes, H=img_size, W=img_size, thickness=35
                    )
                else:
                    pred_mask = np.zeros((img_size, img_size), dtype=np.uint8)

                dice, iou, assd_val = compute_pixel_metrics_from_mask(
                    pred_mask, pixel_masks[b], img_size
                )
                all_dice.append(dice)
                all_iou.append(iou)
                all_assd.append(assd_val)

                # Vector metrics
                M = num_instances[b].item()
                chamfer, frechet = compute_vector_metrics_from_polylines(
                    polylines, classes,
                    gt_pts[b].cpu(), gt_labels[b].cpu(), M
                )
                all_chamfer.append(chamfer)
                all_frechet.append(frechet)

    return {
        'Dice': np.mean(all_dice),
        'IoU': np.mean(all_iou),
        'ASSD': np.mean(all_assd),
        'Chamfer': np.mean(all_chamfer),
        'Frechet': np.mean(all_frechet),
    }


# ──────────────────────────────────────────────
#  Evaluate Pixel-Based Model (TopoNet or Mask2Former)
# ──────────────────────────────────────────────

def evaluate_pixel_model(model, val_loader, device, model_name, img_size=1024,
                         is_mask2former=False):
    """
    Evaluate a pixel-based model with both pixel and vector metrics.
    For vector metrics, predictions are converted via pixel-to-vector.
    """
    model.eval()
    all_dice, all_iou, all_assd = [], [], []
    all_chamfer, all_frechet = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=model_name):
            images, gt_pts, gt_labels, num_instances, pixel_masks, paths = batch
            images = images.to(device)

            if is_mask2former:
                # Mask2Former returns logits directly
                output = model(images)
                pred_mask = torch.argmax(
                    torch.softmax(output, dim=1), dim=1
                ).cpu().numpy()[0]
            else:
                # TopoNet returns (output, aux)
                dummy_depth = torch.zeros(images.shape[0], 1, img_size, img_size).to(device)
                output, _ = model(images)
                pred_mask = torch.argmax(
                    torch.softmax(output, dim=1), dim=1
                ).cpu().numpy()[0]

            pred_mask = pred_mask.astype(np.uint8)

            # Pixel metrics
            dice, iou, assd_val = compute_pixel_metrics_from_mask(
                pred_mask, pixel_masks[0], img_size
            )
            all_dice.append(dice)
            all_iou.append(iou)
            all_assd.append(assd_val)

            # Vector metrics (via pixel-to-vector conversion)
            pred_polylines, pred_classes = prediction_mask_to_vectors(
                pred_mask, K=20, normalize=True
            )
            M = num_instances[0].item()
            chamfer, frechet = compute_vector_metrics_from_polylines(
                pred_polylines, pred_classes,
                gt_pts[0].cpu(), gt_labels[0].cpu(), M
            )
            all_chamfer.append(chamfer)
            all_frechet.append(frechet)

    return {
        'Dice': np.mean(all_dice),
        'IoU': np.mean(all_iou),
        'ASSD': np.mean(all_assd),
        'Chamfer': np.mean(all_chamfer),
        'Frechet': np.mean(all_frechet),
    }


# ──────────────────────────────────────────────
#  Print Comparison Table
# ──────────────────────────────────────────────

def print_comparison_table(results_dict):
    """Print a formatted comparison table."""
    print("\n" + "=" * 90)
    print("UNIFIED EVALUATION RESULTS (Validation Set)")
    print("=" * 90)
    print(f"{'Model':<30} {'Dice↑':>8} {'IoU↑':>8} {'ASSD↓':>8} "
          f"{'Chamfer↓':>10} {'Fréchet↓':>10}")
    print("-" * 90)

    for name, metrics in results_dict.items():
        print(f"{name:<30} {metrics['Dice']:>8.4f} {metrics['IoU']:>8.4f} "
              f"{metrics['ASSD']:>8.2f} {metrics['Chamfer']:>10.4f} "
              f"{metrics['Frechet']:>10.4f}")

    print("=" * 90)
    print("Pixel metrics (Dice, IoU, ASSD): computed on 1024×1024 rasterized masks")
    print("Vector metrics (Chamfer, Fréchet): computed on K=20 normalized polylines")
    print("  - Pixel models: predictions converted via skeletonization + resampling")
    print("  - Vector models: predictions used directly")
    print("=" * 90)


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/kaggle/working/L3D')
    parser.add_argument('--img_size', type=int, default=1024)
    parser.add_argument('--gemap_ckpt', type=str, default=None,
                        help='Path to Surgical-GeMap checkpoint')
    parser.add_argument('--toponet_ckpt', type=str, default=None,
                        help='Path to TopoNet checkpoint')
    parser.add_argument('--mask2former_swin_ckpt', type=str, default=None,
                        help='Path to Mask2Former Swin-Tiny checkpoint')
    parser.add_argument('--mask2former_resnet_ckpt', type=str, default=None,
                        help='Path to Mask2Former ResNet-50 checkpoint')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset
    _, _, val_files = prepare_dataset.get_split(args.data_path)
    val_dataset = VectorLandmarkDataset(
        val_files, N=30, K=20, img_size=args.img_size, mode='val'
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    results = {}

    # ── Surgical-GeMap ──
    if args.gemap_ckpt and os.path.exists(args.gemap_ckpt):
        print("\n🔬 Evaluating Surgical-GeMap (Swin-Tiny)...")
        results['Surgical-GeMap (Swin-T)'] = evaluate_surgical_gemap(
            args.gemap_ckpt, val_loader, device, args.img_size
        )

    # ── TopoNet ──
    if args.toponet_ckpt and os.path.exists(args.toponet_ckpt):
        print("\n🔬 Evaluating TopoNet (ResNet-34 + DAv2)...")
        from models.TopoNet import TopoNet
        toponet = TopoNet(args.img_size, args.img_size).to(device)
        ckpt = torch.load(args.toponet_ckpt, map_location=device)
        toponet.load_state_dict(ckpt)
        results['TopoNet (ResNet-34+DAv2)'] = evaluate_pixel_model(
            toponet, val_loader, device, 'TopoNet', args.img_size
        )

    # Print results
    if results:
        print_comparison_table(results)
    else:
        print("\nNo model checkpoints provided. Use --gemap_ckpt, --toponet_ckpt, etc.")


if __name__ == '__main__':
    main()
