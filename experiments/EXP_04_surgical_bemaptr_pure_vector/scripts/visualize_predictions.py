"""
Visualization & Prediction Inspection Script for Surgical-BeMapTR (EXP_04).

Loads a trained Surgical-BeMapTR checkpoint, runs inference on validation samples,
and outputs:
1. Printed query probability distribution & raw confidence scores.
2. Side-by-side visual overlays:
   [ RGB Surgical Image | Ground Truth Polylines | Predicted Vector Polylines ]
   Saved to val_visualizations/
"""

import argparse
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
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
    from utils import prepare_dataset
except ImportError:
    from shared.utils.vector_dataset import VectorLandmarkDataset, rasterize_normalized_polylines
    from shared.utils import prepare_dataset

from surgical_bemaptr import SurgicalBeMapTR


# Color mapping for classes: 1: Ridge (Green), 2: Silhouette (Blue), 3: Ligament (Red)
CLASS_COLORS = {
    1: (0, 255, 0),    # Ridge -> Green
    2: (255, 0, 0),    # Silhouette -> Blue
    3: (0, 0, 255),    # Ligament -> Red
}
CLASS_NAMES = {1: 'Ridge', 2: 'Silhouette', 3: 'Ligament'}


def draw_polylines_on_img(img_bgr, polylines_norm, classes, thickness=4):
    """Draws normalized [0, 1] polylines directly onto a BGR image."""
    H, W = img_bgr.shape[:2]
    canvas = img_bgr.copy()

    for pts_norm, cls in zip(polylines_norm, classes):
        if cls not in CLASS_COLORS:
            continue
        color = CLASS_COLORS[cls]
        pts_px = (pts_norm * np.array([W, H])).astype(np.int32)

        for i in range(len(pts_px) - 1):
            cv2.line(canvas, tuple(pts_px[i]), tuple(pts_px[i + 1]), color, thickness)
            cv2.circle(canvas, tuple(pts_px[i]), 3, (255, 255, 255), -1)

    return canvas


def inspect_and_visualize(ckpt_path, data_path, output_dir='val_visualizations', num_samples=5, device='cuda'):
    print("=" * 85)
    print(f"🔍 VISUALIZING SURGICAL-BEMAPTR PREDICTIONS FROM: {ckpt_path}")
    print("=" * 85)

    if not os.path.exists(ckpt_path):
        print(f"❌ Checkpoint file not found: {ckpt_path}")
        return

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 1. Load Checkpoint (with weights_only=False for PyTorch 2.6 compatibility)
    try:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location=device)
    print(f"Loaded Epoch {checkpoint.get('epoch', 'N/A')} checkpoint with Best Val Dice: {checkpoint.get('best_dice', 0.0):.4f}")

    # 2. Build Model (auto-detecting coord_feat_size from checkpoint)
    state_dict = checkpoint['model_state_dict']
    coord_feat_size = 64
    if 'spatial_coord_head.coords' in state_dict:
        coord_feat_size = state_dict['spatial_coord_head.coords'].shape[-1]
    print(f"Auto-detected spatial coord grid size: {coord_feat_size}x{coord_feat_size}")

    model = SurgicalBeMapTR(
        img_size=1024, num_classes=4, N=30, K_dense=20,
        bezier_k=3, bezier_n=3, embed_dim=256, coord_feat_size=coord_feat_size,
        pretrained_backbone=False
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # 3. Load Validation Split
    _, _, val_files = prepare_dataset.get_split(data_path)
    val_dataset = VectorLandmarkDataset(val_files, N=30, K=20)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    os.makedirs(output_dir, exist_ok=True)

    sample_count = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if sample_count >= num_samples:
                break

            images, gt_pts, gt_labels, num_instances, pixel_masks, paths = batch
            images = images.to(device)
            img_path = paths[0]
            base_name = os.path.splitext(os.path.basename(img_path))[0]

            # Run Inference
            pred_logits, pred_ctrl_pts, pred_restored_pts = model(images)
            pred_logits = pred_logits[0]            # (30, 4)
            pred_restored = pred_restored_pts[0]    # (30, 20, 2)

            probs = F.softmax(pred_logits, dim=-1)  # (30, 4)
            fg_probs = probs[:, 1:]                 # (30, 3) for classes 1, 2, 3
            max_fg_prob, max_fg_cls = torch.max(fg_probs, dim=-1)
            max_fg_cls = max_fg_cls + 1            # Convert 0,1,2 -> 1,2,3

            print("\n" + "-" * 75)
            print(f"📷 Sample [{sample_count+1}/{num_samples}]: {base_name}")
            print("-" * 75)
            print(f"GT Instances: {num_instances[0].item()}")

            # Print top 10 query probabilities
            print("\n  Top Query Confidence Scores:")
            sorted_indices = torch.argsort(max_fg_prob, descending=True)
            for rank in range(min(10, len(sorted_indices))):
                idx = sorted_indices[rank].item()
                bg_p = probs[idx, 0].item()
                fg_p = max_fg_prob[idx].item()
                cls_name = CLASS_NAMES[max_fg_cls[idx].item()]
                print(f"   • Query #{idx:02d}: P(Background)={bg_p:.3f} | P({cls_name})={fg_p:.3f} ──► Argmax Class = {torch.argmax(probs[idx]).item()}")

            # Extract Predictions using 2 strategies:
            # Strategy A: Conf > 0.05
            keep_a = max_fg_prob > 0.05
            pred_polys_a = pred_restored[keep_a].cpu().numpy()
            pred_cls_a = max_fg_cls[keep_a].cpu().numpy().tolist()

            # Strategy B: Conf > 0.15
            keep_b = max_fg_prob > 0.15
            pred_polys_b = pred_restored[keep_b].cpu().numpy()
            pred_cls_b = max_fg_cls[keep_b].cpu().numpy().tolist()

            # Load original image for visualization
            raw_img = cv2.imread(img_path)
            raw_img = cv2.resize(raw_img, (1024, 1024))

            # GT polylines
            M = num_instances[0].item()
            gt_polys = gt_pts[0, :M].cpu().numpy()
            gt_cls = gt_labels[0, :M].cpu().numpy().tolist()

            img_gt = draw_polylines_on_img(raw_img, gt_polys, gt_cls, thickness=3)
            img_pred_a = draw_polylines_on_img(raw_img, pred_polys_a, pred_cls_a, thickness=3)
            img_pred_b = draw_polylines_on_img(raw_img, pred_polys_b, pred_cls_b, thickness=3)

            # Add labels
            cv2.putText(img_gt, "Ground Truth", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            cv2.putText(img_pred_a, f"Pred (Conf > 0.05, N={len(pred_cls_a)})", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.putText(img_pred_b, f"Pred (Conf > 0.15, N={len(pred_cls_b)})", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

            # Combine side by side
            comparison = np.hstack([img_gt, img_pred_a, img_pred_b])
            out_file = os.path.join(output_dir, f"sample_{sample_count+1}_{base_name}.png")
            cv2.imwrite(out_file, comparison)
            print(f"\n  📸 Visual Comparison Saved to: {out_file}")

            sample_count += 1

    print("\n" + "=" * 85)
    print("✅ VISUALIZATION COMPLETE!")
    print("=" * 85)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inspect & Visualize Predictions")
    parser.add_argument('--ckpt_path', type=str, default='checkpoints_bemaptr/best_surgical_bemaptr.pth', help='Path to checkpoint')
    parser.add_argument('--data_path', type=str, default='/kaggle/working/L3D', help='Path to dataset root')
    parser.add_argument('--output_dir', type=str, default='val_visualizations', help='Output folder for visualization images')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of val samples to visualize')
    args = parser.parse_args()

    inspect_and_visualize(args.ckpt_path, args.data_path, args.output_dir, args.num_samples)
