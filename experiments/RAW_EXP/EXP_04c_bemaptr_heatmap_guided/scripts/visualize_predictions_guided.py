"""
Visualization Script for Surgical-BeMapTR v4 (EXP_04c Guided Heatmap Model).

Extracts predicted polyline curves and guided edge heatmaps, overlays them on surgical images,
and saves comparison figures.
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(os.path.dirname(EXP_DIR))

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
    from utils.prepare_dataset import get_split
except ImportError:
    from shared.utils.vector_dataset import VectorLandmarkDataset, rasterize_normalized_polylines
    from shared.utils.prepare_dataset import get_split

from surgical_bemaptr_guided import SurgicalBeMapTRGuided

CLASS_COLORS = {
    1: (0, 255, 0),      # Green: Ridge
    2: (255, 0, 0),      # Blue: Silhouette
    3: (0, 165, 255),    # Orange: Ligament
}


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Surgical-BeMapTR v4 Guided Predictions")
    parser.add_argument('--ckpt_path', type=str, default='checkpoints_bemaptr_guided/best_surgical_bemaptr_guided.pth')
    parser.add_argument('--data_path', type=str, default='/kaggle/working/L3D')
    parser.add_argument('--output_dir', type=str, default='/kaggle/working/vis_results_guided')
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--conf_threshold', type=float, default=0.05)
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()


def draw_polylines_on_image(img_rgb, polylines, classes, thickness=3):
    vis_img = img_rgb.copy()
    H, W, _ = vis_img.shape

    for poly, cls_id in zip(polylines, classes):
        color = CLASS_COLORS.get(cls_id, (255, 255, 255))
        pts_px = (poly * np.array([W, H])).astype(np.int32)
        cv2.polylines(vis_img, [pts_px], isClosed=False, color=color, thickness=thickness)

    return vis_img


def inspect_and_visualize(ckpt_path, data_path, output_dir, num_samples=5, conf_threshold=0.05, device='cuda'):
    print("=" * 85)
    print(f"🔍 VISUALIZING SURGICAL-BEMAPTR v4 GUIDED PREDICTIONS FROM: {ckpt_path}")
    print("=" * 85)

    if not os.path.exists(ckpt_path):
        print(f"❌ Checkpoint file not found: {ckpt_path}")
        return

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    try:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location=device)

    print(f"Loaded Epoch {checkpoint.get('epoch', 'N/A')} checkpoint with Best Val Dice: {checkpoint.get('best_dice', 0.0):.4f}")

    state_dict = checkpoint['model_state_dict']
    coord_feat_size = 64
    if 'spatial_coord_head.coords' in state_dict:
        coord_feat_size = state_dict['spatial_coord_head.coords'].shape[-1]
    print(f"Auto-detected spatial coord grid size: {coord_feat_size}x{coord_feat_size}")

    model = SurgicalBeMapTRGuided(
        img_size=1024, num_classes=4, N=30, K_dense=20,
        bezier_k=3, bezier_n=3, embed_dim=256, coord_feat_size=coord_feat_size,
        pretrained_backbone=False
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    _, _, val_files = get_split(data_path)
    val_dataset = VectorLandmarkDataset(val_files, N=30, K=20)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    os.makedirs(output_dir, exist_ok=True)
    sample_count = 0

    with torch.no_grad():
        for batch in val_loader:
            if sample_count >= num_samples:
                break

            images, gt_pts, gt_labels, num_instances, pixel_masks, paths = batch
            images = images.to(device)

            pred_logits, pred_ctrl_pts, pred_restored_pts, aux_edge_logits = model(images)

            raw_img = cv2.imread(paths[0])
            raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
            raw_img = cv2.resize(raw_img, (1024, 1024))

            M = num_instances[0].item()
            gt_polylines = gt_pts[0, :M].cpu().numpy()
            gt_classes = gt_labels[0, :M].cpu().numpy().tolist()

            probs = F.softmax(pred_logits[0], dim=-1)
            pred_cls = torch.argmax(probs, dim=-1)
            top_prob = probs.max(dim=-1)[0]

            keep = (pred_cls > 0) & (top_prob > conf_threshold)
            pred_polylines = pred_restored_pts[0][keep].cpu().numpy()
            pred_classes = pred_cls[keep].cpu().numpy().tolist()

            vis_gt = draw_polylines_on_image(raw_img, gt_polylines, gt_classes, thickness=4)
            vis_pred = draw_polylines_on_image(raw_img, pred_polylines, pred_classes, thickness=4)

            aux_probs = torch.sigmoid(aux_edge_logits[0]).cpu().numpy()
            aux_vis = np.max(aux_probs[1:], axis=0)
            aux_vis = (aux_vis * 255).astype(np.uint8)
            aux_vis = cv2.applyColorMap(aux_vis, cv2.COLORMAP_JET)
            aux_vis = cv2.resize(aux_vis, (1024, 1024))

            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            axes[0].imshow(vis_gt)
            axes[0].set_title(f"Ground Truth ({M} lines)", fontsize=11)
            axes[0].axis('off')

            axes[1].imshow(vis_pred)
            axes[1].set_title(f"Surgical-BeMapTR v4 Guided ({len(pred_classes)} lines)", fontsize=11)
            axes[1].axis('off')

            axes[2].imshow(aux_vis)
            axes[2].set_title("Guided Heatmap Feature Map (P2_guided)", fontsize=11)
            axes[2].axis('off')

            plt.tight_layout()
            out_file = os.path.join(output_dir, f"sample_{sample_count+1}_{Path(paths[0]).stem}.png")
            plt.savefig(out_file, dpi=150, bbox_inches='tight')
            plt.close()

            print(f" Saved visualization [{sample_count+1}/{num_samples}]: {out_file}")
            sample_count += 1

    print(f"\n Visualizations completed! Results saved to: {output_dir}\n")


if __name__ == '__main__':
    args = parse_args()
    inspect_and_visualize(args.ckpt_path, args.data_path, args.output_dir, args.num_samples, args.conf_threshold, args.device)
