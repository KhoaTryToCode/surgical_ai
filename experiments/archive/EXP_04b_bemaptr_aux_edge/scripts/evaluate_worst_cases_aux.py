"""
Worst Case Evaluation & Failure Analysis Script for Surgical-BeMapTR v3 (EXP_04b).

1. Identifies specific target images (e.g. Patient_40 failure cases from TopoNet benchmark).
2. Ranks all validation samples to find the WORST performing predictions of EXP_04b.
3. Generates multi-panel visual comparison figures showing RGB input, Ground Truth, EXP_04b Predictions, Aux Edge Heatmap, and Error Maps.
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

from surgical_bemaptr_aux import SurgicalBeMapTRAux

CLASS_NAMES = {1: 'Ridge', 2: 'Silhouette', 3: 'Ligament'}
CLASS_COLORS = {
    1: (0, 255, 0),      # Green: Ridge
    2: (255, 0, 0),      # Blue: Silhouette
    3: (0, 165, 255),    # Orange: Ligament
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Worst Cases & Target Patient Images for EXP_04b")
    parser.add_argument('--ckpt_path', type=str, default='checkpoints_bemaptr_aux/best_surgical_bemaptr_aux.pth')
    parser.add_argument('--data_path', type=str, default='/kaggle/working/L3D')
    parser.add_argument('--output_dir', type=str, default='/kaggle/working/worst_cases_analysis')
    parser.add_argument('--target_patient', type=str, default='Patient_40', help='Target patient ID or keyword to analyze')
    parser.add_argument('--top_k_worst', type=int, default=3, help='Number of worst predictions to plot')
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


def create_error_map(pred_mask, gt_mask_4ch, img_size=1024):
    gt_4ch = gt_mask_4ch.cpu().numpy()
    gt_class = np.argmax(gt_4ch, axis=0)

    pred_bin = (pred_mask > 0).astype(np.uint8)
    gt_bin = (gt_class > 0).astype(np.uint8)

    error_rgb = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    
    # Green: Match (True Positive)
    tp = (pred_bin == 1) & (gt_bin == 1)
    error_rgb[tp] = [0, 255, 0]

    # Red: False Positive (Predicted edge where no GT)
    fp = (pred_bin == 1) & (gt_bin == 0)
    error_rgb[fp] = [255, 0, 0]

    # Yellow: False Negative (Missed GT edge)
    fn = (pred_bin == 0) & (gt_bin == 1)
    error_rgb[fn] = [255, 255, 0]

    return error_rgb


def evaluate_and_plot(ckpt_path, data_path, output_dir, target_patient='Patient_40', top_k_worst=3, conf_threshold=0.05, device='cuda'):
    print("=" * 85)
    print(f"🔍 EVALUATING EXP_04b WORST CASES & TARGET PATIENT: {target_patient}")
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

    model = SurgicalBeMapTRAux(
        img_size=1024, num_classes=4, N=30, K_dense=20,
        bezier_k=3, bezier_n=3, embed_dim=256, coord_feat_size=coord_feat_size,
        pretrained_backbone=False
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    _, _, val_files = prepare_dataset.get_split(data_path)
    val_dataset = VectorLandmarkDataset(val_files, N=30, K=20)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    os.makedirs(output_dir, exist_ok=True)

    results = []

    print("\nEvaluating all validation samples...")
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            images, gt_pts, gt_labels, num_instances, pixel_masks, paths = batch
            images = images.to(device)

            pred_logits, pred_ctrl_pts, pred_restored_pts, aux_edge_logits = model(images)

            probs = F.softmax(pred_logits[0], dim=-1)
            pred_cls = torch.argmax(probs, dim=-1)
            top_prob = probs.max(dim=-1)[0]

            keep = (pred_cls > 0) & (top_prob > conf_threshold)
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

            path_str = str(paths[0])
            filename = Path(path_str).name
            is_target = target_patient.lower() in filename.lower()

            results.append({
                'index': i,
                'path': path_str,
                'filename': filename,
                'dice': dice,
                'iou': iou,
                'is_target': is_target,
                'pred_polylines': pred_polylines,
                'pred_classes': pred_classes,
                'pred_raster': pred_raster,
                'gt_pts': gt_pts[0],
                'gt_labels': gt_labels[0],
                'num_instances': num_instances[0].item(),
                'pixel_masks': pixel_masks[0],
                'aux_edge_logits': aux_edge_logits[0].cpu(),
            })

    # Sort results by Dice ascending (worst first)
    results_sorted = sorted(results, key=lambda x: x['dice'])

    # Find target patient cases (e.g. Patient_40)
    target_results = [r for r in results if r['is_target']]
    if not target_results:
        # Fallback to any matching sample
        target_results = results_sorted[:top_k_worst]

    print(f"\nFound {len(target_results)} target patient samples for '{target_patient}'.")

    def save_5panel_figure(item, title_prefix, out_filename):
        raw_img = cv2.imread(item['path'])
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        raw_img = cv2.resize(raw_img, (1024, 1024))

        M = item['num_instances']
        gt_polylines = item['gt_pts'][:M].cpu().numpy()
        gt_classes = item['gt_labels'][:M].cpu().numpy().tolist()

        vis_gt = draw_polylines_on_image(raw_img, gt_polylines, gt_classes, thickness=4)
        vis_pred = draw_polylines_on_image(raw_img, item['pred_polylines'], item['pred_classes'], thickness=4)

        aux_probs = torch.sigmoid(item['aux_edge_logits']).numpy()
        aux_vis = np.max(aux_probs[1:], axis=0)
        aux_vis = (aux_vis * 255).astype(np.uint8)
        aux_vis = cv2.applyColorMap(aux_vis, cv2.COLORMAP_JET)
        aux_vis = cv2.resize(aux_vis, (1024, 1024))

        error_map = create_error_map(item['pred_raster'], item['pixel_masks'])

        fig, axes = plt.subplots(1, 5, figsize=(26, 5.5))

        axes[0].imshow(raw_img)
        axes[0].set_title(f"{title_prefix}\n{item['filename']}", fontsize=9)
        axes[0].axis('off')

        axes[1].imshow(vis_gt)
        axes[1].set_title(f"Ground Truth ({M} lines)\n(Green:Ridge, Blue:Silh, Orange:Lig)", fontsize=9)
        axes[1].axis('off')

        axes[2].imshow(vis_pred)
        axes[2].set_title(f"BeMapTR v3 Vector Pred\nDice: {item['dice']:.4f} | IoU: {item['iou']:.4f}", fontsize=9)
        axes[2].axis('off')

        axes[3].imshow(aux_vis)
        axes[3].set_title("Branch B Aux Edge Heatmap\n(Pixel-Level Activations)", fontsize=9)
        axes[3].axis('off')

        axes[4].imshow(error_map)
        axes[4].set_title("Error Map\n(Green:Match, Red:FP, Yellow:FN)", fontsize=9)
        axes[4].axis('off')

        plt.tight_layout()
        out_path = os.path.join(output_dir, out_filename)
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f" Saved figure: {out_path}")

    # 1. Plot Target Patient Cases (e.g. Patient_40 TopoNet Failure Cases)
    print("\n--- Plotting Target Patient Failure Analysis ---")
    for idx, item in enumerate(target_results[:5]):
        save_5panel_figure(item, f"Target Case {idx+1}", f"target_{target_patient}_{idx+1}_{item['filename']}.png")

    # 2. Plot Top-K Worst Cases of EXP_04b overall
    print(f"\n--- Plotting Top-{top_k_worst} Worst Predictions of EXP_04b ---")
    for rank, item in enumerate(results_sorted[:top_k_worst]):
        save_5panel_figure(item, f"Rank {rank+1} Worst", f"worst_rank_{rank+1}_{item['filename']}.png")

    print(f"\n✅ All analysis figures saved to: {output_dir}\n")


if __name__ == '__main__':
    args = parse_args()
    evaluate_and_plot(
        args.ckpt_path, args.data_path, args.output_dir,
        target_patient=args.target_patient, top_k_worst=args.top_k_worst,
        conf_threshold=args.conf_threshold, device=args.device
    )
