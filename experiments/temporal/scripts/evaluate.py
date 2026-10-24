"""
Evaluation & Inspection Script for MinVIS (Experiment 17).
Evaluates a trained fold checkpoint on Validation or Frozen Test splits,
and generates multi-panel qualitative inspection images.
"""

import os
import sys
import argparse
import json
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
EXPERIMENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(EXPERIMENT_ROOT))

from shared.utils.split_loader import get_fold_patients, get_frozen_test_patients
from models.minvis_mask2former import MinVISMask2Former
from models.tracker import MinVISTracker
from utils.dataset import PatientVideoL3DDataset, resolve_split_dir
from utils.metrics import evaluate_frame_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MinVIS Checkpoint")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint .pth')
    parser.add_argument('--fold_idx', type=int, default=0, help='Fold index (0 to 4)')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'], help='Split to evaluate')
    parser.add_argument('--data-root', type=str, default='data/L3D', help='Root dataset directory')
    parser.add_argument('--output-dir', type=str, default='visualizations', help='Visual output directory')
    parser.add_argument('--save-vis', action='store_true', help='Save visual comparison panels')
    parser.add_argument('--num-vis', type=int, default=20, help='Maximum number of visual panels to save')
    parser.add_argument('--device', type=str, default=None, help='Device override (cuda/mps/cpu)')
    return parser.parse_args()


def get_device(requested_device=None):
    if requested_device:
        return torch.device(requested_device)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def render_overlay(rgb: np.ndarray, pred_map: np.ndarray, gt_map: np.ndarray) -> np.ndarray:
    """
    Renders 4-panel visual comparison:
    1. Laparoscopy RGB
    2. Ground Truth Mask
    3. MinVIS Predicted Mask
    4. Composite Overlay (Green=Match, Red=False Positive, Blue=False Negative)
    """
    h, w = rgb.shape[:2]

    # Color palette: 1: Ridge (Cyan [0, 255, 255]), 2: Silhouette (Yellow [255, 255, 0]), 3: Ligament (Magenta [255, 0, 255])
    palette = {
        1: np.array([0, 230, 230], dtype=np.uint8),
        2: np.array([230, 230, 0], dtype=np.uint8),
        3: np.array([230, 0, 230], dtype=np.uint8)
    }

    gt_vis = np.zeros_like(rgb)
    for c, col in palette.items():
        gt_vis[gt_map == c] = col

    pred_vis = np.zeros_like(rgb)
    for c, col in palette.items():
        pred_vis[pred_map == c] = col

    overlay = rgb.copy()
    # Match: Green [0, 255, 0]
    match = (pred_map > 0) & (pred_map == gt_map)
    # False Positive: Red [255, 0, 0]
    fp = (pred_map > 0) & (pred_map != gt_map)
    # False Negative: Blue [0, 100, 255]
    fn = (gt_map > 0) & (pred_map == 0)

    overlay[match] = (0.4 * overlay[match] + 0.6 * np.array([0, 255, 0])).astype(np.uint8)
    overlay[fp] = (0.4 * overlay[fp] + 0.6 * np.array([255, 50, 50])).astype(np.uint8)
    overlay[fn] = (0.4 * overlay[fn] + 0.6 * np.array([50, 100, 255])).astype(np.uint8)

    panel = np.hstack([rgb, gt_vis, pred_vis, overlay])
    return panel


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"🔍 Evaluating MinVIS Checkpoint: '{args.checkpoint}' on split: {args.split.upper()} (Device: {device})")

    try:
        train_dir = resolve_split_dir('train', args.data_root)
        data_root = str(Path(train_dir).parent)
    except FileNotFoundError:
        data_root = args.data_root

    if args.split == 'val':
        _, patients = get_fold_patients(args.fold_idx)
    else:
        patients = get_frozen_test_patients()

    dataset = PatientVideoL3DDataset(
        patient_ids=patients,
        data_root=data_root,
        image_size=1024,
        stroke_width=35,
        clip_len=3
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    model = MinVISMask2Former(num_classes=3, hidden_dim=256).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    tracker = MinVISTracker(num_classes=3)
    os.makedirs(args.output_dir, exist_ok=True)

    all_metrics = []
    vis_count = 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            pv = batch['pixel_values'][0].to(device)   # (T, 3, H, W)
            tm = batch['target_masks'][0].cpu().numpy()  # (3, H, W)
            gt_map = batch['gt_map'][0].cpu().numpy()   # (H, W)
            pat = batch['patient'][0]
            fid = batch['frame_ids'][-1][0].item() if torch.is_tensor(batch['frame_ids'][-1]) else batch['frame_ids'][-1]

            outputs = model(pv)
            class_logits = outputs['class_queries_logits']
            query_embds = outputs['query_embeddings']
            mask_logits = outputs['masks_queries_logits']

            mask_probs = torch.nn.functional.interpolate(
                mask_logits.sigmoid(),
                size=(1024, 1024),
                mode='bilinear',
                align_corners=False
            )

            tracked_sem_seg, _ = tracker.track_sequence(query_embds, class_logits, mask_probs)
            pred_prob = tracked_sem_seg[-1].cpu().numpy()

            metrics = evaluate_frame_metrics(pred_sem_seg=pred_prob, gt_input=tm, threshold=0.5)
            metrics['patient'] = pat
            metrics['frame_id'] = fid
            all_metrics.append(metrics)

            if args.save_vis and vis_count < args.num_vis:
                # Reconstruct unnormalized RGB image
                norm_img = pv[-1].permute(1, 2, 0).cpu().numpy()
                rgb = ((norm_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255.0).clip(0, 255).astype(np.uint8)

                max_prob = np.max(pred_prob, axis=0)
                argmax_cls = np.argmax(pred_prob, axis=0) + 1
                pred_map = np.where(max_prob >= 0.5, argmax_cls, 0).astype(np.uint8)

                panel = render_overlay(rgb, pred_map, gt_map)
                vis_path = os.path.join(args.output_dir, f"{args.split}_{pat}_{fid:07d}_dice{metrics['macro_dice']:.1f}.png")
                cv2.imwrite(vis_path, cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
                vis_count += 1

    macro_dice = float(np.mean([m['macro_dice'] for m in all_metrics]))
    micro_dice = float(np.mean([m['micro_dice'] for m in all_metrics]))
    ridge_dice = float(np.mean([m['ridge_dice'] for m in all_metrics]))
    silh_dice = float(np.mean([m['silhouette_dice'] for m in all_metrics]))
    lig_dice = float(np.mean([m['ligament_dice'] for m in all_metrics]))
    macro_assd = float(np.mean([m['macro_assd'] for m in all_metrics]))
    macro_hd95 = float(np.mean([m['macro_hd95'] for m in all_metrics]))

    print(f"\n================ EVALUATION SUMMARY [{args.split.upper()}] ================")
    print(f"Total Frames: {len(all_metrics)} across {len(set(m['patient'] for m in all_metrics))} patients")
    print(f"Macro Dice:  {macro_dice:.2f}% | Micro Dice: {micro_dice:.2f}%")
    print(f"Ridge Dice:  {ridge_dice:.2f}% | Silhouette: {silh_dice:.2f}% | Ligament: {lig_dice:.2f}%")
    print(f"Macro ASSD:  {macro_assd:.2f}px | Macro HD95: {macro_hd95:.2f}px")
    print(f"============================================================")


if __name__ == '__main__':
    main()
