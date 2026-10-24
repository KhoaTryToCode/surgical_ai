"""
Universal Evaluation & Visualizer CLI for experiments/temporal.
Evaluates any model checkpoint on Validation or Frozen Test splits,
and renders multi-panel qualitative inspection images.

Usage Examples:
  python evaluate.py --model minvis --checkpoint checkpoints/minvis/best_minvis_fold0.pth --split val --save-vis
  python evaluate.py --model minvis --checkpoint checkpoints/minvis/best_minvis_fold0.pth --split test --save-vis
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

EXP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_DIR.parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(EXP_DIR))

from shared.utils.split_loader import get_fold_patients, get_frozen_test_patients
from models.registry import get_model_entry
from core.dataset import PatientVideoL3DDataset, resolve_split_dir, resolve_all_split_dirs
from core.metrics import evaluate_frame_metrics
from core.trainer import get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Temporal Video Model Checkpoint")
    parser.add_argument('--model', type=str, default='minvis', choices=['minvis', 'cutie', 'sam2'], help='Model name')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint .pth file')
    parser.add_argument('--fold_idx', type=int, default=0, help='Fold index (0 to 4)')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'], help='Split to evaluate')
    parser.add_argument('--data-root', type=str, default='data/L3D', help='Root dataset directory')
    parser.add_argument('--output-dir', type=str, default=None, help='Visualizations directory')
    parser.add_argument('--save-vis', action='store_true', help='Save 4-panel comparison images')
    parser.add_argument('--num-vis', type=int, default=20, help='Number of inspection images to save')
    parser.add_argument('--device', type=str, default=None, help='Device override')
    return parser.parse_args()


def render_overlay(rgb: np.ndarray, pred_map: np.ndarray, gt_map: np.ndarray) -> np.ndarray:
    palette = {
        1: np.array([0, 230, 230], dtype=np.uint8),   # Ridge: Cyan
        2: np.array([230, 230, 0], dtype=np.uint8),   # Silhouette: Yellow
        3: np.array([230, 0, 230], dtype=np.uint8)    # Ligament: Magenta
    }
    gt_vis = np.zeros_like(rgb)
    for c, col in palette.items():
        gt_vis[gt_map == c] = col

    pred_vis = np.zeros_like(rgb)
    for c, col in palette.items():
        pred_vis[pred_map == c] = col

    overlay = rgb.copy()
    match = (pred_map > 0) & (pred_map == gt_map)
    fp = (pred_map > 0) & (pred_map != gt_map)
    fn = (gt_map > 0) & (pred_map == 0)

    overlay[match] = (0.4 * overlay[match] + 0.6 * np.array([0, 255, 0])).astype(np.uint8)
    overlay[fp] = (0.4 * overlay[fp] + 0.6 * np.array([255, 50, 50])).astype(np.uint8)
    overlay[fn] = (0.4 * overlay[fn] + 0.6 * np.array([50, 100, 255])).astype(np.uint8)

    return np.hstack([rgb, gt_vis, pred_vis, overlay])


def main():
    args = parse_args()
    device = get_device(args.device)
    entry = get_model_entry(args.model)
    print(f"🔍 Evaluating {args.model.upper()} ({args.checkpoint}) on {args.split.upper()} (Device: {device})")

    split_dirs_map = resolve_all_split_dirs(args.data_root)
    all_split_dirs = list(split_dirs_map.values())
    print("📁 Resolved L3D Dataset Splits:")
    for split_k, split_p in split_dirs_map.items():
        print(f"   [{split_k.upper()}]: {split_p}")

    patients = get_fold_patients(args.fold_idx)[1] if args.split == 'val' else get_frozen_test_patients()
    dataset = PatientVideoL3DDataset(
        patient_ids=patients,
        split_dirs=all_split_dirs,
        image_size=1024,
        stroke_width=35,
        clip_len=3
    )
    eval_workers = 2 if device.type == 'cuda' else 0
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=eval_workers)

    model = entry['model_class'](**entry['default_model_kwargs']).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state)
    model.eval()

    tracker = entry['tracker_class'](**entry['default_tracker_kwargs']) if entry['tracker_class'] is not None else None

    out_dir = args.output_dir if args.output_dir else os.path.join(str(EXP_DIR), "visualizations", args.model, f"fold{args.fold_idx}_{args.split}")
    os.makedirs(out_dir, exist_ok=True)

    all_metrics = []
    vis_count = 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            pv = batch['pixel_values'][0].to(device)
            tm = batch['target_masks'][0].cpu().numpy()
            gt_map = batch['gt_map'][0].cpu().numpy()
            pat = batch['patient'][0]
            fid = batch['frame_ids'][-1][0].item() if torch.is_tensor(batch['frame_ids'][-1]) else batch['frame_ids'][-1]
            h, w = pv.shape[-2:]

            if tracker is not None:
                outputs = model(pv)
                mask_probs = torch.nn.functional.interpolate(outputs['masks_queries_logits'].sigmoid(), size=(h, w), mode='bilinear', align_corners=False)
                res = tracker.track_sequence(outputs['query_embeddings'], outputs['class_queries_logits'], mask_probs)
                tracked_sem = res[0] if isinstance(res, (tuple, list)) else res
                pred_prob = tracked_sem[-1].cpu().numpy()
            else:
                outputs = model(pv.unsqueeze(0))
                sem = outputs['sem_seg']
                pred_prob = sem[0, -1].cpu().numpy() if sem.ndim == 5 else sem[0].cpu().numpy()

            metrics = evaluate_frame_metrics(pred_sem_seg=pred_prob, gt_input=tm, threshold=0.5)
            metrics['patient'] = pat
            metrics['frame_id'] = fid
            all_metrics.append(metrics)

            if args.save_vis and vis_count < args.num_vis:
                norm_img = pv[-1].permute(1, 2, 0).cpu().numpy()
                rgb = ((norm_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255.0).clip(0, 255).astype(np.uint8)

                max_p = np.max(pred_prob, axis=0)
                pred_map = np.where(max_p >= 0.5, np.argmax(pred_prob, axis=0) + 1, 0).astype(np.uint8)

                panel = render_overlay(rgb, pred_map, gt_map)
                vis_path = os.path.join(out_dir, f"{pat}_{fid:07d}_dice{metrics['macro_dice']:.1f}.png")
                cv2.imwrite(vis_path, cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
                vis_count += 1

    macro_dice = float(np.mean([m['macro_dice'] for m in all_metrics]))
    micro_dice = float(np.mean([m['micro_dice'] for m in all_metrics]))
    ridge_dice = float(np.mean([m['ridge_dice'] for m in all_metrics]))
    silh_dice = float(np.mean([m['silhouette_dice'] for m in all_metrics]))
    lig_dice = float(np.mean([m['ligament_dice'] for m in all_metrics]))
    macro_assd = float(np.mean([m['macro_assd'] for m in all_metrics]))
    macro_hd95 = float(np.mean([m['macro_hd95'] for m in all_metrics]))

    print(f"\n================ EVALUATION SUMMARY [{args.model.upper()} - {args.split.upper()}] ================")
    print(f"Total Frames: {len(all_metrics)} across {len(set(m['patient'] for m in all_metrics))} patients")
    print(f"Macro Dice:  {macro_dice:.2f}% | Micro Dice: {micro_dice:.2f}%")
    print(f"Ridge Dice:  {ridge_dice:.2f}% | Silhouette: {silh_dice:.2f}% | Ligament: {lig_dice:.2f}%")
    print(f"Macro ASSD:  {macro_assd:.2f}px | Macro HD95: {macro_hd95:.2f}px")
    print(f"============================================================")


if __name__ == '__main__':
    main()
