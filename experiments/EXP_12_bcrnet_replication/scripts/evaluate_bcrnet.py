"""
Comprehensive Evaluation Script for EXP_12: BCRNet Replication
==============================================================
Evaluates trained BCRNet model on both **Val** and **Test** splits.
Computes all benchmark metrics reported in the BCRNet paper:
  - DSC (%)  — Dice Similarity Coefficient
  - IoU (%)  — Intersection over Union
  - ASSD (px) — Average Symmetric Surface Distance

Usage:
  python experiments/EXP_12_bcrnet_replication/scripts/evaluate_bcrnet.py \
      --model_path /kaggle/working/checkpoints/EXP_12_bcrnet_replication/best_model.pt \
      --data_path /kaggle/working/L3D \
      --split Test \
      --save_path /kaggle/working/results/EXP_12_bcrnet_replication
"""

import os
import sys
import argparse
import json
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ws_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
bcrnet_root = os.path.join(ws_root, "repos/BCRNet")
for p in [bcrnet_root, ws_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Automatic PyTorch fallback if CUDA C extension is not compiled
try:
    from adet import _C
except ImportError:
    import adet.layers.ms_deform_attn as ms_module
    class _MockC:
        @staticmethod
        def ms_deform_attn_forward(value, shapes, starts, locs, weights, step):
            return ms_module.ms_deform_attn_core_pytorch(value, shapes, locs, weights)
        @staticmethod
        def ms_deform_attn_backward(*args, **kwargs):
            return None
    ms_module._C = _MockC

from utils.bezier_dataset import BezierDataset, collate_fun
from adet.modeling.bezier_detection import TransformerPureDetector
from utils.config_utils import load_config


def compute_assd(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Computes Average Symmetric Surface Distance (ASSD) in pixels.
    Uses medpy if available, otherwise exact distance-transform boundary method.
    """
    if np.sum(pred_mask) == 0 or np.sum(gt_mask) == 0:
        return np.nan

    try:
        from medpy.metric.binary import assd
        return float(assd(pred_mask, gt_mask))
    except Exception:
        from scipy.ndimage import distance_transform_edt
        # Extract boundaries
        kernel = np.ones((3, 3), np.uint8)
        pred_boundary = pred_mask & ~cv2.erode(pred_mask.astype(np.uint8), kernel, iterations=1)
        gt_boundary = gt_mask & ~cv2.erode(gt_mask.astype(np.uint8), kernel, iterations=1)

        if np.sum(pred_boundary) == 0 or np.sum(gt_boundary) == 0:
            return np.nan

        d_gt = distance_transform_edt(~gt_boundary)
        d_pred = distance_transform_edt(~pred_boundary)

        assd_val = (np.mean(d_gt[pred_boundary]) + np.mean(d_pred[gt_boundary])) / 2.0
        return float(assd_val)


def render_prediction_and_gt(results, targets):
    """
    Renders 30px thick landmark strokes matching official BCRNet test.py metrix().
    """
    gt = np.stack([lm['landmark_mask'].cpu().numpy() for lm in targets[0]], -1)
    pred = np.zeros_like(gt)
    for m, lm in enumerate(results[0]):
        curves = lm['ctrl_points'].cpu().numpy()
        for curve_points in curves:
            for i in range(1, len(curve_points)):
                pt1 = tuple(map(int, curve_points[i - 1]))
                pt2 = tuple(map(int, curve_points[i]))
                pred[:, :, m] = cv2.line(pred[:, :, m].copy(), pt1, pt2, [1], 30)
    return pred, gt


def evaluate_split(model, dataset_dir, split_name, save_dir, device):
    split_cap = split_name.capitalize()
    data_split_dir = os.path.join(dataset_dir, split_cap)
    if not os.path.exists(data_split_dir):
        print(f"⚠️ Directory for split '{split_name}' does not exist at {data_split_dir}. Skipping.")
        return None

    dataset = BezierDataset(data_split_dir, device=device)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fun)

    class_names = ['silhouette', 'ligament', 'ridge']
    sample_metrics = []
    seg_save_dir = os.path.join(save_dir, split_cap, 'seg_results')
    os.makedirs(seg_save_dir, exist_ok=True)

    print(f"\n" + "=" * 70)
    print(f"🔍 EVALUATING BCRNET ON: [{split_cap.upper()}] ({len(dataset)} frames)")
    print(f"=" * 70)

    pbar = tqdm(loader, desc=f"Evaluating {split_cap}")
    for batch_data in pbar:
        img, depth, sam_feature, targets, info = batch_data
        item_name = info[0]['item_name']

        with torch.no_grad():
            results = model(batch_data)

        pred, gt = render_prediction_and_gt(results, targets)

        # Overall Dice & IoU
        smooth = 1e-5
        intersection = np.sum(pred * gt)
        sample_dice = (2.0 * intersection + smooth) / (np.sum(pred) + np.sum(gt) + smooth)
        sample_iou = sample_dice / (2.0 - sample_dice)

        # Overall ASSD
        pred_flat = (np.sum(pred, axis=-1) > 0).astype(np.uint8)
        gt_flat = (np.sum(gt, axis=-1) > 0).astype(np.uint8)
        sample_assd = compute_assd(pred_flat, gt_flat)

        # Per-class Dice & ASSD
        class_dices = {}
        for c_idx, c_name in enumerate(class_names):
            p_c = pred[:, :, c_idx]
            g_c = gt[:, :, c_idx]
            inter_c = np.sum(p_c * g_c)
            d_c = (2.0 * inter_c + smooth) / (np.sum(p_c) + np.sum(g_c) + smooth)
            class_dices[f"{c_name}_dice"] = float(d_c)

        sample_record = {
            'item_name': item_name,
            'dice': float(sample_dice),
            'iou': float(sample_iou),
            'assd': float(sample_assd) if not np.isnan(sample_assd) else None,
            **class_dices,
        }
        sample_metrics.append(sample_record)

        # Save Visual Overlays matching test.py
        pred_bgr = np.stack([pred[:, :, 1], pred[:, :, 0], pred[:, :, 2]], -1) * 255
        gt_bgr = np.stack([gt[:, :, 1], gt[:, :, 0], gt[:, :, 2]], -1) * 255
        cv2.imwrite(os.path.join(seg_save_dir, f"{item_name}-{sample_dice:.3f}.png"), pred_bgr.astype(np.uint8))
        cv2.imwrite(os.path.join(seg_save_dir, f"{item_name}-gt.png"), gt_bgr.astype(np.uint8))

        pbar.set_postfix({'DSC': f"{sample_dice*100:.2f}%", 'IoU': f"{sample_iou*100:.2f}%"})

    # Aggregate metrics
    mean_dice = float(np.mean([s['dice'] for s in sample_metrics]))
    mean_iou = float(np.mean([s['iou'] for s in sample_metrics]))
    valid_assds = [s['assd'] for s in sample_metrics if s['assd'] is not None]
    mean_assd = float(np.mean(valid_assds)) if valid_assds else float('nan')

    summary = {
        'split': split_cap,
        'num_samples': len(sample_metrics),
        'mean_dice': mean_dice * 100.0,
        'mean_iou': mean_iou * 100.0,
        'mean_assd': mean_assd,
        'paper_target_dice': 69.57,
        'paper_target_iou': 54.16,
        'paper_target_assd': 43.55,
    }

    for c_name in class_names:
        c_mean = float(np.mean([s[f"{c_name}_dice"] for s in sample_metrics]))
        summary[f"{c_name}_dice"] = c_mean * 100.0

    print(f"\n📊 Summary for {split_cap}:")
    print(f"   • Mean DSC:  {summary['mean_dice']:.2f}%  (Paper: 69.57%)")
    print(f"   • Mean IoU:  {summary['mean_iou']:.2f}%  (Paper: 54.16%)")
    print(f"   • Mean ASSD: {summary['mean_assd']:.2f} px (Paper: 43.55 px)")
    for c_name in class_names:
        print(f"     - {c_name.capitalize()}: {summary[f'{c_name}_dice']:.2f}% DSC")

    with open(os.path.join(save_dir, f"metrics_{split_cap.lower()}.json"), "w") as f:
        json.dump({'summary': summary, 'samples': sample_metrics}, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="EXP_12: Evaluate BCRNet Model")
    default_cfg = os.path.join(os.path.dirname(__file__), "../configs/bcrnet_l3d.yaml")
    default_data = "/kaggle/working/L3D" if os.path.exists("/kaggle") else os.path.join(ws_root, "data/L3D")
    default_save = "/kaggle/working/results/EXP_12_bcrnet_replication" if os.path.exists("/kaggle") else os.path.join(ws_root, "experiments/EXP_12_bcrnet_replication/results")

    parser.add_argument('--config', default=default_cfg)
    parser.add_argument('--model_path', required=True, help="Path to checkpoint (.pt)")
    parser.add_argument('--data_path', default=default_data)
    parser.add_argument('--split', default='both', choices=['Val', 'Test', 'both'])
    parser.add_argument('--save_path', default=default_save)
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Loading BCRNet model from: {args.model_path}")
    cfg = load_config(args.config)
    cfg.MODEL.DEVICE = device
    model = TransformerPureDetector(cfg).to(device)

    checkpoint = torch.load(args.model_path, map_location=device)
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    print("✅ Model loaded successfully.")

    splits_to_eval = ['Val', 'Test'] if args.split == 'both' else [args.split]

    all_summaries = {}
    for sp in splits_to_eval:
        summary = evaluate_split(model, args.data_path, sp, args.save_path, device)
        if summary:
            all_summaries[sp] = summary

    # Print comparative Markdown table
    print("\n" + "=" * 80)
    print("📋 REPLICATION BENCHMARK SUMMARY TABLE:")
    print("=" * 80)
    print("| Split | DSC (%) | IoU (%) | ASSD (px) | Silhouette (%) | Ligament (%) | Ridge (%) |")
    print("|:---|:---:|:---:|:---:|:---:|:---:|:---:|")
    print(f"| **Paper Target (Test)** | **69.57** | **54.16** | **43.55** | -- | -- | -- |")
    for sp, s in all_summaries.items():
        print(f"| {sp} | {s['mean_dice']:.2f} | {s['mean_iou']:.2f} | {s['mean_assd']:.2f} | {s['silhouette_dice']:.2f} | {s['ligament_dice']:.2f} | {s['ridge_dice']:.2f} |")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
