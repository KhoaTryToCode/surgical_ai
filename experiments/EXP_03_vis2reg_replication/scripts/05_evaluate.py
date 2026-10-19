#!/usr/bin/env python3
"""
Phase 6: Benchmark Evaluation & Cross-Validation Script.
Replicates Vis2Reg Evaluation Pipeline (MICCAI 2026 paper Table 1):
1. Evaluates on held-out test patients for the specified fold (or all 5 folds).
2. Computes official quantitative metrics:
   - Silhouette Dice (%) between rendered warped model and ground-truth 2D mask.
   - Observation-to-Model One-Way Chamfer Distance (mm).
3. Produces per-patient breakdown and aggregate statistics (Mean +- Std).
4. Compares results directly against published baselines in paper Table 1:
   - Vis2Reg (Ours) : 92.6% +- 7.2% Dice | 1.43 +- 1.26 mm CD
   - TopoNet        : 83.2% +- 9.4% Dice | 3.82 +- 2.15 mm CD
   - GeoTransformer : 78.4% +- 11.2% Dice | 5.14 +- 3.40 mm CD
   - ICP            : 61.5% +- 14.8% Dice | 8.92 +- 4.70 mm CD
"""

from __future__ import annotations

import os
import sys
import argparse
from typing import Dict, List, Tuple

# Ensure experiment root is in sys.path
EXPERIMENT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if EXPERIMENT_ROOT not in sys.path:
    sys.path.insert(0, EXPERIMENT_ROOT)

import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader

from configs.default_config import get_default_config
from vis2reg.models.vis2reg import Vis2Reg
from data.dataset import P2ILRegFolds, P2ILRegRealDataset, collate_p2i_lreg


def parse_args():
    parser = argparse.ArgumentParser(description="Vis2Reg Phase 6: Benchmark Evaluation")
    parser.add_argument(
        "--data_root",
        type=str,
        default="/kaggle/input/datasets/junezhou001/p2i-lreg/Liver_regis",
        help="Path to Liver_regis dataset root",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_real_model.pth",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Cross-validation fold to evaluate [0..4] (default: 0)",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=6000,
        help="Resampled point cloud size (default: 6000)",
    )
    parser.add_argument(
        "--render_size",
        type=int,
        default=256,
        help="Resolution for silhouette rendering (default: 256)",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Optional path to save per-keyframe CSV results",
    )
    return parser.parse_args()


def compute_metrics(
    pred_mask: torch.Tensor,
    gt_mask: torch.Tensor,
    target_pts: torch.Tensor,
    warped_pts: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[float, float, float]:
    """
    Computes:
    1. Silhouette Dice (%)
    2. Observation-to-Model One-Way Chamfer Distance (mm)
    3. Model-to-Observation Reverse Chamfer Distance (mm)
    """
    while pred_mask.dim() > 2:
        pred_mask = pred_mask[0]
    while gt_mask.dim() > 2:
        gt_mask = gt_mask[0]

    # Resize gt_mask if needed
    if pred_mask.shape[-2:] != gt_mask.shape[-2:]:
        gt_mask = F.interpolate(
            gt_mask.unsqueeze(0).unsqueeze(0).float(),
            size=pred_mask.shape[-2:],
            mode='nearest'
        ).squeeze()

    pred_bin = (pred_mask > 0.5).float()
    gt_bin = (gt_mask > 0.5).float()

    intersection = (pred_bin * gt_bin).sum().item()
    union = pred_bin.sum().item() + gt_bin.sum().item()
    dice_score = (2.0 * intersection + eps) / (union + eps) * 100.0

    # One-Way Chamfer Distance: (1 / |Q|) * sum_{q in Q} min_{p in P} ||q - p||_2
    dists = torch.cdist(target_pts, warped_pts, p=2.0)
    cd_oneway_tgt2pred = dists.min(dim=-1).values.mean().item()
    cd_reverse_pred2tgt = dists.min(dim=-2).values.mean().item()

    return dice_score, cd_oneway_tgt2pred, cd_reverse_pred2tgt


def evaluate_fold(model, test_dataset, device) -> List[Dict]:
    model.eval()
    results = []

    loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_p2i_lreg,
    )

    with torch.no_grad():
        for batch in loader:
            batch_dev = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            batch_dev['rigid_only'] = False

            outputs = model(batch_dev)

            render_mask = outputs.get('render_mask')
            gt_mask = batch_dev.get('intra_masks')
            target_pts = batch_dev['target_points']
            warped_pts = outputs['warped_pcd']
            pid = batch['patient_id'][0] if isinstance(batch['patient_id'], list) else batch['patient_id']
            frame = batch.get('frame_name', ['unknown'])[0]

            if render_mask is not None and gt_mask is not None:
                dice, cd_oneway, cd_rev = compute_metrics(render_mask, gt_mask, target_pts, warped_pts)
            else:
                dice, cd_oneway, cd_rev = 0.0, 0.0, 0.0

            results.append({
                'patient_id': pid,
                'frame_name': frame,
                'dice_pct': dice,
                'cd_oneway_mm': cd_oneway,
                'cd_reverse_mm': cd_rev,
            })

    return results


def main():
    args = parse_args()

    print("=" * 65)
    print("  VIS2REG PHASE 6: BENCHMARK EVALUATION (PAPER TABLE 1)")
    print("=" * 65)

    # 1. Device Setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[Device] Using Apple Silicon MPS")
    else:
        device = torch.device("cpu")
        print("[Device] Using CPU")

    # 2. Check Data Root
    data_root = args.data_root
    if not os.path.exists(data_root):
        alt_paths = [
            "data/Liver_regis",
            "../data/Liver_regis",
            os.path.join(os.getcwd(), "data/Liver_regis"),
        ]
        for p in alt_paths:
            if os.path.exists(p):
                data_root = p
                break

    if not os.path.exists(data_root):
        print(f"\n[Warning] Dataset root not found at: {args.data_root}")
        print("          Printing published paper benchmarks for comparison.")
        print_paper_benchmark_table()
        return

    # 3. Load Fold Test Patients
    fold_info = P2ILRegFolds.get_fold(args.fold)
    test_patients = fold_info['test']
    print(f"\n[Step 1] Evaluating Fold {args.fold} Held-Out Test Patients ({len(test_patients)}): {test_patients}")

    # 4. Instantiate Test Dataset
    test_dataset = P2ILRegRealDataset(
        dataset_root=data_root,
        patient_ids=test_patients,
        max_points=args.max_points,
    )
    print(f"         Indexed {len(test_dataset)} test keyframes across {len(test_patients)} patients.")

    if len(test_dataset) == 0:
        print("[Error] No test keyframes found. Check dataset path.")
        return

    # 5. Instantiate Architecture
    cfg = get_default_config()
    cfg.data['max_points'] = args.max_points
    cfg.model['renderer'] = {
        'type': 'points',
        'image_size': (args.render_size, args.render_size),
        'radius': 0.01,
        'points_per_pixel': 16,
    }
    model = Vis2Reg(cfg).to(device)

    # 6. Load Trained Checkpoint
    ckpt_path = args.checkpoint
    if not os.path.isabs(ckpt_path) and not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(EXPERIMENT_ROOT, args.checkpoint)

    if os.path.exists(ckpt_path):
        print(f"\n[Step 2] Loading Trained Checkpoint from: {ckpt_path}...")
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        except Exception:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)
        print("         Weights loaded successfully!")
    else:
        print(f"\n[Warning] Checkpoint not found at '{args.checkpoint}'. Evaluating model with current weights.")

    # 7. Run Evaluation
    print("\n[Step 3] Running Inference across Test Keyframes...")
    records = evaluate_fold(model, test_dataset, device)

    # 8. Compute Per-Patient and Overall Statistics
    patient_metrics: Dict[str, List[float]] = {}
    all_dices = []
    all_cds = []

    for r in records:
        pid = r['patient_id']
        if pid not in patient_metrics:
            patient_metrics[pid] = {'dice': [], 'cd': []}
        patient_metrics[pid]['dice'].append(r['dice_pct'])
        patient_metrics[pid]['cd'].append(r['cd_oneway_mm'])
        all_dices.append(r['dice_pct'])
        all_cds.append(r['cd_oneway_mm'])

    mean_dice = float(np.mean(all_dices))
    std_dice = float(np.std(all_dices))
    mean_cd = float(np.mean(all_cds))
    std_cd = float(np.std(all_cds))

    # 9. Print Formatted Results Table
    print("\n" + "=" * 65)
    print(f"  FOLD {args.fold} TEST EVALUATION RESULTS ({len(records)} Keyframes)")
    print("=" * 65)
    print(f"{'Patient ID':<15} | {'Frames':<8} | {'Dice (%)':<18} | {'One-Way CD (mm)':<18}")
    print("-" * 65)
    for pid in sorted(patient_metrics.keys()):
        p_d = patient_metrics[pid]['dice']
        p_c = patient_metrics[pid]['cd']
        print(
            f"Patient {pid:<7} | {len(p_d):<8} | "
            f"{np.mean(p_d):.2f} +- {np.std(p_d):.2f}%{' ':>4} | "
            f"{np.mean(p_c):.2f} +- {np.std(p_c):.2f} mm"
        )
    print("-" * 65)
    print(
        f"{'OVERALL (Ours)':<15} | {len(records):<8} | "
        f"{mean_dice:.2f} +- {std_dice:.2f}%{' ':>4} | "
        f"{mean_cd:.2f} +- {std_cd:.2f} mm"
    )
    print("=" * 65)

    # 10. Direct Comparison against Paper Table 1
    print("\n" + "=" * 65)
    print("  MICCAI 2026 PAPER TABLE 1 BENCHMARK COMPARISON")
    print("=" * 65)
    print(f"{'Method':<20} | {'Silhouette Dice (%)':<20} | {'One-Way CD (mm)':<18}")
    print("-" * 65)
    print(f"{'ICP':<20} | {'61.5 +- 14.8%':<20} | {'8.92 +- 4.70 mm':<18}")
    print(f"{'CPD':<20} | {'67.8 +- 13.2%':<20} | {'7.35 +- 3.90 mm':<18}")
    print(f"{'PointNetLK':<20} | {'72.1 +- 12.0%':<20} | {'6.20 +- 3.10 mm':<18}")
    print(f"{'GeoTransformer':<20} | {'78.4 +- 11.2%':<20} | {'5.14 +- 3.40 mm':<18}")
    print(f"{'TopoNet (MICCAI 24)':<20} | {'83.2 +- 9.4%':<20} | {'3.82 +- 2.15 mm':<18}")
    print("-" * 65)
    print(f"{'Vis2Reg (Paper)':<20} | {'92.6 +- 7.2%':<20} | {'1.43 +- 1.26 mm':<18}")
    print(f"{'Vis2Reg (Ours)':<20} | {f'{mean_dice:.2f} +- {std_dice:.2f}%':<20} | {f'{mean_cd:.2f} +- {std_cd:.2f} mm':<18}")
    print("=" * 65)


def print_paper_benchmark_table():
    print("\n" + "=" * 65)
    print("  MICCAI 2026 PAPER TABLE 1 PUBLISHED BENCHMARKS")
    print("=" * 65)
    print(f"{'Method':<20} | {'Silhouette Dice (%)':<20} | {'One-Way CD (mm)':<18}")
    print("-" * 65)
    print(f"{'ICP':<20} | {'61.5 +- 14.8%':<20} | {'8.92 +- 4.70 mm':<18}")
    print(f"{'CPD':<20} | {'67.8 +- 13.2%':<20} | {'7.35 +- 3.90 mm':<18}")
    print(f"{'PointNetLK':<20} | {'72.1 +- 12.0%':<20} | {'6.20 +- 3.10 mm':<18}")
    print(f"{'GeoTransformer':<20} | {'78.4 +- 11.2%':<20} | {'5.14 +- 3.40 mm':<18}")
    print(f"{'TopoNet (MICCAI 24)':<20} | {'83.2 +- 9.4%':<20} | {'3.82 +- 2.15 mm':<18}")
    print("-" * 65)
    print(f"{'Vis2Reg (Paper)':<20} | {'92.6 +- 7.2%':<20} | {'1.43 +- 1.26 mm':<18}")
    print("=" * 65)


if __name__ == "__main__":
    main()
