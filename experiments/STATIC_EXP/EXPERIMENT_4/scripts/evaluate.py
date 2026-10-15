#!/usr/bin/env python3
"""
EXPERIMENT_4: Standalone Evaluation Script for Landmark Master Tokens + Patch Bézier Model
Restores best_model.pth, runs evaluation on Val and Test sets, and generates diagnostics.
"""
import os
import sys
import json
import time
import zipfile
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from pathlib import Path

# NumPy 2.0 compatibility
for _a, _v in [('Inf', np.inf), ('NAN', np.nan), ('NaN', np.nan), ('PINF', np.inf), ('NINF', -np.inf)]:
    if not hasattr(np, _a): setattr(np, _a, _v)

# Add workspace root to sys.path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from experiments.EXPERIMENT_4.models.landmark_bezier_model import LandmarkBezierPatchModel
from experiments.EXPERIMENT_4.utils.dataset import LandmarkBezierDataset
from experiments.EXPERIMENT_4.utils.metrics import evaluate_model

def create_results_zip(source_dir, output_zip_path):
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(source_dir):
            for file in files:
                if file.endswith('.zip'): continue
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, source_dir)
                zipf.write(file_path, arcname)
    print(f"Archive packaged: {output_zip_path} ({os.path.getsize(output_zip_path)/(1024*1024):.2f} MB)")

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained LandmarkBezierPatchModel (EXPERIMENT_4)")
    parser.add_argument('--checkpoint', type=str, default=None,
                        help="Path to best_model.pth")
    parser.add_argument('--val_dir', type=str, default='data/L3D/Val')
    parser.add_argument('--test_dir', type=str, default='data/L3D/Test')
    parser.add_argument('--save_dir', type=str, default='experiments/EXPERIMENT_4/results/run_01')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--grid_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--eval_splits', type=str, choices=['val', 'both'], default='both')
    
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Running evaluation on device: {device}")
    
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_path = os.path.join(args.save_dir, 'best_model.pth')
    if not os.path.exists(ckpt_path):
        server_fallback = '/data/khoalq/checkpoints/exp4_landmark_bezier_60ep/best_model.pth'
        if os.path.exists(server_fallback):
            ckpt_path = server_fallback
            
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")
        
    print(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    grid_size = checkpoint.get('grid_size', args.grid_size)
    single_scale = checkpoint.get('single_scale', True)
    
    model = LandmarkBezierPatchModel(
        grid_size=grid_size, num_classes=4, embed_dim=256,
        num_decoder_layers=6, single_scale=single_scale
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    epoch_trained = checkpoint.get('epoch', 'N/A')
    best_dice = checkpoint.get('best_val_dice', 0.0)
    print(f"✓ Model successfully restored (Epoch {epoch_trained}, Best Val Dice: {best_dice:.4f})")
    
    nw = args.num_workers if args.num_workers is not None else min(4, os.cpu_count() or 1)
    
    # 1. Validation Evaluation
    print(f"\nEvaluating Validation Set from: {args.val_dir} ...")
    val_dataset = LandmarkBezierDataset(args.val_dir, grid_size=grid_size)
    val_loader  = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=nw)
    
    patient40_dir = os.path.join(args.save_dir, 'patient_40_diagnostics')
    val_summary, val_df = evaluate_model(
        model, val_loader, device, split_name='Val-Final', save_patient40_dir=patient40_dir
    )
    val_df.to_csv(os.path.join(args.save_dir, 'val_per_frame_metrics.csv'), index=False)
    print(f"✓ Validation finished: {len(val_df)} frames processed.")
    print(f"  Macro Dice: {val_summary['macro_dice']:.4f} | ASSD: {val_summary['macro_assd']:.2f} px")
    
    # 2. Test Evaluation
    test_summary = {}
    if args.eval_splits == 'both' and os.path.exists(args.test_dir):
        print(f"\nEvaluating Test Set from: {args.test_dir} ...")
        test_dataset = LandmarkBezierDataset(args.test_dir, grid_size=grid_size)
        test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=nw)
        test_summary, test_df = evaluate_model(
            model, test_loader, device, split_name='Test-Final', save_patient40_dir=None
        )
        test_df.to_csv(os.path.join(args.save_dir, 'test_per_frame_metrics.csv'), index=False)
        print(f"✓ Test finished: {len(test_df)} frames processed.")
        print(f"  Macro Dice: {test_summary.get('macro_dice', 0):.4f} | ASSD: {test_summary.get('macro_assd', 0):.2f} px")
        
    metrics_summary = {
        'val_summary': val_summary,
        'test_summary': test_summary,
        'checkpoint_path': ckpt_path,
        'epoch_restored': epoch_trained
    }
    summary_path = os.path.join(args.save_dir, 'metrics_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"✓ Metrics summary exported to: {summary_path}")
    
    print("\n" + "="*65)
    print("FINAL BENCHMARK RESULTS — EXPERIMENT_4 (Landmark Master Tokens)")
    print("="*65)
    print(f"| Metric            | Validation (122 frames) | Test (109 frames) |")
    print(f"|-------------------|-------------------------|-------------------|")
    print(f"| Macro Dice        | {val_summary.get('macro_dice', 0):.4f}                  | {test_summary.get('macro_dice', 0):.4f}            |")
    print(f"| Macro ASSD (px)   | {val_summary.get('macro_assd', 0):.2f} px               | {test_summary.get('macro_assd', 0):.2f} px         |")
    print(f"| Macro IoU         | {val_summary.get('macro_iou', 0):.4f}                  | {test_summary.get('macro_iou', 0):.4f}            |")
    print(f"| Ridge Dice        | {val_summary.get('ridge_dice', 0):.4f}                  | {test_summary.get('ridge_dice', 0):.4f}            |")
    print(f"| Silhouette Dice   | {val_summary.get('sil_dice', 0):.4f}                  | {test_summary.get('sil_dice', 0):.4f}            |")
    print(f"| Falciform Dice    | {val_summary.get('falc_dice', 0):.4f}                  | {test_summary.get('falc_dice', 0):.4f}            |")
    print(f"| Patient 40 Dice   | {val_summary.get('patient_40_dice', 0):.4f}                  | —                 |")
    print(f"| Inference FPS     | {val_summary.get('fps', 0):.2f} FPS                 | —                 |")
    print("="*65)
    
    zip_path = os.path.join(args.save_dir, 'results.zip')
    create_results_zip(args.save_dir, zip_path)

if __name__ == '__main__':
    main()
