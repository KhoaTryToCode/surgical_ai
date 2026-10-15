#!/usr/bin/env python3
"""
EXPERIMENT_3: Standalone Evaluation Script for Mask2Former Backbone + Patch Bézier Decoder
Loads an existing checkpoint (best_model.pth) and performs full validation and test set evaluation,
generates Patient 40 diagnostics, exports per-frame CSVs, and packages results into a zip archive.
"""
import os, sys, json, time, zipfile, argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from pathlib import Path

# NumPy 2.0 monkeypatch for legacy metrics
for _a, _v in [('Inf', np.inf), ('NAN', np.nan), ('NaN', np.nan), ('PINF', np.inf), ('NINF', -np.inf)]:
    if not hasattr(np, _a): setattr(np, _a, _v)

# Add workspace root to sys.path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from experiments.EXPERIMENT_3.models.bezier_patch_model import BezierPatchModel
from experiments.EXPERIMENT_3.utils.dataset import BezierPatchDataset
from experiments.EXPERIMENT_3.utils.metrics import evaluate_model

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
    parser = argparse.ArgumentParser(description="Evaluate trained BezierPatchModel checkpoint.")
    parser.add_argument('--checkpoint', type=str, default=None,
                        help="Path to best_model.pth. Defaults to save_dir/best_model.pth")
    parser.add_argument('--val_dir', type=str, default='data/L3D/Val',
                        help="Path to Validation dataset directory.")
    parser.add_argument('--test_dir', type=str, default='data/L3D/Test',
                        help="Path to Test dataset directory.")
    parser.add_argument('--save_dir', type=str, default='experiments/EXPERIMENT_3/results/run_01',
                        help="Directory to save evaluation artifacts and metrics.")
    parser.add_argument('--batch_size', type=int, default=4,
                        help="Batch size for evaluation.")
    parser.add_argument('--grid_size', type=int, default=8,
                        help="Patch grid size (8x8 = 64 patches).")
    parser.add_argument('--num_workers', type=int, default=4,
                        help="DataLoader worker processes.")
    parser.add_argument('--eval_splits', type=str, choices=['val', 'both'], default='both',
                        help="Splits to evaluate.")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Running evaluation on device: {device}")

    # Resolve checkpoint path
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_path = os.path.join(args.save_dir, 'best_model.pth')
    if not os.path.exists(ckpt_path):
        # Fallback check server standard path
        server_fallback = '/data/khoalq/checkpoints/exp3_patch_bezier_60ep/best_model.pth'
        if os.path.exists(server_fallback):
            ckpt_path = server_fallback

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    print(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    grid_size = checkpoint.get('grid_size', args.grid_size)
    model = BezierPatchModel(grid_size=grid_size, num_classes=4)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    epoch_trained = checkpoint.get('epoch', 'N/A')
    best_dice = checkpoint.get('best_val_dice', 0.0)
    print(f"✓ Model successfully restored (Epoch {epoch_trained}, Best Val Dice: {best_dice:.4f})")

    nw = args.num_workers if args.num_workers is not None else min(4, os.cpu_count() or 1)

    # 1. Validation Evaluation
    print(f"\nEvaluating Validation Set from: {args.val_dir} ...")
    val_dataset = BezierPatchDataset(args.val_dir, grid_size=grid_size)
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
        test_dataset = BezierPatchDataset(args.test_dir, grid_size=grid_size)
        test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=nw)
        test_summary, test_df = evaluate_model(
            model, test_loader, device, split_name='Test-Final', save_patient40_dir=None
        )
        test_df.to_csv(os.path.join(args.save_dir, 'test_per_frame_metrics.csv'), index=False)
        print(f"✓ Test finished: {len(test_df)} frames processed.")
        print(f"  Macro Dice: {test_summary.get('macro_dice', 0):.4f} | ASSD: {test_summary.get('macro_assd', 0):.2f} px")

    # 3. Export Summary JSON
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

    # 4. Print Benchmark Table
    print("\n" + "="*65)
    print("FINAL BENCHMARK RESULTS — EXPERIMENT_3 (Patch Bézier Decoder)")
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

    # 5. Create Archive
    zip_path = os.path.join(args.save_dir, 'results.zip')
    create_results_zip(args.save_dir, zip_path)

if __name__ == '__main__':
    main()
