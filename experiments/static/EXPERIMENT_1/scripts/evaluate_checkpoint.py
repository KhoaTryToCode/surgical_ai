import os
import sys
import json
import argparse
import time
import pandas as pd
import torch
from torch.utils.data import DataLoader

EXPERIMENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
WORKSPACE_ROOT = os.path.abspath(os.path.join(EXPERIMENT_DIR, '../..'))
REPO_ROOT = os.path.join(WORKSPACE_ROOT, 'repos/TopoNet')

if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from experiments.EXPERIMENT_1.utils.dataset import TopoNetDataset
from experiments.EXPERIMENT_1.models.toponet_ablation import TopoNetAblationModel
from experiments.EXPERIMENT_1.scripts.train_toponet import run_evaluation


def main():
    parser = argparse.ArgumentParser(description="Standalone TopoNet Checkpoint Evaluator")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to best_model.pth checkpoint")
    parser.add_argument('--ablation', type=str, required=True, 
                        choices=['full', 'baseline', 'wo_lper', 'wo_lcl', 'wo_lper_lcl', 'wo_btf'],
                        help="Ablation mode of the checkpoint")
    parser.add_argument('--val_dir', type=str, default='data/L3D/Val', help="Path to Val directory")
    parser.add_argument('--test_dir', type=str, default='data/L3D/Test', help="Path to Test directory")
    parser.add_argument('--val_depth_dir', type=str, default='data/L3D/Depth/val', help="Path to Val depth directory")
    parser.add_argument('--test_depth_dir', type=str, default='data/L3D/Depth/test', help="Path to Test depth directory")
    parser.add_argument('--save_dir', type=str, default=None, help="Directory to save evaluation results")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")

    if args.save_dir is None:
        args.save_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    os.makedirs(args.save_dir, exist_ok=True)
    patient40_dir = os.path.join(args.save_dir, 'patient_40_diagnostics')

    # Device selection: CUDA -> CPU (Apple Silicon MPS has an adaptive_avg_pool2d non-divisible size bug in PyTorch PPM)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 80)
    print(f"🔬 TOPONET BENCHMARK EVALUATOR")
    print(f"   Checkpoint:       {args.checkpoint}")
    print(f"   Ablation Mode:    {args.ablation}")
    print(f"   Compute Device:   {device}")
    print(f"   Output Directory: {args.save_dir}")
    print("=" * 80)

    # 1. Build Model & Load Weights
    print("📦 Initializing TopoNet model architecture...")
    model = TopoNetAblationModel(ablation_mode=args.ablation).to(device)
    
    print(f"📥 Loading weights from {args.checkpoint}...")
    checkpoint_state = torch.load(args.checkpoint, map_location=device)
    if isinstance(checkpoint_state, dict) and 'state_dict' in checkpoint_state:
        checkpoint_state = checkpoint_state['state_dict']
    
    # Handle module prefix if trained with DataParallel
    cleaned_state = {}
    for k, v in checkpoint_state.items():
        key = k.replace('module.', '') if k.startswith('module.') else k
        cleaned_state[key] = v
        
    model.load_state_dict(cleaned_state)
    model.eval()
    print("✅ Model weights loaded successfully!")

    # 2. Evaluate Validation Set (122 frames)
    val_dataset = TopoNetDataset(args.val_dir, depth_dir=args.val_depth_dir, mode='val')
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    print(f"\n🚀 Evaluating Validation Set ({len(val_dataset)} frames)...")
    val_summary, df_val = run_evaluation(model, val_loader, device, split_name='Val', save_patient40_dir=patient40_dir)
    
    # Save per-frame validation CSV
    val_csv_path = os.path.join(args.save_dir, 'validation_per_frame_results.csv')
    df_val.to_csv(val_csv_path, index=False)
    print(f"📄 Validation per-frame metrics saved: {val_csv_path}")

    # 3. Evaluate Test Set (109 frames) if available
    test_summary = None
    if args.test_dir and os.path.exists(args.test_dir):
        test_dataset = TopoNetDataset(args.test_dir, depth_dir=args.test_depth_dir, mode='test')
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
        print(f"\n🚀 Evaluating Test Set ({len(test_dataset)} frames)...")
        test_summary, df_test = run_evaluation(model, test_loader, device, split_name='Test', save_patient40_dir=None)
        
        test_csv_path = os.path.join(args.save_dir, 'test_per_frame_results.csv')
        df_test.to_csv(test_csv_path, index=False)
        print(f"📄 Test per-frame metrics saved: {test_csv_path}")

    # 4. Save Combined Summary JSON
    summary_path = os.path.join(args.save_dir, 'summary_metrics.json')
    combined_summary = {
        'ablation_mode': args.ablation,
        'checkpoint_path': os.path.abspath(args.checkpoint),
        'val_metrics': val_summary,
        'test_metrics': test_summary
    }
    with open(summary_path, 'w') as f:
        json.dump(combined_summary, f, indent=2)
    print(f"📊 Summary metrics JSON saved: {summary_path}")

    # 5. Print Formatted Leaderboard Markdown Table
    print("\n" + "=" * 80)
    print("🏆 FINAL BENCHMARK EVALUATION RESULTS")
    print("=" * 80)
    print(f"Ablation Mode:        {args.ablation.upper()}")
    print(f"Validation Macro DSC: {val_summary['macro_dice']*100:.2f}%")
    print(f"Validation Mean IoU:  {val_summary['macro_iou']*100:.2f}%")
    print(f"Validation ASSD:      {val_summary['macro_assd']:.2f} px")
    print(f"Foreground DSC:       {val_summary['fg_dice']*100:.2f}%")
    print(f"Silhouette DSC:       {val_summary['sil_dice']*100:.2f}%")
    print(f"Ridge DSC:            {val_summary['ridge_dice']*100:.2f}%")
    print(f"Falciform DSC:        {val_summary['falc_dice']*100:.2f}%")
    print(f"Patient 40 DSC:       {val_summary['patient_40_dice']*100:.2f}%")
    if test_summary:
        print(f"Test Macro DSC:       {test_summary['macro_dice']*100:.2f}%")
        print(f"Test Mean IoU:        {test_summary['macro_iou']*100:.2f}%")
        print(f"Test ASSD:            {test_summary['macro_assd']:.2f} px")
    print("=" * 80)

if __name__ == '__main__':
    main()
