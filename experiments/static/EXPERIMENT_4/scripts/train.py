#!/usr/bin/env python3
"""
EXPERIMENT_4: Main Training Script for Landmark Master Tokens + Patch Bézier Decoder
Integrates 3 Landmark Master Tokens with 64 Spatial Patch Queries in a 67-token Transformer Decoder.
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
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# NumPy 2.0 compatibility
for _a, _v in [('Inf', np.inf), ('NAN', np.nan), ('NaN', np.nan), ('PINF', np.inf), ('NINF', -np.inf)]:
    if not hasattr(np, _a): setattr(np, _a, _v)

# Ensure workspace root is on path for cross-environment portability
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from experiments.EXPERIMENT_4.models.landmark_bezier_model import LandmarkBezierPatchModel
from experiments.EXPERIMENT_4.models.landmark_losses import LandmarkBezierLoss
from experiments.EXPERIMENT_4.utils.dataset import LandmarkBezierDataset
from experiments.EXPERIMENT_4.utils.metrics import evaluate_model

def get_autocast_ctx(device):
    if device.type == 'cuda':
        return torch.amp.autocast('cuda')
    elif device.type == 'mps':
        return torch.amp.autocast('cpu')
    else:
        return torch.amp.autocast('cpu', enabled=False)

def get_grad_scaler(device):
    if device.type == 'cuda':
        return torch.amp.GradScaler('cuda')
    return None

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
    parser = argparse.ArgumentParser(description="Train LandmarkBezierPatchModel (EXPERIMENT_4)")
    parser.add_argument('--train_dir', type=str, default='data/L3D/Train')
    parser.add_argument('--val_dir', type=str, default='data/L3D/Val')
    parser.add_argument('--test_dir', type=str, default='data/L3D/Test')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--accumulation_steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=8e-5)
    parser.add_argument('--weight_decay', type=float, default=3e-5)
    parser.add_argument('--grid_size', type=int, default=8)
    parser.add_argument('--save_dir', type=str, default='experiments/EXPERIMENT_4/results/run_01')
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--eval_splits', type=str, choices=['val', 'both'], default='both')
    parser.add_argument('--single_scale', action='store_true', default=True,
                        help="Lock all decoder layers to stride-16 features (prevents scale hopping)")
    parser.add_argument('--multi_scale', dest='single_scale', action='store_false',
                        help="Enable multi-scale cycling across stride-32, 16, 8")
    
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    autocast_ctx = get_autocast_ctx(device)
    scaler = get_grad_scaler(device)
    
    train_dataset = LandmarkBezierDataset(args.train_dir, grid_size=args.grid_size)
    val_dataset   = LandmarkBezierDataset(args.val_dir, grid_size=args.grid_size)
    
    nw = args.num_workers if args.num_workers is not None else min(4, os.cpu_count() or 1)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=nw, pin_memory=(device.type=='cuda'))
    val_loader   = DataLoader(val_dataset,   batch_size=max(1, args.batch_size), shuffle=False, num_workers=nw, pin_memory=(device.type=='cuda'))
    
    print(f"Loaded datasets: Train={len(train_dataset)} frames, Val={len(val_dataset)} frames")
    print(f"Decoder setup: single_scale={args.single_scale}, grid_size={args.grid_size} (64 patches + 3 landmark tokens = 67 tokens)")
    
    model = LandmarkBezierPatchModel(
        grid_size=args.grid_size,
        num_classes=4,
        embed_dim=256,
        num_decoder_layers=6,
        single_scale=args.single_scale
    ).to(device)
    
    criterion = LandmarkBezierLoss(
        lambda_cls=2.0,
        lambda_ctrl=5.0,
        lambda_sample=2.0,
        lambda_cont=1.0,
        lambda_tan=0.5,
        lambda_lm=1.0,
        continuity_phase_epoch=31
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    best_val_dice = 0.0
    training_log = []
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        step_count = 0
        optimizer.zero_grad()
        
        loss_accum = {'cls': 0.0, 'ctrl': 0.0, 'sample': 0.0, 'cont': 0.0, 'tan': 0.0, 'lm_bce': 0.0, 'lm_com': 0.0}
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{args.epochs:02d} [Train]")
        for step, batch in enumerate(pbar):
            pixel_values, _, target_class, target_bezier, active_mask, target_lm_pres, target_lm_cent, _ = batch
            
            pixel_values      = pixel_values.to(device)
            target_class      = target_class.to(device)
            target_bezier     = target_bezier.to(device)
            active_mask       = active_mask.to(device)
            target_lm_pres    = target_lm_pres.to(device)
            target_lm_cent    = target_lm_cent.to(device)
            
            with autocast_ctx:
                pred_class, pred_bezier, pred_presence, pred_centroid = model(pixel_values)
                loss, loss_dict = criterion(
                    pred_class, pred_bezier, pred_presence, pred_centroid,
                    target_class, target_bezier, active_mask,
                    target_lm_pres, target_lm_cent, epoch=epoch
                )
                loss_scaled = loss / args.accumulation_steps
                
            if scaler:
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()
                
            if (step + 1) % args.accumulation_steps == 0 or (step + 1) == len(train_loader):
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                optimizer.zero_grad()
                
            epoch_loss += loss_dict['loss']
            step_count += 1
            loss_accum['cls']    += loss_dict['cls_loss']
            loss_accum['ctrl']   += loss_dict['ctrl_loss']
            loss_accum['cont']   += loss_dict['cont_loss']
            loss_accum['lm_com'] += loss_dict['lm_com']
            
            pbar.set_postfix({
                'loss':   f"{epoch_loss / step_count:.4f}",
                'cls':    f"{loss_accum['cls'] / step_count:.4f}",
                'ctrl':   f"{loss_accum['ctrl'] / step_count:.4f}",
                'lm_com': f"{loss_accum['lm_com'] / step_count:.4f}",
                'phase':  'P2' if epoch >= 31 else 'P1'
            })
            
        scheduler.step()
        
        # Validation Evaluation
        val_summary, _ = evaluate_model(model, val_loader, device, split_name=f'Val-Ep{epoch}')
        current_val_dice = val_summary['macro_dice']
        
        print(f"\n Epoch {epoch:02d}: TrainLoss={epoch_loss/step_count:.4f} | "
              f"Val MacroDice={current_val_dice:.4f} | Ridge={val_summary['ridge_dice']:.4f} | "
              f"Sil={val_summary['sil_dice']:.4f} | Falc={val_summary['falc_dice']:.4f} | "
              f"ASSD={val_summary['macro_assd']:.2f}px")
              
        if current_val_dice > best_val_dice:
            best_val_dice = current_val_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_dice': best_val_dice,
                'grid_size': args.grid_size,
                'single_scale': args.single_scale,
            }, os.path.join(args.save_dir, 'best_model.pth'))
            print(f'   ⭐ New best model saved! Val MacroDice: {best_val_dice:.4f}')
            
        training_log.append({
            'epoch': epoch,
            'train_loss': epoch_loss / step_count,
            'val_macro_dice': current_val_dice,
            'val_macro_assd': val_summary['macro_assd'],
            'val_ridge_dice': val_summary['ridge_dice'],
            'val_sil_dice':   val_summary['sil_dice'],
            'val_falc_dice':  val_summary['falc_dice'],
            'val_p40_dice':   val_summary['patient_40_dice'],
            'lr': scheduler.get_last_lr()[0],
        })
        
    pd.DataFrame(training_log).to_csv(os.path.join(args.save_dir, 'training_log.csv'), index=False)
    
    # ── Final Benchmark Evaluation ───────────────────────────────────────────────
    print("\n" + "="*70)
    print("Running Final Benchmark Evaluation on Best Model Checkpoint...")
    print("="*70)
    
    checkpoint = torch.load(os.path.join(args.save_dir, 'best_model.pth'), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']} with Val MacroDice {checkpoint['best_val_dice']:.4f}")
    
    patient40_dir = os.path.join(args.save_dir, 'patient_40_diagnostics_final')
    final_val_summary, final_val_df = evaluate_model(
        model, val_loader, device, split_name='Val-Final', save_patient40_dir=patient40_dir
    )
    final_val_df.to_csv(os.path.join(args.save_dir, 'val_predictions.csv'), index=False)
    
    final_test_summary = {}
    if args.eval_splits == 'both' and os.path.exists(args.test_dir):
        test_dataset = LandmarkBezierDataset(args.test_dir, grid_size=args.grid_size)
        test_loader  = DataLoader(test_dataset, batch_size=max(1, args.batch_size), shuffle=False, num_workers=nw)
        final_test_summary, test_df = evaluate_model(model, test_loader, device, split_name='Test-Final')
        test_df.to_csv(os.path.join(args.save_dir, 'test_predictions.csv'), index=False)
        
    metrics_summary = {
        'val_summary': final_val_summary,
        'test_summary': final_test_summary,
        'best_epoch': checkpoint['epoch'],
    }
    with open(os.path.join(args.save_dir, 'metrics_summary.json'), 'w') as f:
        json.dump(metrics_summary, f, indent=4)
        
    print("\n" + "="*65)
    print("FINAL BENCHMARK RESULTS — EXPERIMENT_4 (Landmark Master Tokens)")
    print("="*65)
    print(f"| Metric            | Validation (122 frames) | Test (109 frames) |")
    print(f"|-------------------|-------------------------|-------------------|")
    print(f"| Macro Dice        | {final_val_summary.get('macro_dice', 0):.4f}                  | {final_test_summary.get('macro_dice', 0):.4f}            |")
    print(f"| Macro ASSD (px)   | {final_val_summary.get('macro_assd', 0):.2f} px               | {final_test_summary.get('macro_assd', 0):.2f} px         |")
    print(f"| Macro IoU         | {final_val_summary.get('macro_iou', 0):.4f}                  | {final_test_summary.get('macro_iou', 0):.4f}            |")
    print(f"| Ridge Dice        | {final_val_summary.get('ridge_dice', 0):.4f}                  | {final_test_summary.get('ridge_dice', 0):.4f}            |")
    print(f"| Silhouette Dice   | {final_val_summary.get('sil_dice', 0):.4f}                  | {final_test_summary.get('sil_dice', 0):.4f}            |")
    print(f"| Falciform Dice    | {final_val_summary.get('falc_dice', 0):.4f}                  | {final_test_summary.get('falc_dice', 0):.4f}            |")
    print(f"| Patient 40 Dice   | {final_val_summary.get('patient_40_dice', 0):.4f}                  | —                 |")
    print(f"| Inference FPS     | {final_val_summary.get('fps', 0):.2f} FPS                 | —                 |")
    print("="*65)
    
    create_results_zip(args.save_dir, os.path.join(args.save_dir, 'results.zip'))

if __name__ == '__main__':
    main()
