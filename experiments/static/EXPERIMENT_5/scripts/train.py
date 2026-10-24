"""
Training Script for EXPERIMENT_5 (Junction-Steered Mask2Former).
Features:
  - Pretrained ADE20K Swin-Tiny Backbone + MSDeformAttn Pixel Decoder
  - 4-Query Junction Anchor Head + Dynamic Query Steering Block
  - Multi-Task Optimization (Hungarian Loss + Junction Coordinate L1 + Visibility BCE)
  - Differential Learning Rates (Backbone: 1e-5, Decoder & Heads: 1e-4)
  - CosineAnnealingLR Scheduler
  - Validation tracking with automated best_model.pth checkpointing
  - Fast --smoke_test mode for pipeline verification
"""
import os
import sys
import time
import json
import zipfile
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

# Ensure workspace root is on sys.path
_WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)

from experiments.EXPERIMENT_5.utils.dataset import L3DDataset
from experiments.EXPERIMENT_5.models.junction_steered_mask2former import JunctionSteeredMask2Former
from experiments.EXPERIMENT_5.models.losses import JunctionSteeredLoss
from experiments.EXPERIMENT_5.scripts.evaluate import run_evaluation

def collate_fn_l3d(batch):
    pixel_values = torch.stack([b['pixel_values'] for b in batch])
    masks = torch.stack([b['mask'] for b in batch])
    junction_coords = torch.stack([b['junction_coords'] for b in batch])
    junction_vis = torch.stack([b['junction_vis'] for b in batch])
    filenames = [b['filename'] for b in batch]
    is_p40s = [b['is_patient_40'] for b in batch]
    
    # Prepare binary mask_labels and class_labels list for Mask2Former Hungarian matching
    mask_labels_list = []
    class_labels_list = []
    
    for b in range(len(batch)):
        m = masks[b] # (H, W)
        unique_classes = torch.unique(m)
        unique_classes = unique_classes[unique_classes > 0] # Filter out background (0)
        
        if len(unique_classes) == 0:
            # Dummy target if empty frame
            mask_labels_list.append(torch.zeros((1, m.shape[0], m.shape[1]), dtype=torch.float32))
            class_labels_list.append(torch.zeros(1, dtype=torch.int64))
        else:
            binary_masks = torch.stack([(m == c).float() for c in unique_classes]) # (K, H, W)
            mask_labels_list.append(binary_masks)
            class_labels_list.append(unique_classes.long())
            
    return {
        'pixel_values': pixel_values,
        'mask': masks,
        'mask_labels': mask_labels_list,
        'class_labels': class_labels_list,
        'junction_coords': junction_coords,
        'junction_vis': junction_vis,
        'filename': filenames,
        'is_patient_40': is_p40s
    }

def build_optimizer_and_scheduler(model, lr_backbone=1e-5, lr_head=1e-4, weight_decay=1e-4, epochs=60):
    """
    Differential learning rates: Lower for pretrained backbone, standard for decoder & junction head.
    """
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'pixel_level_module.encoder' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
            
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': lr_backbone},
        {'params': head_params, 'lr': lr_head}
    ], weight_decay=weight_decay)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    return optimizer, scheduler

def create_results_zip(source_dir, output_zip_path):
    """
    Packages evaluation outputs, checkpoints, and Patient 40 diagnostic montages into a zip archive
    matching the project standard of EXPERIMENT_3 and EXPERIMENT_4.
    """
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(source_dir):
            for file in files:
                if file.endswith('.zip'):
                    continue
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, source_dir)
                zipf.write(file_path, arcname)
    print(f"📦 Results archive packaged: {output_zip_path} ({os.path.getsize(output_zip_path)/(1024*1024):.2f} MB)")

def main():
    parser = argparse.ArgumentParser(description="Train Junction-Steered Mask2Former (EXPERIMENT_5)")
    parser.add_argument('--data_dir', type=str, default=None, help="L3D dataset root")
    parser.add_argument('--out_dir', type=str, default=None, help="Output directory for checkpoints and logs")
    parser.add_argument('--epochs', type=int, default=60, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=2, help="Batch size per GPU step")
    parser.add_argument('--accum_steps', type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument('--lr_head', type=float, default=1e-4, help="Learning rate for heads & decoder")
    parser.add_argument('--lr_backbone', type=float, default=1e-5, help="Learning rate for Swin backbone")
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--lambda_coord', type=float, default=5.0, help="Weight for junction coordinate loss")
    parser.add_argument('--lambda_vis', type=float, default=1.0, help="Weight for junction visibility loss")
    parser.add_argument('--num_workers', type=int, default=4, help="DataLoader workers")
    parser.add_argument('--eval_splits', type=str, default='both', choices=['val', 'both'], help="Splits to evaluate after training")
    parser.add_argument('--smoke_test', action='store_true', help="Run 1 step of train and val for verification")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    if args.out_dir is None:
        args.out_dir = os.path.join(_WORKSPACE_ROOT, 'experiments/EXPERIMENT_5/results')
    os.makedirs(args.out_dir, exist_ok=True)
    
    device = torch.device(args.device)
    print(f"🖥️ Using device: {device}")
    
    # 1. Datasets & DataLoaders
    train_dataset = L3DDataset(split='Train', data_dir=args.data_dir)
    val_dataset = L3DDataset(split='Val', data_dir=args.data_dir)
    test_dataset = L3DDataset(split='Test', data_dir=args.data_dir) if args.eval_splits == 'both' else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn_l3d,
        pin_memory=(device.type == 'cuda')
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_l3d,
        pin_memory=(device.type == 'cuda')
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_l3d,
        pin_memory=(device.type == 'cuda')
    ) if test_dataset else None
    
    # 2. Build Model, Loss, Optimizer & AMP Scaler
    model = JunctionSteeredMask2Former(num_labels=4).to(device)
    criterion = JunctionSteeredLoss(lambda_m2f=1.0, lambda_coord=args.lambda_coord, lambda_vis=args.lambda_vis)
    optimizer, scheduler = build_optimizer_and_scheduler(
        model,
        lr_backbone=args.lr_backbone,
        lr_head=args.lr_head,
        weight_decay=args.weight_decay,
        epochs=args.epochs
    )
    use_amp = (device.type == 'cuda')
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    if use_amp:
        print(f"⚡ AMP enabled: dtype={amp_dtype}")
    
    # Smoke Test
    if args.smoke_test:
        print("\n🧪 Running Smoke Test (1 iteration)...")
        model.train()
        for batch in train_loader:
            pixel_values = batch['pixel_values'].to(device)
            mask_labels = [m.to(device) for m in batch['mask_labels']]
            class_labels = [c.to(device) for c in batch['class_labels']]
            gt_j_coords = batch['junction_coords'].to(device)
            gt_j_vis = batch['junction_vis'].to(device)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                outputs = model(pixel_values, mask_labels=mask_labels, class_labels=class_labels)
                total_loss, loss_dict = criterion(outputs, gt_j_coords, gt_j_vis)
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            print(f"   [Smoke Test Loss] Total: {loss_dict['loss']:.4f} | M2F: {loss_dict['m2f_loss']:.4f} | Coord: {loss_dict['coord_loss']:.4f} | Vis: {loss_dict['vis_loss']:.4f}")
            break
            
        val_summary, val_df = run_evaluation(model, val_loader, device, split_name="SmokeVal", out_dir=args.out_dir)
        print(f"✅ Smoke Test Passed! Val Frames: {len(val_df)} | Macro Dice: {val_summary['macro_dice']:.2%}")
        return

    # 3. Main Training Loop
    print(f"\n🚀 Starting Training: {args.epochs} Epochs | Effective Batch Size: {args.batch_size * args.accum_steps}...")
    best_val_dice = 0.0
    training_log = []
    
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        epoch_losses = {'loss': [], 'm2f_loss': [], 'coord_loss': [], 'vis_loss': []}
        
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            pixel_values = batch['pixel_values'].to(device)
            mask_labels = [m.to(device) for m in batch['mask_labels']]
            class_labels = [c.to(device) for c in batch['class_labels']]
            gt_j_coords = batch['junction_coords'].to(device)
            gt_j_vis = batch['junction_vis'].to(device)
            
            with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                outputs = model(pixel_values, mask_labels=mask_labels, class_labels=class_labels)
                total_loss, loss_dict = criterion(outputs, gt_j_coords, gt_j_vis)
                # Gradient accumulation
                loss_scaled = total_loss / float(args.accum_steps)
            
            scaler.scale(loss_scaled).backward()
            
            for k in epoch_losses.keys():
                epoch_losses[k].append(loss_dict[k])
                
            if (step + 1) % args.accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            if (step + 1) % 50 == 0 or (step + 1) == len(train_loader):
                print(f"Epoch [{epoch}/{args.epochs}] Step [{step+1}/{len(train_loader)}] Loss: {np.mean(epoch_losses['loss']):.4f} (M2F: {np.mean(epoch_losses['m2f_loss']):.4f}, Coord: {np.mean(epoch_losses['coord_loss']):.4f})")
                
        scheduler.step()
        epoch_duration = time.time() - t0
        
        # Validation Evaluation
        val_summary, val_df = run_evaluation(model, val_loader, device, split_name="Val", out_dir=args.out_dir)
        val_dice = val_summary['macro_dice']
        val_assd = val_summary['macro_assd']
        p40_dice = val_summary['patient_40_dice']
        
        log_entry = {
            'epoch': epoch,
            'train_loss': float(np.mean(epoch_losses['loss'])),
            'm2f_loss': float(np.mean(epoch_losses['m2f_loss'])),
            'coord_loss': float(np.mean(epoch_losses['coord_loss'])),
            'vis_loss': float(np.mean(epoch_losses['vis_loss'])),
            'val_macro_dice': val_dice,
            'val_macro_assd': val_assd,
            'val_ridge_dice': val_summary['ridge_dice'],
            'val_sil_dice': val_summary['sil_dice'],
            'val_falc_dice': val_summary['falc_dice'],
            'val_p40_dice': p40_dice,
            'val_p40_assd': val_summary['patient_40_assd'],
            'duration_sec': epoch_duration
        }
        training_log.append(log_entry)
        pd.DataFrame(training_log).to_csv(os.path.join(args.out_dir, 'training_log.csv'), index=False)
        
        print(f"\n📊 [Epoch {epoch}/{args.epochs}] Val Dice: {val_dice:.2%} | ASSD: {val_assd:.2f}px | P40 Dice: {p40_dice:.2%} | Time: {epoch_duration:.1f}s")
        
        # Save Best Checkpoint
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_ckpt_path = os.path.join(args.out_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_macro_dice': val_dice,
                'val_macro_assd': val_assd,
                'val_summary': val_summary
            }, best_ckpt_path)
            print(f"⭐ New Best Model Saved (Macro Dice: {val_dice:.2%}): {best_ckpt_path}")

    # 4. Final Comprehensive Evaluation on Best Checkpoint
    print("\n" + "="*70)
    print(f"🏁 TRAINING COMPLETE! Loading best checkpoint: {best_ckpt_path}")
    print("="*70)
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    
    val_summary, val_df = run_evaluation(model, val_loader, device, split_name="Val-Final", out_dir=args.out_dir)
    val_csv_path = os.path.join(args.out_dir, 'val_predictions.csv')
    val_df.to_csv(val_csv_path, index=False)
    print(f"💾 Saved Val Predictions: {val_csv_path}")
    
    test_summary = {}
    if test_loader:
        test_summary, test_df = run_evaluation(model, test_loader, device, split_name="Test-Final", out_dir=args.out_dir)
        test_csv_path = os.path.join(args.out_dir, 'test_predictions.csv')
        test_df.to_csv(test_csv_path, index=False)
        print(f"💾 Saved Test Predictions: {test_csv_path}")
        
    full_summary = {
        'val_summary': val_summary,
        'test_summary': test_summary,
        'best_epoch': ckpt.get('epoch', None),
        'checkpoint_path': best_ckpt_path
    }
    summary_json_path = os.path.join(args.out_dir, 'metrics_summary.json')
    with open(summary_json_path, 'w') as f:
        json.dump(full_summary, f, indent=4)
    print(f"🎉 Saved Metrics Summary JSON: {summary_json_path}")
    
    # 5. Package results into results.zip matching project standard
    zip_path = os.path.join(args.out_dir, 'results.zip')
    create_results_zip(args.out_dir, zip_path)

if __name__ == '__main__':
    main()
