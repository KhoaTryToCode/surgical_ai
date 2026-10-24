"""
Training Script for EXPERIMENT_12 (Dual-Decoder Multi-Task Mask2Former).
Foundation Learning across L3D and CholecSeg8k:
  1. Base Model: DualDecoderMask2Former
     - Shared Swin-Tiny Backbone + MSDeformAttn Pixel Decoder
     - Scene Decoder (CholecSeg8k 8,080 frames, 13 classes)
     - Landmark Decoder (L3D 921 frames, 3 landmark lines + 4 continuous junctions)
  2. Loss Balancing: L_total = L_L3D + lambda_cholec * L_cholec (lambda_cholec = 0.5)
  3. Strict L3D Evaluation: Validation macro dice on 122 L3D validation frames governs model selection.
  4. Fully conforms to Trap 14 archival standard (unified SAVE_DIR and complete results.zip).
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
from torch.utils.data import DataLoader, WeightedRandomSampler
from pathlib import Path

# Ensure workspace root is on sys.path
_WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)

from experiments.EXPERIMENT_12.utils.dataset import L3DDataset, collate_fn_l3d
from experiments.EXPERIMENT_12.utils.cholec_dataset import CholecSeg8kDataset, collate_fn_cholec
from experiments.EXPERIMENT_12.utils.dual_dataloader import DualStreamDataLoader
from experiments.EXPERIMENT_12.models.dual_decoder_mask2former import DualDecoderMask2Former
from experiments.EXPERIMENT_12.models.losses import MultiTaskDualLoss
from experiments.EXPERIMENT_12.scripts.evaluate import run_evaluation


def build_optimizer_and_scheduler(model, lr_backbone=1e-5, lr_head=1e-4, weight_decay=1e-4, epochs=60):
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
    Packages evaluation outputs, logs, CSVs, and Patient 40 diagnostic montages into a zip archive
    matching the Trap 14 project standard (excluding heavy model weights which reside adjacent in SAVE_DIR).
    """
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(source_dir):
            for file in files:
                if file.endswith('.zip') or file.endswith('.pth'):
                    continue
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, source_dir)
                zipf.write(file_path, arcname)
    print(f"📦 Results archive packaged: {output_zip_path} ({os.path.getsize(output_zip_path)/(1024*1024):.2f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Train Dual-Decoder Multi-Task Mask2Former (EXPERIMENT_12)")
    parser.add_argument('--data_dir', type=str, default=None, help="L3D dataset root")
    parser.add_argument('--cholec_dir', type=str, default=None, help="CholecSeg8k dataset root")
    parser.add_argument('--out_dir', type=str, default=None, help="Output directory for checkpoints and logs")
    parser.add_argument('--epochs', type=int, default=60, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=2, help="Batch size per GPU step")
    parser.add_argument('--accum_steps', type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument('--lr_head', type=float, default=1e-4, help="Learning rate for heads & decoders")
    parser.add_argument('--lr_backbone', type=float, default=1e-5, help="Learning rate for Swin backbone")
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--lambda_cholec', type=float, default=0.5, help="Weight for CholecSeg8k scene loss")
    parser.add_argument('--lambda_coord', type=float, default=5.0, help="Weight for junction coordinate loss")
    parser.add_argument('--lambda_vis', type=float, default=1.0, help="Weight for junction visibility loss")
    parser.add_argument('--tier1_weight', type=float, default=15.0, help="Sampling weight multiplier for Tier-1 deformed frames")
    parser.add_argument('--no_augment', action='store_true', help="Disable photometric ColorJitter during training")
    parser.add_argument('--num_workers', type=int, default=4, help="DataLoader workers")
    parser.add_argument('--eval_splits', type=str, default='both', choices=['val', 'both'], help="Splits to evaluate after training")
    parser.add_argument('--smoke_test', action='store_true', help="Run 1 step of train and val for verification")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    if args.out_dir is None:
        args.out_dir = os.path.join(_WORKSPACE_ROOT, 'experiments/EXPERIMENT_12/results')
    os.makedirs(args.out_dir, exist_ok=True)
    
    device = torch.device(args.device)
    print(f"🖥️ Using device: {device}")
    
    # 1. Datasets & DataLoaders
    use_augment = not args.no_augment
    train_l3d_dataset = L3DDataset(split='Train', data_dir=args.data_dir, augment=use_augment)
    val_l3d_dataset = L3DDataset(split='Val', data_dir=args.data_dir, augment=False)
    
    train_cholec_dataset = CholecSeg8kDataset(data_dir=args.cholec_dir)
    
    # L3D Sampler
    if args.tier1_weight > 1.0:
        sample_weights = train_l3d_dataset.get_sample_weights(
            deformed_weight=args.tier1_weight,
            base_weight=1.0
        )
        l3d_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        l3d_train_loader = DataLoader(
            train_l3d_dataset,
            batch_size=args.batch_size,
            sampler=l3d_sampler,
            num_workers=args.num_workers,
            pin_memory=(device.type == 'cuda'),
            collate_fn=collate_fn_l3d,
            drop_last=True
        )
    else:
        l3d_train_loader = DataLoader(
            train_l3d_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(device.type == 'cuda'),
            collate_fn=collate_fn_l3d,
            drop_last=True
        )
        
    # Cholec DataLoader (Shuffled stream)
    cholec_train_loader = DataLoader(
        train_cholec_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        collate_fn=collate_fn_cholec,
        drop_last=True
    )
    
    # Dual-Stream Synchronized Loader
    dual_train_loader = DualStreamDataLoader(l3d_train_loader, cholec_train_loader)
    
    # 2. Build Model & Multi-Task Loss
    model = DualDecoderMask2Former(
        model_name="facebook/mask2former-swin-tiny-ade-semantic",
        num_landmark_labels=4,
        num_scene_labels=13
    ).to(device)
    
    criterion = MultiTaskDualLoss(
        lambda_m2f=1.0,
        lambda_coord=args.lambda_coord,
        lambda_vis=args.lambda_vis,
        lambda_cholec=args.lambda_cholec
    )
    
    optimizer, scheduler = build_optimizer_and_scheduler(
        model,
        lr_backbone=args.lr_backbone,
        lr_head=args.lr_head,
        weight_decay=args.weight_decay,
        epochs=args.epochs
    )
    
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    
    print("\n🚀 Starting EXPERIMENT_12 Training...")
    print(f"   Epochs          : {args.epochs}")
    print(f"   Batch Size      : {args.batch_size} (effective {args.batch_size * args.accum_steps} with accum={args.accum_steps})")
    print(f"   L3D Steps/Epoch : {len(dual_train_loader)}")
    print(f"   lambda_cholec   : {args.lambda_cholec}")
    print(f"   Target Out Dir  : {args.out_dir}\n")
    
    best_val_dice = -1.0
    best_model_path = os.path.join(args.out_dir, 'best_model.pth')
    training_history = []
    
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        running_l3d_loss = 0.0
        running_cholec_loss = 0.0
        running_coord_loss = 0.0
        running_vis_loss = 0.0
        optimizer.zero_grad()
        
        step_count = 0
        for step, (batch_l3d, batch_cholec) in enumerate(dual_train_loader):
            step_count += 1
            
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                # 1. Forward L3D Landmark Batch
                out_l3d = model.forward_landmark(
                    pixel_values=batch_l3d['pixel_values'].to(device),
                    mask_labels=[m.to(device) for m in batch_l3d['mask_labels']],
                    class_labels=[c.to(device) for c in batch_l3d['class_labels']]
                )
                loss_l3d, loss_dict_l3d = criterion.compute_landmark_loss(
                    out_l3d,
                    gt_junction_coords=batch_l3d['junction_coords'].to(device),
                    gt_junction_vis=batch_l3d['junction_vis'].to(device)
                )
                
                # 2. Forward CholecSeg8k Scene Batch
                out_cholec = model.forward_scene(
                    pixel_values=batch_cholec['pixel_values'].to(device),
                    mask_labels=[m.to(device) for m in batch_cholec['mask_labels']],
                    class_labels=[c.to(device) for c in batch_cholec['class_labels']]
                )
                loss_cholec, loss_dict_cholec = criterion.compute_scene_loss(out_cholec)
                
                # 3. Total Balanced Multi-Task Loss
                loss_total = criterion.combine_losses(loss_l3d, loss_cholec)
                loss_step = loss_total / args.accum_steps
                
            scaler.scale(loss_step).backward()
            
            if (step + 1) % args.accum_steps == 0 or (step + 1) == len(dual_train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            running_loss += loss_total.item()
            running_l3d_loss += loss_l3d.item()
            running_cholec_loss += loss_cholec.item()
            running_coord_loss += loss_dict_l3d['coord_loss']
            running_vis_loss += loss_dict_l3d['vis_loss']
            
            if args.smoke_test:
                print("⚡ Smoke test: single training step completed successfully!")
                break
                
        scheduler.step()
        epoch_time = time.time() - t0
        
        avg_loss = running_loss / max(step_count, 1)
        avg_l3d_loss = running_l3d_loss / max(step_count, 1)
        avg_cholec_loss = running_cholec_loss / max(step_count, 1)
        avg_coord_loss = running_coord_loss / max(step_count, 1)
        avg_vis_loss = running_vis_loss / max(step_count, 1)
        
        # Periodic validation strictly on L3D validation frames
        is_eval_epoch = (epoch % 5 == 0) or (epoch == args.epochs) or args.smoke_test
        val_macro_dice = 0.0
        
        if is_eval_epoch:
            print(f"\n--- [Epoch {epoch:02d}/{args.epochs:02d}] Validating on L3D Val Set ---")
            val_summary = run_evaluation(
                model=model,
                data_dir=args.data_dir,
                out_dir=args.out_dir,
                device=device,
                eval_splits='val',
                max_frames=2 if args.smoke_test else None,
                num_workers=args.num_workers
            )
            val_macro_dice = val_summary['val_macro_dice']
            
            if val_macro_dice > best_val_dice:
                best_val_dice = val_macro_dice
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_dice': best_val_dice,
                    'args': vars(args)
                }, best_model_path)
                print(f"⭐ NEW BEST CHECKPOINT: Val Macro Dice = {best_val_dice:.4f} -> Saved to {best_model_path}")
                
        epoch_record = {
            'epoch': epoch,
            'train_loss': avg_loss,
            'train_l3d_loss': avg_l3d_loss,
            'train_cholec_loss': avg_cholec_loss,
            'train_coord_loss': avg_coord_loss,
            'train_vis_loss': avg_vis_loss,
            'val_macro_dice': val_macro_dice,
            'best_val_dice': best_val_dice,
            'lr_head': optimizer.param_groups[1]['lr'],
            'epoch_time_s': epoch_time
        }
        training_history.append(epoch_record)
        
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({epoch_time:.1f}s) | "
              f"L_tot: {avg_loss:.4f} | L_l3d: {avg_l3d_loss:.4f} | L_cholec: {avg_cholec_loss:.4f} | "
              f"L_coord: {avg_coord_loss:.4f} | ValDice: {val_macro_dice:.4f} (Best: {best_val_dice:.4f})")
              
        if args.smoke_test:
            break
            
    # Save training history
    history_df = pd.DataFrame(training_history)
    history_df.to_csv(os.path.join(args.out_dir, 'training_log.csv'), index=False)
    
    # Final evaluation on best model
    if os.path.exists(best_model_path):
        print(f"\n🏆 Loading BEST model from {best_model_path} for final full evaluation...")
        ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        
    final_summary = run_evaluation(
        model=model,
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        device=device,
        eval_splits=args.eval_splits,
        max_frames=2 if args.smoke_test else None,
        num_workers=args.num_workers
    )
    
    # Trap 14 Compliance: Package results.zip directly inside args.out_dir
    zip_path = os.path.join(args.out_dir, 'results.zip')
    create_results_zip(args.out_dir, zip_path)
    
    print("\n==================================================================")
    print("✅ EXPERIMENT_12 TRAINING & EVALUATION COMPLETED")
    print(f"   Best Val Macro Dice: {best_val_dice:.4f}")
    print(f"   Packaged Archive   : {zip_path}")
    print("==================================================================\n")


if __name__ == '__main__':
    main()
