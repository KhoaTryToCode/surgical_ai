"""
Full Training Script for EXPERIMENT_10 (Depth-Geometric Junction-Steered Mask2Former).

Features:
  - Pretrained ADE20K Swin-Tiny Backbone (strictly 3-channel RGB, preserving 100% ImageNet weights)
  - Dedicated 4-Stage DepthGeometryEncoder extracting 3D surface geometry at stride 16
  - Gated Residual Feature Fusion Layer at stride 16
  - Continuous 4-Query Junction Anchor Head (from EXPERIMENT_5) with auxiliary visibility
  - Multi-Task Optimization (Hungarian Loss + Smooth-L1 Coordinate Loss + Auxiliary BCE)
  - Differential Learning Rates (Backbone: 1e-5, Depth Encoder & Heads: 1e-4)
  - CosineAnnealingLR Scheduler
  - Automated exp10_best_model.pth checkpointing on Validation Macro Dice
  - Complete post-training evaluation, Patient 40 diagnostic rendering, and results.zip packaging
  - Fast --smoke-test mode for pipeline verification
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

# NumPy 2.0 compatibility
for _a, _v in [('Inf', np.inf), ('NAN', np.nan), ('NaN', np.nan), ('PINF', np.inf), ('NINF', -np.inf)]:
    if not hasattr(np, _a):
        setattr(np, _a, _v)

# Ensure workspace root is on sys.path
_WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)

from experiments.EXPERIMENT_10.utils.dataset import L3DDataset
from experiments.EXPERIMENT_10.models.depth_junction_steered_mask2former import DepthJunctionSteeredMask2Former
from experiments.EXPERIMENT_10.models.losses import DepthJunctionLoss
from experiments.EXPERIMENT_10.scripts.evaluate import run_evaluation, rasterize_class_map
from experiments.EXPERIMENT_10.utils.metrics import evaluate_frame_metrics


def collate_fn_l3d(batch):
    pixel_values = torch.stack([b['pixel_values'] for b in batch])
    depth_maps = torch.stack([b['depth_map'] for b in batch])
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
            mask_labels_list.append(torch.zeros((1, m.shape[0], m.shape[1]), dtype=torch.float32))
            class_labels_list.append(torch.tensor([0], dtype=torch.long))
        else:
            binary_masks = []
            classes = []
            for c in unique_classes:
                binary_masks.append((m == c).float())
                classes.append(c.long())
            mask_labels_list.append(torch.stack(binary_masks))
            class_labels_list.append(torch.tensor(classes, dtype=torch.long))
            
    return {
        'pixel_values': pixel_values,
        'depth_map': depth_maps,
        'masks': masks,
        'junction_coords': junction_coords,
        'junction_vis': junction_vis,
        'filenames': filenames,
        'is_patient_40': is_p40s,
        'mask_labels': mask_labels_list,
        'class_labels': class_labels_list
    }


def train_experiment_10(args):
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    print("=" * 65)
    print("🚀 Initiating Training: EXPERIMENT_10 (Depth-Geometric Junction-Steered M2F)")
    print(f"Device: {device} | Canvas: 1024x1024 | Batch Size: {args.batch_size} (Accum: {args.accum_steps})")
    print(f"LR Backbone: {args.lr_backbone} | LR Heads & Depth Encoder: {args.lr_head}")
    print(f"Loss Balance: Hungarian=1.0 | Coord L1={args.lambda_coord} | Vis BCE={args.lambda_vis}")
    print("=" * 65)
    
    # 1. Dataset & DataLoaders
    train_dataset = L3DDataset(split='Train', data_dir=args.data_dir)
    val_dataset = L3DDataset(split='Val', data_dir=args.data_dir)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn_l3d,
        num_workers=2 if args.device == 'cuda' else 0,
        pin_memory=(args.device == 'cuda')
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1 if args.device == 'cuda' else 0
    )
    
    # 2. Build Model
    model = DepthJunctionSteeredMask2Former(num_labels=4)
    model.to(device)
    
    # 3. Parameter Grouping with Differential Learning Rates
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'm2f.model.pixel_level_module.encoder' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
            
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.lr_backbone},
        {'params': head_params, 'lr': args.lr_head}
    ], weight_decay=args.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )
    
    criterion = DepthJunctionLoss(
        lambda_m2f=1.0,
        lambda_coord=args.lambda_coord,
        lambda_vis=args.lambda_vis
    )
    
    use_amp = (args.device == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    
    best_val_dice = 0.0
    best_ckpt_path = os.path.join(args.checkpoint_dir, "exp10_best_model.pth")
    training_log = []
    
    epochs_to_run = 1 if args.smoke_test else args.epochs
    
    # 4. Training Loop
    for epoch in range(1, epochs_to_run + 1):
        model.train()
        epoch_losses = []
        epoch_m2f = []
        epoch_coord = []
        epoch_vis = []
        start_time = time.time()
        
        optimizer.zero_grad()
        
        for step, batch in enumerate(train_loader):
            pixel_values = batch['pixel_values'].to(device)
            depth_map = batch['depth_map'].to(device)
            gt_coords = batch['junction_coords'].to(device)
            gt_vis = batch['junction_vis'].to(device)
            
            mask_labels = [m.to(device) for m in batch['mask_labels']]
            class_labels = [c.to(device) for c in batch['class_labels']]
            
            with torch.amp.autocast(device_type='cuda' if args.device == 'cuda' else 'cpu', enabled=use_amp):
                outputs = model(
                    pixel_values=pixel_values,
                    depth_map=depth_map,
                    mask_labels=mask_labels,
                    class_labels=class_labels
                )
                loss, loss_dict = criterion(outputs, gt_coords, gt_vis)
                loss_scaled = loss / args.accum_steps
                
            scaler.scale(loss_scaled).backward()
            
            if (step + 1) % args.accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            epoch_losses.append(loss_dict['loss'])
            epoch_m2f.append(loss_dict['m2f_loss'])
            epoch_coord.append(loss_dict['coord_loss'])
            epoch_vis.append(loss_dict['vis_loss'])
            
            if step % 20 == 0:
                print(f"Epoch [{epoch:02d}/{epochs_to_run:02d}] Step [{step:03d}/{len(train_loader):03d}] "
                      f"Loss: {loss_dict['loss']:.4f} (M2F: {loss_dict['m2f_loss']:.3f}, "
                      f"Coord: {loss_dict['coord_loss']:.4f}, Vis: {loss_dict['vis_loss']:.4f})")
                      
            if args.smoke_test and step >= 2:
                print("⚡ Smoke test: successfully completed 2 training steps.")
                break
                
        scheduler.step()
        elapsed = time.time() - start_time
        
        # 5. Validation Check
        val_dices = []
        val_assds = []
        p40_dices = []
        model.eval()
        
        with torch.no_grad():
            for v_step, v_batch in enumerate(val_loader):
                v_pixel = v_batch['pixel_values'].to(device)
                v_depth = v_batch['depth_map'].to(device)
                v_gt_mask = v_batch['mask'][0].numpy()
                v_is_p40 = v_batch['is_patient_40'][0]
                
                v_out = model(v_pixel, v_depth)
                pred_map = rasterize_class_map(v_out['masks_queries_logits'][0], v_out['class_queries_logits'][0])
                
                v_metrics = evaluate_frame_metrics(pred_map, v_gt_mask)
                val_dices.append(v_metrics['macro_dice'])
                val_assds.append(v_metrics['macro_assd'])
                if v_is_p40:
                    p40_dices.append(v_metrics['macro_dice'])
                    
                if args.smoke_test and v_step >= 2:
                    print("⚡ Smoke test: successfully completed 2 validation steps.")
                    break
                    
        mean_val_dice = float(np.mean(val_dices))
        mean_val_assd = float(np.mean(val_assds))
        mean_p40_dice = float(np.mean(p40_dices)) if len(p40_dices) > 0 else 0.0
        
        print(f"\n🌟 Epoch [{epoch:02d}/{epochs_to_run:02d}] Completed in {elapsed:.1f}s | "
              f"Val Macro Dice: {mean_val_dice*100:.2f}% | Val ASSD: {mean_val_assd:.2f} px | "
              f"Patient 40 Dice: {mean_p40_dice*100:.2f}%")
              
        epoch_record = {
            'epoch': epoch,
            'train_loss': float(np.mean(epoch_losses)),
            'train_m2f': float(np.mean(epoch_m2f)),
            'train_coord': float(np.mean(epoch_coord)),
            'train_vis': float(np.mean(epoch_vis)),
            'val_dice': mean_val_dice,
            'val_assd': mean_val_assd,
            'p40_dice': mean_p40_dice,
            'lr_backbone': scheduler.get_last_lr()[0],
            'lr_head': scheduler.get_last_lr()[1],
            'time_sec': elapsed
        }
        training_log.append(epoch_record)
        
        # Save Best Checkpoint
        if mean_val_dice > best_val_dice and not args.smoke_test:
            best_val_dice = mean_val_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': mean_val_dice,
                'val_assd': mean_val_assd,
                'p40_dice': mean_p40_dice
            }, best_ckpt_path)
            print(f"🏆 New Best Model Saved to: {best_ckpt_path} (Val Dice: {best_val_dice*100:.2f}%)")
            
    # Save training history
    log_df = pd.DataFrame(training_log)
    log_csv = os.path.join(args.output_dir, "training_history.csv")
    log_df.to_csv(log_csv, index=False)
    print(f"📈 Saved training history to: {log_csv}")
    
    # 6. Standalone Full Evaluation
    if not args.smoke_test and os.path.exists(best_ckpt_path):
        print("\n🏁 Executing Final Post-Training Standalone Evaluation on Best Checkpoint...")
        run_evaluation(
            model_path=best_ckpt_path,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            device=args.device
        )
    print("✅ EXPERIMENT_10 Pipeline Completed Successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train DepthJunctionSteeredMask2Former (EXP_10)")
    parser.add_argument('--data-dir', type=str, default=None, help="Root path to L3D dataset")
    parser.add_argument('--output-dir', type=str, default="experiments/EXPERIMENT_10/results")
    parser.add_argument('--checkpoint-dir', type=str, default="checkpoints")
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--accum-steps', type=int, default=2)
    parser.add_argument('--lr-backbone', type=float, default=1e-5)
    parser.add_argument('--lr-head', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--lambda-coord', type=float, default=5.0)
    parser.add_argument('--lambda-vis', type=float, default=1.0)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--smoke-test', action='store_true', help="Run 2-step verification dry run")
    args = parser.parse_args()
    
    train_experiment_10(args)
