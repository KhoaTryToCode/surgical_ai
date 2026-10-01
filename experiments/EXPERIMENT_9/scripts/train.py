"""
Training Script for EXPERIMENT_9: RGB-D Human-Anchor-Steered Mask2Former.

Strict Controlled Ablation on EXPERIMENT_8:
  - Architecture: RGB-D Swin-Tiny Backbone (4 input channels: RGB + Depth Anything v2)
  - Anchor Head: Exact same 4-Query Biological Junction Anchor Head
  - Query Steering: Exact same Dynamic Query Steering Block (Gate alpha=0.1)
  - Optimizer & Scheduler: AdamW (lr_backbone=1e-5, lr_head=1e-4), CosineAnnealingLR
  - Losses: Same Hungarian Matching Loss + lambda_coord=5.0 + lambda_vis=1.0
  - Batching: batch_size=2, accum_steps=2 (effective batch size 4) or configurable via CLI
"""
import os
import sys
import time
import json
import zipfile
import argparse
import numpy as np
import pandas as pd
import torch
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

from experiments.EXPERIMENT_9.utils.dataset import L3DRGBDDataset
from experiments.EXPERIMENT_9.models.rgbd_junction_steered_mask2former import RGBDJunctionSteeredMask2Former
from experiments.EXPERIMENT_9.models.losses import JunctionSteeredLoss
from experiments.EXPERIMENT_9.scripts.evaluate import run_evaluation

def collate_fn_l3d(batch):
    pixel_values = torch.stack([b['pixel_values'] for b in batch]) # (B, 4, H, W)
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
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(source_dir):
            for file in files:
                if file.endswith('.zip'):
                    continue
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, source_dir)
                zipf.write(file_path, arcname)
    print(f"📦 Results archive packaged: {output_zip_path} ({os.path.getsize(output_zip_path)/(1024*1024):.2f} MB)")

def train(args):
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("=" * 75)
    print("🚀 EXPERIMENT_9: RGB-D Human-Anchor-Steered Mask2Former Training")
    print(f"   Output dir: {args.out_dir}")
    print(f"   Device    : {device}")
    print(f"   Epochs    : {args.epochs}")
    print(f"   Batch size: {args.batch_size} (accum: {args.accum_steps})")
    print(f"   LRs       : Backbone={args.lr_backbone}, Head={args.lr_head}")
    print(f"   Loss Wgts : Coord={args.lambda_coord}, Vis={args.lambda_vis}")
    print("=" * 75)
    
    # 1. Dataset & DataLoaders
    train_dataset = L3DRGBDDataset(split='Train', data_dir=args.data_dir, anchor_json_path=args.anchor_json)
    val_dataset   = L3DRGBDDataset(split='Val', data_dir=args.data_dir)
    test_dataset  = L3DRGBDDataset(split='Test', data_dir=args.data_dir)
    
    if args.smoke_test:
        print("⚠️ SMOKE TEST MODE: Truncating datasets to 4 samples")
        train_dataset.json_files = train_dataset.json_files[:4]
        val_dataset.json_files = val_dataset.json_files[:4]
        test_dataset.json_files = test_dataset.json_files[:4]
        args.epochs = 1
        
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn_l3d,
        pin_memory=(device.type == 'cuda'),
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(4, args.num_workers),
        collate_fn=collate_fn_l3d
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(4, args.num_workers),
        collate_fn=collate_fn_l3d
    )
    
    # 2. Build Model & Criterion
    model = RGBDJunctionSteeredMask2Former(num_labels=4).to(device)
    criterion = JunctionSteeredLoss(
        lambda_m2f=1.0,
        lambda_coord=args.lambda_coord,
        lambda_vis=args.lambda_vis
    ).to(device)
    
    optimizer, scheduler = build_optimizer_and_scheduler(
        model,
        lr_backbone=args.lr_backbone,
        lr_head=args.lr_head,
        weight_decay=args.weight_decay,
        epochs=args.epochs
    )
    
    use_amp = (device.type == 'cuda')
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and amp_dtype == torch.float16))
    
    best_val_dice = -1.0
    best_epoch = -1
    best_checkpoint_path = os.path.join(args.out_dir, "best_model.pth")
    training_history = []
    
    # 3. Training Loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        epoch_m2f = []
        epoch_coord = []
        epoch_vis = []
        
        t0 = time.time()
        optimizer.zero_grad()
        
        for step, batch in enumerate(train_loader):
            pixel_values = batch['pixel_values'].to(device) # (B, 4, H, W)
            mask_labels = [m.to(device) for m in batch['mask_labels']]
            class_labels = [c.to(device) for c in batch['class_labels']]
            gt_j_coords = batch['junction_coords'].to(device)
            gt_j_vis = batch['junction_vis'].to(device)
            
            with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                outputs = model(
                    pixel_values=pixel_values,
                    mask_labels=mask_labels,
                    class_labels=class_labels
                )
                loss, loss_dict = criterion(outputs, gt_j_coords, gt_j_vis)
                loss_scaled = loss / args.accum_steps
                
            if scaler.is_enabled():
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()
                
            if (step + 1) % args.accum_steps == 0 or (step + 1) == len(train_loader):
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()
                optimizer.zero_grad()
                
            epoch_losses.append(loss_dict['loss'])
            epoch_m2f.append(loss_dict['m2f_loss'])
            epoch_coord.append(loss_dict['coord_loss'])
            epoch_vis.append(loss_dict['vis_loss'])
            
        scheduler.step()
        epoch_time = time.time() - t0
        
        mean_loss = float(np.mean(epoch_losses))
        mean_m2f = float(np.mean(epoch_m2f))
        mean_coord = float(np.mean(epoch_coord))
        mean_vis = float(np.mean(epoch_vis))
        
        # Periodic Validation (every 5 epochs, last 10 epochs, or smoke test)
        eval_this_epoch = (epoch % 5 == 0) or (epoch > (args.epochs - 10)) or args.smoke_test
        
        if eval_this_epoch:
            val_summary, _ = run_evaluation(model, val_loader, device, split_name="Val", out_dir=None)
            val_dice = val_summary['macro_dice']
            val_assd = val_summary['macro_assd']
            p40_dice = val_summary.get('patient_40_dice', 0.0)
            c08730_dice = val_summary.get('case_08730_dice', 0.0)
            
            print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({epoch_time:.1f}s) | "
                  f"Loss: {mean_loss:.4f} (M2F: {mean_m2f:.4f}, Coord: {mean_coord:.4f}, Vis: {mean_vis:.4f}) | "
                  f"Val Dice: {val_dice:.2%} (P40: {p40_dice:.2%}, Case 08730: {c08730_dice:.2%}) | ASSD: {val_assd:.2f}px")
                  
            record = {
                'epoch': epoch,
                'train_loss': mean_loss,
                'val_dice': val_dice,
                'val_assd': val_assd,
                'p40_dice': p40_dice,
                'case_08730_dice': c08730_dice
            }
            training_history.append(record)
            
            # Save best model
            if val_dice > best_val_dice:
                best_val_dice = val_dice
                best_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_dice': val_dice,
                    'val_assd': val_assd,
                    'summary': val_summary
                }, best_checkpoint_path)
                print(f"   🏆 New best checkpoint saved to: {best_checkpoint_path} (Val Dice: {best_val_dice:.2%})")
        else:
            print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({epoch_time:.1f}s) | "
                  f"Loss: {mean_loss:.4f} (M2F: {mean_m2f:.4f}, Coord: {mean_coord:.4f}, Vis: {mean_vis:.4f})")
                  
    print("\n" + "=" * 75)
    print(f"🏁 Training Finished! Best Epoch: {best_epoch} with Val Macro Dice: {best_val_dice:.2%}")
    print("=" * 75)
    
    # 4. Final Full Evaluation with Best Checkpoint
    if os.path.exists(best_checkpoint_path):
        print(f"\n📦 Loading best checkpoint from {best_checkpoint_path} for final evaluation...")
        ckpt = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
    
    print("\n📊 Running final full benchmark evaluation on Val and Test splits...")
    val_summary, val_df = run_evaluation(model, val_loader, device, split_name="Val", out_dir=args.out_dir)
    test_summary, test_df = run_evaluation(model, test_loader, device, split_name="Test", out_dir=args.out_dir)
    
    val_df.to_csv(os.path.join(args.out_dir, "val_predictions.csv"), index=False)
    val_df.to_csv(os.path.join(args.out_dir, "val_per_frame_results.csv"), index=False)
    test_df.to_csv(os.path.join(args.out_dir, "test_predictions.csv"), index=False)
    test_df.to_csv(os.path.join(args.out_dir, "test_per_frame_results.csv"), index=False)
    
    full_summary = {
        'experiment': 'EXPERIMENT_9_RGBD_Human_Anchor_Steered',
        'best_epoch': best_epoch,
        'val_summary': val_summary,
        'test_summary': test_summary,
        'training_history': training_history
    }
    with open(os.path.join(args.out_dir, "metrics_summary.json"), 'w') as f:
        json.dump(full_summary, f, indent=4)
    with open(os.path.join(args.out_dir, "summary_metrics.json"), 'w') as f:
        json.dump(full_summary, f, indent=4)
        
    zip_path = os.path.join(args.out_dir, "results.zip")
    create_results_zip(args.out_dir, zip_path)

def main():
    parser = argparse.ArgumentParser(description="Train EXPERIMENT_9 RGB-D Human-Anchor-Steered Mask2Former")
    parser.add_argument('--data_dir', type=str, default=None, help="Path to L3D dataset root")
    parser.add_argument('--anchor_json', type=str, default=None, help="Path to train_biological_anchors_human.json")
    parser.add_argument('--out_dir', type=str, default=None, help="Directory to save checkpoints and results")
    parser.add_argument('--epochs', type=int, default=60, help="Total training epochs")
    parser.add_argument('--batch_size', type=int, default=2, help="Per-step batch size")
    parser.add_argument('--accum_steps', type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument('--lr_backbone', type=float, default=1e-5, help="Backbone learning rate")
    parser.add_argument('--lr_head', type=float, default=1e-4, help="Heads & decoder learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="Weight decay")
    parser.add_argument('--lambda_coord', type=float, default=5.0, help="Coordinate loss weight")
    parser.add_argument('--lambda_vis', type=float, default=1.0, help="Visibility loss weight")
    default_workers = 0 if sys.platform == 'darwin' else 4
    parser.add_argument('--num_workers', type=int, default=default_workers, help="DataLoader workers")
    parser.add_argument('--eval_splits', type=str, default='both', help="Splits to evaluate after training")
    parser.add_argument('--smoke_test', action='store_true', help="Run 1-epoch / 4-sample test")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    if args.out_dir is None:
        args.out_dir = os.path.join(_WORKSPACE_ROOT, 'experiments/EXPERIMENT_9/results')
        
    train(args)

if __name__ == '__main__':
    main()
