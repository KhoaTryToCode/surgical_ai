"""
Training Pipeline for EXPERIMENT_15: Temporal-Augmented Mask2Former (T=3).
Features:
  - Pure vanilla Mask2Former base (ImageNet/ADE20k), zero junction priors
  - Patient-wise T=3 chronological clip dataloader with PatientStratifiedClipSampler
  - Gated Spatiotemporal Query Cross-Attention
  - Gradient accumulation, mixed precision (AMP), and validation after each epoch
  - Automatic results.zip packaging
"""
import os
import sys
import time
import json
import argparse
import zipfile
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path

# Ensure workspace root is on sys.path
_WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)

from experiments.EXPERIMENT_15_temporal_augmented_mask2former.utils.dataset_temporal import PatientTemporalL3DDataset
from experiments.EXPERIMENT_15_temporal_augmented_mask2former.utils.clip_sampler import PatientStratifiedClipSampler
from experiments.EXPERIMENT_15_temporal_augmented_mask2former.utils.metrics import evaluate_frame_metrics
from experiments.EXPERIMENT_15_temporal_augmented_mask2former.models.temporal_mask2former import TemporalMask2Former

def collate_clip_fn(batch):
    """
    Collate function for batches of (pixel_values, masks, meta).
    pixel_values: (B, T, 3, H, W)
    masks: (B, T, H, W)
    """
    pixel_values = torch.stack([b[0] for b in batch]) # (B, T, 3, H, W)
    masks = torch.stack([b[1] for b in batch])        # (B, T, H, W)
    metas = [b[2] for b in batch]
    
    B, T, H, W = masks.shape
    mask_labels_clip = []
    class_labels_clip = []
    
    for b in range(B):
        sample_masks = []
        sample_classes = []
        for t in range(T):
            m = masks[b, t] # (H, W)
            unique_classes = torch.unique(m)
            unique_classes = unique_classes[unique_classes > 0] # Filter out background (0)
            
            if len(unique_classes) == 0:
                sample_masks.append(torch.zeros((1, H, W), dtype=torch.float32))
                sample_classes.append(torch.zeros(1, dtype=torch.int64))
            else:
                binary_masks = torch.stack([(m == c).float() for c in unique_classes]) # (K, H, W)
                sample_masks.append(binary_masks)
                sample_classes.append(unique_classes.long())
                
        mask_labels_clip.append(sample_masks)
        class_labels_clip.append(sample_classes)
        
    return {
        'pixel_values': pixel_values,
        'masks': masks,
        'mask_labels_clip': mask_labels_clip,
        'class_labels_clip': class_labels_clip,
        'metas': metas
    }

def rasterize_class_map(masks_queries_logits, class_queries_logits, canvas_size=1024):
    """
    Argmax post-processing to obtain integer prediction map (0: BG, 1: Ridge, 2: Sil, 3: Falc).
    """
    masks_queries_logits = masks_queries_logits.float()
    class_queries_logits = class_queries_logits.float()
    
    # Resize mask logits to canvas size
    masks_prob = F.interpolate(
        masks_queries_logits.unsqueeze(0),
        size=(canvas_size, canvas_size),
        mode="bilinear",
        align_corners=False
    ).squeeze(0).sigmoid().cpu().numpy() # (100, H, W)
    
    class_probs = F.softmax(class_queries_logits, dim=-1).cpu().numpy() # (100, num_classes + 1)
    
    # Exclude background class (index 0)
    foreground_probs = class_probs[:, 1:] # (100, 3) -> col 0: Ridge, 1: Silhouette, 2: Falciform
    
    sem_seg = np.zeros((3, canvas_size, canvas_size), dtype=np.float32)
    for q_idx in range(masks_prob.shape[0]):
        m = masks_prob[q_idx]
        for c_idx in range(3):
            c_score = foreground_probs[q_idx, c_idx]
            if c_score > 0.2:
                sem_seg[c_idx] = np.maximum(sem_seg[c_idx], m * c_score)
                
    pred_map = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    for c_idx in range(3):
        bin_mask = sem_seg[c_idx] > 0.4
        pred_map[bin_mask] = (c_idx + 1)
    return pred_map

def run_evaluation(model, dataloader, device, split_name="Val", max_batches=None):
    model.eval()
    records = []
    
    with torch.no_grad():
        for b_idx, batch in enumerate(dataloader):
            if max_batches and b_idx >= max_batches:
                break
                
            pixel_values = batch['pixel_values'].to(device) # (B, T, 3, H, W)
            masks = batch['masks'].cpu().numpy()            # (B, T, H, W)
            metas = batch['metas']
            
            outputs = model(pixel_values)
            masks_logits = outputs['masks_queries_logits'] # (B, T, 100, H/4, W/4)
            class_logits = outputs['class_queries_logits'] # (B, T, 100, 4)
            
            B, T = masks_logits.shape[:2]
            target_t = T - 1 # Evaluate the target frame t given context t-2, t-1
            
            for b in range(B):
                m_pred = rasterize_class_map(masks_logits[b, target_t], class_logits[b, target_t], canvas_size=masks.shape[-1])
                m_gt = masks[b, target_t]
                
                metrics = evaluate_frame_metrics(m_pred, m_gt)
                metrics['patient'] = metas[b]['patient']
                metrics['frame_id'] = metas[b]['frame_ids'][target_t]
                metrics['is_patient_40'] = (metas[b]['patient'] == 'Patient_40')
                records.append(metrics)
                
    df = pd.DataFrame(records)
    summary = {
        'split': split_name,
        'frames': len(df),
        'macro_dice': float(df['macro_dice'].mean()),
        'macro_iou': float(df['macro_iou'].mean()),
        'macro_assd': float(df['macro_assd'].mean()),
        'dice_ridge': float(df['dice_ridge'].mean()),
        'dice_silhouette': float(df['dice_silhouette'].mean()),
        'dice_falciform': float(df['dice_falciform'].mean()),
    }
    if 'is_patient_40' in df and df['is_patient_40'].sum() > 0:
        p40_df = df[df['is_patient_40']]
        summary['p40_macro_dice'] = float(p40_df['macro_dice'].mean())
        summary['p40_macro_assd'] = float(p40_df['macro_assd'].mean())
    return summary, df

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

def main():
    parser = argparse.ArgumentParser(description="Train Temporal-Augmented Mask2Former (EXPERIMENT_15)")
    parser.add_argument('--data_dir', type=str, default=None, help="L3D dataset root")
    parser.add_argument('--train_dir', type=str, default=None)
    parser.add_argument('--val_dir', type=str, default=None)
    parser.add_argument('--test_dir', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--clip_len', type=int, default=3)
    parser.add_argument('--max_gap', type=int, default=240)
    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=1, help="Clips per step")
    parser.add_argument('--accum_steps', type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument('--lr_head', type=float, default=1e-4)
    parser.add_argument('--lr_backbone', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--eval_splits', type=str, default='both', choices=['val', 'both'])
    parser.add_argument('--smoke_test', action='store_true')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    if args.out_dir is None:
        args.out_dir = os.path.join(_WORKSPACE_ROOT, 'experiments/EXPERIMENT_15_temporal_augmented_mask2former/results')
    os.makedirs(args.out_dir, exist_ok=True)
    
    device = torch.device(args.device)
    print(f"🖥️ Using device: {device} | Base: facebook/mask2former-swin-tiny-ade-semantic")
    
    # 1. Datasets & Loaders
    train_dataset = PatientTemporalL3DDataset('Train', data_dir=args.data_dir, split_dir=args.train_dir, clip_len=args.clip_len, max_gap=args.max_gap, image_size=args.image_size)
    val_dataset = PatientTemporalL3DDataset('Val', data_dir=args.data_dir, split_dir=args.val_dir, clip_len=args.clip_len, max_gap=args.max_gap, image_size=args.image_size)
    test_dataset = PatientTemporalL3DDataset('Test', data_dir=args.data_dir, split_dir=args.test_dir, clip_len=args.clip_len, max_gap=args.max_gap, image_size=args.image_size) if args.eval_splits == 'both' else None
    
    train_sampler = PatientStratifiedClipSampler(train_dataset)
    num_workers = args.num_workers if device.type == 'cuda' else 0
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=num_workers, collate_fn=collate_clip_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_clip_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_clip_fn) if test_dataset else None
    
    # 2. Model, Optimizer & Scheduler
    model = TemporalMask2Former(
        model_name="facebook/mask2former-swin-tiny-ade-semantic",
        num_labels=4,
        clip_len=args.clip_len
    ).to(device)
    
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
        {'params': backbone_params, 'lr': args.lr_backbone},
        {'params': head_params, 'lr': args.lr_head}
    ], weight_decay=args.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    use_amp = (device.type == 'cuda')
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    
    # Smoke Test
    if args.smoke_test:
        print("\n🧪 Running Smoke Test (1 step)...")
        model.train()
        for batch in train_loader:
            pixel_values = batch['pixel_values'].to(device)
            mask_labels = [[m.to(device) for m in t_list] for t_list in batch['mask_labels_clip']]
            class_labels = [[c.to(device) for c in t_list] for t_list in batch['class_labels_clip']]
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                outputs = model(pixel_values, mask_labels_clip=mask_labels, class_labels_clip=class_labels)
                loss = outputs['loss']
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            print(f"   [Smoke Test Loss]: {loss.item():.4f} | Gamma: {outputs['gamma']:.4f}")
            break
        val_summary, _ = run_evaluation(model, val_loader, device, split_name="SmokeVal", max_batches=2)
        print(f"✅ Smoke Test Passed! Val Dice: {val_summary['macro_dice']:.2%}")
        return

    # 3. Main Training Loop
    print(f"\n🚀 Starting Training: {args.epochs} Epochs | Effective Batch Size: {args.batch_size * args.accum_steps} clips...")
    best_val_dice = -1.0
    best_ckpt_path = os.path.join(args.out_dir, 'best_model.pth')
    training_log = []
    
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_losses = []
        
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            pixel_values = batch['pixel_values'].to(device)
            mask_labels = [[m.to(device) for m in t_list] for t_list in batch['mask_labels_clip']]
            class_labels = [[c.to(device) for c in t_list] for t_list in batch['class_labels_clip']]
            
            with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                outputs = model(pixel_values, mask_labels_clip=mask_labels, class_labels_clip=class_labels)
                loss = outputs['loss'] / args.accum_steps
                
            scaler.scale(loss).backward()
            
            if (step + 1) % args.accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            train_losses.append(outputs['loss'].item())
            
        scheduler.step()
        epoch_time = time.time() - t0
        mean_train_loss = float(np.mean(train_losses))
        
        # Validation Evaluation
        val_summary, val_df = run_evaluation(model, val_loader, device, split_name="Val")
        p40_str = f" | P40 Dice: {val_summary.get('p40_macro_dice', 0):.2%}" if 'p40_macro_dice' in val_summary else ""
        
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] Train Loss: {mean_train_loss:.4f} | Val Dice: {val_summary['macro_dice']:.2%} | ASSD: {val_summary['macro_assd']:.2f}px{p40_str} ({epoch_time:.1f}s)")
        
        log_entry = {
            'epoch': epoch,
            'train_loss': mean_train_loss,
            'val_macro_dice': val_summary['macro_dice'],
            'val_macro_assd': val_summary['macro_assd'],
            'val_p40_dice': val_summary.get('p40_macro_dice', np.nan),
            'gamma': outputs['gamma'],
            'time_sec': epoch_time
        }
        training_log.append(log_entry)
        pd.DataFrame(training_log).to_csv(os.path.join(args.out_dir, 'training_log.csv'), index=False)
        
        # Save Checkpoint
        if val_summary['macro_dice'] > best_val_dice:
            best_val_dice = val_summary['macro_dice']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_summary': val_summary,
                'args': vars(args)
            }, best_ckpt_path)
            print(f"⭐ New best model saved: {best_val_dice:.2%}")
            
    # Final Testing & Packaging
    print("\n🏁 Training Complete! Evaluating Best Checkpoint...")
    best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt['model_state_dict'])
    
    final_val_summary, final_val_df = run_evaluation(model, val_loader, device, split_name="FinalVal")
    final_val_df.to_csv(os.path.join(args.out_dir, 'val_predictions.csv'), index=False)
    
    metrics_summary = {'val': final_val_summary}
    if test_loader:
        final_test_summary, final_test_df = run_evaluation(model, test_loader, device, split_name="FinalTest")
        final_test_df.to_csv(os.path.join(args.out_dir, 'test_predictions.csv'), index=False)
        metrics_summary['test'] = final_test_summary
        print(f"🏆 Final Test Macro Dice: {final_test_summary['macro_dice']:.2%} | ASSD: {final_test_summary['macro_assd']:.2f}px")
        
    with open(os.path.join(args.out_dir, 'metrics_summary.json'), 'w') as f:
        json.dump(metrics_summary, f, indent=2)
        
    create_results_zip(args.out_dir, os.path.join(args.out_dir, 'results.zip'))
    print("✅ EXPERIMENT_15 Finished Successfully!")

if __name__ == '__main__':
    main()
