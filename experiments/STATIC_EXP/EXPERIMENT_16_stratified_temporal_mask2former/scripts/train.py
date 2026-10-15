import os
import sys
import time
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.temporal_mask2former import TemporalMask2Former
from utils.dataset_temporal import StratifiedTemporalL3DDataset, resolve_split_dir
from utils.metrics import evaluate_frame_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train Stratified Temporal Mask2Former on L3D Benchmark (EXP_16)")
    parser.add_argument('--config', type=str, default='experiments/EXPERIMENT_16_stratified_temporal_mask2former/configs/temporal_m2f_t3.yaml', help='Path to config YAML')
    parser.add_argument('--save-dir', type=str, default='checkpoints/exp16_stratified_temporal_m2f', help='Directory to save checkpoints')
    parser.add_argument('--epochs', type=int, default=None, help='Override number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None, help='Override batch size')
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def run_validation(model, val_loader, device, num_classes=3, threshold=0.5):
    model.eval()
    all_metrics = []
    p38_macro_dices = []
    p40_macro_dices = []

    with torch.no_grad():
        for batch in val_loader:
            pixel_values = batch['pixel_values'].to(device)  # (B, T, 3, H, W)
            target_masks = batch['target_masks'].to(device)  # (B, 3, H, W)
            patients = batch['patient']

            outputs = model(pixel_values)

            # Extract predicted masks
            class_logits = outputs.class_queries_logits  # (B, 100, 4)
            mask_logits = outputs.masks_queries_logits    # (B, 100, H/4, W/4)

            # Compute semantic segmentation
            mask_probs = mask_logits.sigmoid()
            class_probs = class_logits.softmax(dim=-1)[..., :num_classes]
            sem_seg = torch.einsum('bqc, bqhw -> bchw', class_probs, mask_probs)

            # Upsample to ground truth resolution
            sem_seg_full = nn.functional.interpolate(
                sem_seg,
                size=(target_masks.shape[-2], target_masks.shape[-1]),
                mode='bilinear',
                align_corners=False
            )

            for b in range(pixel_values.shape[0]):
                pred_prob = sem_seg_full[b].cpu().numpy()
                gt_in = batch['gt_map'][b].cpu().numpy() if 'gt_map' in batch else target_masks[b].cpu().numpy()

                m = evaluate_frame_metrics(pred_prob, gt_in, num_classes=num_classes, threshold=threshold)
                all_metrics.append(m)

                if patients[b] == 'Patient_38':
                    p38_macro_dices.append(m['macro_dice'])
                elif patients[b] == 'Patient_40':
                    p40_macro_dices.append(m['macro_dice'])

    summary = {
        'val_macro_dice': float(np.mean([m['macro_dice'] for m in all_metrics])),
        'val_micro_dice': float(np.mean([m['micro_dice'] for m in all_metrics])),
        'val_macro_iou': float(np.mean([m['macro_iou'] for m in all_metrics])),
        'val_mean_assd': float(np.nanmean([m['mean_assd'] for m in all_metrics])),
        'p38_macro_dice': float(np.mean(p38_macro_dices)) if len(p38_macro_dices) > 0 else 0.0,
        'p40_macro_dice': float(np.mean(p40_macro_dices)) if len(p40_macro_dices) > 0 else 0.0
    }
    return summary


def main():
    args = parse_args()
    cfg = load_config(args.config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"🚀 Using Device: {device}")

    # Dataset paths
    train_dir = resolve_split_dir(cfg['dataset']['train_split'])
    val_dir = resolve_split_dir(cfg['dataset']['val_split'])
    print(f"📁 Resolved Train Directory: {train_dir}")
    print(f"📁 Resolved Val Directory:   {val_dir}")

    # Stratified datasets (Option 1: Patient_40 in Train, Patient_38 in Val)
    train_ds = StratifiedTemporalL3DDataset(
        split='train',
        train_dir=train_dir,
        val_dir=val_dir,
        clip_len=cfg['dataset']['clip_len'],
        max_gap=cfg['dataset']['max_gap'],
        image_size=cfg['dataset']['image_size'],
        stroke_width=cfg['dataset']['stroke_width']
    )
    val_ds = StratifiedTemporalL3DDataset(
        split='val',
        train_dir=train_dir,
        val_dir=val_dir,
        clip_len=cfg['dataset']['clip_len'],
        max_gap=cfg['dataset']['max_gap'],
        image_size=cfg['dataset']['image_size'],
        stroke_width=cfg['dataset']['stroke_width']
    )

    print(f"📦 Stratified Train clips: {len(train_ds)} (Expected: 923) | Stratified Val clips: {len(val_ds)} (Expected: 120)")

    batch_size = args.batch_size or cfg['training']['batch_size']
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Initialize Model
    model = TemporalMask2Former(
        base_checkpoint=cfg['model']['base_checkpoint'],
        num_classes=cfg['model']['num_classes'],
        hidden_dim=cfg['model']['hidden_dim'],
        temporal_gate_init=cfg['model']['temporal_gate_init']
    ).to(device)

    # Optimizer with differential learning rates
    backbone_params = []
    decoder_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'pixel_level_module.encoder' in name:
            backbone_params.append(param)
        else:
            decoder_params.append(param)

    optimizer = AdamW([
        {'params': backbone_params, 'lr': cfg['training']['lr_backbone']},
        {'params': decoder_params, 'lr': cfg['training']['lr_decoder']}
    ], weight_decay=cfg['training']['weight_decay'])

    num_epochs = args.epochs or cfg['training']['num_epochs']
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1.0e-6)

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_macro = 0.0
    grad_accum_steps = cfg['training']['grad_accum_steps']

    print(f"🏋️ Starting {num_epochs} Epochs Training (Accumulation: {grad_accum_steps})...")
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            pixel_values = batch['pixel_values'].to(device)  # (B, T, 3, H, W)
            mask_labels = batch['mask_labels'].to(device)    # (B, T, 3, H, W) - 100% GT for Teacher Forcing!

            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                outputs = model(pixel_values, mask_labels=mask_labels)
                loss = outputs.loss / grad_accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['clip_grad_norm'])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += loss.item() * grad_accum_steps

            if (step + 1) % 50 == 0 or (step + 1) == len(train_loader):
                step_time = time.time() - start_time
                steps_done = step + 1
                total_steps = len(train_loader)
                speed = step_time / steps_done
                eta_m = (speed * (total_steps - steps_done)) / 60.0
                print(f"  [Epoch {epoch:02d}/{num_epochs:02d}] Step [{steps_done:03d}/{total_steps:03d}] Loss: {loss.item()*grad_accum_steps:.4f} | Time: {step_time:.1f}s | ETA: {eta_m:.1f}m", flush=True)

        scheduler.step()
        train_loss = epoch_loss / len(train_loader)
        elapsed = time.time() - start_time

        # Run Validation
        val_summary = run_validation(model, val_loader, device, num_classes=cfg['model']['num_classes'], threshold=cfg['metrics']['threshold'])

        print(
            f"Epoch [{epoch:02d}/{num_epochs:02d}] "
            f"Loss: {train_loss:.4f} | "
            f"Val Macro: {val_summary['val_macro_dice']*100:.2f}% | "
            f"Val Micro: {val_summary['val_micro_dice']*100:.2f}% | "
            f"ASSD: {val_summary['val_mean_assd']:.2f}px | "
            f"P38 Macro: {val_summary['p38_macro_dice']*100:.2f}% "
            f"({elapsed:.1f}s)"
        )

        # Save Best Checkpoint
        if val_summary['val_macro_dice'] > best_val_macro:
            best_val_macro = val_summary['val_macro_dice']
            save_path = os.path.join(args.save_dir, 'best_temporal_m2f.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_summary': val_summary,
                'config': cfg
            }, save_path)
            print(f"  ⭐ New best checkpoint saved: Val Macro {best_val_macro*100:.2f}%")

    print(f"🏁 Training Complete! Best Val Macro: {best_val_macro*100:.2f}%")


if __name__ == '__main__':
    main()
