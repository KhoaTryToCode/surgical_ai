import os
import sys
import time
import json
import zipfile
import argparse
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add experiment and workspace roots to PYTHONPATH
EXPERIMENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
WORKSPACE_ROOT = os.path.abspath(os.path.join(EXPERIMENT_DIR, '../..'))

if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from experiments.EXPERIMENT_2.utils.dataset import Mask2FormerDataset
from experiments.EXPERIMENT_2.utils.metrics import evaluate_batch
from experiments.EXPERIMENT_2.models.mask2former_ablation import (
    Mask2FormerAblationModel,
    Mask2FormerLoss,
)


def get_autocast_context(device):
    """Returns modern torch.amp.autocast or falls back gracefully."""
    if device.type == 'cuda':
        try:
            return torch.amp.autocast('cuda')
        except (AttributeError, TypeError):
            return torch.cuda.amp.autocast()
    import contextlib
    return contextlib.nullcontext()


def get_grad_scaler(device):
    """Returns modern torch.amp.GradScaler or legacy GradScaler without deprecation warnings."""
    if device.type == 'cuda':
        try:
            return torch.amp.GradScaler('cuda')
        except (AttributeError, TypeError):
            return torch.cuda.amp.GradScaler()
    try:
        return torch.amp.GradScaler('cpu', enabled=False)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=False)


def render_patient40_panels(img_t, gt_t, pred_logits_t, filename, output_dir):
    """
    Renders 4-panel visual comparison: [RGB | GT Mask | Mask2Former Pred | Error Map]
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. RGB
    rgb = (img_t.permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # 2. GT & Pred Class Maps
    gt_class = torch.argmax(gt_t, dim=0).cpu().numpy().astype(np.uint8)
    pred_class = torch.argmax(pred_logits_t, dim=0).cpu().numpy().astype(np.uint8)

    color_map = {
        0: (30, 30, 30),     # Background
        1: (0, 0, 255),      # Ridge: Red
        2: (0, 255, 0),      # Silhouette: Green
        3: (255, 0, 0),      # Falciform: Blue
    }

    gt_vis = np.zeros_like(rgb_bgr)
    pred_vis = np.zeros_like(rgb_bgr)

    for c, col in color_map.items():
        gt_vis[gt_class == c] = col
        pred_vis[pred_class == c] = col

    # 3. Error Map: Green = True Positive, Blue = False Positive, Red = False Negative
    error_vis = np.zeros_like(rgb_bgr)
    fg_gt = (gt_class > 0)
    fg_pred = (pred_class > 0)

    tp = np.logical_and(fg_gt, fg_pred)
    fp = np.logical_and(fg_pred, ~fg_gt)
    fn = np.logical_and(fg_gt, ~fg_pred)

    error_vis[tp] = (0, 255, 0)   # TP: Green
    error_vis[fp] = (255, 0, 0)   # FP: Blue
    error_vis[fn] = (0, 0, 255)   # FN: Red

    # Stitch into 1x4 panel
    h, w, _ = rgb_bgr.shape
    panel = np.zeros((h, w * 4, 3), dtype=np.uint8)
    panel[:, 0:w] = rgb_bgr
    panel[:, w:2*w] = gt_vis
    panel[:, 2*w:3*w] = pred_vis
    panel[:, 3*w:4*w] = error_vis

    # Add text banners
    cv2.putText(panel, "RGB Input", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(panel, "Ground Truth", (w + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(panel, "Mask2Former Pred", (2 * w + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(panel, "Error (G:TP, B:FP, R:FN)", (3 * w + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    save_name = os.path.splitext(filename)[0] + "_diag.png"
    cv2.imwrite(os.path.join(output_dir, save_name), panel)


def run_evaluation(model, dataloader, device, split_name='Val', save_patient40_dir=None):
    """Evaluates Mask2Former model and aggregates standardized benchmark metrics."""
    model.eval()
    all_metrics = []
    latencies = []
    warmup_count = 0

    with torch.no_grad():
        with get_autocast_context(device):
            for batch_idx, (images, masks, filenames) in enumerate(tqdm(dataloader, desc=f"Evaluating {split_name}")):
                images = images.to(device)
                masks = masks.to(device)

                if device.type == 'cuda':
                    torch.cuda.synchronize()
                start_t = time.perf_counter()

                outputs = model(images)
                semantic_logits = outputs["semantic_logits"]

                if device.type == 'cuda':
                    torch.cuda.synchronize()
                end_t = time.perf_counter()

                if warmup_count >= 5:
                    latencies.append((end_t - start_t) * 1000.0 / images.size(0))
                else:
                    warmup_count += 1

                # Batch metrics
                batch_m = evaluate_batch(semantic_logits.float(), masks)
                for i, m in enumerate(batch_m):
                    m['filename'] = filenames[i]
                    m['patient'] = filenames[i].split('_')[1] if 'Patient_' in filenames[i] else 'unknown'
                    all_metrics.append(m)

                    # Patient 40 visual diagnostics
                    if save_patient40_dir and ('Patient_40_' in filenames[i] or '_40_' in filenames[i]):
                        render_patient40_panels(
                            images[i], masks[i], semantic_logits[i].float(), filenames[i], save_patient40_dir
                        )

    if device.type == 'cuda':
        torch.cuda.empty_cache()

    df = pd.DataFrame(all_metrics)
    mean_dice = float(df['macro_dice'].mean())
    mean_iou = float(df['macro_iou'].mean())
    mean_assd = float(df['macro_assd'].mean())

    mean_fg_dice = float(df['fg_dice'].mean())
    mean_fg_iou = float(df['fg_iou'].mean())
    mean_fg_assd = float(df['fg_assd'].mean())

    ridge_dice = float(df['ridge_dice'].mean())
    sil_dice = float(df['sil_dice'].mean())
    falc_dice = float(df['falc_dice'].mean())

    # Patient 40 subset
    p40_df = df[df['patient'] == '40']
    p40_dice = float(p40_df['macro_dice'].mean()) if len(p40_df) > 0 else 0.0
    p40_fg_dice = float(p40_df['fg_dice'].mean()) if len(p40_df) > 0 else 0.0
    p40_assd = float(p40_df['macro_assd'].mean()) if len(p40_df) > 0 else 80.0

    mean_latency = float(np.mean(latencies)) if len(latencies) > 0 else 0.0
    fps = float(1000.0 / mean_latency) if mean_latency > 0 else 0.0

    summary = {
        'split': split_name,
        'total_frames': len(df),
        'macro_dice': mean_dice,
        'macro_iou': mean_iou,
        'macro_assd': mean_assd,
        'fg_dice': mean_fg_dice,
        'fg_iou': mean_fg_iou,
        'fg_assd': mean_fg_assd,
        'ridge_dice': ridge_dice,
        'sil_dice': sil_dice,
        'falc_dice': falc_dice,
        'patient_40_dice': p40_dice,
        'patient_40_fg_dice': p40_fg_dice,
        'patient_40_assd': p40_assd,
        'patient_40_count': len(p40_df),
        'mean_latency_ms': mean_latency,
        'fps': fps,
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    }
    return summary, df


def create_results_zip(source_dir, output_zip_path):
    """Packages all results in source_dir into a single clean zip file."""
    print(f"\n📦 Packaging experiment artifacts into: {output_zip_path}")
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(source_dir):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, source_dir)
                zipf.write(file_path, arcname=rel_path)
    print(f"✅ Packaging complete ({os.path.getsize(output_zip_path) / (1024 * 1024):.2f} MB). Ready for one-click download!")


def main():
    parser = argparse.ArgumentParser(description="Mask2Former Component Ablation Runner — EXPERIMENT_2")
    parser.add_argument('--train_dir', type=str, default='data/L3D/Train', help="Path to Train directory")
    parser.add_argument('--val_dir', type=str, default='data/L3D/Val', help="Path to Val directory")
    parser.add_argument('--test_dir', type=str, default='data/L3D/Test', help="Path to Test directory")
    parser.add_argument('--ablation', type=str, default='baseline',
                        choices=['baseline', 'wo_masked_attn', 'wo_multiscale', 'wo_self_attn'],
                        help="Ablation mode to execute")
    parser.add_argument('--epochs', type=int, default=50, help="Training epochs (standard: 50)")
    parser.add_argument('--batch_size', type=int, default=1, help="Micro-batch size (default: 1)")
    parser.add_argument('--accumulation_steps', type=int, default=4, help="Gradient accumulation steps (default: 4 -> eff batch = 4)")
    parser.add_argument('--lr', type=float, default=8e-5, help="Learning rate (default: 8e-5)")
    parser.add_argument('--weight_decay', type=float, default=3e-5, help="Weight decay (default: 3e-5)")
    parser.add_argument('--save_dir', type=str, default='results/mask2former_baseline', help="Output results directory")
    parser.add_argument('--zip_output', type=str, default=None, help="Path to output zip archive (default: <save_dir>.zip)")
    parser.add_argument('--eval_splits', type=str, default='both', choices=['val', 'both'], help="Splits to evaluate at end")
    parser.add_argument('--num_workers', type=int, default=None, help="DataLoader workers (default: auto)")
    parser.add_argument('--smoke_test', action='store_true', help="Run 2-batch sanity check and exit")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    patient40_dir = os.path.join(args.save_dir, 'patient_40_diagnostics')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 80)
    print(f"🚀 MASK2FORMER COMPONENT ABLATION RUNNER — EXPERIMENT_2")
    print(f"   Ablation Mode:        {args.ablation}")
    print(f"   Epochs:               {args.epochs}")
    print(f"   Micro Batch Size:     {args.batch_size} (Accumulation: {args.accumulation_steps} -> Effective Batch: {args.batch_size * args.accumulation_steps})")
    print(f"   Learning Rate:        {args.lr}")
    print(f"   Device:               {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Local'})")
    print(f"   Mixed Precision (AMP):{'Enabled (FP16)' if device.type == 'cuda' else 'Disabled'}")
    print(f"   Depth Image Usage:    NONE (Pure RGB-Only Standard Mask2Former)")
    print(f"   Train Directory:      {args.train_dir}")
    print(f"   Val Directory:        {args.val_dir}")
    print(f"   Save Directory:       {args.save_dir}")
    print("=" * 80)

    # 1. Build Datasets
    train_dataset = Mask2FormerDataset(args.train_dir, mode='train')
    val_dataset = Mask2FormerDataset(args.val_dir, mode='val')

    workers = args.num_workers if args.num_workers is not None else (min(4, os.cpu_count() or 2) if device.type == 'cuda' else 0)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=(device.type == 'cuda'),
        drop_last=True,
        persistent_workers=(workers > 0)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(workers > 0)
    )

    test_loader = None
    if args.test_dir and os.path.exists(args.test_dir):
        test_dataset = Mask2FormerDataset(args.test_dir, mode='test')
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=workers,
            persistent_workers=(workers > 0)
        )

    # 2. Build Model & Loss
    model = Mask2FormerAblationModel(ablation_mode=args.ablation).to(device)
    criterion = Mask2FormerLoss(lambda_cls=2.0, lambda_bce=5.0, lambda_dice=5.0).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = get_grad_scaler(device)

    best_val_dice = -1.0
    best_checkpoint_path = os.path.join(args.save_dir, "best_model.pth")
    training_log = []

    # Smoke Test Sanity Check
    if args.smoke_test:
        print("\n🧪 Running Local Smoke Test (1 iteration)...")
        model.train()
        images, masks, _ = next(iter(train_loader))
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        with get_autocast_context(device):
            outputs = model(images)
            loss, loss_dict = criterion(outputs, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        print(f"   Smoke forward/backward passed! Loss: {loss.item():.4f}")
        return

    # 3. Main Training Loop
    total_start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_cls_loss = 0.0
        epoch_bce_loss = 0.0
        epoch_dice_loss = 0.0
        step_count = 0

        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{args.epochs:02d} [Train]")

        for batch_idx, (images, masks, _) in enumerate(pbar):
            images = images.to(device)
            masks = masks.to(device)

            with get_autocast_context(device):
                outputs = model(images)
                loss, loss_dict = criterion(outputs, masks)
                loss_scaled = loss / args.accumulation_steps

            scaler.scale(loss_scaled).backward()

            if (batch_idx + 1) % args.accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += loss_dict['loss']
            epoch_cls_loss += loss_dict['cls_loss']
            epoch_bce_loss += loss_dict['bce_loss']
            epoch_dice_loss += loss_dict['dice_loss']
            step_count += 1

            pbar.set_postfix({
                'loss': f"{epoch_loss / step_count:.4f}",
                'dice_loss': f"{epoch_dice_loss / step_count:.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}"
            })

        scheduler.step()

        # Validation at each epoch
        val_summary, _ = run_evaluation(model, val_loader, device, split_name=f"Val-Ep{epoch}")
        current_val_dice = val_summary['macro_dice']
        print(f"\n📊 Epoch {epoch:02d} Summary: Train Loss={epoch_loss/step_count:.4f} | Val Macro Dice={current_val_dice:.4f} | Ridge={val_summary['ridge_dice']:.4f} | Sil={val_summary['sil_dice']:.4f} | Falc={val_summary['falc_dice']:.4f} | ASSD={val_summary['macro_assd']:.2f}px")

        # Save Checkpoint
        is_best = current_val_dice > best_val_dice
        if is_best:
            best_val_dice = current_val_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_dice': best_val_dice,
                'ablation': args.ablation,
            }, best_checkpoint_path)
            print(f"   🏆 New best model saved! (Val Macro Dice: {best_val_dice:.4f})")

        training_log.append({
            'epoch': epoch,
            'train_loss': epoch_loss / step_count,
            'train_cls_loss': epoch_cls_loss / step_count,
            'train_bce_loss': epoch_bce_loss / step_count,
            'train_dice_loss': epoch_dice_loss / step_count,
            'val_macro_dice': current_val_dice,
            'val_macro_iou': val_summary['macro_iou'],
            'val_macro_assd': val_summary['macro_assd'],
            'val_ridge_dice': val_summary['ridge_dice'],
            'val_sil_dice': val_summary['sil_dice'],
            'val_falc_dice': val_summary['falc_dice'],
            'val_p40_dice': val_summary['patient_40_dice'],
            'lr': scheduler.get_last_lr()[0],
        })

    # Save training log
    pd.DataFrame(training_log).to_csv(os.path.join(args.save_dir, 'training_log.csv'), index=False)

    # 4. Final Evaluation with Best Model
    print("\n" + "=" * 80)
    print("🎯 FINAL BENCHMARK EVALUATION (Loading Best Checkpoint)")
    print("=" * 80)
    if os.path.exists(best_checkpoint_path):
        ckpt = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded best checkpoint from epoch {ckpt['epoch']} with Val Dice: {ckpt['best_val_dice']:.4f}")

    final_results = {'ablation_mode': args.ablation, 'best_epoch': ckpt['epoch'] if os.path.exists(best_checkpoint_path) else args.epochs}

    # Val split final evaluation with Patient 40 diagnostic rendering
    val_summary, val_df = run_evaluation(
        model, val_loader, device, split_name='Val', save_patient40_dir=patient40_dir
    )
    final_results['val'] = val_summary
    val_df.to_csv(os.path.join(args.save_dir, 'val_per_frame_metrics.csv'), index=False)

    # Test split evaluation if available
    if test_loader is not None and args.eval_splits == 'both':
        test_summary, test_df = run_evaluation(
            model, test_loader, device, split_name='Test', save_patient40_dir=patient40_dir
        )
        final_results['test'] = test_summary
        test_df.to_csv(os.path.join(args.save_dir, 'test_per_frame_metrics.csv'), index=False)

    # Save final JSON summary
    with open(os.path.join(args.save_dir, 'metrics_summary.json'), 'w') as f:
        json.dump(final_results, f, indent=4)

    # Output Markdown Summary Table
    print("\n" + "=" * 80)
    print("📋 FINAL BENCHMARK RESULTS TABLE")
    print("=" * 80)
    print(f"| Split | Macro Dice | Macro IoU | Macro ASSD | Ridge Dice | Sil Dice | Falc Dice | Patient 40 Dice |")
    print(f"| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    v = final_results['val']
    print(f"| **Val (122)** | **{v['macro_dice']*100:.2f}%** | {v['macro_iou']*100:.2f}% | {v['macro_assd']:.2f} px | {v['ridge_dice']*100:.2f}% | {v['sil_dice']*100:.2f}% | {v['falc_dice']*100:.2f}% | {v['patient_40_dice']*100:.2f}% |")
    if 'test' in final_results:
        t = final_results['test']
        print(f"| **Test (109)** | **{t['macro_dice']*100:.2f}%** | {t['macro_iou']*100:.2f}% | {t['macro_assd']:.2f} px | {t['ridge_dice']*100:.2f}% | {t['sil_dice']*100:.2f}% | {t['falc_dice']*100:.2f}% | {t['patient_40_dice']*100:.2f}% |")
    print("=" * 80)

    # 5. Automated Packaging into .zip for One-Click Kaggle Download
    zip_path = args.zip_output or f"{args.save_dir.rstrip('/')}.zip"
    create_results_zip(args.save_dir, zip_path)


if __name__ == '__main__':
    main()
