import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

# Add experiment and repo roots to PYTHONPATH
EXPERIMENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
WORKSPACE_ROOT = os.path.abspath(os.path.join(EXPERIMENT_DIR, '../..'))
REPO_ROOT = os.path.join(WORKSPACE_ROOT, 'repos/TopoNet')

if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from experiments.EXPERIMENT_1.utils.dataset import TopoNetDataset
from experiments.EXPERIMENT_1.utils.metrics import evaluate_batch
from experiments.EXPERIMENT_1.models.toponet_ablation import TopoNetAblationModel

# TopoNet Loss Suite
from cldice.cldice import soft_dice_cldice

# Check Betti Matching availability gracefully
HAS_BETTI = False
try:
    from utils.betti_loss import FastBettiMatchingLoss, FiltrationType
    HAS_BETTI = True
except Exception as e:
    # Notice for local Mac test or if C++ build is pending
    HAS_BETTI = False


def dice_loss_fn(pred, target, smooth=1e-5):
    """Standard Soft Multi-Class Dice Loss"""
    pred = torch.softmax(pred, dim=1)
    target_one_hot = target.float()
    intersection = (pred * target_one_hot).sum(dim=(2, 3))
    total = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (total + smooth)
    return 1.0 - dice[:, 1:].mean()  # Exclude background class 0


def render_patient40_panels(img_t, gt_t, pred_t, filename, output_dir):
    """
    Renders 4-panel visual comparison: [RGB | GT Mask | Pred Mask | Error Map]
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. RGB
    rgb = (img_t.permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # 2. GT & Pred
    gt_class = torch.argmax(gt_t, dim=0).cpu().numpy().astype(np.uint8)
    pred_class = torch.argmax(pred_t, dim=0).cpu().numpy().astype(np.uint8)

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

    # 3. Error Map: Green = TP, Blue = FP, Red = FN
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
    cv2.putText(panel, "TopoNet Prediction", (2 * w + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(panel, "Error (G:TP, B:FP, R:FN)", (3 * w + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    save_name = os.path.splitext(filename)[0] + "_diag.png"
    cv2.imwrite(os.path.join(output_dir, save_name), panel)


def run_evaluation(model, dataloader, device, split_name='Val', save_patient40_dir=None):
    """Evaluates model, measures CUDA latency, and collects per-frame metrics."""
    model.eval()
    all_metrics = []
    latencies = []

    # Warmup for latency timing
    warmup_count = 0

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            for batch_idx, (images, depths, masks, filenames) in enumerate(tqdm(dataloader, desc=f"Evaluating {split_name}")):
                images = images.to(device)
                masks = masks.to(device)

                # CUDA Synchronized latency timing
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                start_t = time.perf_counter()

                logits, _ = model(images)

                if device.type == 'cuda':
                    torch.cuda.synchronize()
                end_t = time.perf_counter()

                if warmup_count >= 5:
                    latencies.append((end_t - start_t) * 1000.0 / images.size(0))
                else:
                    warmup_count += 1

                # Batch metrics
                batch_m = evaluate_batch(logits.float(), masks)
                for i, m in enumerate(batch_m):
                    m['filename'] = filenames[i]
                    m['patient'] = filenames[i].split('_')[1] if 'Patient_' in filenames[i] else 'unknown'
                    all_metrics.append(m)

                    # Patient 40 diagnostic rendering
                    if save_patient40_dir and ('Patient_40_' in filenames[i] or '_40_' in filenames[i]):
                        render_patient40_panels(images[i], masks[i], logits[i].float(), filenames[i], save_patient40_dir)

    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # Aggregate summaries
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


def main():
    parser = argparse.ArgumentParser(description="TopoNet Replication & Systematic Ablation Runner")
    parser.add_argument('--train_dir', type=str, default='data/L3D/Train', help="Path to Train directory")
    parser.add_argument('--val_dir', type=str, default='data/L3D/Val', help="Path to Val directory")
    parser.add_argument('--test_dir', type=str, default='data/L3D/Test', help="Path to Test directory")
    parser.add_argument('--depth_ckpt', '--depth_weights', dest='depth_ckpt', type=str, 
                        default='checkpoints/depth_anything_v2_vitb.pth', help="Path to depth checkpoint")
    parser.add_argument('--ablation', type=str, default='full', 
                        choices=['full', 'baseline', 'wo_lper', 'wo_lcl', 'wo_lper_lcl', 'wo_btf'],
                        help="Ablation mode to execute")
    parser.add_argument('--epochs', type=int, default=100, help="Training epochs (paper: 100)")
    parser.add_argument('--batch_size', type=int, default=1, help="Micro-batch size (default: 1 for 16GB VRAM safety)")
    parser.add_argument('--accumulation_steps', type=int, default=4, help="Gradient accumulation steps (default: 4 -> eff batch = 4)")
    parser.add_argument('--lr', type=float, default=8e-5, help="Learning rate (paper: 8e-5)")
    parser.add_argument('--weight_decay', type=float, default=3e-5, help="Weight decay (paper: 3e-5)")
    parser.add_argument('--save_dir', type=str, default='results/toponet_full', help="Output results directory")
    parser.add_argument('--eval_splits', type=str, default='both', choices=['val', 'both'], help="Splits to evaluate at end")
    parser.add_argument('--smoke_test', action='store_true', help="Run 2-batch sanity check and exit")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    patient40_dir = os.path.join(args.save_dir, 'patient_40_diagnostics')

    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print("=" * 80)
    print(f"🚀 TOPONET ABLATION RUNNER — EXPERIMENT_1")
    print(f"   Ablation Mode:        {args.ablation}")
    print(f"   Epochs:               {args.epochs}")
    print(f"   Micro Batch Size:     {args.batch_size} (Accumulation: {args.accumulation_steps} -> Effective Batch: {args.batch_size * args.accumulation_steps})")
    print(f"   Learning Rate:        {args.lr}")
    print(f"   Device:               {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Local'})")
    print(f"   Mixed Precision (AMP):{'Enabled (FP16)' if device.type == 'cuda' else 'Disabled (Local non-CUDA)'}")
    print(f"   Train Directory:      {args.train_dir}")
    print(f"   Val Directory:        {args.val_dir}")
    print(f"   Save Directory:       {args.save_dir}")
    print("=" * 80)

    # 1. Build Datasets
    train_dataset = TopoNetDataset(args.train_dir, mode='train')
    val_dataset = TopoNetDataset(args.val_dir, mode='val')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=(device.type == 'cuda'), drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=(device.type == 'cuda'))

    test_loader = None
    if args.test_dir and os.path.exists(args.test_dir):
        test_dataset = TopoNetDataset(args.test_dir, mode='test')
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    # 2. Build Model
    model = TopoNetAblationModel(ablation_mode=args.ablation, depth_path=args.depth_ckpt).to(device)

    # 3. Setup Loss Functions
    cl_dice_loss = soft_dice_cldice(exclude_background=True)
    betti_loss = None
    if args.ablation in ['full', 'wo_lcl', 'wo_btf']:
        if HAS_BETTI:
            betti_loss = FastBettiMatchingLoss(
                filtration_type=FiltrationType.SUPERLEVEL,
                num_processes=4,
                convert_to_one_vs_rest=False,
                ignore_background=True,
                push_unmatched_to_1_0=True,
                barcode_length_threshold=0.1,
                topology_weights=[0.5, 0.5]
            )
            print("✅ Betti Matching Loss initialized.")
        else:
            print("⚠️  BettiMatching C++ module not compiled. Running without Betti loss for this test.")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    best_val_dice = -1.0
    best_checkpoint_path = os.path.join(args.save_dir, "best_model.pth")

    # Smoke Test Short-Circuit
    if args.smoke_test:
        print("\n🧪 Running Local Smoke Test (1 iteration)...")
        model.train()
        for images, depths, masks, names in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                logits, _ = model(images)
                loss = dice_loss_fn(logits.float(), masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            print(f"   [Smoke Test] Forward/Backward Loss: {loss.item():.4f}")
            break

        print("\n🧪 Running Validation Smoke Test & Patient 40 Diagnostics...")
        val_summary, df_val = run_evaluation(model, val_loader, device, split_name='Val_Smoke', save_patient40_dir=patient40_dir)
        print(f"   [Smoke Test] Val Frames Evaluated: {len(df_val)} | Macro DSC: {val_summary['macro_dice']:.4f}")
        print("✅ Local Smoke Test Passed with Zero Errors!\n")
        return

    # 4. Main Training Loop
    total_iters = len(train_loader)
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch [{epoch+1}/{args.epochs}]")
        for batch_idx, (images, depths, masks, names) in pbar:
            images = images.to(device)
            masks = masks.to(device)

            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                logits, _ = model(images)
                logits = logits.float()

                # Compute Loss based on Ablation Mode & Epoch
                if epoch >= 5 and args.ablation in ['full', 'wo_btf']:
                    # Full TopoNet Dynamic Betti Warmup
                    p = float(batch_idx + (epoch + 1) * total_iters) / (args.epochs * total_iters)
                    alpha = (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0) * 0.05
                    seg_loss = cl_dice_loss(masks, logits)
                    if betti_loss is not None:
                        b_out = betti_loss(logits, masks)
                        betti = b_out[0] if isinstance(b_out, (tuple, list)) else b_out
                    else:
                        betti = 0.0
                    raw_loss = betti * alpha + seg_loss * (1.0 - alpha)
                elif epoch >= 5 and args.ablation == 'wo_lper':
                    raw_loss = cl_dice_loss(masks, logits)
                elif epoch >= 5 and args.ablation == 'wo_lcl':
                    p = float(batch_idx + (epoch + 1) * total_iters) / (args.epochs * total_iters)
                    alpha = (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0) * 0.05
                    d_loss = dice_loss_fn(logits, masks)
                    if betti_loss is not None:
                        b_out = betti_loss(logits, masks)
                        betti = b_out[0] if isinstance(b_out, (tuple, list)) else b_out
                    else:
                        betti = 0.0
                    raw_loss = betti * alpha + d_loss * (1.0 - alpha)
                else:
                    # Baseline, wo_lper_lcl, or Warmup epochs (0-4)
                    raw_loss = dice_loss_fn(logits, masks)

                # Gradient Accumulation: scale loss
                loss = raw_loss / args.accumulation_steps

            scaler.scale(loss).backward()

            epoch_loss += raw_loss.item()

            if (batch_idx + 1) % args.accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            pbar.set_postfix({'loss': f"{raw_loss.item():.4f}"})

        scheduler.step()

        # Validation at epoch end
        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.epochs:
            val_summary, df_val = run_evaluation(model, val_loader, device, split_name='Val', save_patient40_dir=patient40_dir)
            print(f"\n📊 Epoch {epoch+1} Val DSC: {val_summary['macro_dice']*100:.2f}% | IoU: {val_summary['macro_iou']*100:.2f}% | ASSD: {val_summary['macro_assd']:.2f}px | Patient 40 DSC: {val_summary['patient_40_dice']*100:.2f}%\n")

            if val_summary['macro_dice'] > best_val_dice:
                best_val_dice = val_summary['macro_dice']
                torch.save(model.state_dict(), best_checkpoint_path)
                print(f"🌟 Best model saved to {best_checkpoint_path} (DSC: {best_val_dice*100:.2f}%)")

        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # 5. Final Comprehensive Evaluation
    print("\n" + "=" * 80)
    print("🏁 FINAL COMPREHENSIVE BENCHMARK EVALUATION")
    print("=" * 80)

    if os.path.exists(best_checkpoint_path):
        model.load_state_dict(torch.load(best_checkpoint_path, map_location=device))
        print(f"Loaded best checkpoint: {best_checkpoint_path}")

    # Evaluate Val
    final_val_summary, final_val_df = run_evaluation(model, val_loader, device, split_name='Val', save_patient40_dir=patient40_dir)
    final_val_df.to_csv(os.path.join(args.save_dir, "validation_per_frame_results.csv"), index=False)

    # Evaluate Test if requested
    final_test_summary = None
    if (args.eval_splits == 'both' or args.ablation == 'full') and test_loader is not None:
        final_test_summary, final_test_df = run_evaluation(model, test_loader, device, split_name='Test')
        final_test_df.to_csv(os.path.join(args.save_dir, "test_per_frame_results.csv"), index=False)

    # Save summary JSON
    results_json = {
        'ablation_mode': args.ablation,
        'epochs': args.epochs,
        'effective_batch_size': args.batch_size * args.accumulation_steps,
        'val_metrics': final_val_summary,
        'test_metrics': final_test_summary,
    }
    with open(os.path.join(args.save_dir, "summary_metrics.json"), 'w') as f:
        json.dump(results_json, f, indent=2)

    # Print Formatted Markdown Table for immediate viewing
    print("\n" + "=" * 80)
    print(f"🏆 BENCHMARK RESULTS SUMMARY: {args.ablation.upper()}")
    print("=" * 80)
    print(f"| Metric             | Validation (122 frames) | Test (109 frames) | Patient 40 Subset (Val) |")
    print(f"|:-------------------|:------------------------|:------------------|:------------------------|")
    print(f"| **Macro Mean DSC** | **{final_val_summary['macro_dice']*100:.2f}%**          | **{final_test_summary['macro_dice']*100 if final_test_summary else 0.0:.2f}%**       | **{final_val_summary['patient_40_dice']*100:.2f}%**              |")
    print(f"| **Mean IoU**       | {final_val_summary['macro_iou']*100:.2f}%          | {final_test_summary['macro_iou']*100 if final_test_summary else 0.0:.2f}%       | {final_val_summary.get('patient_40_iou', 0.0)*100:.2f}%              |")
    print(f"| **ASSD (px)**      | {final_val_summary['macro_assd']:.2f} px          | {final_test_summary['macro_assd'] if final_test_summary else 0.0:.2f} px       | {final_val_summary['patient_40_assd']:.2f} px              |")
    print(f"| **Ridge DSC**      | {final_val_summary['ridge_dice']*100:.2f}%          | {final_test_summary['ridge_dice']*100 if final_test_summary else 0.0:.2f}%       | --                      |")
    print(f"| **Silhouette DSC** | {final_val_summary['sil_dice']*100:.2f}%          | {final_test_summary['sil_dice']*100 if final_test_summary else 0.0:.2f}%       | --                      |")
    print(f"| **Falciform DSC**  | {final_val_summary['falc_dice']*100:.2f}%          | {final_test_summary['falc_dice']*100 if final_test_summary else 0.0:.2f}%       | --                      |")
    print(f"| **Latency / FPS**  | {final_val_summary['mean_latency_ms']:.1f} ms ({final_val_summary['fps']:.1f} FPS) | --                | Device: {final_val_summary['gpu_name']} |")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
