#!/usr/bin/env python3
"""
Mask2Former HPC Cluster & Server Training Runner (EXPERIMENT_2 — Set A Suite)
----------------------------------------------------------------------------
Complies strictly with the gpu-a240 Slurm cluster standard established in EXPERIMENT_1:
  - Uses the authentic pretrained Hugging Face model: 'facebook/mask2former-swin-tiny-ade-semantic'
  - Supports the 4 Set A internal component ablations via --ablation:
      1. 'baseline' (or 'Swin_MaskedAttn') : Full Mask2Former (Masked Attn + Multi-Scale + Query Self-Attn -> ~0.68 DSC)
      2. 'wo_masked_attn'                   : Spatial Gating Ablation (Full Global Cross-Attention, attn_mask=None)
      3. 'wo_multiscale'                    : Scale Engine Ablation (Single-Scale stride-16 decoding, level_index=1)
      4. 'wo_self_attn'                     : Query Interaction Ablation (Independent parallel queries, bypass self-attn)
  - Native deep supervision Hungarian loss across all 9 decoder layers.
  - Multi-query semantic ensemble post-processing.
  - Generates:
      - best_model.pth (saved on validation improvement)
      - summary_metrics.json (complete benchmark report)
      - validation_per_frame_results.csv
      - test_per_frame_results.csv (when --eval_splits both)
      - Patient 40 diagnostic visual panels
      - Formatted Markdown benchmark summary table at job termination.
"""

import os
import sys
import gc
import json
import time
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# NumPy 2.0+ compatibility
for attr, val in [("Inf", np.inf), ("NAN", np.nan), ("NaN", np.nan)]:
    if not hasattr(np, attr):
        setattr(np, attr, val)

# Surface distance evaluation
try:
    import surface_distance
    from surface_distance import metrics as sd_metrics
    HAS_SURFACE_DIST = True
except Exception:
    HAS_SURFACE_DIST = False

# Transformers check
try:
    import transformers
    from transformers import (
        AutoImageProcessor,
        Mask2FormerForUniversalSegmentation,
    )
except ImportError:
    print("❌ Error: 'transformers' library not found. Ensure 'conda activate surgical_ai' is active.")
    sys.exit(1)


# ==============================================================================
# Metric Evaluation Functions (Exact TopoNet & Mask2Former Formulation)
# ==============================================================================

def evaluation(pred, gt):
    smooth = 1e-5
    intersection = np.sum(pred * gt)
    dice = (2.0 * intersection + smooth) / (np.sum(pred) + np.sum(gt) + smooth)
    iou = dice / (2.0 - dice)
    return float(iou), float(dice)


def compute_frame_metrics(pred_map, gt_2d):
    """
    Computes Macro Dice, IoU, per-class Dice, and ASSD.
    Classes: 0: Background, 1: Ridge, 2: Silhouette, 3: Falciform.
    """
    pred_channels = np.array([pred_map == i for i in range(4)]).astype(np.uint8)
    gt_channels = np.array([gt_2d == i for i in range(4)]).astype(np.uint8)

    # Macro foreground metric (Classes 1, 2, 3)
    iou, dice = evaluation(pred_channels[1:].flatten(), gt_channels[1:].flatten())

    class_dices = {}
    class_names = {1: "ridge", 2: "sil", 3: "falc"}
    for c, name in class_names.items():
        _, c_dice = evaluation(pred_channels[c].flatten(), gt_channels[c].flatten())
        class_dices[name] = float(c_dice)

    assd = None
    if HAS_SURFACE_DIST:
        try:
            if np.count_nonzero(pred_channels[1:]) == 0:
                assd = 80.0
            else:
                temp_assd = []
                for i in range(3):
                    gt_c = np.array(gt_channels[i + 1], dtype=bool)
                    pred_c = np.array(pred_channels[i + 1], dtype=bool)
                    if not gt_c.any() or not pred_c.any():
                        temp_assd.append(80.0)
                        continue
                    sd = sd_metrics.compute_surface_distances(gt_c, pred_c, (1.0, 1.0))
                    avg_sd = surface_distance.compute_average_surface_distance(sd)
                    val = avg_sd[1]
                    temp_assd.append(val if not np.isnan(val) and val < 500.0 else 80.0)
                mean_dist = float(np.mean(temp_assd)) if temp_assd else 80.0
                assd = mean_dist if mean_dist < 500.0 else 80.0
        except Exception:
            assd = 80.0

    return dice, iou, class_dices, assd


# ==============================================================================
# Set A Component Ablation Helpers
# ==============================================================================

def apply_architectural_ablation(model, ablation_type):
    """
    Surgically patches the 9 decoder layers of the pretrained Mask2Former model.
    """
    decoder_layers = model.model.transformer_module.decoder.layers

    if ablation_type in ["wo_masked_attn", "swin_fullattn"]:
        # Ablation 1: Disable Masked Cross-Attention (Set attn_mask = None -> Full Global Attention)
        for l in decoder_layers:
            orig = l.forward_pre
            def patch_fn(fn):
                def patched(*args, **kwargs):
                    kwargs['encoder_attention_mask'] = None
                    return fn(*args, **kwargs)
                return patched
            l.forward_pre = patch_fn(orig)
            l.forward = patch_fn(l.forward)
        print("⚠️ [Ablation 1 Active]: Masked Attention DISABLED (All 9 decoder layers running in FULL GLOBAL CROSS-ATTENTION mode).")

    elif ablation_type == "wo_multiscale":
        # Ablation 2: Disable Multi-Scale Feature Cycling (Lock all layers to single stride-16 feature level)
        orig_tm_forward = model.model.transformer_module.forward
        def single_scale_tm_forward(multi_scale_features, mask_features, output_hidden_states=False, output_attentions=False):
            single_feat = multi_scale_features[1]  # Lock to stride 16 (24x24)
            single_multi_scale = [single_feat, single_feat, single_feat]
            return orig_tm_forward(single_multi_scale, mask_features, output_hidden_states=output_hidden_states, output_attentions=output_attentions)
        model.model.transformer_module.forward = single_scale_tm_forward
        print("⚠️ [Ablation 2 Active]: Multi-Scale Feature Cycling DISABLED (All 9 decoder layers locked to single stride-16 level).")

    elif ablation_type == "wo_self_attn":
        # Ablation 3: Disable Query Self-Attention (Bypass self_attn -> Independent Parallel Queries)
        for l in decoder_layers:
            def patch_self_attn():
                def patched(*args, **kwargs):
                    hs = kwargs.get('hidden_states', args[0] if len(args) > 0 else None)
                    return torch.zeros_like(hs), None
                return patched
            l.self_attn.forward = patch_self_attn()
        print("⚠️ [Ablation 3 Active]: Query Self-Attention DISABLED (Queries operating as independent parallel detectors).")

    else:
        print("✅ [Run 0 Active]: Full Standard Mask2Former (Masked Cross-Attention + Multi-Scale Deformable Cycling + Query Self-Attention).")

    return model


# ==============================================================================
# Standardized L3D Dataset Reader with Patient 32 4K Canvas Preservation
# ==============================================================================

class L3DSurgicalDataset(Dataset):
    """
    Standardized Dataset for Laparoscopic Liver Landmark Detection.
    - Preserves Patient 32 4K canvas dynamic resolution from JSON metadata.
    - Standardizes line thickness to 35.
    - Returns raw RGB (1024, 1024, 3) and discrete 2D ground-truth (1024, 1024).
    """
    def __init__(self, root_dir):
        root_path = Path(root_dir)
        if (root_path / "images").exists():
            img_dir = root_path / "images"
        else:
            img_dir = root_path

        valid_exts = {'.png', '.jpg', '.jpeg', '.bmp'}
        self.file_paths = sorted([
            p for p in img_dir.iterdir() if p.suffix.lower() in valid_exts
        ])
        if len(self.file_paths) == 0:
            self.file_paths = sorted([
                Path(p) for p in Path(root_dir).rglob("*") if p.suffix.lower() in valid_exts
            ])

        print(f"   Found {len(self.file_paths)} frames in: {root_dir}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        rgb_img = self.load_image(img_path)
        gt_2d = self.load_mask_2d(img_path)
        return rgb_img, gt_2d, str(img_path)

    @staticmethod
    def load_image(path):
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[0] != 1024 or img.shape[1] != 1024:
            img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        return img

    @staticmethod
    def load_mask_2d(path):
        path_str = str(path)
        json_path = os.path.splitext(path_str)[0] + '.json'
        if not os.path.exists(json_path):
            parent = os.path.dirname(path_str)
            base_stem = os.path.splitext(os.path.basename(path_str))[0]
            candidates = [
                os.path.join(parent, '..', 'labels', base_stem + '.json'),
                os.path.join(parent, 'labels', base_stem + '.json'),
                os.path.join(parent.replace('images', 'labels'), base_stem + '.json')
            ]
            for c in candidates:
                if os.path.exists(c):
                    json_path = c
                    break

        if not os.path.exists(json_path):
            return np.zeros((1024, 1024), dtype=np.int32)

        with open(json_path, 'r') as f:
            data = json.load(f)

        # Dynamic canvas size extraction (Fixes Patient 32 4K canvas bug)
        img_h = data.get('imageHeight', 1080)
        img_w = data.get('imageWidth', 1920)
        canvas = np.zeros((img_h, img_w), dtype=np.uint8)

        for shape in data.get('shapes', []):
            label = str(shape.get('label', '')).lower()
            if label.startswith('r') or 'ridge' in label or 'rigde' in label:
                color = 1
            elif label.startswith('s') or 'sil' in label:
                color = 2
            elif label.startswith('l') or 'lig' in label or 'falc' in label:
                color = 3
            else:
                color = 0

            if color > 0:
                points = shape.get('points', [])
                for i in range(1, len(points)):
                    pt1 = tuple(map(int, points[i - 1]))
                    pt2 = tuple(map(int, points[i]))
                    cv2.line(canvas, pt1, pt2, color, 35)

        if canvas.shape[0] != 1024 or canvas.shape[1] != 1024:
            canvas = cv2.resize(canvas, (1024, 1024), interpolation=cv2.INTER_NEAREST)

        return canvas.astype(np.int32)


def collate_fn_l3d(batch):
    images = [item[0] for item in batch]
    masks = [item[1] for item in batch]
    paths = [item[2] for item in batch]
    return images, masks, paths


# ==============================================================================
# Patient 40 Visual Diagnostics Generator
# ==============================================================================

def render_patient40_panels(rgb_img, gt_2d, pred_map, save_path):
    """
    Renders 4-panel visual comparison: [RGB | Ground Truth | Prediction | Error Map].
    """
    colors = {
        0: (30, 30, 30),
        1: (255, 0, 0),     # Ridge (Red)
        2: (0, 255, 0),     # Silhouette (Green)
        3: (0, 0, 255),     # Falciform (Blue)
    }

    h, w = 1024, 1024
    gt_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    pred_rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for c, col in colors.items():
        if c == 0:
            continue
        gt_rgb[gt_2d == c] = col
        pred_rgb[pred_map == c] = col

    # Error Map: Green = TP, Red = FP, Blue = FN
    error_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    gt_fg = gt_2d > 0
    pred_fg = pred_map > 0
    tp = gt_fg & pred_fg & (gt_2d == pred_map)
    fp = pred_fg & (~gt_fg | (gt_2d != pred_map))
    fn = gt_fg & (~pred_fg | (gt_2d != pred_map))

    error_rgb[tp] = (0, 255, 0)
    error_rgb[fp] = (255, 0, 0)
    error_rgb[fn] = (0, 0, 255)

    blend_gt = cv2.addWeighted(rgb_img, 0.6, gt_rgb, 0.4, 0)
    blend_pred = cv2.addWeighted(rgb_img, 0.6, pred_rgb, 0.4, 0)

    panel = np.hstack([rgb_img, blend_gt, blend_pred, error_rgb])
    panel = cv2.resize(panel, (2048, 512), interpolation=cv2.INTER_AREA)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))


# ==============================================================================
# Evaluation Function
# ==============================================================================

def run_evaluation(model, processor, loader, device, use_amp=True, save_patient40_dir=None):
    model.eval()
    frame_records = []
    latencies = []

    with torch.no_grad():
        for batch_rgb, batch_gt, batch_paths in loader:
            t0 = time.time()
            inputs = processor(images=batch_rgb, segmentation_maps=batch_gt, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)

            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    outputs = model(pixel_values=pixel_values)
            else:
                outputs = model(pixel_values=pixel_values)

            target_sizes = [(1024, 1024)] * len(batch_rgb)
            pred_maps = processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)
            dt = (time.time() - t0) / len(batch_rgb)
            latencies.append(dt)

            for p_tensor, gt_arr, rgb_arr, p_path in zip(pred_maps, batch_gt, batch_rgb, batch_paths):
                p_arr = p_tensor.cpu().numpy()
                fname = Path(p_path).name
                dice, iou, c_dices, assd = compute_frame_metrics(p_arr, gt_arr)

                is_p40 = ("patient40" in p_path.lower() or "patient_40" in p_path.lower() or "p40" in p_path.lower())

                frame_records.append({
                    "filename": fname,
                    "macro_dice": dice,
                    "macro_iou": iou,
                    "assd_px": assd if assd is not None else 80.0,
                    "ridge_dice": c_dices["ridge"],
                    "sil_dice": c_dices["sil"],
                    "falc_dice": c_dices["falc"],
                    "is_patient40": is_p40
                })

                if save_patient40_dir and is_p40:
                    panel_path = os.path.join(save_patient40_dir, f"{Path(fname).stem}_diag.png")
                    render_patient40_panels(rgb_arr, gt_arr, p_arr, panel_path)

    df = pd.DataFrame(frame_records)
    p40_df = df[df["is_patient40"]]

    summary = {
        "macro_dice": float(df["macro_dice"].mean()),
        "macro_iou": float(df["macro_iou"].mean()),
        "macro_assd": float(df["assd_px"].mean()),
        "ridge_dice": float(df["ridge_dice"].mean()),
        "sil_dice": float(df["sil_dice"].mean()),
        "falc_dice": float(df["falc_dice"].mean()),
        "patient_40_dice": float(p40_df["macro_dice"].mean()) if len(p40_df) > 0 else 0.0,
        "patient_40_iou": float(p40_df["macro_iou"].mean()) if len(p40_df) > 0 else 0.0,
        "patient_40_assd": float(p40_df["assd_px"].mean()) if len(p40_df) > 0 else 0.0,
        "mean_latency_ms": float(np.mean(latencies) * 1000.0),
        "fps": float(1.0 / np.mean(latencies)) if np.mean(latencies) > 0 else 0.0,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    }

    return summary, df


# ==============================================================================
# Main Training Orchestrator
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Mask2Former High-Throughput Server Runner (EXPERIMENT_2 — Set A)")
    parser.add_argument("--ablation", "--mode", dest="ablation", type=str, default="baseline",
                        help="Ablation mode: baseline (or Swin_MaskedAttn), wo_masked_attn, wo_multiscale, wo_self_attn")
    parser.add_argument("--train_dir", type=str, default="/data/khoalq/data/L3D/Train", help="Train dataset directory")
    parser.add_argument("--val_dir", type=str, default="/data/khoalq/data/L3D/Val", help="Val dataset directory")
    parser.add_argument("--test_dir", type=str, default="/data/khoalq/data/L3D/Test", help="Test dataset directory")
    parser.add_argument("--save_dir", type=str, default="/data/khoalq/checkpoints/mask2former_baseline_60ep", help="Output checkpoint directory")
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs (default: 60)")
    parser.add_argument("--batch_size", type=int, default=4, help="Micro-batch size per GPU step (default: 4)")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="Gradient accumulation steps (default: 1)")
    parser.add_argument("--lr", type=float, default=8e-5, help="Learning rate (default: 8e-5)")
    parser.add_argument("--weight_decay", type=float, default=3e-5, help="Weight decay (default: 3e-5)")
    parser.add_argument("--eval_splits", type=str, default="both", choices=["val", "both"], help="Splits to evaluate at end")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader worker processes")
    parser.add_argument("--no_amp", action="store_true", help="Disable PyTorch AMP mixed precision")
    parser.add_argument("--smoke_test", action="store_true", help="Run 1-batch sanity check and exit")
    args = parser.parse_args()

    # Normalize ablation name alias
    mode_aliases = {
        "swin_maskedattn": "baseline",
        "swin_fullattn": "wo_masked_attn",
    }
    ablation_mode = mode_aliases.get(args.ablation.lower(), args.ablation.lower())

    os.makedirs(args.save_dir, exist_ok=True)
    patient40_dir = os.path.join(args.save_dir, "patient_40_diagnostics")
    best_checkpoint_path = os.path.join(args.save_dir, "best_model.pth")
    latest_checkpoint_path = os.path.join(args.save_dir, "latest_model.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    use_amp = (not args.no_amp) and (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    print("=" * 80)
    print(f"🚀 MASK2FORMER CLUSTER RUNNER — EXPERIMENT_2 (Set A Suite)")
    print(f"   Ablation Mode:        {ablation_mode}")
    print(f"   Epochs:               {args.epochs}")
    print(f"   Micro Batch Size:     {args.batch_size} (Accumulation: {args.accumulation_steps} -> Effective Batch: {args.batch_size * args.accumulation_steps})")
    print(f"   Learning Rate:        {args.lr}")
    print(f"   Device:               {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'Local'})")
    print(f"   Mixed Precision (AMP):{'Enabled (FP16)' if use_amp else 'Disabled'}")
    print(f"   Train Directory:      {args.train_dir}")
    print(f"   Val Directory:        {args.val_dir}")
    print(f"   Test Directory:       {args.test_dir}")
    print(f"   Save Directory:       {args.save_dir}")
    print("=" * 80)

    # 1. Datasets & Loaders
    train_dataset = L3DSurgicalDataset(args.train_dir)
    val_dataset = L3DSurgicalDataset(args.val_dir)
    test_dataset = L3DSurgicalDataset(args.test_dir) if args.test_dir and os.path.exists(args.test_dir) else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn_l3d,
        pin_memory=(device.type == "cuda"),
        drop_last=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_l3d,
        pin_memory=(device.type == "cuda")
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_l3d,
        pin_memory=(device.type == "cuda")
    ) if test_dataset else None

    # 2. Build Pretrained Model & ImageProcessor
    model_name = "facebook/mask2former-swin-tiny-ade-semantic"
    print(f"📦 Initializing Pretrained Mask2Former from '{model_name}'...")
    processor = AutoImageProcessor.from_pretrained(model_name, reduce_labels=False, ignore_index=255)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        model_name,
        num_labels=4,
        ignore_mismatched_sizes=True
    ).to(device)

    # Apply Set A Component Ablation
    model = apply_architectural_ablation(model, ablation_mode)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Smoke Test Short-Circuit
    if args.smoke_test:
        print("\n🧪 Running Local Smoke Test (1 iteration)...")
        model.train()
        for batch_rgb, batch_gt, _ in train_loader:
            inputs = processor(images=batch_rgb, segmentation_maps=batch_gt, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            mask_labels = [m.to(device) for m in inputs["mask_labels"]]
            class_labels = [c.to(device) for c in inputs["class_labels"]]
            optimizer.zero_grad()
            outputs = model(pixel_values=pixel_values, mask_labels=mask_labels, class_labels=class_labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            print(f"   [Smoke Test] Loss: {loss.item():.4f}")
            break

        val_summary, df_val = run_evaluation(model, processor, val_loader, device, use_amp=use_amp, save_patient40_dir=patient40_dir)
        print(f"   [Smoke Test] Val Frames Evaluated: {len(df_val)} | Macro DSC: {val_summary['macro_dice']:.4f}")
        print("✅ Local Smoke Test Passed with Zero Errors!\n")
        return

    # 3. Main Training Loop
    best_val_dice = 0.0
    metrics_history = []

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch [{epoch}/{args.epochs}]")
        for step, (batch_rgb, batch_gt, _) in pbar:
            inputs = processor(images=batch_rgb, segmentation_maps=batch_gt, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            mask_labels = [m.to(device) for m in inputs["mask_labels"]]
            class_labels = [c.to(device) for c in inputs["class_labels"]]

            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    outputs = model(pixel_values=pixel_values, mask_labels=mask_labels, class_labels=class_labels)
                    loss = outputs.loss / args.accumulation_steps
                scaler.scale(loss).backward()
            else:
                outputs = model(pixel_values=pixel_values, mask_labels=mask_labels, class_labels=class_labels)
                loss = outputs.loss / args.accumulation_steps
                loss.backward()

            total_loss += outputs.loss.item() * len(batch_rgb)

            if (step + 1) % args.accumulation_steps == 0 or (step + 1) == len(train_loader):
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                optimizer.zero_grad()

            pbar.set_postfix({"loss": f"{outputs.loss.item():.4f}"})

        scheduler.step()
        epoch_time = time.time() - epoch_start
        avg_train_loss = total_loss / len(train_dataset)

        # Validation at epoch end (every 5 epochs and final epoch)
        if (epoch % 5 == 0) or (epoch == args.epochs):
            val_summary, df_val = run_evaluation(model, processor, val_loader, device, use_amp=use_amp, save_patient40_dir=patient40_dir)
            print(
                f"\n📊 Epoch {epoch} Val DSC: {val_summary['macro_dice']*100:.2f}% | "
                f"IoU: {val_summary['macro_iou']*100:.2f}% | ASSD: {val_summary['macro_assd']:.2f}px | "
                f"Patient 40 DSC: {val_summary['patient_40_dice']*100:.2f}%\n"
            )

            if val_summary["macro_dice"] > best_val_dice:
                best_val_dice = val_summary["macro_dice"]
                torch.save(model.state_dict(), best_checkpoint_path)
                print(f"🌟 Best model saved to {best_checkpoint_path} (DSC: {best_val_dice*100:.2f}%)")

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_dice": best_val_dice,
                "ablation_mode": ablation_mode
            }, latest_checkpoint_path)

            metrics_history.append({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_summary": val_summary
            })

    # 4. Final Comprehensive Evaluation (Matching EXPERIMENT_1 Schema)
    print("\n" + "=" * 80)
    print("🏁 FINAL COMPREHENSIVE BENCHMARK EVALUATION")
    print("=" * 80)

    if os.path.exists(best_checkpoint_path):
        model.load_state_dict(torch.load(best_checkpoint_path, map_location=device))
        print(f"Loaded best checkpoint: {best_checkpoint_path}")

    final_val_summary, final_val_df = run_evaluation(model, processor, val_loader, device, use_amp=use_amp, save_patient40_dir=patient40_dir)
    final_val_df.to_csv(os.path.join(args.save_dir, "validation_per_frame_results.csv"), index=False)

    final_test_summary = None
    if (args.eval_splits == "both") and (test_loader is not None):
        final_test_summary, final_test_df = run_evaluation(model, processor, test_loader, device, use_amp=use_amp)
        final_test_df.to_csv(os.path.join(args.save_dir, "test_per_frame_results.csv"), index=False)

    results_json = {
        "ablation_mode": ablation_mode,
        "epochs": args.epochs,
        "effective_batch_size": args.batch_size * args.accumulation_steps,
        "val_metrics": final_val_summary,
        "test_metrics": final_test_summary,
    }
    with open(os.path.join(args.save_dir, "summary_metrics.json"), "w") as f:
        json.dump(results_json, f, indent=2)

    # Print Formatted Markdown Table for immediate viewing (Bit-for-Bit EXPERIMENT_1 parity)
    print("\n" + "=" * 80)
    print(f"🏆 BENCHMARK RESULTS SUMMARY: MASK2FORMER ({ablation_mode.upper()})")
    print("=" * 80)
    print(f"| Metric             | Validation (122 frames) | Test (109 frames) | Patient 40 Subset (Val) |")
    print(f"|:-------------------|:------------------------|:------------------|:------------------------|")
    print(f"| **Macro Mean DSC** | **{final_val_summary['macro_dice']*100:.2f}%**          | **{final_test_summary['macro_dice']*100 if final_test_summary else 0.0:.2f}%**       | **{final_val_summary['patient_40_dice']*100:.2f}%**              |")
    print(f"| **Mean IoU**       | {final_val_summary['macro_iou']*100:.2f}%          | {final_test_summary['macro_iou']*100 if final_test_summary else 0.0:.2f}%       | {final_val_summary['patient_40_iou']*100:.2f}%              |")
    print(f"| **ASSD (px)**      | {final_val_summary['macro_assd']:.2f} px          | {final_test_summary['macro_assd'] if final_test_summary else 0.0:.2f} px       | {final_val_summary['patient_40_assd']:.2f} px              |")
    print(f"| **Ridge DSC**      | {final_val_summary['ridge_dice']*100:.2f}%          | {final_test_summary['ridge_dice']*100 if final_test_summary else 0.0:.2f}%       | --                      |")
    print(f"| **Silhouette DSC** | {final_val_summary['sil_dice']*100:.2f}%          | {final_test_summary['sil_dice']*100 if final_test_summary else 0.0:.2f}%       | --                      |")
    print(f"| **Falciform DSC**  | {final_val_summary['falc_dice']*100:.2f}%          | {final_test_summary['falc_dice']*100 if final_test_summary else 0.0:.2f}%       | --                      |")
    print(f"| **Latency / FPS**  | {final_val_summary['mean_latency_ms']:.1f} ms ({final_val_summary['fps']:.1f} FPS) | --                | Device: {final_val_summary['gpu_name']} |")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
