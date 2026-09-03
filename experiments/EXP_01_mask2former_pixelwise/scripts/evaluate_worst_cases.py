#!/usr/bin/env python3
"""
Worst-to-Best Failure Mode Analysis for Mask2Former on Laparoscopic Liver Landmark Dataset.

1. Evaluates trained Mask2Former checkpoint on all validation samples.
2. Computes overall foreground Dice, IoU, and per-landmark class Dice (Ridge, Silhouette, Ligament).
3. Ranks and prints all validation samples sorted from WORST to BEST.
4. Diagnoses failure patterns (missing classes, over-segmentation, fragmentation, complete collapse).
5. Generates 4-panel visual comparison figures (RGB, GT, Pred, Error Map) for the worst-K cases.
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np

# NumPy 2.0+ compatibility for surface_distance and legacy libraries
if not hasattr(np, "Inf"):
    np.Inf = np.inf
if not hasattr(np, "NAN"):
    np.NAN = np.nan
if not hasattr(np, "NaN"):
    np.NaN = np.nan

import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ── Setup paths ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(os.path.dirname(EXP_DIR))

for p in [SCRIPT_DIR, os.path.join(EXP_DIR, 'models'), os.path.join(REPO_ROOT, 'shared'), REPO_ROOT]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

try:
    from shared.utils.prepare_dataset import get_split
    from shared.utils.dataset import load_image, load_depth, load_mask
except ImportError:
    from utils.prepare_dataset import get_split
    from utils.dataset import load_image, load_depth, load_mask

try:
    from transformers import (
        AutoImageProcessor,
        Mask2FormerForUniversalSegmentation,
        MaskFormerForInstanceSegmentation
    )
except ImportError:
    os.system("pip install -q transformers")
    from transformers import (
        AutoImageProcessor,
        Mask2FormerForUniversalSegmentation,
        MaskFormerForInstanceSegmentation
    )

CLASS_NAMES = {
    1: "Ridge",
    2: "Silhouette",
    3: "Ligament"
}

CLASS_COLORS = {
    1: [255, 40, 40],    # Red: Ridge
    2: [40, 255, 40],    # Green: Silhouette
    3: [40, 140, 255],   # Blue: Ligament
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Mask2Former worst-to-best validation failure cases")
    parser.add_argument("--mode", type=str, default="Swin_MaskedAttn",
                        choices=["Swin_MaskedAttn", "ResNet_MaskedAttn", "Swin_FullAttn", "ResNet_FullAttn"],
                        help="Ablation mode used during training")
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="Path to trained checkpoint (.pth). Auto-detected if omitted.")
    parser.add_argument("--data_path", type=str, default="/kaggle/working/L3D",
                        help="Dataset root containing val split")
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/results_ablation/worst_cases",
                        help="Directory to save visual diagnostics and CSV summary")
    parser.add_argument("--rgbd", action="store_true", default=False,
                        help="Enable 4-channel RGB-D evaluation using Depth Anything V2 (SUB_02)")
    parser.add_argument("--top_k_save", type=int, default=15,
                        help="Number of worst-case multi-panel visual plots to export")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run inference on (cuda/cpu/mps)")
    return parser.parse_args()


def resolve_checkpoint_path(ckpt_arg, mode, is_rgbd=False):
    if ckpt_arg and os.path.exists(ckpt_arg):
        return ckpt_arg

    mode_lower = mode.lower()
    candidates = []
    if is_rgbd:
        candidates.extend([
            "/kaggle/working/results_rgbd/best_swin_rgbd.pth",
            "/kaggle/working/results_rgbd/latest_swin_rgbd.pth",
            "checkpoints/best_swin_rgbd.pth",
            "checkpoints/latest_swin_rgbd.pth",
        ])
    candidates.extend([
        f"/kaggle/working/results_ablation/{mode_lower}/best_{mode_lower}.pth",
        f"/kaggle/working/results_ablation/{mode_lower}/latest_{mode_lower}.pth",
        f"/kaggle/working/results_ablation/best_{mode_lower}.pth",
        "/kaggle/working/results_rgbd/best_swin_rgbd.pth",
        "/kaggle/working/results_rgbd/latest_swin_rgbd.pth",
        f"checkpoints/mask2former_baseline/{mode_lower}/best_{mode_lower}.pth",
        f"checkpoints/mask2former_baseline/{mode_lower}/latest_{mode_lower}.pth",
        f"checkpoints/best_{mode_lower}.pth",
    ])
    for c in candidates:
        if os.path.exists(c):
            return c
    return ckpt_arg


def resolve_dataset_path(data_arg):
    if os.path.exists(data_arg):
        return data_arg
    fallbacks = [
        "/kaggle/working/L3D",
        "data/laparoscopic_liver",
        "/content/L3D"
    ]
    for f in fallbacks:
        if os.path.exists(f):
            return f
    return data_arg


def disable_masked_attention(model):
    disabled_count = 0
    for module in model.modules():
        if module.__class__.__name__ == "Mask2FormerTransformerDecoderLayer":
            original_forward = module.forward
            def make_unmasked_forward(orig_fn):
                def unmasked_forward(*args, **kwargs):
                    kwargs['attn_mask'] = None
                    return orig_fn(*args, **kwargs)
                return unmasked_forward
            module.forward = make_unmasked_forward(original_forward)
            disabled_count += 1
        elif hasattr(module, "cross_attn"):
            original_cross_attn = module.cross_attn.forward
            def make_unmasked_cross_attn(orig_fn):
                def unmasked_cross_attn(*args, **kwargs):
                    kwargs['attn_mask'] = None
                    if 'attention_mask' in kwargs:
                        kwargs['attention_mask'] = None
                    return orig_fn(*args, **kwargs)
                return unmasked_cross_attn
            module.cross_attn.forward = make_unmasked_cross_attn(original_cross_attn)
    return model


def compute_binary_dice_iou(pred_bin, gt_bin):
    smooth = 1e-5
    intersection = np.sum(pred_bin * gt_bin)
    sum_pred = np.sum(pred_bin)
    sum_gt = np.sum(gt_bin)

    if sum_gt == 0 and sum_pred == 0:
        return 1.0, 1.0  # Both empty -> true negative agreement
    if sum_gt == 0 and sum_pred > 0:
        return 0.0, 0.0  # Hallucinated prediction on non-existent class

    dice = (2.0 * intersection + smooth) / (sum_pred + sum_gt + smooth)
    iou = dice / (2.0 - dice)
    return float(dice), float(iou)


def overlay_mask_on_rgb(rgb_img, mask_2d, alpha=0.55):
    color_mask = np.zeros_like(rgb_img, dtype=np.uint8)
    for cls_id, rgb_color in CLASS_COLORS.items():
        color_mask[mask_2d == cls_id] = rgb_color

    fg_mask = (mask_2d > 0)
    overlay = rgb_img.copy()
    overlay[fg_mask] = cv2.addWeighted(rgb_img, 1.0 - alpha, color_mask, alpha, 0)[fg_mask]
    return overlay


def create_error_map(rgb_img, pred_mask, gt_mask, alpha=0.6):
    pred_bin = (pred_mask > 0).astype(np.uint8)
    gt_bin = (gt_mask > 0).astype(np.uint8)

    tp = (pred_bin == 1) & (gt_bin == 1)      # Green: True Positive
    fp = (pred_bin == 1) & (gt_bin == 0)      # Red: False Positive (Hallucination)
    fn = (pred_bin == 0) & (gt_bin == 1)      # Yellow: False Negative (Missed Edge)

    error_overlay = rgb_img.copy()
    error_layer = np.zeros_like(rgb_img, dtype=np.uint8)
    error_layer[tp] = [40, 255, 40]    # Green
    error_layer[fp] = [255, 40, 40]    # Red
    error_layer[fn] = [255, 230, 0]    # Yellow

    active = (tp | fp | fn)
    error_overlay[active] = cv2.addWeighted(rgb_img, 1.0 - alpha, error_layer, alpha, 0)[active]
    return error_overlay


def diagnose_failure(pred_map, gt_2d, overall_dice, dice_c1, dice_c2, dice_c3):
    pred_fg = (pred_map > 0).sum()
    gt_fg = (gt_2d > 0).sum()

    issues = []
    if pred_fg == 0:
        return "Complete Collapse (0 pixels predicted)"

    if overall_dice < 0.20:
        if pred_fg > 3 * gt_fg:
            issues.append("Severe Hallucination / Over-segmentation")
        elif pred_fg < 0.2 * gt_fg:
            issues.append("Severe Under-segmentation / Dropout")
        else:
            issues.append("Spatial Misalignment / Inverted Boundary")

    # Check per-class missing
    for c_id, c_name, c_dice in [(1, "Ridge", dice_c1), (2, "Silhouette", dice_c2), (3, "Ligament", dice_c3)]:
        gt_c = (gt_2d == c_id).sum()
        pred_c = (pred_map == c_id).sum()
        if gt_c > 100 and pred_c == 0:
            issues.append(f"Missed {c_name}")
        elif gt_c > 100 and c_dice < 0.15:
            issues.append(f"Poor {c_name} ({c_dice:.2f})")
        elif gt_c == 0 and pred_c > 500:
            issues.append(f"Spurious {c_name}")

    if not issues:
        if overall_dice >= 0.70:
            return "Good Segmentation"
        elif overall_dice >= 0.50:
            return "Moderate Boundary Jitter"
        else:
            return "Boundary Noise / Partial Miss"

    return "; ".join(issues)


def generate_multipanel_figure(rgb_img, gt_2d, pred_map, sample_info, save_path):
    H, W, _ = rgb_img.shape
    vis_gt = overlay_mask_on_rgb(rgb_img, gt_2d, alpha=0.55)
    vis_pred = overlay_mask_on_rgb(rgb_img, pred_map, alpha=0.55)
    vis_error = create_error_map(rgb_img, pred_map, gt_2d, alpha=0.65)

    # Add header titles
    def add_title(img, text, bg_color=(30, 30, 30)):
        banner_h = 45
        banner = np.full((banner_h, img.shape[1], 3), bg_color, dtype=np.uint8)
        cv2.putText(banner, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        return np.vstack([banner, img])

    panel1 = add_title(rgb_img, f"1. RGB Input ({sample_info['filename']})")
    panel2 = add_title(vis_gt, "2. Ground Truth Landmarks (Red:Ridge, Green:Silh, Blue:Lig)")
    panel3 = add_title(vis_pred, f"3. Mask2Former Pred (Dice: {sample_info['dice']:.4f})")
    panel4 = add_title(vis_error, "4. Error Map (Green:TP, Red:FP/Halluc, Yellow:FN/Missed)")

    top_row = np.hstack([panel1, panel2])
    bot_row = np.hstack([panel3, panel4])
    canvas = np.vstack([top_row, bot_row])

    # Add top diagnosis banner
    header_h = 55
    header = np.full((header_h, canvas.shape[1], 3), (20, 20, 20), dtype=np.uint8)
    summary_text = (f"Rank #{sample_info['rank']} (Worst) | File: {sample_info['filename']} | "
                    f"Dice: {sample_info['dice']:.4f} | IoU: {sample_info['iou']:.4f} | "
                    f"R:{sample_info['dice_r']:.2f} S:{sample_info['dice_s']:.2f} L:{sample_info['dice_l']:.2f} | "
                    f"Diagnosis: {sample_info['diagnosis']}")
    cv2.putText(header, summary_text, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2, cv2.LINE_AA)
    final_canvas = np.vstack([header, canvas])

    cv2.imwrite(str(save_path), cv2.cvtColor(final_canvas, cv2.COLOR_RGB2BGR))


def adapt_model_to_rgbd(model):
    """
    Adapts Swin patch embedding projection from 3 to 4 channels for RGB-D weights.
    """
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) and module.in_channels == 3:
            old_conv = module
            new_conv = torch.nn.Conv2d(
                in_channels=4,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                dilation=old_conv.dilation,
                groups=old_conv.groups,
                bias=(old_conv.bias is not None)
            )
            with torch.no_grad():
                new_conv.weight[:, :3, :, :] = old_conv.weight
                new_conv.weight[:, 3:4, :, :] = old_conv.weight.mean(dim=1, keepdim=True)
                if old_conv.bias is not None:
                    new_conv.bias.copy_(old_conv.bias)

            parent_name, child_name = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
            setattr(parent, child_name, new_conv)
            print(f"✅ Adapted '{name}' to 4 channels for RGB-D evaluation.")
            return model
    return model


def main():
    args = parse_args()
    device = torch.device(args.device)

    print("=" * 80)
    print("🔍 MASK2FORMER VALIDATION INFERENCE & WORST-TO-BEST ANALYSIS")
    print(f"   Mode: {args.mode} | RGB-D Flag: {args.rgbd}")
    print(f"   Device: {device}")
    print("=" * 80)

    # 1. Resolve Checkpoint
    ckpt_path = resolve_checkpoint_path(args.ckpt_path, args.mode, is_rgbd=args.rgbd)
    if not ckpt_path or not os.path.exists(ckpt_path):
        print(f"❌ Error: Checkpoint not found at '{ckpt_path}'.")
        print("   Please pass --ckpt_path /path/to/your/checkpoint.pth explicitly.")
        sys.exit(1)
    print(f"📦 Loading weights from: '{ckpt_path}'")

    # 2. Resolve Dataset
    data_path = resolve_dataset_path(args.data_path)
    if not os.path.exists(data_path):
        print(f"❌ Error: Dataset path '{data_path}' not found.")
        print("   Please ensure dataset exists at /kaggle/working/L3D or pass --data_path.")
        sys.exit(1)
    print(f"📁 Dataset root: '{data_path}'")

    train_files, test_files, val_files = get_split(data_path)
    print(f"📊 Found {len(val_files)} validation samples.")
    if len(val_files) == 0:
        print("❌ Error: No validation samples found. Check dataset folder structure.")
        sys.exit(1)

    # 3. Model & Processor Setup
    if "ResNet" in args.mode:
        model_name = "facebook/maskformer-resnet50-ade"
        model = MaskFormerForInstanceSegmentation.from_pretrained(
            model_name, num_labels=4, ignore_mismatched_sizes=True
        ).to(device)
    else:
        model_name = "facebook/mask2former-swin-tiny-ade-semantic"
        model = Mask2FormerForUniversalSegmentation.from_pretrained(
            model_name, num_labels=4, ignore_mismatched_sizes=True
        ).to(device)

    if "FullAttn" in args.mode:
        model = disable_masked_attention(model)

    processor = AutoImageProcessor.from_pretrained(model_name, reduce_labels=False, ignore_index=255)

    # Load weights
    ckpt_obj = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt_obj, dict) and "model_state_dict" in ckpt_obj:
        state_dict = ckpt_obj["model_state_dict"]
        best_dice_logged = ckpt_obj.get("best_val_dice", None)
        epoch_logged = ckpt_obj.get("epoch", None)
        print(f"   Checkpoint metadata: Epoch={epoch_logged}, Best Val Dice={best_dice_logged}")
    else:
        state_dict = ckpt_obj

    # Auto-detect if checkpoint has 4 channels (RGB-D)
    is_rgbd = args.rgbd
    for k, v in state_dict.items():
        if "patch_embeddings.projection.weight" in k and v.shape[1] == 4:
            is_rgbd = True
            break

    if is_rgbd:
        print("💡 Detected 4-channel RGB-D checkpoint (Depth Anything V2 active).")
        model = adapt_model_to_rgbd(model)
        if args.output_dir == "/kaggle/working/results_ablation/worst_cases":
            args.output_dir = "/kaggle/working/results_rgbd/worst_cases"

    model.load_state_dict(state_dict)
    model.eval()
    print("✅ Model loaded successfully in evaluation mode.")

    # 4. Inference Loop
    os.makedirs(args.output_dir, exist_ok=True)
    results = []

    print("\n🚀 Running inference across validation split...")
    with torch.no_grad():
        for idx, img_path in enumerate(tqdm(val_files, desc="Validating")):
            rgb_img = load_image(img_path)
            gt_masks = load_mask(img_path)
            gt_2d = np.argmax(gt_masks, axis=0).astype(np.int32)

            inputs = processor(images=[rgb_img], segmentation_maps=[gt_2d], return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)

            if is_rgbd:
                depth_img = load_depth(img_path)
                depth_tensor = torch.from_numpy(depth_img).float() / 255.0
                depth_tensor = (depth_tensor - 0.5) / 0.25
                depth_tensor = depth_tensor.unsqueeze(0).unsqueeze(0).to(device)
                if depth_tensor.shape[-2:] != pixel_values.shape[-2:]:
                    depth_tensor = F.interpolate(
                        depth_tensor,
                        size=pixel_values.shape[-2:],
                        mode="bilinear",
                        align_corners=False
                    )
                pixel_values = torch.cat([pixel_values, depth_tensor], dim=1)

            mask_labels = [m.to(device) for m in inputs["mask_labels"]]
            class_labels = [c.to(device) for c in inputs["class_labels"]]

            outputs = model(
                pixel_values=pixel_values,
                mask_labels=mask_labels,
                class_labels=class_labels
            )

            pred_map = processor.post_process_semantic_segmentation(
                outputs, target_sizes=[(1024, 1024)]
            )[0].cpu().numpy()

            # Compute overall foreground Dice & IoU (classes 1, 2, 3)
            pred_channels = np.array([pred_map == c for c in range(4)]).astype(np.uint8)
            gt_channels = np.array([gt_2d == c for c in range(4)]).astype(np.uint8)

            overall_dice, overall_iou = compute_binary_dice_iou(
                pred_channels[1:].flatten(), gt_channels[1:].flatten()
            )

            # Per-class Dice
            dice_c1, iou_c1 = compute_binary_dice_iou(pred_channels[1], gt_channels[1])
            dice_c2, iou_c2 = compute_binary_dice_iou(pred_channels[2], gt_channels[2])
            dice_c3, iou_c3 = compute_binary_iou(dice_c3_dummy := (pred_channels[3]), gt_channels[3]) if False else compute_binary_dice_iou(pred_channels[3], gt_channels[3])

            filename = Path(str(img_path)).name
            patient_id = Path(str(img_path)).parts[-3] if len(Path(str(img_path)).parts) >= 3 else "Unknown"

            diagnosis = diagnose_failure(pred_map, gt_2d, overall_dice, dice_c1, dice_c2, dice_c3)

            results.append({
                "index": idx,
                "path": str(img_path),
                "filename": filename,
                "patient_id": patient_id,
                "dice": overall_dice,
                "iou": overall_iou,
                "dice_r": dice_c1,
                "dice_s": dice_c2,
                "dice_l": dice_c3,
                "diagnosis": diagnosis,
                "rgb_img": rgb_img,
                "gt_2d": gt_2d,
                "pred_map": pred_map
            })

    # 5. Sort from WORST to BEST (ascending by Dice)
    results_sorted = sorted(results, key=lambda x: x["dice"])
    for rank, item in enumerate(results_sorted, 1):
        item["rank"] = rank

    # 6. Pretty Print Full Table to Terminal
    print("\n" + "=" * 125)
    print("📋 VALIDATION SAMPLES RANKED: WORST ──► BEST")
    print("=" * 125)
    header_fmt = "{:<5} | {:<24} | {:<10} | {:<7} | {:<7} | {:<7} | {:<7} | {:<7} | {:<30}"
    row_fmt    = "{:<5} | {:<24} | {:<10} | {:<7.4f} | {:<7.4f} | {:<7.4f} | {:<7.4f} | {:<7.4f} | {:<30}"
    print(header_fmt.format("Rank", "Filename", "Patient", "Dice", "IoU", "Ridge", "Silh", "Lig", "Diagnosis"))
    print("-" * 125)

    for item in results_sorted:
        print(row_fmt.format(
            f"#{item['rank']}",
            item["filename"][:24],
            item["patient_id"][:10],
            item["dice"],
            item["iou"],
            item["dice_r"],
            item["dice_s"],
            item["dice_l"],
            item["diagnosis"][:30]
        ))
    print("=" * 125)

    # 7. Summary Statistics
    all_dices = [r["dice"] for r in results_sorted]
    all_ious = [r["iou"] for r in results_sorted]
    print("\n📊 AGGREGATE VALIDATION SUMMARY:")
    print(f"   Total Validation Samples: {len(all_dices)}")
    print(f"   Mean Val Dice:   {np.mean(all_dices):.4f} (± {np.std(all_dices):.4f})")
    print(f"   Median Val Dice: {np.median(all_dices):.4f}")
    print(f"   Mean Val IoU:    {np.mean(all_ious):.4f}")
    print(f"   Worst Case Dice: {results_sorted[0]['dice']:.4f} ({results_sorted[0]['filename']})")
    print(f"   Best Case Dice:  {results_sorted[-1]['dice']:.4f} ({results_sorted[-1]['filename']})")
    print(f"   Severe Failures (Dice < 0.20): {sum(1 for d in all_dices if d < 0.20)} / {len(all_dices)}")
    print(f"   Poor Cases      (Dice < 0.50): {sum(1 for d in all_dices if d < 0.50)} / {len(all_dices)}")
    print(f"   High Quality    (Dice >= 0.70): {sum(1 for d in all_dices if d >= 0.70)} / {len(all_dices)}")

    # 8. Export CSV Summary
    csv_path = os.path.join(args.output_dir, "validation_worst_to_best_summary.csv")
    with open(csv_path, "w") as f:
        f.write("Rank,Filename,Patient,Dice,IoU,Ridge_Dice,Silhouette_Dice,Ligament_Dice,Diagnosis,FullPath\n")
        for item in results_sorted:
            f.write(f"{item['rank']},{item['filename']},{item['patient_id']},{item['dice']:.4f},"
                    f"{item['iou']:.4f},{item['dice_r']:.4f},{item['dice_s']:.4f},{item['dice_l']:.4f},"
                    f"\"{item['diagnosis']}\",{item['path']}\n")
    print(f"\n💾 Saved full sorted results table to: '{csv_path}'")

    # 9. Save Visual Diagnostics for Top-K Worst Cases & Top-3 Best Cases
    top_k = min(args.top_k_save, len(results_sorted))
    print(f"\n🎨 Exporting {top_k} worst-case 4-panel visual failure maps...")
    vis_dir = os.path.join(args.output_dir, "visual_diagnostics")
    os.makedirs(vis_dir, exist_ok=True)

    for item in results_sorted[:top_k]:
        out_name = f"rank{item['rank']:03d}_dice{item['dice']:.4f}_{Path(item['filename']).stem}.png"
        save_path = os.path.join(vis_dir, out_name)
        generate_multipanel_figure(item["rgb_img"], item["gt_2d"], item["pred_map"], item, save_path)

    # Also save top 3 best cases for comparison
    print("🎨 Exporting 3 best-case comparison plots...")
    for item in results_sorted[-3:]:
        out_name = f"BEST_rank{item['rank']:03d}_dice{item['dice']:.4f}_{Path(item['filename']).stem}.png"
        save_path = os.path.join(vis_dir, out_name)
        generate_multipanel_figure(item["rgb_img"], item["gt_2d"], item["pred_map"], item, save_path)

    print(f"✅ Visual diagnostics saved to: '{vis_dir}'")
    print("=" * 80)


if __name__ == "__main__":
    main()
