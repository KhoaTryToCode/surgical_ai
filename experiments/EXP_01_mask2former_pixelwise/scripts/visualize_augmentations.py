#!/usr/bin/env python3
"""
EXP_01 Diagnostic Utility: Visualize Targeted Surgical Augmentations Before vs. After.

Demonstrates the 4 targeted surgical augmentations on real laparoscopic liver frames:
  1. 3D Perspective / Projective Warp (Simulating laparoscope tilt and out-of-plane angles)
  2. Wide-Angle Affine & Retraction Rotation (Simulating tissue retraction & anatomical inversion)
  3. Endoscopic Photometric & White-Balance Jitter (Simulating Olympus vs. Storz domain shift)
  4. Synthetic Tool Occlusion / CutMix (Suppressing false landmark hallucinations on metal shafts)
  5. Full Pipeline Combination (All 4 active in sequence)

Outputs high-resolution 2-row x 6-column comparison figures for 3 selected surgical frames.
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt

# ── Path Resolution ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(os.path.dirname(EXP_DIR))

for p in [SCRIPT_DIR, os.path.join(EXP_DIR, 'models'), os.path.join(REPO_ROOT, 'shared'), REPO_ROOT]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

try:
    from shared.utils.prepare_dataset import get_split
    from shared.utils.dataset import load_image, load_mask
except ImportError:
    from utils.prepare_dataset import get_split
    from utils.dataset import load_image, load_mask

# Multi-class colors (RGB)
CLASS_COLORS = {
    1: [255, 40, 40],    # Red: Ridge
    2: [40, 255, 40],    # Green: Silhouette
    3: [40, 140, 255],   # Blue/Cyan: Falciform Ligament
}


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize surgical augmentations Before vs. After")
    parser.add_argument("--data_path", type=str, default="/kaggle/working/L3D",
                        help="Dataset root containing images and labels")
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/augmentation_visualizations",
                        help="Directory to save visual comparison figures")
    parser.add_argument("--num_samples", type=int, default=3,
                        help="Number of surgical patient samples to visualize (default: 3)")
    return parser.parse_args()


# ==================== OVERLAY HELPER ====================
def overlay_landmarks(image_rgb, mask_2d, alpha=0.75, dilate_px=2):
    """
    Overlays multi-class surgical landmarks on top of an RGB image.
    Multi-class mask (0: Background, 1: Ridge, 2: Silhouette, 3: Ligament).
    """
    vis = image_rgb.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))

    for cls_id in [1, 2, 3]:
        cls_mask = (mask_2d == cls_id).astype(np.uint8)
        if np.count_nonzero(cls_mask) == 0:
            continue
        if dilate_px > 0:
            cls_mask = cv2.dilate(cls_mask, kernel)

        color = np.array(CLASS_COLORS[cls_id], dtype=np.uint8)
        bool_mask = cls_mask > 0
        vis[bool_mask] = np.clip(
            (1.0 - alpha) * vis[bool_mask].astype(np.float32) + alpha * color.astype(np.float32),
            0, 255
        ).astype(np.uint8)

    return vis


# ==================== AUGMENTATION 1: PERSPECTIVE WARP ====================
def aug_perspective_warp(image, mask, max_tilt=0.18):
    """
    Simulates laparoscopic 30-degree angled endoscope tilt and out-of-plane rotation.
    Uses 4-point projective transformation.
    """
    h, w = image.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    dx1 = np.random.uniform(-max_tilt, max_tilt) * w
    dy1 = np.random.uniform(-max_tilt, max_tilt) * h
    dx2 = np.random.uniform(-max_tilt, max_tilt) * w
    dy2 = np.random.uniform(-max_tilt, max_tilt) * h
    dx3 = np.random.uniform(-max_tilt, max_tilt) * w
    dy3 = np.random.uniform(-max_tilt, max_tilt) * h
    dx4 = np.random.uniform(-max_tilt, max_tilt) * w
    dy4 = np.random.uniform(-max_tilt, max_tilt) * h

    dst = np.float32([
        [max(0, dx1), max(0, dy1)],
        [min(w, w + dx2), max(0, dy2)],
        [min(w, w + dx3), min(h, h + dy3)],
        [max(0, dx4), min(h, h + dy4)]
    ])

    M = cv2.getPerspectiveTransform(src, dst)
    aug_img = cv2.warpPerspective(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    aug_mask = cv2.warpPerspective(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return aug_img, aug_mask


# ==================== AUGMENTATION 2: WIDE-ANGLE AFFINE & RETRACTION ====================
def aug_affine_retraction(image, mask, max_angle=45, scale_range=(0.85, 1.20)):
    """
    Simulates surgical grasper liver retraction (rotation up to ±90 degrees & scale shifts).
    """
    h, w = image.shape[:2]
    angle = np.random.uniform(-max_angle, max_angle)
    # 40% chance of extreme rotation (90° anatomical retraction flip)
    if np.random.rand() < 0.4:
        angle += np.random.choice([90, -90])

    scale = np.random.uniform(*scale_range)
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)

    aug_img = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    aug_mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # 50% chance horizontal flip
    if np.random.rand() > 0.5:
        aug_img = cv2.flip(aug_img, 1)
        aug_mask = cv2.flip(aug_mask, 1)

    return aug_img, aug_mask


# ==================== AUGMENTATION 3: PHOTOMETRIC & WHITE-BALANCE ====================
def aug_photometric_jitter(image, brightness=0.25, contrast=0.25, saturation=0.35, hue=0.08):
    """
    Simulates inter-patient hospital camera domain shifts (Olympus vs. Karl Storz vs. Stryker),
    desaturation, cool/gray tint, and endoscopic light falloff.
    """
    img_float = image.astype(np.float32) / 255.0

    # 1. Channel shift (White balance perturbation)
    channel_gains = np.random.uniform(0.88, 1.12, size=(1, 1, 3)).astype(np.float32)
    img_float = np.clip(img_float * channel_gains, 0.0, 1.0)

    # 2. Brightness
    b_factor = 1.0 + np.random.uniform(-brightness, brightness)
    img_float = np.clip(img_float * b_factor, 0.0, 1.0)

    # 3. Contrast around image mean
    c_factor = 1.0 + np.random.uniform(-contrast, contrast)
    mean_val = np.mean(img_float, axis=(0, 1), keepdims=True)
    img_float = np.clip((img_float - mean_val) * c_factor + mean_val, 0.0, 1.0)

    # 4. Saturation and Hue via HSV space
    img_uint8 = (img_float * 255.0).astype(np.uint8)
    hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + np.random.uniform(-hue, hue) * 180.0) % 180.0
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 + np.random.uniform(-saturation, saturation)), 0, 255)
    aug_img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    return aug_img


# ==================== AUGMENTATION 4: SYNTHETIC TOOL OCCLUSION ====================
def aug_tool_occlusion(image, mask, max_tools=2):
    """
    Pastes synthetic metallic straight bars/graspers across the surgical field.
    Forces model to ignore metallic straight edges and not hallucinate false ridges on tools.
    """
    aug_img = image.copy()
    aug_mask = mask.copy()
    h, w = image.shape[:2]

    num_tools = np.random.randint(1, max_tools + 1)
    for _ in range(num_tools):
        # Choose starting border
        border = np.random.choice(["top", "bottom", "left", "right"])
        if border == "top":
            pt1 = (np.random.randint(0, w), 0)
        elif border == "bottom":
            pt1 = (np.random.randint(0, w), h)
        elif border == "left":
            pt1 = (0, np.random.randint(0, h))
        else:
            pt1 = (w, np.random.randint(0, h))

        # Target endpoint extending towards liver center
        pt2 = (
            np.random.randint(int(w * 0.25), int(w * 0.75)),
            np.random.randint(int(h * 0.25), int(h * 0.75))
        )
        thickness = np.random.randint(35, 65)

        # Metallic tone (grasper shaft: silvery gray with highlight strip)
        val = np.random.randint(150, 210) if np.random.rand() > 0.3 else np.random.randint(40, 75)
        color = (val, val, val)

        cv2.line(aug_img, pt1, pt2, color, thickness)
        # Add highlight core along instrument axis
        cv2.line(aug_img, pt1, pt2, (min(255, val + 50), min(255, val + 50), min(255, val + 50)), thickness // 4)

        # Occlude ground-truth landmarks directly under tool
        cv2.line(aug_mask, pt1, pt2, 0, thickness)

    return aug_img, aug_mask


# ==================== TECHNIQUE 5: FULL COMBINED PIPELINE ====================
def aug_full_pipeline(image, mask):
    """
    Applies the full surgical augmentation pipeline in stochastic sequence.
    """
    img_curr, mask_curr = image.copy(), mask.copy()

    # 1. Perspective Tilt (p=0.8)
    if np.random.rand() < 0.8:
        img_curr, mask_curr = aug_perspective_warp(img_curr, mask_curr)

    # 2. Affine / Retraction (p=0.8)
    if np.random.rand() < 0.8:
        img_curr, mask_curr = aug_affine_retraction(img_curr, mask_curr)

    # 3. Photometric / Color Domain (p=0.9)
    if np.random.rand() < 0.9:
        img_curr = aug_photometric_jitter(img_curr)

    # 4. Tool Occlusion (p=0.5)
    if np.random.rand() < 0.5:
        img_curr, mask_curr = aug_tool_occlusion(img_curr, mask_curr)

    return img_curr, mask_curr


# ==================== MAIN VISUALIZATION ====================
def main():
    args = parse_args()
    print("=" * 80)
    print("🔬 SURGICAL AUGMENTATION DIAGNOSTIC VISUALIZER")
    print(f"   Data Path:   {args.data_path}")
    print(f"   Output Dir:  {args.output_dir}")
    print("=" * 80)

    train_files, test_files, val_files = get_split(args.data_path)
    # Prefer validation or train files that represent worst-case patients (Patient_40, Patient_32)
    candidate_pool = val_files if len(val_files) > 0 else train_files
    if len(candidate_pool) == 0:
        print(f"❌ Error: No samples found in '{args.data_path}'. Check path.")
        sys.exit(1)

    # Prioritize key test cases if available
    selected_samples = []
    for target_patient in ["Patient_40", "Patient_32", "Patient_12"]:
        matched = [f for f in candidate_pool if target_patient in str(f)]
        if len(matched) > 0:
            selected_samples.append(matched[0])

    # Fill remaining from candidate pool if needed
    for f in candidate_pool:
        if len(selected_samples) >= args.num_samples:
            break
        if f not in selected_samples:
            selected_samples.append(f)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"📊 Visualizing {len(selected_samples)} representative surgical frames...\n")

    for idx, img_path in enumerate(selected_samples, 1):
        raw_img = load_image(img_path)
        raw_masks = load_mask(img_path)
        raw_mask_2d = np.argmax(raw_masks, axis=0).astype(np.int32)

        file_name = Path(str(img_path)).stem

        # Seed random generator deterministically per sample for reproducible visualization
        np.random.seed(42 + idx * 7)

        # 1. Perspective
        p_img, p_mask = aug_perspective_warp(raw_img, raw_mask_2d)

        # 2. Retraction & Affine
        a_img, a_mask = aug_affine_retraction(raw_img, raw_mask_2d)

        # 3. Photometric Jitter
        c_img = aug_photometric_jitter(raw_img)
        c_mask = raw_mask_2d.copy()

        # 4. Tool Occlusion
        t_img, t_mask = aug_tool_occlusion(raw_img, raw_mask_2d)

        # 5. Full Pipeline
        f_img, f_mask = aug_full_pipeline(raw_img, raw_mask_2d)

        # Generate overlays
        over_orig = overlay_landmarks(raw_img, raw_mask_2d)
        over_p = overlay_landmarks(p_img, p_mask)
        over_a = overlay_landmarks(a_img, a_mask)
        over_c = overlay_landmarks(c_img, c_mask)
        over_t = overlay_landmarks(t_img, t_mask)
        over_f = overlay_landmarks(f_img, f_mask)

        # ── Build 2x6 Multi-Panel Comparison Canvas ──
        # Row 1: RGB Images
        # Row 2: Overlaid Landmark Ground Truths
        fig, axes = plt.subplots(2, 6, figsize=(28, 9.5), dpi=130)

        cols = [
            ("1. Original Baseline", raw_img, over_orig, "Unmodified Camera Frame"),
            ("2. Perspective Tilt", p_img, over_p, "30° Endoscope Port Angle"),
            ("3. Retraction / Flip", a_img, over_a, "Grasper Lobe Retraction (±90°)"),
            ("4. Color Domain Shift", c_img, over_c, "Olympus/Storz Light Fluctuation"),
            ("5. Tool Occlusion", t_img, over_t, "Metallic Grasper CutMix"),
            ("6. Combined Pipeline", f_img, over_f, "Full Stochastic Sequence"),
        ]

        for col_idx, (title, img_panel, over_panel, note) in enumerate(cols):
            # Row 0: RGB Image
            axes[0, col_idx].imshow(img_panel)
            axes[0, col_idx].set_title(title, fontsize=12, fontweight="bold", pad=8)
            axes[0, col_idx].axis("off")

            # Row 1: Landmark Overlay
            axes[1, col_idx].imshow(over_panel)
            axes[1, col_idx].set_title(f"Overlay: {title.split('.')[1].strip()}", fontsize=11, pad=6)
            axes[1, col_idx].axis("off")

            # Subtitle annotation
            axes[1, col_idx].text(
                0.5, -0.08, note,
                transform=axes[1, col_idx].transAxes,
                ha='center', va='top', fontsize=9, style='italic', color='#333333'
            )

        plt.suptitle(
            f"Surgical Landmark Augmentation Diagnostics — Sample #{idx}: {file_name}\n"
            f"[Red: Ridge | Green: Silhouette | Blue: Falciform Ligament]",
            fontsize=15, fontweight="bold", y=0.98
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.94])

        out_path = os.path.join(args.output_dir, f"aug_comparison_sample_{idx}_{file_name}.png")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved comparison figure {idx}/{len(selected_samples)}: '{out_path}'")

    print("\n🎉 Augmentation diagnostic complete!")
    print(f"   Visual results saved in: '{args.output_dir}'")


if __name__ == "__main__":
    main()
