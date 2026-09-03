import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="torch.amp.*")
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import sys
import glob
import argparse
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt

# Ensure experiment root is in sys.path
exp_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if exp_root not in sys.path:
    sys.path.insert(0, exp_root)

from configs.exp09_config import config
from models.patch_vector_vit import PatchBezierViT
from models.patch_merger import merge_patch_beziers_to_image, CLASS_COLORS_RGB
from utils.dataset_patch_vit import PatchBezierLandmarkDataset


def parse_args():
    parser = argparse.ArgumentParser(description="EXP_09: Post-Process with Skeleton Thinning (Technique B) & Evaluate against Real GT Masks")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/EXP_09_base/best_model.pth", help="Path to trained ViT-Base checkpoint")
    parser.add_argument("--backbone", type=str, default=config.backbone_name, help="ViT backbone (default: vit_base_patch16_224)")
    parser.add_argument("--dataset_dir", type=str, default=config.dataset_dir, help="Root path to surgical dataset (/kaggle/working/L3D)")
    parser.add_argument("--masks_gt_dir", type=str, default="", help="Path to real masks_gt folder (auto-detected if empty)")
    parser.add_argument("--split", type=str, default="val", help="Dataset split (val or train)")
    parser.add_argument("--threshold", type=float, default=config.confidence_thresh, help="Patch confidence threshold (default: 0.20)")
    parser.add_argument("--stroke_thickness", type=int, default=2, help="Restroked line thickness after thinning (default: 2px)")
    parser.add_argument("--max_eval_samples", type=int, default=122, help="Number of validation samples to evaluate")
    parser.add_argument("--output_dir", type=str, default="outputs/eval_skeleton_thinning", help="Directory to save clean visual figures")
    parser.add_argument("--use_depth", action="store_true", default=config.use_depth, help="Ingest Depth Anything V2 as 4th channel")
    return parser.parse_args()


# =========================================================================
# Technique B: Morphological Skeletonization / Medial Axis Thinning
# =========================================================================

def apply_skeleton_thinning(mask_binary: np.ndarray, stroke_thickness: int = 2) -> np.ndarray:
    """
    Applies Zhang-Suen morphological thinning to collapse multi-patch burst bands
    into a strictly 1-pixel wide topological centerline, then dilates to match
    the target ground truth stroke thickness.
    """
    if mask_binary.sum() == 0:
        return np.zeros_like(mask_binary, dtype=np.uint8)

    uint8_bin = (mask_binary > 0).astype(np.uint8) * 255
    skeleton = None

    # Method 1: OpenCV Extra Modules (cv2.ximgproc)
    try:
        if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
            skeleton = cv2.ximgproc.thinning(uint8_bin, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    except Exception:
        pass

    # Method 2: Scikit-Image skeletonize
    if skeleton is None:
        try:
            from skimage.morphology import skeletonize
            skeleton = (skeletonize(uint8_bin > 0).astype(np.uint8)) * 255
        except ImportError:
            pass

    # Method 3: Pure OpenCV Morphological Erosion Loop (Guaranteed Fallback)
    if skeleton is None:
        skeleton = np.zeros(uint8_bin.shape, dtype=np.uint8)
        elem = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        temp = uint8_bin.copy()
        while True:
            eroded = cv2.erode(temp, elem)
            temp_open = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, elem)
            subset = cv2.subtract(eroded, temp_open)
            skeleton = cv2.bitwise_or(skeleton, subset)
            temp = eroded.copy()
            if cv2.countNonZero(temp) == 0:
                break

    if skeleton is None or skeleton.sum() == 0:
        skeleton = uint8_bin

    # Restroke by target stroke thickness to align with GT mask thickness
    if stroke_thickness > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (stroke_thickness, stroke_thickness))
        clean_mask = cv2.dilate(skeleton, kernel, iterations=1)
    else:
        clean_mask = skeleton

    return (clean_mask > 0).astype(np.uint8)


def thin_multiclass_mask(pred_class_mask: np.ndarray, stroke_thickness: int = 2) -> np.ndarray:
    """
    Applies skeleton thinning to each landmark class independently
    so that distinct classes are preserved without interference.
    """
    h, w = pred_class_mask.shape
    clean_multiclass = np.zeros((h, w), dtype=np.uint8)

    for c in range(1, 5):
        class_bin = (pred_class_mask == c).astype(np.uint8)
        if class_bin.sum() == 0:
            continue
        thinned_bin = apply_skeleton_thinning(class_bin, stroke_thickness=stroke_thickness)
        clean_multiclass[thinned_bin > 0] = c

    return clean_multiclass


# =========================================================================
# Ground Truth Mask Discovery & Loading
# =========================================================================

def resolve_masks_gt_dir(specified_dir: str, dataset_dir: str, split: str = "val") -> str:
    """Finds the directory containing the original benchmark ground truth masks."""
    if specified_dir and os.path.isdir(specified_dir):
        return specified_dir

    candidates = [
        # Kaggle input paths
        f"/kaggle/input/datasets/khoatrytopublish/l3d-{split}/Val/masks_gt",
        f"/kaggle/input/datasets/khoatrytopublish/l3d-{split}/val/masks_gt",
        f"/kaggle/input/l3d-{split}/Val/masks_gt",
        f"/kaggle/input/l3d-{split}/val/masks_gt",
        # Local workspace / symlink paths
        os.path.join(dataset_dir, split, "masks_gt"),
        os.path.join(dataset_dir, "Val", "masks_gt"),
        os.path.join(dataset_dir, "val", "masks_gt"),
        f"data/laparoscopic_liver/{split}/masks_gt",
    ]

    for cand in candidates:
        if os.path.isdir(cand):
            print(f"🎯 Found real ground truth masks at: {cand}")
            return cand

    print(f"⚠️ Warning: masks_gt folder not found in default paths. Checked: {candidates[:3]}...")
    return ""


def load_real_gt_mask(masks_gt_dir: str, image_filename: str, target_size: tuple = (512, 512)):
    """Loads the benchmark ground truth mask matching the image filename."""
    if not masks_gt_dir or not os.path.isdir(masks_gt_dir):
        return None

    stem = os.path.splitext(os.path.basename(image_filename))[0]
    # Try .png and .jpg
    candidates = [
        os.path.join(masks_gt_dir, f"{stem}.png"),
        os.path.join(masks_gt_dir, f"{stem}.jpg"),
        os.path.join(masks_gt_dir, f"{stem}_mask.png")
    ]

    gt_path = None
    for cand in candidates:
        if os.path.exists(cand):
            gt_path = cand
            break

    if gt_path is None:
        return None

    gt_raw = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
    if gt_raw is None:
        return None

    # Handle 3-channel or 1-channel
    if gt_raw.ndim == 3:
        gt_gray = cv2.cvtColor(gt_raw, cv2.COLOR_BGR2GRAY)
    else:
        gt_gray = gt_raw

    # Resize to target size with nearest-neighbor to preserve discrete lines
    if gt_gray.shape[:2] != target_size:
        gt_gray = cv2.resize(gt_gray, target_size, interpolation=cv2.INTER_NEAREST)

    gt_binary = (gt_gray > 0).astype(np.uint8)
    return {
        "gt_binary": gt_binary,
        "gt_multiclass": gt_gray,
        "path": gt_path
    }


def compute_dice(pred_bin: np.ndarray, gt_bin: np.ndarray, eps: float = 1e-6) -> float:
    """Computes standard binary Dice similarity score."""
    inter = np.sum((pred_bin > 0) & (gt_bin > 0))
    total = np.sum(pred_bin > 0) + np.sum(gt_bin > 0)
    if total < eps:
        return 1.0 if np.sum(gt_bin > 0) == 0 else 0.0
    return float((2.0 * inter + eps) / (total + eps))


# =========================================================================
# Main Evaluation with Post-Processing
# =========================================================================

def main():
    args = parse_args()
    in_chans = 4 if args.use_depth else 3

    # Auto-resolve checkpoint path
    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        for candidate in ["checkpoints/EXP_09_base/best_model.pth", "checkpoints/EXP_09/best_model.pth"]:
            if os.path.exists(candidate):
                checkpoint_path = candidate
                break

    masks_gt_dir = resolve_masks_gt_dir(args.masks_gt_dir, args.dataset_dir, args.split)

    print("=" * 85)
    print("🔥 [EXP_09] Evaluation with Technique B: Morphological Skeletonization / Thinning")
    print(f"📦 Checkpoint:     {checkpoint_path}")
    print(f"🏛️ Backbone:       {args.backbone}")
    print(f"📂 Dataset:        {args.dataset_dir} (Split: {args.split})")
    print(f"🎯 Real GT Masks:  {masks_gt_dir or '[Fallback to dataset annotations]'}")
    print(f"📐 Target Stroke:  {args.stroke_thickness} px")
    print(f"🎯 Threshold:      {args.threshold}")
    print("=" * 85)

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Initialize ViT-Base Model
    model = PatchBezierViT(
        backbone_name=args.backbone,
        in_chans=in_chans,
        pretrained=False,
        image_size=config.image_size,
        patch_size=config.patch_size,
        num_classes=config.num_classes,
        embed_dim=config.embed_dim
    ).to(device)

    if os.path.exists(checkpoint_path):
        print(f"✅ Loading checkpoint weights from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
    else:
        print(f"⚠️ Checkpoint not found at '{checkpoint_path}'. Initializing random weights.")

    model.eval()

    # 2. Load Dataset
    val_dataset = PatchBezierLandmarkDataset(
        dataset_dir=args.dataset_dir,
        mode=args.split,
        image_size=config.image_size,
        patch_size=config.patch_size,
        spline_step_px=config.spline_step_px,
        stroke_thickness=config.stroke_thickness,
        use_depth=args.use_depth
    )

    num_samples = min(len(val_dataset), args.max_eval_samples)
    print(f"🚀 Evaluating {num_samples} validation frames...\n")

    raw_dices = []
    clean_dices = []

    for idx in range(num_samples):
        sample = val_dataset[idx]
        image_tensor = sample["image"].unsqueeze(0).to(device)
        img_filename = sample.get("filename", f"sample_{idx+1:03d}.png")
        stem = os.path.splitext(os.path.basename(img_filename))[0]

        # 3. Model Forward Pass
        with torch.no_grad():
            pred_dict = model(image_tensor)

        # 4. Rasterize Raw Patch Béziers (Contains Parallel Bursts)
        patch_probs = torch.softmax(pred_dict["patch_logits"][0], dim=-1).cpu().numpy()
        patch_beziers = pred_dict["patch_beziers"][0].cpu().numpy()

        raw_pred_canvas, raw_class_masks = merge_patch_beziers_to_image(
            patch_classes=patch_probs,
            patch_beziers=patch_beziers,
            patch_size=config.patch_size,
            img_size=config.image_size,
            threshold=args.threshold,
            stroke_thickness=args.stroke_thickness,
            return_class_masks=True
        )
        raw_class_mask = np.zeros((config.image_size, config.image_size), dtype=np.uint8)
        for c in range(1, 5):
            raw_class_mask[raw_class_masks[c - 1] > 0] = c
        raw_bin_mask = (raw_class_mask > 0).astype(np.uint8)

        # 5. Apply Technique B: Morphological Skeletonization / Thinning
        clean_class_mask = thin_multiclass_mask(raw_class_mask, stroke_thickness=args.stroke_thickness)
        clean_bin_mask = (clean_class_mask > 0).astype(np.uint8)

        # 6. Load Benchmark Ground Truth Mask (from masks_gt) or fallback to sample target_masks
        real_gt_data = load_real_gt_mask(masks_gt_dir, img_filename, target_size=(config.image_size, config.image_size))
        if real_gt_data is not None:
            gt_bin_mask = real_gt_data["gt_binary"]
            gt_source = "Real masks_gt"
        else:
            # Fallback to vector-rendered target mask from sample
            gt_bin_mask = (sample["target_masks"].sum(dim=0) > 0).cpu().numpy().astype(np.uint8)
            gt_source = "Spline JSON"

        # 7. Compute Dice: Before vs After Thinning
        raw_dice = compute_dice(raw_bin_mask, gt_bin_mask)
        clean_dice = compute_dice(clean_bin_mask, gt_bin_mask)

        raw_dices.append(raw_dice)
        clean_dices.append(clean_dice)
        dice_gain = (clean_dice - raw_dice)

        # Log progress every 10 samples
        if (idx + 1) % 10 == 0 or idx == 0 or idx == num_samples - 1:
            print(f"[{idx+1:03d}/{num_samples:03d}] {stem} | Raw Dice: {raw_dice:.4f} ➔ Clean Thinned Dice: {clean_dice:.4f} ({dice_gain:+.4f}) [{gt_source}]")

        # 8. Render 5-Panel High-Resolution Diagnostic Figure
        # Denormalize input frame
        img_np = sample["image"][:3].cpu().numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        frame_rgb = np.clip((img_np * std + mean) * 255.0, 0, 255).astype(np.uint8)

        # Build clean colored prediction canvas
        clean_canvas = np.zeros((config.image_size, config.image_size, 3), dtype=np.uint8)
        for c in range(1, 5):
            clean_canvas[clean_class_mask == c] = CLASS_COLORS_RGB.get(c, (255, 255, 255))

        # Build clean alignment overlay (Green: Real GT, Magenta: Clean Pred, White: Overlap)
        overlay = np.zeros((config.image_size, config.image_size, 3), dtype=np.uint8)
        overlay[:, :, 0] = np.clip(clean_bin_mask * 255, 0, 255)                  # Red channel (Magenta)
        overlay[:, :, 1] = np.clip(gt_bin_mask * 255, 0, 255)                     # Green channel (GT)
        overlay[:, :, 2] = np.clip(clean_bin_mask * 255, 0, 255)                  # Blue channel (Magenta)

        fig, axes = plt.subplots(1, 5, figsize=(25, 5), dpi=150)
        axes[0].imshow(frame_rgb)
        axes[0].set_title(f"Input Frame: {stem}", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(gt_bin_mask, cmap="gray")
        axes[1].set_title(f"Benchmark GT Mask ({gt_source})", fontsize=11, fontweight="bold")
        axes[1].axis("off")

        axes[2].imshow(raw_pred_canvas)
        axes[2].set_title(f"Raw ViT-Base (Burst: {raw_bin_mask.sum()} px)\nRaw Dice: {raw_dice:.4f}", fontsize=11, fontweight="bold")
        axes[2].axis("off")

        axes[3].imshow(clean_canvas)
        axes[3].set_title(f"Technique B Thinned (Clean: {clean_bin_mask.sum()} px)\nClean Dice: {clean_dice:.4f}", fontsize=11, fontweight="bold", color="darkgreen")
        axes[3].axis("off")

        axes[4].imshow(overlay)
        axes[4].set_title(f"Clean Alignment Overlay\nGreen: GT | Magenta: Pred | White: Fit", fontsize=11, fontweight="bold")
        axes[4].axis("off")

        plt.suptitle(
            f"Sample #{idx+1:03d} | Post-Processing Gain: {raw_dice:.4f} ➔ {clean_dice:.4f} ({dice_gain:+.4f})",
            fontsize=13, fontweight="bold", y=1.02
        )
        plt.tight_layout()

        out_fig_path = os.path.join(args.output_dir, f"thinned_sample_{idx+1:03d}_{stem}.png")
        plt.savefig(out_fig_path, bbox_inches="tight")
        plt.close()

    # =========================================================================
    # Final Benchmark Summary
    # =========================================================================
    mean_raw_dice = np.mean(raw_dices)
    mean_clean_dice = np.mean(clean_dices)
    mean_gain = mean_clean_dice - mean_raw_dice

    print("\n" + "=" * 85)
    print("🏆 [TECHNIQUE B: SKELETON THINNING RESULTS]")
    print(f"📊 Evaluated Samples:            {num_samples} validation frames")
    print(f"🔴 Mean Raw Dice (Burst lines):  {mean_raw_dice:.4f}")
    print(f"🟢 Mean Clean Dice (1-line):     {mean_clean_dice:.4f}")
    print(f"🚀 Average Improvement:          {mean_gain:+.4f} ({(mean_gain / max(mean_raw_dice, 1e-6))*100:+.1f}%)")
    print(f"📁 High-Res Visuals Saved to:    {args.output_dir}")
    print("=" * 85)


if __name__ == "__main__":
    main()
