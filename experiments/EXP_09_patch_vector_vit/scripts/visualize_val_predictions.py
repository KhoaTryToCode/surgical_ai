import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="torch.amp.*")
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import sys
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
    parser = argparse.ArgumentParser(description="EXP_09: Visualize Patch-Bézier ViT Predictions on Validation Set")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/EXP_09/best_model.pth", help="Path to trained model checkpoint")
    parser.add_argument("--dataset_dir", type=str, default=config.dataset_dir, help="Path to surgical dataset")
    parser.add_argument("--split", type=str, default="val", help="Dataset split (val or train)")
    parser.add_argument("--num_samples", type=int, default=6, help="Number of samples to visualize")
    parser.add_argument("--threshold", type=float, default=0.25, help="Confidence threshold for active patches (e.g. 0.25 or 0.50)")
    parser.add_argument("--output_dir", type=str, default="outputs/val_visualizations", help="Directory to save visual PNGs")
    parser.add_argument("--use_depth", action="store_true", default=config.use_depth, help="Ingest Depth Anything V2 as 4th channel")
    parser.add_argument("--show", action="store_true", help="Display interactive matplotlib figures (for Jupyter notebooks)")
    return parser.parse_args()


def compute_dice(pred_bin: np.ndarray, gt_bin: np.ndarray, eps: float = 1e-6) -> float:
    inter = np.sum(pred_bin * gt_bin)
    total = np.sum(pred_bin) + np.sum(gt_bin)
    if total < eps:
        return 1.0
    return float((2.0 * inter + eps) / (total + eps))


def main():
    args = parse_args()
    print("=" * 75)
    print("🎨 [EXP_09] Visualizing Patch-Bézier ViT Predictions on Validation Split")
    print(f"📦 Checkpoint:  {args.checkpoint}")
    print(f"📂 Dataset:     {args.dataset_dir} (Split: {args.split})")
    print(f"🎯 Threshold:   {args.threshold}")
    print(f"🖼️  Num Samples: {args.num_samples}")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    os.makedirs(args.output_dir, exist_ok=True)
    in_chans = 4 if args.use_depth else 3

    # 1. Initialize Model
    model = PatchBezierViT(
        backbone_name=config.backbone_name,
        in_chans=in_chans,
        pretrained=False,
        image_size=config.image_size,
        patch_size=config.patch_size,
        num_classes=config.num_classes,
        embed_dim=config.embed_dim
    ).to(device)

    # 2. Load Checkpoint Weights
    if os.path.exists(args.checkpoint):
        print(f"✅ Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
        if "val_bin_dice" in ckpt:
            print(f"   🏆 Checkpoint Val Dice: {ckpt['val_bin_dice']:.4f} (Epoch {ckpt.get('epoch', '?')})")
    else:
        print(f"⚠️ Warning: Checkpoint '{args.checkpoint}' not found! Using initialized weights.")

    model.eval()

    # 3. Load Dataset
    dataset = PatchBezierLandmarkDataset(
        dataset_dir=args.dataset_dir,
        mode=args.split,
        image_size=config.image_size,
        patch_size=config.patch_size,
        spline_step_px=config.spline_step_px,
        stroke_thickness=config.stroke_thickness,
        use_depth=args.use_depth
    )

    num_samples = min(len(dataset), args.num_samples)
    print(f"🔎 Visualizing {num_samples} validation samples...")

    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    for i in range(num_samples):
        sample = dataset[i]
        img_tensor = sample["image"].unsqueeze(0).to(device)
        tgt_classes = sample["target_classes"].numpy()
        tgt_beziers = sample["target_beziers"].numpy()
        act_mask = sample["active_mask"].numpy()
        tgt_masks = sample["target_masks"].numpy()

        with torch.no_grad():
            pred_dict = model(img_tensor)
            logits = pred_dict["patch_logits"].squeeze(0).cpu().numpy()
            beziers = pred_dict["patch_beziers"].squeeze(0).cpu().numpy()

        probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = probs / np.sum(probs, axis=-1, keepdims=True)
        max_conf = float(np.max(probs[:, :, 1:]))

        # Denormalize RGB Image
        img_np = img_tensor.squeeze(0)[:3].cpu().numpy() * std + mean
        img_np = np.clip(img_np.transpose(1, 2, 0), 0.0, 1.0)

        # Merge Ground Truth Curves
        gt_canvas, gt_class_masks = merge_patch_beziers_to_image(
            patch_classes=tgt_classes,
            patch_beziers=tgt_beziers,
            patch_size=config.patch_size,
            img_size=config.image_size,
            stroke_thickness=config.stroke_thickness,
            return_class_masks=True
        )

        # Merge Predicted Curves
        pred_canvas, pred_class_masks = merge_patch_beziers_to_image(
            patch_classes=probs,
            patch_beziers=beziers,
            patch_size=config.patch_size,
            img_size=config.image_size,
            threshold=args.threshold,
            stroke_thickness=config.stroke_thickness,
            return_class_masks=True
        )

        # Compute Dice Score
        pred_bin = (pred_class_masks.sum(axis=0) > 0).astype(np.float32)
        gt_bin = (gt_class_masks.sum(axis=0) > 0).astype(np.float32)
        dice = compute_dice(pred_bin, gt_bin)

        # Overlays
        gt_overlay = img_np.copy()
        gt_active_px = gt_canvas.sum(axis=-1) > 0
        gt_overlay[gt_active_px] = gt_canvas[gt_active_px] / 255.0 * 0.85 + gt_overlay[gt_active_px] * 0.15

        pred_overlay = img_np.copy()
        pred_active_px = pred_canvas.sum(axis=-1) > 0
        num_pred_patches = int(np.sum((np.argmax(probs, axis=-1) > 0) & (np.max(probs[:, :, 1:], axis=-1) >= args.threshold)))
        pred_overlay[pred_active_px] = pred_canvas[pred_active_px] / 255.0 * 0.85 + pred_overlay[pred_active_px] * 0.15

        # Visual Alignment Panel (Green = GT, Magenta = Pred, Yellow/White = Coincidence)
        align_overlay = img_np.copy() * 0.4
        align_overlay[gt_active_px, 1] += 0.6          # Green for Ground Truth
        align_overlay[pred_active_px, 0] += 0.6        # Red/Magenta for Prediction
        align_overlay[pred_active_px, 2] += 0.6
        align_overlay = np.clip(align_overlay, 0.0, 1.0)

        # Plot 4-Panel Figure
        fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))
        axes[0].imshow(img_np)
        axes[0].set_title(f"Sample {i+1}: Input Surgical Frame", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(gt_overlay)
        axes[1].set_title(f"Ground Truth Curves ({np.sum(act_mask)} patches)", fontsize=11, fontweight="bold")
        axes[1].axis("off")

        axes[2].imshow(pred_overlay)
        axes[2].set_title(f"Predicted Curves (Dice: {dice:.3f} | Active: {num_pred_patches})", fontsize=11, fontweight="bold")
        axes[2].axis("off")

        axes[3].imshow(align_overlay)
        axes[3].set_title("Alignment (Green: GT, Magenta: Pred)", fontsize=11, fontweight="bold")
        axes[3].axis("off")

        plt.suptitle(
            f"EXP_09 Patch-Bézier ViT | Val Sample #{i+1} | Threshold: {args.threshold} | Max Conf: {max_conf:.3f} | Dice: {dice:.3f}",
            fontsize=13, y=0.98
        )
        plt.tight_layout()

        save_path = os.path.join(args.output_dir, f"val_prediction_sample_{i+1:02d}.png")
        plt.savefig(save_path, dpi=160, bbox_inches="tight")
        print(f"   📸 Saved: {save_path} (Dice: {dice:.4f}, Active Patches: {num_pred_patches})")

        if args.show:
            plt.show()
        plt.close()

    print("=" * 75)
    print(f"🎉 Visualization complete! All figures saved to: {args.output_dir}")
    print("=" * 75)


if __name__ == "__main__":
    main()
