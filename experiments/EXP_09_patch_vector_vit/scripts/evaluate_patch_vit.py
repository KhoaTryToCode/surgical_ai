import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import sys
import argparse
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Ensure experiment root is in sys.path
exp_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if exp_root not in sys.path:
    sys.path.insert(0, exp_root)

from configs.exp09_config import config
from models.patch_vector_vit import PatchBezierViT
from models.patch_merger import merge_patch_beziers_to_image
from utils.dataset_patch_vit import PatchBezierLandmarkDataset


def parse_args():
    parser = argparse.ArgumentParser(description="EXP_09: Evaluate Patch-Bézier ViT")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to model checkpoint (.pth)")
    parser.add_argument("--dataset_dir", type=str, default=config.dataset_dir, help="Dataset directory")
    parser.add_argument("--split", type=str, default="val", help="Dataset split (val or train)")
    parser.add_argument("--threshold", type=float, default=config.confidence_thresh, help="Patch confidence threshold")
    parser.add_argument("--output_dir", type=str, default="outputs/eval_results", help="Directory to save evaluation figures")
    parser.add_argument("--max_eval_samples", type=int, default=50, help="Maximum samples to evaluate")
    parser.add_argument("--use_depth", action="store_true", default=config.use_depth, help="Ingest Depth Anything V2 as 4th channel")
    return parser.parse_args()


def compute_dice_score(pred_mask: np.ndarray, gt_mask: np.ndarray, eps: float = 1e-6) -> float:
    """Computes Dice similarity coefficient between two binary/continuous masks."""
    intersection = np.sum(pred_mask * gt_mask)
    union = np.sum(pred_mask) + np.sum(gt_mask)
    if union < eps:
        return 1.0  # Both empty
    return float((2.0 * intersection + eps) / (union + eps))


def main():
    args = parse_args()
    in_chans = 4 if args.use_depth else 3
    print("=" * 75)
    print(f"📊 [EXP_09] Evaluating Patch-Level Bézier Vision Transformer (RGB-D: {args.use_depth}, Channels: {in_chans})")
    print(f"📂 Dataset:    {args.dataset_dir} (Split: {args.split})")
    print(f"🎯 Threshold:  {args.threshold}")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize model
    model = PatchBezierViT(
        backbone_name=config.backbone_name,
        in_chans=in_chans,
        pretrained=False,
        image_size=config.image_size,
        patch_size=config.patch_size,
        num_classes=config.num_classes,
        embed_dim=config.embed_dim
    ).to(device)

    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"📦 Loading weights from: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
    else:
        print("⚠️ No checkpoint found. Running evaluation in diagnostic mode.")

    model.eval()

    # Load dataset
    dataset = PatchBezierLandmarkDataset(
        dataset_dir=args.dataset_dir,
        mode=args.split,
        image_size=config.image_size,
        patch_size=config.patch_size,
        spline_step_px=config.spline_step_px,
        stroke_thickness=config.stroke_thickness,
        use_depth=args.use_depth
    )

    num_samples = min(len(dataset), args.max_eval_samples)
    print(f"🔍 Evaluating {num_samples} samples...")

    total_tp = 0
    total_fp = 0
    total_fn = 0
    ctrl_errors_px = []
    dice_scores = []

    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    with torch.no_grad():
        for i in range(num_samples):
            sample = dataset[i]
            img_tensor = sample["image"].unsqueeze(0).to(device)
            tgt_classes = sample["target_classes"].numpy()      # (32, 32)
            tgt_beziers = sample["target_beziers"].numpy()      # (32, 32, 4, 2)
            act_mask = sample["active_mask"].numpy()            # (32, 32)
            tgt_masks = sample["target_masks"].numpy()          # (4, 512, 512)

            pred_dict = model(img_tensor)
            # (1, 32, 32, C+1)
            logits = pred_dict["patch_logits"].squeeze(0).cpu().numpy()
            beziers = pred_dict["patch_beziers"].squeeze(0).cpu().numpy()

            probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
            pred_classes = np.argmax(probs, axis=-1)
            pred_active = (pred_classes > 0) & (np.max(probs[:, :, 1:], axis=-1) >= args.threshold)

            # Patch Detection Classification Metrics
            tp = np.sum(pred_active & act_mask)
            fp = np.sum(pred_active & (~act_mask))
            fn = np.sum((~pred_active) & act_mask)
            total_tp += tp
            total_fp += fp
            total_fn += fn

            # Control Point Error on True Positive Patches
            tp_mask = pred_active & act_mask
            if np.sum(tp_mask) > 0:
                pred_ctrl_tp = beziers[tp_mask] * float(config.patch_size)      # in pixels
                tgt_ctrl_tp = tgt_beziers[tp_mask] * float(config.patch_size)
                err = np.mean(np.linalg.norm(pred_ctrl_tp - tgt_ctrl_tp, axis=-1))
                ctrl_errors_px.append(err)

            # Reconstruct Final Image & Calculate Dice Score
            pred_canvas, pred_class_masks = merge_patch_beziers_to_image(
                patch_classes=probs,
                patch_beziers=beziers,
                patch_size=config.patch_size,
                img_size=config.image_size,
                threshold=args.threshold,
                stroke_thickness=config.stroke_thickness,
                return_class_masks=True
            )

            # Image Dice
            pred_bin = (pred_class_masks.sum(axis=0) > 0).astype(np.float32)
            gt_bin = (tgt_masks.sum(axis=0) > 0).astype(np.float32)
            sample_dice = compute_dice_score(pred_bin, gt_bin)
            dice_scores.append(sample_dice)

            # Save qualitative visual figure for first 5 samples
            if i < 5:
                img_np = img_tensor.squeeze(0).cpu().numpy() * std + mean
                img_np = np.clip(img_np.transpose(1, 2, 0), 0.0, 1.0)

                gt_canvas = merge_patch_beziers_to_image(
                    patch_classes=tgt_classes,
                    patch_beziers=tgt_beziers,
                    patch_size=config.patch_size,
                    img_size=config.image_size,
                    stroke_thickness=config.stroke_thickness
                )

                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                axes[0].imshow(img_np)
                axes[0].set_title("Input Frame", fontsize=12)
                axes[0].axis("off")

                axes[1].imshow(gt_canvas)
                axes[1].set_title(f"Ground Truth Bézier Merge (Active: {np.sum(act_mask)})", fontsize=12)
                axes[1].axis("off")

                axes[2].imshow(pred_canvas)
                axes[2].set_title(f"Predicted Bézier Merge (Dice: {sample_dice:.3f})", fontsize=12)
                axes[2].axis("off")

                fig_path = os.path.join(args.output_dir, f"sample_{i:02d}.png")
                plt.tight_layout()
                plt.savefig(fig_path, dpi=150, bbox_inches="tight")
                plt.close()

    # Aggregate Metrics
    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    mean_ctrl_err = np.mean(ctrl_errors_px) if len(ctrl_errors_px) > 0 else 0.0
    mean_dice = np.mean(dice_scores) if len(dice_scores) > 0 else 0.0

    print("=" * 75)
    print("📈 Evaluation Results:")
    print(f"   - Patch Precision:        {precision:.4f}")
    print(f"   - Patch Recall:           {recall:.4f}")
    print(f"   - Patch F1-Score:         {f1:.4f}")
    print(f"   - Mean Bézier Error (px): {mean_ctrl_err:.2f} px (at 512×512)")
    print(f"   - Merged Image Mean Dice: {mean_dice:.4f}")
    print(f"📁 Visual overlays saved to: {args.output_dir}")
    print("=" * 75)


if __name__ == "__main__":
    main()
