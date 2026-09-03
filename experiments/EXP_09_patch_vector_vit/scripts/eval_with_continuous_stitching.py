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
    parser = argparse.ArgumentParser(description="EXP_09: Post-Processing via Continuous Trajectory Stitching & Peak NMS")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/EXP_09_base/best_model.pth", help="Path to ViT-Base checkpoint")
    parser.add_argument("--backbone", type=str, default=config.backbone_name, help="ViT backbone (default: vit_base_patch16_224)")
    parser.add_argument("--dataset_dir", type=str, default=config.dataset_dir, help="Root path to surgical dataset")
    parser.add_argument("--masks_gt_dir", type=str, default="", help="Path to real masks_gt folder (auto-detected if empty)")
    parser.add_argument("--split", type=str, default="val", help="Dataset split (val or train)")
    parser.add_argument("--threshold", type=float, default=config.confidence_thresh, help="Patch confidence threshold (default: 0.20)")
    parser.add_argument("--stroke_thickness", type=int, default=2, help="Stroke thickness in pixels (default: 2)")
    parser.add_argument("--max_eval_samples", type=int, default=122, help="Number of validation samples to evaluate")
    parser.add_argument("--output_dir", type=str, default="outputs/eval_continuous_stitching", help="Directory to save visual figures")
    parser.add_argument("--use_depth", action="store_true", default=config.use_depth, help="Ingest Depth Anything V2 as 4th channel")
    return parser.parse_args()


# =========================================================================
# Core Algorithm: Directional Peak NMS + Continuous Polyline Stitching
# =========================================================================

def stitch_patch_curves_to_continuous_lines(
    patch_classes: np.ndarray,
    patch_beziers: np.ndarray,
    patch_size: int = 16,
    img_size: int = 512,
    threshold: float = 0.20,
    stroke_thickness: int = 2,
    num_samples_per_curve: int = 10
):
    """
    Transforms disjoint patch Bézier predictions into smooth, solid, continuous polylines.
    1. Suppresses parallel ghost bursts via Directional Orthogonal NMS.
    2. Sorts and chains adjacent patch curve endpoints end-to-end (P_exit -> P_entry).
    3. Renders single, unbroken continuous polylines matching TopoNet dataset GT format.
    """
    if patch_classes.ndim == 3:
        cls_map = np.argmax(patch_classes, axis=-1)
        conf_map = np.max(patch_classes[:, :, 1:], axis=-1)
    else:
        cls_map = patch_classes.astype(np.int32)
        conf_map = np.ones_like(cls_map, dtype=np.float32)

    grid_h, grid_w = cls_map.shape[:2]
    canvas_rgb = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    class_masks = np.zeros((4, img_size, img_size), dtype=np.uint8)

    # Cubic Bernstein polynomial basis matrix (S, 4)
    s = np.linspace(0.0, 1.0, num_samples_per_curve)
    M_basis = np.column_stack([(1 - s)**3, 3 * s * (1 - s)**2, 3 * s**2 * (1 - s), s**3])

    for c_id in range(1, 5):
        # 1. Identify active patches for this class
        active_mask = (cls_map == c_id) & (conf_map >= threshold)
        if not np.any(active_mask):
            continue

        active_coords = np.argwhere(active_mask)  # (N, 2) array of [row, col]

        # 2. Directional Orthogonal NMS (Kills parallel bursts)
        # Suppress patches perpendicular to the curve trajectory
        surviving_patches = []
        for (r, c) in active_coords:
            local_ctrl = patch_beziers[r, c]  # (4, 2) in [0, 1]^2
            dx = local_ctrl[-1, 0] - local_ctrl[0, 0]
            dy = local_ctrl[-1, 1] - local_ctrl[0, 1]
            my_conf = conf_map[r, c]

            is_peak = True
            if abs(dx) >= abs(dy):
                # Horizontal line: check vertical neighbors (r-1, c) and (r+1, c)
                for dr in [-1, 1]:
                    nr = r + dr
                    if 0 <= nr < grid_h and (cls_map[nr, c] == c_id) and (conf_map[nr, c] > my_conf):
                        is_peak = False
                        break
            else:
                # Vertical line: check horizontal neighbors (r, c-1) and (r, c+1)
                for dc in [-1, 1]:
                    nc = c + dc
                    if 0 <= nc < grid_w and (cls_map[r, nc] == c_id) and (conf_map[r, nc] > my_conf):
                        is_peak = False
                        break

            if is_peak:
                surviving_patches.append((r, c))

        if not surviving_patches:
            continue

        # 3. Sort surviving patches along principal axis
        surviving_arr = np.array(surviving_patches)
        r_span = surviving_arr[:, 0].max() - surviving_arr[:, 0].min()
        c_span = surviving_arr[:, 1].max() - surviving_arr[:, 1].min()

        if c_span >= r_span:
            # Sort by column (left to right)
            order = np.argsort(surviving_arr[:, 1])
        else:
            # Sort by row (top to bottom)
            order = np.argsort(surviving_arr[:, 0])

        sorted_patches = surviving_arr[order]

        # 4. Chain into continuous polylines (bridging patch seams)
        chains = []
        curr_chain = []

        for (r, c) in sorted_patches:
            offset = np.array([c * patch_size, r * patch_size], dtype=np.float32)
            global_ctrl = offset + patch_beziers[r, c] * float(patch_size)
            curve_pts = M_basis @ global_ctrl  # (S, 2)

            if len(curr_chain) == 0:
                curr_chain.append(curve_pts)
            else:
                last_pt = curr_chain[-1][-1]
                start_pt = curve_pts[0]
                dist = np.linalg.norm(start_pt - last_pt)

                # If patches are neighbors (within 2 patch widths = 32px), link them
                if dist <= float(patch_size) * 2.2:
                    curr_chain.append(curve_pts)
                else:
                    # Discontinuity: save completed chain and start new
                    chains.append(np.vstack(curr_chain))
                    curr_chain = [curve_pts]

        if curr_chain:
            chains.append(np.vstack(curr_chain))

        # 5. Draw solid, continuous, unbroken polylines
        color = CLASS_COLORS_RGB.get(c_id, (255, 255, 255))
        for chain in chains:
            if len(chain) < 2:
                continue
            pts_int = np.clip(np.round(chain), 0, img_size - 1).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas_rgb, [pts_int], isClosed=False, color=color, thickness=stroke_thickness, lineType=cv2.LINE_AA)
            cv2.polylines(class_masks[c_id - 1], [pts_int], isClosed=False, color=1, thickness=stroke_thickness, lineType=cv2.LINE_AA)

    return canvas_rgb, class_masks


# =========================================================================
# Ground Truth Loader & Metrics
# =========================================================================

def resolve_masks_gt_dir(specified_dir: str, dataset_dir: str, split: str = "val") -> str:
    if specified_dir and os.path.isdir(specified_dir):
        return specified_dir

    candidates = [
        f"/kaggle/input/datasets/khoatrytopublish/l3d-{split}/Val/masks_gt",
        f"/kaggle/input/datasets/khoatrytopublish/l3d-{split}/val/masks_gt",
        f"/kaggle/input/l3d-{split}/Val/masks_gt",
        f"/kaggle/input/l3d-{split}/val/masks_gt",
        os.path.join(dataset_dir, split, "masks_gt"),
        os.path.join(dataset_dir, "Val", "masks_gt"),
        os.path.join(dataset_dir, "val", "masks_gt"),
        f"data/laparoscopic_liver/{split}/masks_gt",
    ]

    for cand in candidates:
        if os.path.isdir(cand):
            return cand
    return ""


def load_real_gt_mask(masks_gt_dir: str, image_filename: str, target_size: tuple = (512, 512)):
    if not masks_gt_dir or not os.path.isdir(masks_gt_dir):
        return None

    stem = os.path.splitext(os.path.basename(image_filename))[0]
    for ext in [".png", ".jpg", "_mask.png"]:
        path = os.path.join(masks_gt_dir, f"{stem}{ext}")
        if os.path.exists(path):
            gt_raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if gt_raw is None:
                continue
            if gt_raw.ndim == 3:
                gt_gray = cv2.cvtColor(gt_raw, cv2.COLOR_BGR2GRAY)
            else:
                gt_gray = gt_raw
            if gt_gray.shape[:2] != target_size:
                gt_gray = cv2.resize(gt_gray, target_size, interpolation=cv2.INTER_NEAREST)
            return (gt_gray > 0).astype(np.uint8)
    return None


def compute_dice(pred_bin: np.ndarray, gt_bin: np.ndarray, eps: float = 1e-6) -> float:
    inter = np.sum((pred_bin > 0) & (gt_bin > 0))
    total = np.sum(pred_bin > 0) + np.sum(gt_bin > 0)
    if total < eps:
        return 1.0 if np.sum(gt_bin > 0) == 0 else 0.0
    return float((2.0 * inter + eps) / (total + eps))


# =========================================================================
# Main Execution
# =========================================================================

def main():
    args = parse_args()
    in_chans = 4 if args.use_depth else 3

    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        for candidate in ["checkpoints/EXP_09_base/best_model.pth", "checkpoints/EXP_09/best_model.pth"]:
            if os.path.exists(candidate):
                checkpoint_path = candidate
                break

    masks_gt_dir = resolve_masks_gt_dir(args.masks_gt_dir, args.dataset_dir, args.split)

    print("=" * 85)
    print("🚀 [EXP_09] Post-Processing: Continuous Trajectory Stitching & Peak NMS")
    print(f"📦 Checkpoint:     {checkpoint_path}")
    print(f"🏛️ Backbone:       {args.backbone}")
    print(f"📂 Dataset:        {args.dataset_dir} (Split: {args.split})")
    print(f"🎯 Real GT Masks:  {masks_gt_dir or '[Fallback to dataset Spline JSON]'}")
    print(f"📐 Stroke Width:   {args.stroke_thickness} px")
    print(f"🎯 Threshold:      {args.threshold}")
    print("=" * 85)

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Initialize ViT-Base
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
        print(f"⚠️ Checkpoint not found at '{checkpoint_path}'.")

    model.eval()

    # 2. Dataset
    val_dataset = PatchBezierLandmarkDataset(
        dataset_dir=args.dataset_dir,
        mode=args.split,
        image_size=config.image_size,
        patch_size=config.patch_size,
        spline_step_px=config.spline_step_px,
        stroke_thickness=args.stroke_thickness,
        use_depth=args.use_depth
    )

    num_samples = min(len(val_dataset), args.max_eval_samples)
    print(f"🚀 Evaluating {num_samples} validation frames...\n")

    raw_dices = []
    stitched_dices = []

    for idx in range(num_samples):
        sample = val_dataset[idx]
        image_tensor = sample["image"].unsqueeze(0).to(device)
        img_filename = sample.get("filename", f"sample_{idx+1:03d}.png")
        stem = os.path.splitext(os.path.basename(img_filename))[0]

        # 3. Inference
        with torch.no_grad():
            pred_dict = model(image_tensor)

        patch_probs = torch.softmax(pred_dict["patch_logits"][0], dim=-1).cpu().numpy()
        patch_beziers = pred_dict["patch_beziers"][0].cpu().numpy()

        # 4. Baseline Raw Drawing (Dashed Bursts)
        raw_canvas, raw_class_masks = merge_patch_beziers_to_image(
            patch_classes=patch_probs,
            patch_beziers=patch_beziers,
            patch_size=config.patch_size,
            img_size=config.image_size,
            threshold=args.threshold,
            stroke_thickness=args.stroke_thickness,
            return_class_masks=True
        )
        raw_bin_mask = (raw_class_masks.sum(axis=0) > 0).astype(np.uint8)

        # 5. Post-Processing: Continuous Trajectory Stitching & Directional NMS
        stitched_canvas, stitched_class_masks = stitch_patch_curves_to_continuous_lines(
            patch_classes=patch_probs,
            patch_beziers=patch_beziers,
            patch_size=config.patch_size,
            img_size=config.image_size,
            threshold=args.threshold,
            stroke_thickness=args.stroke_thickness
        )
        stitched_bin_mask = (stitched_class_masks.sum(axis=0) > 0).astype(np.uint8)

        # 6. Load Ground Truth Mask (from masks_gt if present, else dataset target_masks)
        real_gt_bin = load_real_gt_mask(masks_gt_dir, img_filename, target_size=(config.image_size, config.image_size))
        if real_gt_bin is not None:
            gt_bin_mask = real_gt_bin
            gt_source = "Real masks_gt"
        else:
            gt_bin_mask = (sample["target_masks"].sum(dim=0) > 0).cpu().numpy().astype(np.uint8)
            gt_source = "Spline JSON"

        # 7. Compute Dice: Raw vs Stitched
        raw_dice = compute_dice(raw_bin_mask, gt_bin_mask)
        stitched_dice = compute_dice(stitched_bin_mask, gt_bin_mask)

        raw_dices.append(raw_dice)
        stitched_dices.append(stitched_dice)
        gain = stitched_dice - raw_dice

        if (idx + 1) % 10 == 0 or idx == 0 or idx == num_samples - 1:
            print(f"[{idx+1:03d}/{num_samples:03d}] {stem} | Raw Dashes: {raw_dice:.4f} ➔ Stitched Continuous: {stitched_dice:.4f} ({gain:+.4f}) [{gt_source}]")

        # 8. Render 5-Panel High-Resolution Diagnostic Figure
        img_np = sample["image"][:3].cpu().numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        frame_rgb = np.clip((img_np * std + mean) * 255.0, 0, 255).astype(np.uint8)

        # Overlay (Green: GT, Magenta: Stitched Pred, White: Exact Fit)
        overlay = np.zeros((config.image_size, config.image_size, 3), dtype=np.uint8)
        overlay[:, :, 0] = np.clip(stitched_bin_mask * 255, 0, 255)
        overlay[:, :, 1] = np.clip(gt_bin_mask * 255, 0, 255)
        overlay[:, :, 2] = np.clip(stitched_bin_mask * 255, 0, 255)

        fig, axes = plt.subplots(1, 5, figsize=(25, 5), dpi=150)
        axes[0].imshow(frame_rgb)
        axes[0].set_title(f"Input Frame: {stem}", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(gt_bin_mask, cmap="gray")
        axes[1].set_title(f"Benchmark GT Mask ({gt_source})", fontsize=11, fontweight="bold")
        axes[1].axis("off")

        axes[2].imshow(raw_canvas)
        axes[2].set_title(f"Raw ViT-Base (Dashed Bursts: {raw_bin_mask.sum()} px)\nRaw Dice: {raw_dice:.4f}", fontsize=11, fontweight="bold")
        axes[2].axis("off")

        axes[3].imshow(stitched_canvas)
        axes[3].set_title(f"Stitched Continuous (Clean: {stitched_bin_mask.sum()} px)\nClean Dice: {stitched_dice:.4f}", fontsize=11, fontweight="bold", color="darkgreen")
        axes[3].axis("off")

        axes[4].imshow(overlay)
        axes[4].set_title(f"Continuous Alignment Overlay\nGreen: GT | Magenta: Pred | White: Fit", fontsize=11, fontweight="bold")
        axes[4].axis("off")

        plt.suptitle(
            f"Sample #{idx+1:03d} | Trajectory Stitching Gain: {raw_dice:.4f} ➔ {stitched_dice:.4f} ({gain:+.4f})",
            fontsize=13, fontweight="bold", y=1.02
        )
        plt.tight_layout()

        out_fig_path = os.path.join(args.output_dir, f"stitched_sample_{idx+1:03d}_{stem}.png")
        plt.savefig(out_fig_path, bbox_inches="tight")
        plt.close()

    # 9. Summary
    mean_raw = np.mean(raw_dices)
    mean_stitched = np.mean(stitched_dices)
    mean_gain = mean_stitched - mean_raw

    print("\n" + "=" * 85)
    print("🏆 [CONTINUOUS TRAJECTORY STITCHING RESULTS]")
    print(f"📊 Evaluated Samples:                {num_samples} validation frames")
    print(f"🔴 Mean Raw Dice (Dashed Bursts):    {mean_raw:.4f}")
    print(f"🟢 Mean Stitched Dice (Continuous):  {mean_stitched:.4f}")
    print(f"🚀 Average Improvement:              {mean_gain:+.4f} ({(mean_gain / max(mean_raw, 1e-6))*100:+.1f}%)")
    print(f"📁 High-Res Visuals Saved to:        {args.output_dir}")
    print("=" * 85)


if __name__ == "__main__":
    main()
