import os
import sys
import argparse
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader

exp_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if exp_root not in sys.path:
    sys.path.insert(0, exp_root)

from configs.exp10_config import EXP10Config, resolve_dataset_dir
from models.macro_patch_vit import MacroPatchViT
from models.macro_merger import merge_macro_beziers_to_image, compute_batch_dice
from utils.dataset_macro_vit import MacroPatchLandmarkDataset


def compute_iou(pred_mask, gt_mask, eps=1e-6):
    pred_bin = (pred_mask > 0.5).astype(np.float32)
    gt_bin = (gt_mask > 0.5).astype(np.float32)
    intersection = np.sum(pred_bin * gt_bin)
    union = np.sum((pred_bin + gt_bin) > 0)
    if union == 0:
        return 1.0
    return float((intersection + eps) / (union + eps))


def evaluate(args):
    print("=" * 80)
    print("🔬 [EXP_10 EVALUATION] Macro-Patch ViT Benchmark Evaluation (Way A)")
    print(f"📦 Checkpoint:     {args.checkpoint}")
    print(f"📂 Dataset:        {args.dataset_dir}")
    print(f"🎯 Threshold:      {args.thresh}")
    print(f"📐 Stroke Width:   {args.stroke_px} px")
    print("=" * 80)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"🖥️ Execution Device: {device}")

    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found at: {args.checkpoint}")
        return

    checkpoint = torch.load(args.checkpoint, map_location=device)
    saved_cfg = checkpoint.get("config", {})
    backbone_name = saved_cfg.get("backbone", "vit_base_patch16_224")
    macro_patch_size = saved_cfg.get("macro_patch_size", 64)
    use_depth = saved_cfg.get("use_depth", True)
    in_chans = 4 if use_depth else 3

    model = MacroPatchViT(
        backbone_name=backbone_name,
        in_chans=in_chans,
        pretrained=False,
        image_size=512,
        micro_patch_size=16,
        macro_patch_size=macro_patch_size,
        num_classes=4,
        embed_dim=768,
        hidden_dim=256,
        macro_depth=2,
        macro_heads=8
    ).to(device)

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    print(f"✅ Loaded weights from epoch {checkpoint.get('epoch', '?')} (Training Val Dice: {checkpoint.get('val_dice', 0.0)*100:.2f}%)")

    val_dataset = MacroPatchLandmarkDataset(
        dataset_dir=args.dataset_dir,
        mode="val",
        image_size=512,
        macro_patch_size=macro_patch_size,
        stroke_thickness=args.stroke_px,
        use_depth=use_depth
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    class_names = ["Anterior Ridge", "Liver Silhouette", "Falciform Ligament", "Gallbladder Boundary"]
    class_dices = {c: [] for c in range(4)}
    class_ious = {c: [] for c in range(4)}
    class_ctrl_errors = {c: [] for c in range(4)}

    print(f"\n🚀 Evaluating {len(val_dataset)} validation frames...")

    with torch.no_grad():
        for batch in val_loader:
            v_img = batch["image"].to(device)
            v_target_classes = batch["target_classes"].cpu().numpy()
            v_target_beziers = batch["target_beziers"].cpu().numpy()
            v_active_mask = batch["active_mask"].cpu().numpy()
            v_eval_masks = batch["target_masks"].cpu().numpy()

            preds = model(v_img)
            macro_logits_np = preds["macro_logits"].cpu().numpy()
            macro_beziers_np = preds["macro_beziers"].cpu().numpy()

            B_val = v_img.shape[0]
            for b in range(B_val):
                render_res = merge_macro_beziers_to_image(
                    macro_logits_np[b],
                    macro_beziers_np[b],
                    macro_patch_size=macro_patch_size,
                    image_size=512,
                    confidence_thresh=args.thresh,
                    stroke_thickness=args.stroke_px
                )
                pred_pixel_masks = render_res["pixel_masks"]

                for c in range(4):
                    d = compute_batch_dice(pred_pixel_masks[c], v_eval_masks[b, c])
                    iou = compute_iou(pred_pixel_masks[c], v_eval_masks[b, c])
                    class_dices[c].append(d)
                    class_ious[c].append(iou)

                    # Control point error for this class
                    cls_mask = v_active_mask[b] & (v_target_classes[b] == (c + 1))
                    if cls_mask.sum() > 0:
                        err = np.linalg.norm(
                            (macro_beziers_np[b, cls_mask] - v_target_beziers[b, cls_mask]) * float(macro_patch_size),
                            axis=-1
                        ).mean()
                        class_ctrl_errors[c].append(err)

    # -------------------------------------------------------------
    # Summary Report Table
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"{'Landmark Class':<26} | {'Dice Score':<12} | {'IoU':<10} | {'Ctrl Err (px)':<14}")
    print("-" * 80)
    all_dices = []
    all_ious = []
    all_errors = []

    for c in range(4):
        c_name = class_names[c]
        d_mean = np.mean(class_dices[c]) * 100 if class_dices[c] else 0.0
        iou_mean = np.mean(class_ious[c]) * 100 if class_ious[c] else 0.0
        err_mean = np.mean(class_ctrl_errors[c]) if class_ctrl_errors[c] else 0.0

        all_dices.extend(class_dices[c])
        all_ious.extend(class_ious[c])
        all_errors.extend(class_ctrl_errors[c])

        print(f"{c_name:<26} | {d_mean:>9.2f}%  | {iou_mean:>8.2f}% | {err_mean:>11.2f} px")

    print("=" * 80)
    overall_dice = np.mean(all_dices) * 100 if all_dices else 0.0
    overall_iou = np.mean(all_ious) * 100 if all_ious else 0.0
    overall_err = np.mean(all_errors) if all_errors else 0.0

    print(f"{'OVERALL BENCHMARK':<26} | {overall_dice:>9.2f}%  | {overall_iou:>8.2f}% | {overall_err:>11.2f} px")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate EXP_10 Macro-Patch ViT")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/EXP_10/best_model.pth", help="Path to checkpoint")
    parser.add_argument("--dataset_dir", type=str, default=resolve_dataset_dir(), help="Path to dataset")
    parser.add_argument("--thresh", type=float, default=0.30, help="Confidence threshold")
    parser.add_argument("--stroke_px", type=int, default=2, help="Stroke thickness in pixels")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--device", type=str, default="", help="Device override")
    args = parser.parse_args()
    evaluate(args)
