"""
EXP_11: Evaluate & Visualize SurgicalCurveFormer on Val Set
=============================================================
Produces 4-panel diagnostic visualizations for all 122 val frames:
  Panel 1: RGB + GT annotation overlay (cyan strokes)
  Panel 2: RGB + predicted Bézier curves (per-class colors) + ctrl pts
  Panel 3: Soft rasterizer masks vs GT masks (overlaid at 512×512)
  Panel 4: ACPI score map (top-K proposal centroids on f4 grid)

Also computes and saves official benchmark metrics:
  - Pixel Dice (30px dilation, standard L3D protocol)
  - IoU (30px dilation)
  - Hausdorff Distance (95th percentile, px)
  - Existence Accuracy

Usage (Kaggle):
  python experiments/EXP_11_surgical_curve_former/scripts/evaluate_exp11.py \\
      --checkpoint /kaggle/working/checkpoints/EXP_11/best_model.pth \\
      --dataset_dir /kaggle/working/L3D \\
      --output_dir /kaggle/working/outputs/EXP_11 \\
      --max_samples 122
"""
import os
import sys
import argparse
import json
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

exp_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ws_root  = os.path.abspath(os.path.join(exp_root, ".."))
for p in [exp_root, ws_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

from models.surgical_curve_former import SurgicalCurveFormer
from models.bezier_ops import evaluate_bezier_torch
from utils.dataset import SurgicalCurveFormerDataset

# ── Class color palette (BGR for OpenCV) ─────────────────────────────────────
CLASS_COLORS = {
    0: (0,   255, 255),   # Ridge      — Cyan
    1: (0,   165, 255),   # Silhouette — Orange
    2: (255,  50, 100),   # Ligament   — Magenta
}
CLASS_NAMES = ["Ridge", "Silhouette", "Ligament"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",   type=str, required=True)
    p.add_argument("--dataset_dir",  type=str, default="/kaggle/working/L3D")
    p.add_argument("--output_dir",   type=str, default="/kaggle/working/outputs/EXP_11")
    p.add_argument("--max_samples",  type=int, default=122)
    p.add_argument("--dilate_px",    type=int, default=30)
    p.add_argument("--thresh",       type=float, default=0.15)
    p.add_argument("--use_depth",    action="store_true", default=True)
    p.add_argument("--device",       type=str, default="")
    return p.parse_args()


def get_device(forced=""):
    if forced:
        return torch.device(forced)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Metric helpers ────────────────────────────────────────────────────────────

def compute_dice(pred: np.ndarray, gt: np.ndarray, eps=1e-6) -> float:
    inter = (pred & gt).sum()
    return float(2 * inter + eps) / float(pred.sum() + gt.sum() + eps)


def compute_iou(pred: np.ndarray, gt: np.ndarray, eps=1e-6) -> float:
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    return float(inter + eps) / float(union + eps)


def compute_hausdorff95(pred: np.ndarray, gt: np.ndarray) -> float:
    """95th-percentile Hausdorff distance (in pixels)."""
    p_pts = np.argwhere(pred)
    g_pts = np.argwhere(gt)
    if len(p_pts) == 0 or len(g_pts) == 0:
        return float("inf")
    try:
        from scipy.spatial import cKDTree
        tree_g = cKDTree(g_pts)
        tree_p = cKDTree(p_pts)
        d_p2g, _ = tree_g.query(p_pts, k=1)
        d_g2p, _ = tree_p.query(g_pts, k=1)
        return float(max(np.percentile(d_p2g, 95), np.percentile(d_g2p, 95)))
    except ImportError:
        return float("nan")


def dilate_mask(mask: np.ndarray, dilate_px: int) -> np.ndarray:
    if dilate_px <= 0:
        return mask
    k = dilate_px * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask.astype(np.uint8), kernel) > 0


# ── Curve drawing ─────────────────────────────────────────────────────────────

def draw_bezier_on_canvas(
    canvas: np.ndarray,
    ctrl_pts_norm: np.ndarray,   # (6, 2) in [0, 1]^2
    color: tuple,
    thickness: int = 2,
    n_samples: int = 100,
    draw_ctrl: bool = True,
) -> np.ndarray:
    """Draw a 5th-order Bézier curve onto canvas (H, W, 3), BGR."""
    H, W = canvas.shape[:2]
    ctrl_t = torch.from_numpy(ctrl_pts_norm).float().unsqueeze(0)   # (1, 6, 2)
    sampled = evaluate_bezier_torch(ctrl_t, num_samples=n_samples)  # (1, 100, 2)
    pts_px = sampled[0].numpy()
    pts_px[:, 0] = np.clip(pts_px[:, 0] * W, 0, W - 1)
    pts_px[:, 1] = np.clip(pts_px[:, 1] * H, 0, H - 1)
    pts_int = pts_px.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(canvas, [pts_int], isClosed=False, color=color,
                  thickness=thickness, lineType=cv2.LINE_AA)
    if draw_ctrl:
        for j in range(len(ctrl_pts_norm)):
            cx = int(ctrl_pts_norm[j, 0] * W)
            cy = int(ctrl_pts_norm[j, 1] * H)
            cv2.circle(canvas, (cx, cy), 3, color, -1)
            if j > 0:
                px = int(ctrl_pts_norm[j-1, 0] * W)
                py = int(ctrl_pts_norm[j-1, 1] * H)
                cv2.line(canvas, (px, py), (cx, cy), color, 1, cv2.LINE_AA)
    return canvas


# ── Main evaluation loop ──────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = get_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    vis_dir = os.path.join(args.output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  EXP_11 Evaluation — SurgicalCurveFormer")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Dataset:    {args.dataset_dir}")
    print(f"  Output:     {args.output_dir}")
    print(f"{'='*65}\n")

    # ── Load model ────────────────────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    saved_args = ckpt.get("args", {})

    model = SurgicalCurveFormer(
        num_classes=3, image_size=512,
        in_chans=4 if args.use_depth else 3,
        fpn_channels=256, bezier_ctrl_pts=6,
        acpi_top_k=saved_args.get("acpi_top_k", 10),
        hcr_stages=saved_args.get("hcr_stages", 3),
        hcr_embed_dim=256, hcr_heads=8, hcr_n_ref_pts=26, hcr_deform_pts=4,
        vit_backbone="vit_base_patch16_224", vit_embed_dim=768,
        vit_pretrained=False,       # weights loaded from checkpoint
        exist_attn_heads=8,
        raster_render_size=128, raster_num_samples=64, raster_sigma_px=2.0,
    ).to(device)

    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    print(f"✅ Loaded checkpoint (epoch {ckpt.get('epoch', '?')}, "
          f"best Dice={ckpt.get('best_dice', 0)*100:.2f}%)")

    # ── Dataset ───────────────────────────────────────────────────────────────
    val_ds = SurgicalCurveFormerDataset(
        dataset_dir=args.dataset_dir, mode="val",
        image_size=512, use_depth=args.use_depth,
        acpi_stride=32, render_size=128, bezier_degree=5,
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=2, pin_memory=True)
    n_samples = min(len(val_ds), args.max_samples)
    print(f"   Evaluating {n_samples} samples...\n")

    # ── Metrics accumulators ──────────────────────────────────────────────────
    all_dice    = []
    all_iou     = []
    all_hd95    = []
    per_class_dice = [[] for _ in range(3)]
    exist_correct = []

    dilate_k = args.dilate_px

    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            if idx >= n_samples:
                break

            x          = batch["image"].to(device)           # (1, 4, 512, 512)
            gt_masks   = batch["target_masks"][0].numpy()    # (M, 512, 512)
            active     = batch["active_mask"][0].numpy()     # (M,)
            img_path   = batch["img_path"][0]

            out = model(x)

            # Soft masks → 512px
            soft_masks = out["soft_masks"]     # (1, M, 128, 128)
            soft_512   = F.interpolate(soft_masks, size=(512, 512),
                                       mode="bilinear", align_corners=False)[0]  # (M, 512, 512)
            soft_np    = soft_512.cpu().numpy()

            # Existence predictions
            exist_probs = out["exist_probs"][0].cpu().numpy()   # (M,)
            pred_exist  = (exist_probs > 0.5).astype(bool)
            exist_correct.append((pred_exist == active).mean())

            # Final HCR ctrl pts: pick best (highest conf) per class
            final_ctrl = out["final_ctrl_pts"][0].cpu().numpy()    # (M, K_top, 6, 2)
            conf_logits = out["all_conf_logits"][-1][0].cpu().numpy()  # (M, K_top)
            best_k     = conf_logits.argmax(axis=1)  # (M,)
            best_ctrl  = np.array([
                final_ctrl[m, best_k[m]] for m in range(3)
            ])  # (M, 6, 2)

            # Score map: ACPI
            score_map = torch.sigmoid(out["score_logits"][0]).cpu().numpy()  # (M, Hf, Wf)

            # ── Per-class metrics ─────────────────────────────────────────────
            sample_dices = []
            sample_ious  = []
            sample_hd95s = []

            for m in range(3):
                if not active[m]:
                    continue
                pred_bin = (soft_np[m] > args.thresh).astype(np.uint8)
                gt_bin   = (gt_masks[m] > 0.5).astype(np.uint8)

                pred_dil = dilate_mask(pred_bin, dilate_k)
                gt_dil   = dilate_mask(gt_bin,   dilate_k)

                d   = compute_dice(pred_dil, gt_dil)
                iou = compute_iou(pred_dil, gt_dil)
                hd  = compute_hausdorff95(pred_bin.astype(bool),
                                          gt_bin.astype(bool))

                all_dice.append(d)
                all_iou.append(iou)
                if np.isfinite(hd):
                    all_hd95.append(hd)
                per_class_dice[m].append(d)
                sample_dices.append(d)
                sample_ious.append(iou)
                sample_hd95s.append(hd)

            mean_dice = np.mean(sample_dices) if sample_dices else 0.0

            # ── Visualization ─────────────────────────────────────────────────
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            img_bgr   = cv2.imread(img_path)
            if img_bgr is None:
                img_bgr = np.zeros((512, 512, 3), dtype=np.uint8)
            if img_bgr.shape[:2] != (512, 512):
                img_bgr = cv2.resize(img_bgr, (512, 512))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # ── Panel 1: RGB + GT annotations (cyan) ─────────────────────────
            p1 = img_rgb.copy()
            for m in range(3):
                if active[m]:
                    gt_bin = (gt_masks[m] > 0.5).astype(np.uint8) * 255
                    colored = np.zeros_like(p1)
                    color_rgb = CLASS_COLORS[m][::-1]  # BGR→RGB
                    colored[:, :] = color_rgb
                    mask3 = np.stack([gt_bin, gt_bin, gt_bin], axis=-1) > 0
                    p1 = np.where(mask3, (0.5 * p1 + 0.5 * colored).astype(np.uint8), p1)
            cv2.putText(p1, "Panel 1: GT Annotations", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            for m in range(3):
                if active[m]:
                    color_rgb = tuple(int(c) for c in CLASS_COLORS[m][::-1])
                    cv2.putText(p1, f"{CLASS_NAMES[m]} (GT)",
                                (10, 50 + m * 18), cv2.FONT_HERSHEY_SIMPLEX,
                                0.45, color_rgb, 1, cv2.LINE_AA)

            # ── Panel 2: Predicted Bézier curves ─────────────────────────────
            p2 = img_rgb.copy()
            for m in range(3):
                if pred_exist[m] or exist_probs[m] > 0.3:
                    color_bgr = CLASS_COLORS[m]
                    color_rgb = tuple(int(c) for c in color_bgr[::-1])
                    p2_bgr = cv2.cvtColor(p2, cv2.COLOR_RGB2BGR)
                    draw_bezier_on_canvas(p2_bgr, best_ctrl[m], color_bgr,
                                         thickness=2, draw_ctrl=True)
                    p2 = cv2.cvtColor(p2_bgr, cv2.COLOR_BGR2RGB)
            cv2.putText(p2, "Panel 2: Predicted Curves", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            for m in range(3):
                label = (f"{CLASS_NAMES[m]}: p={exist_probs[m]:.2f} "
                         f"Dice={per_class_dice[m][-1]*100:.1f}%"
                         if (active[m] and per_class_dice[m]) else
                         f"{CLASS_NAMES[m]}: p={exist_probs[m]:.2f}")
                color_rgb = tuple(int(c) for c in CLASS_COLORS[m][::-1])
                cv2.putText(p2, label, (10, 50 + m * 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_rgb, 1, cv2.LINE_AA)

            # ── Panel 3: Soft mask overlay ────────────────────────────────────
            p3 = img_rgb.copy()
            for m in range(3):
                # GT: tinted overlay; Pred: different tint
                soft_vis = (soft_np[m] * 255).clip(0, 255).astype(np.uint8)
                pred_bin = (soft_vis > int(args.thresh * 255)).astype(np.uint8) * 255

                # GT mask: cyan-ish overlay
                if active[m]:
                    gt_vis = (gt_masks[m] > 0.5).astype(np.uint8) * 200
                    gt_overlay = np.zeros_like(p3)
                    color_rgb = tuple(int(c) for c in CLASS_COLORS[m][::-1])
                    gt_overlay[:, :] = color_rgb
                    mask3 = np.stack([gt_vis, gt_vis, gt_vis], axis=-1) > 0
                    p3 = np.where(mask3,
                                  (0.6 * p3 + 0.4 * gt_overlay).astype(np.uint8), p3)

                # Pred mask: red overlay
                pred_overlay = np.zeros_like(p3)
                pred_overlay[:, :] = (255, 80, 80)  # red-ish
                pmask3 = np.stack([pred_bin, pred_bin, pred_bin], axis=-1) > 0
                p3 = np.where(pmask3,
                              (0.5 * p3 + 0.5 * pred_overlay).astype(np.uint8), p3)

            cv2.putText(p3, "Panel 3: Soft Masks (color=GT, red=Pred)", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(p3, f"Mean Dice={mean_dice*100:.1f}% (30px dil)", (10, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 220, 80), 1, cv2.LINE_AA)

            # ── Panel 4: ACPI score map ───────────────────────────────────────
            Hf, Wf = score_map.shape[-2], score_map.shape[-1]
            score_vis = np.zeros((512, 512, 3), dtype=np.uint8)
            # Draw a grey base image (downsampled)
            score_base = cv2.resize(img_bgr, (512, 512))
            score_base = (score_base * 0.3).astype(np.uint8)
            score_vis = score_base.copy()

            for m in range(3):
                heatmap = (score_map[m] * 255).astype(np.uint8)
                heatmap_up = cv2.resize(heatmap, (512, 512),
                                        interpolation=cv2.INTER_NEAREST)
                color_bgr = CLASS_COLORS[m]
                colored_heat = np.zeros((512, 512, 3), dtype=np.uint8)
                colored_heat[:, :] = color_bgr
                alpha_map = (heatmap_up.astype(float) / 255.0) * 0.8
                for c_idx in range(3):
                    score_vis[:, :, c_idx] = np.clip(
                        score_vis[:, :, c_idx].astype(float)
                        + alpha_map * colored_heat[:, :, c_idx],
                        0, 255
                    ).astype(np.uint8)

            # Draw grid
            cell_h, cell_w = 512 // Hf, 512 // Wf
            for r in range(Hf + 1):
                cv2.line(score_vis, (0, r * cell_h), (512, r * cell_h),
                         (60, 60, 60), 1)
            for c in range(Wf + 1):
                cv2.line(score_vis, (c * cell_w, 0), (c * cell_w, 512),
                         (60, 60, 60), 1)

            p4 = cv2.cvtColor(score_vis, cv2.COLOR_BGR2RGB)
            cv2.putText(p4, f"Panel 4: ACPI Score Map ({Hf}x{Wf} grid)", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # ── Assemble 4-panel grid ─────────────────────────────────────────
            row1 = np.hstack([p1, p2])
            row2 = np.hstack([p3, p4])
            grid = np.vstack([row1, row2])

            # Header bar
            header = np.zeros((40, grid.shape[1], 3), dtype=np.uint8)
            dice_str = f"{mean_dice*100:.1f}%" if sample_dices else "N/A"
            cv2.putText(header,
                        f"EXP_11 | {base_name} | Dice={dice_str} (30px dil) | "
                        f"Ep{ckpt.get('epoch', '?')}",
                        (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (200, 255, 150), 1, cv2.LINE_AA)
            output = np.vstack([header, grid])

            out_path = os.path.join(vis_dir, f"{base_name}_exp11.png")
            cv2.imwrite(out_path, cv2.cvtColor(output, cv2.COLOR_RGB2BGR))

            if (idx + 1) % 10 == 0 or idx == 0:
                print(f"  [{idx+1:>3}/{n_samples}] {base_name}  "
                      f"Dice={mean_dice*100:.1f}%  "
                      f"Exist: {pred_exist.tolist()}  → saved")

    # ── Final metrics ─────────────────────────────────────────────────────────
    overall_dice  = float(np.mean(all_dice))  if all_dice  else 0.0
    overall_iou   = float(np.mean(all_iou))   if all_iou   else 0.0
    overall_hd95  = float(np.mean(all_hd95))  if all_hd95  else float("nan")
    overall_exist = float(np.mean(exist_correct))

    print(f"\n{'='*65}")
    print(f"  EXP_11 Benchmark Results (Val Set, {n_samples} frames)")
    print(f"  Dilation: {dilate_k}px  |  Threshold: {args.thresh}")
    print(f"{'='*65}")
    print(f"  Overall Pixel Dice (30px dil):  {overall_dice*100:.2f}%")
    print(f"  Overall Pixel IoU  (30px dil):  {overall_iou*100:.2f}%")
    print(f"  Overall HD95 (px):              {overall_hd95:.2f}")
    print(f"  Existence Accuracy:             {overall_exist*100:.2f}%")
    print(f"  {'─'*50}")
    for m in range(3):
        if per_class_dice[m]:
            print(f"  {CLASS_NAMES[m]:12s} Dice: {np.mean(per_class_dice[m])*100:.2f}%  "
                  f"(n={len(per_class_dice[m])})")
    print(f"{'='*65}")
    print(f"\n  BCRNet baseline:  69.57% Dice | 54.16% IoU | 43.55px HD95")
    gap = 69.57 - overall_dice * 100
    print(f"  Gap to SOTA:      {gap:+.2f}pp")
    print(f"\n  Visualizations → {vis_dir}")

    results = {
        "checkpoint": args.checkpoint,
        "epoch": int(ckpt.get("epoch", 0)),
        "overall_dice_30px": overall_dice,
        "overall_iou_30px":  overall_iou,
        "overall_hd95_px":   overall_hd95 if np.isfinite(overall_hd95) else None,
        "existence_accuracy": overall_exist,
        "per_class": {
            CLASS_NAMES[m]: float(np.mean(per_class_dice[m])) if per_class_dice[m] else 0.0
            for m in range(3)
        },
        "baseline_bcrnet_dice": 0.6957,
        "gap_to_sota_pp": gap,
        "dilate_px": dilate_k,
        "threshold": args.thresh,
    }
    metrics_path = os.path.join(args.output_dir, "metrics_exp11.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Metrics JSON  → {metrics_path}")


if __name__ == "__main__":
    main()
