"""
EXP_12: SurgicalCurveFormer v2 Evaluation & Visualizer
======================================================
Evaluates trained checkpoints and produces 4-panel clinical visualizations:
1. Input Laparoscopic RGB-D View
2. Ground Truth Landmark Annotations
3. Multi-Level CNN Segmentation Probability Map
4. Refined Bézier Parametric Splines (with ON-Snake Orthogonal Normals)
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

exp_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
workspace_root = os.path.abspath(os.path.join(exp_root, "../.."))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)
if exp_root not in sys.path:
    sys.path.insert(0, exp_root)

from configs.exp12_config import EXP12Config, resolve_dataset_dir
from models.surgical_curve_former_v2 import SurgicalCurveFormerV2
from utils.dataset import SurgicalCurveFormerDataset




def parse_args():
    cfg = EXP12Config()
    p = argparse.ArgumentParser(description="EXP_12: Evaluate SurgicalCurveFormer v2")
    p.add_argument("--dataset_dir",     type=str, default=resolve_dataset_dir())
    p.add_argument("--checkpoint",      type=str, default=os.path.join(cfg.save_dir, "best_model.pth"))
    p.add_argument("--output_dir",      type=str, default=os.path.join(exp_root, "outputs/eval_plots"))
    p.add_argument("--device",          type=str, default="")
    p.add_argument("--batch_size",      type=int, default=1)
    p.add_argument("--max_plots",       type=int, default=25)
    p.add_argument("--target_patient",  type=str, default="", help="Filter specific patient or sample (e.g., '40')")
    p.add_argument("--all_plots",       action="store_true", help="Plot all validation samples")
    return p.parse_args()


def get_device(forced: str = "") -> torch.device:
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def dice_metric(pred_mask: np.ndarray, gt_mask: np.ndarray, eps: float = 1e-6) -> float:
    inter = (pred_mask & gt_mask).sum()
    return float((2.0 * inter + eps) / (pred_mask.sum() + gt_mask.sum() + eps))


def evaluate_and_visualize():
    args = parse_args()
    device = get_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("🔬 [EXP_12] Evaluating SurgicalCurveFormer v2")
    print(f"   Checkpoint:     {args.checkpoint}")
    print(f"   Dataset:        {args.dataset_dir}")
    print(f"   Outputs:        {args.output_dir}")
    if args.target_patient:
        print(f"   Target Patient: {args.target_patient}")
    print("=" * 80)

    val_ds = SurgicalCurveFormerDataset(
        dataset_dir=args.dataset_dir, mode="val",
        image_size=512, use_depth=True,
        acpi_stride=32, render_size=128, bezier_degree=5,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = SurgicalCurveFormerV2(
        num_classes=3, top_k=10, bezier_order=5, fpn_dim=256, grid_size=128, image_size=512
    ).to(device)

    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        state = ckpt.get("model_state", ckpt)
        model.load_state_dict(state, strict=False)
        print(f"✅ Loaded checkpoint from {args.checkpoint}")
    else:
        print(f"⚠️ Checkpoint {args.checkpoint} not found. Running with initialized weights.")

    model.eval()

    class_names = ["Ridge", "Silhouette", "Ligament"]
    class_colors = [(0, 255, 0), (255, 128, 0), (255, 0, 128)]
    dilate_px = 30
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))

    per_class_dice_cnn = {0: [], 1: [], 2: []}
    per_class_dice_ras = {0: [], 1: [], 2: []}
    ghost_lines = {0: 0, 1: 0, 2: 0}
    plots_saved = 0
    target_plots_saved = []

    with torch.no_grad():
        for b_idx, batch in enumerate(val_loader):
            x = batch["image"].to(device)
            out = model(x)

            B = x.shape[0]
            seg_list = out.get("seg_logits_list", [])
            soft_ras = out["raster_masks"].cpu().float()
            exist_p = out["exist_probs"].cpu()
            final_curves = out["final_curves"].cpu()  # (B, M, K, 6, 2)
            gt_masks = batch["target_masks"]
            active = batch["active_mask"]
            img_paths = batch.get("img_path", [f"sample_{b_idx}"] * B)

            if seg_list:
                cnn_512 = torch.sigmoid(seg_list[0].detach().cpu())
            else:
                cnn_512 = F.interpolate(soft_ras, size=(512, 512), mode="bilinear", align_corners=False)

            raster_512 = F.interpolate(soft_ras, size=(512, 512), mode="bilinear", align_corners=False)

            for b in range(B):
                img_path = img_paths[b] if isinstance(img_paths, (list, tuple)) else str(img_paths)
                img_name = os.path.splitext(os.path.basename(img_path))[0]

                # Check if this sample matches target patient
                is_target = False
                if args.target_patient:
                    target_str = str(args.target_patient).lower().strip()
                    is_target = (
                        target_str in img_name.lower() or
                        target_str in img_path.lower() or
                        (target_str.isdigit() and b_idx == int(target_str))
                    )

                sample_dice_cnn = {}
                sample_dice_ras = {}

                for m in range(3):
                    is_active = active[b, m].item()
                    p_exist = exist_p[b, m].item()

                    if is_active:
                        gt_np = (gt_masks[b, m].numpy() > 0.5).astype(np.uint8)
                        gt_dil = cv2.dilate(gt_np, kernel)

                        # CNN Dice
                        pred_c = (cnn_512[b, m].numpy() > 0.3).astype(np.uint8)
                        pred_c_dil = cv2.dilate(pred_c, kernel)
                        d_c = dice_metric(pred_c_dil > 0, gt_dil > 0)
                        per_class_dice_cnn[m].append(d_c)
                        sample_dice_cnn[m] = d_c

                        # Rasterizer Dice
                        pred_r = (raster_512[b, m].numpy() > 0.3).astype(np.uint8)
                        pred_r_dil = cv2.dilate(pred_r, kernel)
                        d_r = dice_metric(pred_r_dil > 0, gt_dil > 0)
                        per_class_dice_ras[m].append(d_r)
                        sample_dice_ras[m] = d_r
                    else:
                        # Check for false positive ghost lines
                        if p_exist > 0.5:
                            ghost_lines[m] += 1

                # Visual Plotting Condition
                should_plot = args.all_plots or is_target or (plots_saved < args.max_plots)
                if should_plot:
                    # Fix: Load original image directly or properly un-normalize from ImageNet mean/std
                    rgb_img = None
                    if os.path.exists(img_path):
                        img_bgr = cv2.imread(img_path)
                        if img_bgr is not None:
                            if img_bgr.shape[:2] != (512, 512):
                                img_bgr = cv2.resize(img_bgr, (512, 512), interpolation=cv2.INTER_LINEAR)
                            rgb_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                    if rgb_img is None:
                        rgb_norm = x[b, :3].cpu().numpy().transpose(1, 2, 0)
                        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                        rgb_denorm = np.clip((rgb_norm * std + mean) * 255.0, 0, 255).astype(np.uint8)
                        rgb_img = rgb_denorm

                    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))


                    # 1. RGB
                    axes[0].imshow(rgb_img)
                    axes[0].set_title(f"1. RGB: {img_name} [Idx {b_idx}]", fontsize=11, fontweight="bold")
                    axes[0].axis("off")

                    # 2. GT Overlays
                    gt_overlay = rgb_img.copy()
                    for m in range(3):
                        if active[b, m].item():
                            gt_m = (gt_masks[b, m].numpy() > 0.5).astype(np.uint8)
                            contours, _ = cv2.findContours(gt_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            cv2.drawContours(gt_overlay, contours, -1, class_colors[m], 2)
                    axes[1].imshow(gt_overlay)
                    axes[1].set_title("2. Ground Truth Landmarks", fontsize=11, fontweight="bold")
                    axes[1].axis("off")

                    # 3. CNN Multi-Level Probability
                    cnn_overlay = rgb_img.copy()
                    c_info = []
                    for m in range(3):
                        prob = cnn_512[b, m].numpy()
                        mask_c = (prob > 0.3).astype(np.uint8)
                        contours, _ = cv2.findContours(mask_c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(cnn_overlay, contours, -1, class_colors[m], 2)
                        if m in sample_dice_cnn:
                            c_info.append(f"{class_names[m][:3]}:{sample_dice_cnn[m]*100:.0f}%")
                    c_str = ", ".join(c_info) if c_info else "None"
                    axes[2].imshow(cnn_overlay)
                    axes[2].set_title(f"3. CNN Decoder ({c_str})", fontsize=11, fontweight="bold")
                    axes[2].axis("off")

                    # 4. Refined Bézier Splines
                    spline_overlay = rgb_img.copy()
                    M_bern = model.hcr.M_bern.cpu().numpy()[:25]
                    s_info = []
                    for m in range(3):
                        if exist_p[b, m].item() > 0.5:
                            best_k = int(torch.argmax(out["final_scores"][b, m]).item()) if "final_scores" in out else 0
                            ctrl = final_curves[b, m, best_k].numpy()  # (6, 2)
                            pts = np.einsum("ng, gy -> ny", M_bern, ctrl) * 512.0

                            pts_int = pts.astype(np.int32).reshape((-1, 1, 2))
                            cv2.polylines(spline_overlay, [pts_int], False, class_colors[m], 3)
                            # Draw control polygon in white
                            ctrl_int = (ctrl * 512.0).astype(np.int32)
                            for cp in ctrl_int:
                                cv2.circle(spline_overlay, tuple(cp), 3, (255, 255, 255), -1)
                            if m in sample_dice_ras:
                                s_info.append(f"{class_names[m][:3]}:{sample_dice_ras[m]*100:.0f}%")
                    s_str = ", ".join(s_info) if s_info else "None"
                    axes[3].imshow(spline_overlay)
                    axes[3].set_title(f"4. v2 Splines ({s_str})", fontsize=11, fontweight="bold")
                    axes[3].axis("off")

                    plt.tight_layout()
                    plot_file = os.path.join(args.output_dir, f"sample_{b_idx:03d}_{img_name}.png")
                    plt.savefig(plot_file, dpi=150, bbox_inches="tight")

                    if is_target:
                        target_file = os.path.join(args.output_dir, f"TARGET_{args.target_patient}_{img_name}.png")
                        plt.savefig(target_file, dpi=150, bbox_inches="tight")
                        target_plots_saved.append(target_file)
                        print(f"  ⭐ Saved TARGET Patient Plot: {target_file}")

                    plt.close(fig)
                    plots_saved += 1


    print("\n" + "=" * 80)
    print("📊 EVALUATION REPORT — SURGICALCURVEFORMER V2")
    print("=" * 80)
    for m in range(3):
        m_dice_c = np.mean(per_class_dice_cnn[m]) * 100.0 if per_class_dice_cnn[m] else 0.0
        m_dice_r = np.mean(per_class_dice_ras[m]) * 100.0 if per_class_dice_ras[m] else 0.0
        print(f"  • {class_names[m]:<12}: CNN Dice = {m_dice_c:6.2f}% | Cauchy Raster Dice = {m_dice_r:6.2f}% | Ghost Lines = {ghost_lines[m]}")

    all_c = [d for m in range(3) for d in per_class_dice_cnn[m]]
    all_r = [d for m in range(3) for d in per_class_dice_ras[m]]
    macro_c = np.mean(all_c) * 100.0 if all_c else 0.0
    macro_r = np.mean(all_r) * 100.0 if all_r else 0.0
    print("-" * 80)
    print(f"  ⭐ MACRO MEAN CNN DICE:            {macro_c:.2f}%")
    print(f"  ⭐ MACRO MEAN CAUCHY RASTER DICE:  {macro_r:.2f}%")
    print(f"  ⭐ TOTAL GHOST DETECTIONS:         {sum(ghost_lines.values())}")
    if target_plots_saved:
        print("-" * 80)
        print(f"  🎯 TARGET PATIENT ({args.target_patient}) PLOTS SAVED:")
        for p_path in target_plots_saved:
            print(f"     • {p_path}")
    print("=" * 80)



if __name__ == "__main__":
    evaluate_and_visualize()
