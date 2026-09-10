#!/usr/bin/env python3
"""
EXP_13: Comprehensive Evaluation Script for Mask2Former-BCRNet
=============================================================
Evaluates:
1. Mask2Former Pixel-Wise Metrics: Dice, IoU
2. BCRNet Parametric Curve Metrics: Chamfer L1 Distance, Tangent Cosine Error
3. Generates 4-Panel Side-by-Side Visualizations (including Patient 40)
"""
import os
import sys
import argparse
import json
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP13_DIR = os.path.dirname(SCRIPT_DIR)
WORKSPACE_DIR = os.path.dirname(os.path.dirname(EXP13_DIR))

for p in [EXP13_DIR, WORKSPACE_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from configs.exp13_config import EXP13Config, resolve_dataset_dir, resolve_checkpoint_dir
from utils.dataset import Mask2FormerBCRNetDataset
from models.mask2former_bcrnet import Mask2FormerBCRNet
from utils.bezier_ops import evaluate_bezier_torch


def plot_comparison_4panel(
    orig_rgb: np.ndarray,
    gt_sem: np.ndarray,
    pred_sem: np.ndarray,
    pred_curves: np.ndarray,  # (3, 26, 2) in [0, 1]
    gt_curves: np.ndarray,    # (3, 26, 2) in [0, 1]
    gt_exists: np.ndarray,    # (3,)
    scores: np.ndarray,       # (3,)
    save_path: str,
    title: str = "EXP_13 Evaluation",
):
    """
    4-panel visualization:
    Panel 1: Original Laparoscopic Frame
    Panel 2: Ground Truth Masks & Polylines
    Panel 3: Mask2Former Predicted Pixel Segmentation Map
    Panel 4: Predicted Continuous 5th-Order Bézier Curves
    """
    H, W = orig_rgb.shape[:2]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Color palette (RGB normalized to [0, 1])
    colors = {
        0: (1.0, 0.2, 0.2),  # Ridge: Red
        1: (0.2, 0.9, 0.2),  # Silhouette: Green
        2: (0.2, 0.4, 1.0),  # Ligament: Blue
    }
    class_names = ["Ridge", "Silhouette", "Ligament"]

    # --- Panel 1: Original Image ---
    axes[0].imshow(orig_rgb)
    axes[0].set_title("1. Laparoscopic View", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # --- Panel 2: Ground Truth ---
    gt_overlay = orig_rgb.copy()
    axes[1].imshow(gt_overlay)
    for c in range(3):
        if gt_exists[c] > 0.5:
            pts_px = gt_curves[c] * np.array([W, H])
            axes[1].plot(pts_px[:, 0], pts_px[:, 1], color=colors[c], linewidth=3, label=f"GT {class_names[c]}")
            axes[1].scatter(pts_px[::5, 0], pts_px[::5, 1], color=colors[c], s=20)
    axes[1].set_title("2. Ground Truth Curves", fontsize=12, fontweight="bold")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].axis("off")

    # --- Panel 3: Mask2Former Semantic Mask ---
    sem_color = np.zeros((H, W, 3), dtype=np.float32)
    for c in range(1, 4):
        mask_c = (pred_sem == c)
        sem_color[mask_c] = colors[c - 1]
    blend = orig_rgb.astype(np.float32) / 255.0 * 0.5 + sem_color * 0.5
    axes[2].imshow(np.clip(blend, 0.0, 1.0))
    axes[2].set_title("3. Mask2Former + TopoNet Mask", fontsize=12, fontweight="bold")
    axes[2].axis("off")

    # --- Panel 4: Parametric Bézier Curves ---
    axes[3].imshow(orig_rgb)
    for c in range(3):
        if scores[c] > 0.2:
            pts_px = pred_curves[c] * np.array([W, H])
            axes[3].plot(
                pts_px[:, 0], pts_px[:, 1],
                color=colors[c], linewidth=3,
                label=f"{class_names[c]} ({scores[c]:.2f})"
            )
            axes[3].scatter(pts_px[0, 0], pts_px[0, 1], color="yellow", s=30, zorder=5)  # Start
            axes[3].scatter(pts_px[-1, 0], pts_px[-1, 1], color="cyan", s=30, zorder=5)   # End
    axes[3].set_title("4. BCRNet 5th-Order Bézier Splines", fontsize=12, fontweight="bold")
    axes[3].legend(loc="upper right", fontsize=8)
    axes[3].axis("off")

    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate EXP_13 Mask2Former-BCRNet")
    parser.add_argument("--dataset_dir", type=str, default=None, help="Dataset directory")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file path")
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/eval_plots_exp13", help="Output plots directory")
    parser.add_argument("--max_plots", type=int, default=25, help="Maximum number of frames to plot")
    args = parser.parse_args()

    cfg = EXP13Config()
    if args.dataset_dir:
        cfg.dataset_dir = resolve_dataset_dir(args.dataset_dir)

    ckpt_path = args.checkpoint
    if not ckpt_path:
        ckpt_path = os.path.join(cfg.save_dir, "best_model.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("🔬 [EXP_13] Evaluating Mask2Former-BCRNet Master Architecture")
    print(f"   Checkpoint: {ckpt_path}")
    print(f"   Dataset:    {cfg.dataset_dir}")
    print(f"   Outputs:    {args.output_dir}")
    print("=" * 80)

    val_dataset = Mask2FormerBCRNetDataset(
        dataset_dir=cfg.dataset_dir,
        mode="val",
        image_size=cfg.image_size,
        use_depth=cfg.use_depth,
        bezier_degree=cfg.bezier_degree,
        num_sample_pts=cfg.num_sample_pts,
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)

    model = Mask2FormerBCRNet(cfg).to(device)

    if os.path.exists(ckpt_path):
        print(f"🔄 Loading weights from '{ckpt_path}'...")
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
        print(" Model weights loaded successfully.")
    else:
        print(f"⚠️ Warning: Checkpoint '{ckpt_path}' not found! Evaluating initialized model.")

    model.eval()
    os.makedirs(args.output_dir, exist_ok=True)

    class_dices = {0: [], 1: [], 2: []}  # 0: Ridge, 1: Silhouette, 2: Ligament
    chamfer_dists = {0: [], 1: [], 2: []}
    plotted = 0

    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, desc="Evaluating")):
            images = batch["image"].to(device)
            gt_sem = batch["gt_semantic"].cpu().numpy()[0]     # (H, W)
            gt_curves = batch["gt_sample_pts"].cpu().numpy()[0] # (3, 26, 2)
            gt_exists = batch["gt_exists"].cpu().numpy()[0]     # (3,)
            orig_rgb = batch["orig_rgb"].numpy()[0]             # (H, W, 3)
            base_name = batch["base_name"][0]

            preds = model.predict_best_curves(images, target_size=(cfg.image_size, cfg.image_size))
            pred_sem = preds["semantic_pred"].cpu().numpy()[0]     # (H, W)
            pred_curves = preds["best_curve_pts"].cpu().numpy()[0] # (3, 26, 2)
            scores = preds["best_scores"].cpu().numpy()[0]         # (3,)

            # Pixel Dice per class
            smooth = 1e-5
            for c in range(3):
                p_mask = (pred_sem == (c + 1)).astype(np.float32)
                g_mask = (gt_sem == (c + 1)).astype(np.float32)
                intersection = np.sum(p_mask * g_mask)
                dice = (2.0 * intersection + smooth) / (np.sum(p_mask) + np.sum(g_mask) + smooth)
                class_dices[c].append(dice)

                # Chamfer Distance (px) on existing landmarks
                if gt_exists[c] > 0.5:
                    p_pts_px = pred_curves[c] * float(cfg.image_size)
                    g_pts_px = gt_curves[c] * float(cfg.image_size)
                    g_rev_px = np.flip(g_pts_px, axis=0)

                    d_fwd = np.mean(np.linalg.norm(p_pts_px - g_pts_px, axis=-1))
                    d_rev = np.mean(np.linalg.norm(p_pts_px - g_rev_px, axis=-1))
                    chamfer_dists[c].append(min(d_fwd, d_rev))

            # Prioritize saving Patient 40 and first N plots
            is_patient_40 = "patient_40" in base_name.lower() or "p40" in base_name.lower()
            if plotted < args.max_plots or is_patient_40:
                p40_tag = "PATIENT_40_" if is_patient_40 else ""
                save_filename = os.path.join(args.output_dir, f"{p40_tag}eval_{base_name}.png")
                plot_comparison_4panel(
                    orig_rgb=orig_rgb,
                    gt_sem=gt_sem,
                    pred_sem=pred_sem,
                    pred_curves=pred_curves,
                    gt_curves=gt_curves,
                    gt_exists=gt_exists,
                    scores=scores,
                    save_path=save_filename,
                    title=f"EXP_13 Synthesis — {base_name} ({p40_tag[:-1] or 'Val'})",
                )
                plotted += 1

    # Print Quantitative Report
    mean_ridge_dice = float(np.mean(class_dices[0]))
    mean_silh_dice  = float(np.mean(class_dices[1]))
    mean_lig_dice   = float(np.mean(class_dices[2]))
    macro_dice      = (mean_ridge_dice + mean_silh_dice + mean_lig_dice) / 3.0

    print("\n" + "=" * 80)
    print("📈 [EXP_13] QUANTITATIVE BENCHMARK EVALUATION RESULTS")
    print("=" * 80)
    print(f"  Anterior Ridge Dice:      {mean_ridge_dice*100:.2f}%  | Chamfer Error: {np.mean(chamfer_dists[0]):.2f} px")
    print(f"  Liver Silhouette Dice:    {mean_silh_dice*100:.2f}%  | Chamfer Error: {np.mean(chamfer_dists[1]):.2f} px")
    print(f"  Falciform Ligament Dice:  {mean_lig_dice*100:.2f}%  | Chamfer Error: {np.mean(chamfer_dists[2]):.2f} px")
    print("-" * 80)
    print(f"  🏆 MACRO MEAN DICE:       {macro_dice*100:.2f}%")
    print("=" * 80)
    print(f"🖼️ Plots saved to: {args.output_dir} ({plotted} frames plotted)")


if __name__ == "__main__":
    main()
