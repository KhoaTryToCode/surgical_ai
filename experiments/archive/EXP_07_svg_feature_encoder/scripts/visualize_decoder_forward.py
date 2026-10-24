"""
EXP_07 — Decoder Layer-by-Layer Forward Pass Visualization.

Visualizes:
  Row 0: Decoder Input  (Swin Geometric Features PCA, Saliency, Tangent HSV, Combined)
  Row 1–6: Each decoder layer showing:
    - Left:  Feature map with probe dots (where each query is sampling)
    - Right: Predicted Bézier curves overlaid on the original image

No loss, no training — pure forward pass with random weights.
"""

import argparse
import os
import sys
import glob
import random

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn.functional as F

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
exp_dir = os.path.dirname(current_dir)
workspace_dir = os.path.dirname(os.path.dirname(exp_dir))
if workspace_dir not in sys.path:
    sys.path.insert(0, workspace_dir)
if exp_dir not in sys.path:
    sys.path.insert(0, exp_dir)

from configs.config_svg_encoder import config
from models.svg_vector_encoder import SVGVectorAwareFeatureEncoder
from models.bezier_decoder import BezierSplineDecoder
from utils.svg_visualizer_utils import (
    feature_pca_rgb,
    vector_field_to_hsv,
    create_synthetic_surgical_frame,
)


QUERY_COLORS = [
    "#ff3366",  # Hot Pink
    "#00ffcc",  # Cyan-Green
    "#ffaa00",  # Amber
    "#7c4dff",  # Purple
    "#00e5ff",  # Electric Cyan
    "#ff6d00",  # Orange
    "#76ff03",  # Lime
    "#ff1744",  # Red
    "#448aff",  # Blue
    "#eeff41",  # Yellow-Green
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="EXP_07: Decoder Layer-by-Layer Forward Pass Visualization"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/kaggle/working/L3D",
        help="Path to surgical dataset directory",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Direct path to input surgical image (optional)",
    )
    parser.add_argument(
        "--num_queries",
        type=int,
        default=5,
        help="Number of Bézier query curves",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=6,
        help="Number of decoder layers",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=os.path.join(config.output_dir, "decoder_layer_progression.png"),
        help="Path to save visualization PNG",
    )
    return parser.parse_args()


def load_image(args):
    """Load or synthesize a surgical image, returns (H, W, 3) float32 in [0, 1]."""
    img_np = None

    # Priority A: Explicit path
    if args.image_path and os.path.exists(args.image_path):
        img_bgr = cv2.imread(args.image_path)
        if img_bgr is not None:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            if img_rgb.shape[:2] != (1024, 1024):
                img_rgb = cv2.resize(img_rgb, (1024, 1024))
            img_np = img_rgb.astype(np.float32) / 255.0
            print(f"🖼️ Loaded surgical image from: {args.image_path}")

    # Priority B: Random sample from dataset
    if img_np is None and os.path.exists(args.dataset_dir):
        candidates = sorted(
            glob.glob(os.path.join(args.dataset_dir, "**", "*.png"), recursive=True)
            + glob.glob(os.path.join(args.dataset_dir, "**", "*.jpg"), recursive=True)
        )
        if candidates:
            sel = random.choice(candidates)
            img_bgr = cv2.imread(sel)
            if img_bgr is not None:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                if img_rgb.shape[:2] != (1024, 1024):
                    img_rgb = cv2.resize(img_rgb, (1024, 1024))
                img_np = img_rgb.astype(np.float32) / 255.0
                print(f"🖼️ Randomly sampled image: {sel}")

    # Priority C: Synthetic
    if img_np is None:
        img_np = create_synthetic_surgical_frame(1024, 1024)
        print("🖼️ Generated synthetic surgical frame.")

    return img_np


def draw_bezier_on_ax(ax, control_points_np, img_np, title, query_colors, num_sample=50):
    """
    Draw Bézier curves for all queries on an axis, overlaid on the surgical image.

    control_points_np: (Q, 4, 2) in [0, 1]
    """
    ax.imshow(img_np, alpha=0.7)
    Q = control_points_np.shape[0]
    t = np.linspace(0.0, 1.0, num_sample).reshape(-1, 1)

    for q in range(Q):
        p0 = control_points_np[q, 0] * 1024.0
        c1 = control_points_np[q, 1] * 1024.0
        c2 = control_points_np[q, 2] * 1024.0
        p3 = control_points_np[q, 3] * 1024.0
        col = query_colors[q % len(query_colors)]

        # Evaluate curve
        curve = (
            ((1.0 - t) ** 3) * p0
            + 3.0 * ((1.0 - t) ** 2) * t * c1
            + 3.0 * (1.0 - t) * (t ** 2) * c2
            + (t ** 3) * p3
        )

        # Draw curve
        ax.plot(curve[:, 0], curve[:, 1], color=col, linewidth=2.5, alpha=0.9)

        # Draw control points
        ax.scatter([p0[0], p3[0]], [p0[1], p3[1]], color="white", s=50, zorder=5, edgecolors=col, linewidths=1.5)
        ax.scatter([c1[0], c2[0]], [c1[1], c2[1]], color=col, s=30, marker="s", zorder=5, alpha=0.8)

        # Draw tangent handles
        ax.plot([p0[0], c1[0]], [p0[1], c1[1]], color=col, linestyle=":", alpha=0.5, linewidth=1.2)
        ax.plot([p3[0], c2[0]], [p3[1], c2[1]], color=col, linestyle=":", alpha=0.5, linewidth=1.2)

    ax.set_title(title, color="white", fontsize=10, fontweight="bold")
    ax.axis("off")


def draw_probe_dots_on_ax(ax, sample_xy_np, saliency_np, img_np, title, query_colors):
    """
    Draw probe sample dots on the saliency map + image overlay.

    sample_xy_np: (Q, T, 2) in [0, 1]
    saliency_np: (H, W) saliency map.
    """
    H_s, W_s = saliency_np.shape

    # Blend saliency as a heat overlay on the surgical image
    saliency_rgb = plt.cm.magma(saliency_np)[:, :, :3]  # (H, W, 3)
    saliency_up = cv2.resize(saliency_rgb.astype(np.float32), (1024, 1024))
    blended = img_np * 0.4 + saliency_up * 0.6
    blended = np.clip(blended, 0.0, 1.0)

    ax.imshow(blended)

    Q, T, _ = sample_xy_np.shape
    for q in range(Q):
        col = query_colors[q % len(query_colors)]
        pts = sample_xy_np[q] * 1024.0  # (T, 2)
        ax.scatter(pts[:, 0], pts[:, 1], color=col, s=12, alpha=0.85, zorder=4, edgecolors="white", linewidths=0.3)

    ax.set_title(title, color="white", fontsize=10, fontweight="bold")
    ax.axis("off")


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Running Decoder Forward Pass Visualization on device: {device}")

    # 1. Load image
    img_np = load_image(args)

    # 2. Normalize
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    img_norm = (img_np - mean) / std
    img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float().to(device)

    # 3. Encoder forward pass
    print("⏳ [1/2] Running SVG Vector Encoder...")
    encoder = SVGVectorAwareFeatureEncoder(config).to(device)
    encoder.eval()
    with torch.no_grad():
        enc_out = encoder(img_tensor)

    # 4. Decoder forward pass (random weights — no training)
    print(f"⏳ [2/2] Running Bézier Decoder ({args.num_layers} layers, {args.num_queries} queries)...")
    decoder = BezierSplineDecoder(
        num_queries=args.num_queries,
        num_layers=args.num_layers,
        d_model=256,
        nhead=8,
        num_sample_t=20,
        num_classes=5,
    ).to(device)
    decoder.eval()
    with torch.no_grad():
        dec_out = decoder(enc_out)

    # 5. Extract data for visualization
    saliency_np = enc_out["saliency_field"][0, 0].cpu().numpy()
    tangent_np = enc_out["tangent_field"][0].cpu().numpy()
    geo_feat = enc_out["geometric_features"][0]  # (128, H, W)
    hsv_map = vector_field_to_hsv(tangent_np, saliency_np)

    layer_states = dec_out["layer_states"]
    num_layers = len(layer_states)

    # 6. Build visualization using GridSpec for mixed column layout
    from matplotlib.gridspec import GridSpec

    total_rows = num_layers + 1  # Row 0 = decoder input (4 cols), Rows 1-6 = layers (2 cols)
    fig = plt.figure(figsize=(20, 5 * total_rows), facecolor="#0d1117")
    fig.suptitle(
        "EXP_07 DECODER LAYER-BY-LAYER FORWARD PASS\n"
        f"(Random Weights — {args.num_queries} Queries × {args.num_layers} Layers × 20 Sample Points)",
        fontsize=18,
        fontweight="bold",
        color="#58a6ff",
        y=0.995,
    )

    gs = GridSpec(total_rows, 4, figure=fig, hspace=0.35, wspace=0.15)

    # ────────────────── ROW 0: DECODER INPUT (4 columns) ──────────────────
    ax0_1 = fig.add_subplot(gs[0, 0])
    ax0_1.imshow(img_np)
    ax0_1.set_title("Input Surgical Image", color="white", fontsize=11, fontweight="bold")
    ax0_1.axis("off")

    ax0_2 = fig.add_subplot(gs[0, 1])
    pca_geo = feature_pca_rgb(geo_feat)
    ax0_2.imshow(pca_geo)
    ax0_2.set_title("Geometric Features PCA\n(128ch → RGB)", color="#7ee787", fontsize=11, fontweight="bold")
    ax0_2.axis("off")

    ax0_3 = fig.add_subplot(gs[0, 2])
    ax0_3.imshow(saliency_np, cmap="magma")
    ax0_3.set_title("Saliency Field S(x,y)", color="#7ee787", fontsize=11, fontweight="bold")
    ax0_3.axis("off")

    ax0_4 = fig.add_subplot(gs[0, 3])
    ax0_4.imshow(hsv_map)
    ax0_4.set_title("Tangent Direction HSV", color="#7ee787", fontsize=11, fontweight="bold")
    ax0_4.axis("off")

    # ────────────────── ROWS 1–N: DECODER LAYERS (2 columns, each spans 2 grid cols) ──────────────────
    for layer_idx, state in enumerate(layer_states):
        cp_after = state["control_points_after"][0].cpu().numpy()  # (Q, 4, 2)
        sample_xy = state["sample_xy"][0].cpu().numpy()            # (Q, T, 2)

        row = layer_idx + 1

        # Left panel (spans grid cols 0–1): Probe dots on saliency + image blend
        ax_left = fig.add_subplot(gs[row, 0:2])
        draw_probe_dots_on_ax(
            ax_left,
            sample_xy,
            saliency_np,
            img_np,
            f"Layer {layer_idx + 1}: Probe Locations on Saliency Map\n"
            f"({args.num_queries} queries × 20 samples = {args.num_queries * 20} probe dots)",
            QUERY_COLORS,
        )

        # Right panel (spans grid cols 2–3): Bézier curves after correction
        ax_right = fig.add_subplot(gs[row, 2:4])
        draw_bezier_on_ax(
            ax_right,
            cp_after,
            img_np,
            f"Layer {layer_idx + 1}: Predicted Bézier Curves After Correction\n"
            f"(ΔP0, ΔC1, ΔC2, ΔP3 applied — {8 * args.num_queries} parameters refined)",
            QUERY_COLORS,
        )

    # Legend
    legend_patches = [
        mpatches.Patch(color=QUERY_COLORS[i], label=f"Query {i + 1}")
        for i in range(args.num_queries)
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=args.num_queries,
        fontsize=11,
        facecolor="#161b22",
        edgecolor="#30363d",
        labelcolor="white",
        framealpha=0.9,
    )

    plt.savefig(args.output_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()

    print(f"\n🎉 Decoder layer-by-layer visualization saved to:\n   👉 {args.output_path}")
    print(f"\n📊 Summary:")
    print(f"   Encoder output: geometric_features (128, 256, 256) + saliency (1, 256, 256) + tangent (2, 256, 256)")
    print(f"   Decoder input:  Combined feature map (131, 256, 256)")
    print(f"   Queries: {args.num_queries} Bézier curves × 4 control points × 2 coords = {args.num_queries * 8} params")
    print(f"   Layers: {args.num_layers} iterative refinement steps")
    print(f"   Sample points per curve: 20 (for probing)")


if __name__ == "__main__":
    main()
