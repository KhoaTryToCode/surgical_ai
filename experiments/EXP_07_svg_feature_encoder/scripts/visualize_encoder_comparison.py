import argparse
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# Ensure workspace root is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
exp_dir = os.path.dirname(current_dir)
workspace_dir = os.path.dirname(os.path.dirname(exp_dir))
if workspace_dir not in sys.path:
    sys.path.insert(0, workspace_dir)
if exp_dir not in sys.path:
    sys.path.insert(0, exp_dir)

from configs.config_svg_encoder import config
from models.swin_standard_encoder import StandardSwinEncoder
from models.svg_vector_encoder import SVGVectorAwareFeatureEncoder
from utils.svg_visualizer_utils import (
    feature_pca_rgb,
    vector_field_to_hsv,
    create_synthetic_surgical_frame
)

def parse_args():
    parser = argparse.ArgumentParser(description="EXP_07: Visual Comparison of Standard Swin vs SVG Feature Encoder")
    parser.add_argument("--image_path", type=str, default=None, help="Path to input surgical image (optional)")
    parser.add_argument("--output_path", type=str, default=os.path.join(config.output_dir, "encoder_feature_comparison.png"), help="Path to save comparison PNG")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Running EXP_07 Encoder Feature Map Comparison on device: {device}")

    # 1. Load or Generate Input Surgical Frame
    if args.image_path and os.path.exists(args.image_path):
        import cv2
        img_bgr = cv2.imread(args.image_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if img_rgb.shape[:2] != (1024, 1024):
            img_rgb = cv2.resize(img_rgb, (1024, 1024))
        img_np = img_rgb.astype(np.float32) / 255.0
        print(f"🖼️ Loaded real surgical frame from: {args.image_path}")
    else:
        img_np = create_synthetic_surgical_frame(1024, 1024)
        print("🖼️ Generated high-fidelity synthetic laparoscopic liver frame with Falciform & Ridge contours.")

    # Normalize with ImageNet stats
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    img_norm = (img_np - mean) / std
    img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float().to(device)

    # 2. Forward Pass: Standard Swin Backbone Encoder (EXP_06)
    print("⏳ [1/2] Extracting Standard Swin Backbone Features (EXP_06)...")
    swin_encoder = StandardSwinEncoder(config).to(device)
    swin_encoder.eval()
    with torch.no_grad():
        swin_out = swin_encoder(img_tensor)

    # 3. Forward Pass: SVG / Vector-Aware Feature Encoder (EXP_07)
    print("⏳ [2/2] Extracting SVG / Vector-Aware Feature Representations (EXP_07)...")
    svg_encoder = SVGVectorAwareFeatureEncoder(config).to(device)
    svg_encoder.eval()
    with torch.no_grad():
        svg_out = svg_encoder(img_tensor)

    # 4. Process Visual Maps
    # A. Standard Swin Features
    feat_s4 = swin_out["stride4_features"][0] # (256, 256, 256)
    feat_s4_energy = torch.mean(torch.abs(feat_s4), dim=0).cpu().numpy()
    feat_s4_energy = (feat_s4_energy - feat_s4_energy.min()) / (feat_s4_energy.max() - feat_s4_energy.min() + 1e-5)

    pca_swin_s4 = feature_pca_rgb(feat_s4) # (256, 256, 3)

    feat_s32 = swin_out["stride32_features"][0] # (256, 32, 32)
    pca_swin_s32 = feature_pca_rgb(feat_s32)

    # B. SVG / Vector-Aware Feature Maps
    saliency_np = svg_out["saliency_field"][0, 0].cpu().numpy() # (256, 256)
    tangent_np = svg_out["tangent_field"][0].cpu().numpy()      # (2, 256, 256)
    curvature_np = svg_out["curvature_field"][0, 0].cpu().numpy() # (256, 256)
    hsv_vector_map = vector_field_to_hsv(tangent_np, saliency_np) # (256, 256, 3)
    bezier_curves = svg_out["bezier_curves"][0]

    # 5. Render 8-Panel Comparison Figure
    fig = plt.figure(figsize=(24, 12), facecolor='#0d1117')
    plt.suptitle("EXP_07 ENCODER ARCHITECTURAL COMPARISON:\nStandard Swin Feature Map (EXP_06) vs SVG Vector-Aware Feature Map (EXP_07)",
                 fontsize=18, fontweight='bold', color='#58a6ff', y=0.98)

    # Define 2x4 Subplots
    axes = fig.subplots(2, 4)

    # ------------------ ROW 1: STANDARD SWIN BACKBONE ENCODER (EXP_06) ------------------
    # [1, 1] Input RGB Image
    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title("[1] Input Surgical RGB Frame (1024x1024)\n(Moist Liver Lobe + Anatomical Contours)", color='white', fontsize=11, fontweight='bold')
    axes[0, 0].axis('off')

    # [1, 2] Stride-4 Feature Energy
    im1 = axes[0, 1].imshow(feat_s4_energy, cmap='inferno')
    axes[0, 1].set_title("[2] Standard Swin Feature Energy (Stride-4: 256x256)\n(Broad, diffuse scalar pixel activations)", color='white', fontsize=11, fontweight='bold')
    axes[0, 1].axis('off')

    # [1, 3] Stride-4 PCA Semantic Projection
    axes[0, 2].imshow(pca_swin_s4)
    axes[0, 2].set_title("[3] Standard Swin PCA Projection (256x256)\n(Multi-channel regional organ representations)", color='white', fontsize=11, fontweight='bold')
    axes[0, 2].axis('off')

    # [1, 4] Stride-32 Cross-Attention Token Grid
    axes[0, 3].imshow(pca_swin_s32)
    axes[0, 3].set_title("[4] Stride-32 Cross-Attention Memory Grid (32x32)\n(Low-res discrete tokens fed to decoder queries)", color='white', fontsize=11, fontweight='bold')
    axes[0, 3].axis('off')

    # ------------------ ROW 2: SVG / VECTOR-AWARE FEATURE ENCODER (EXP_07) ------------------
    # [2, 1] Landmark Saliency & Skeleton Field
    axes[1, 0].imshow(saliency_np, cmap='magma')
    axes[1, 0].set_title("[5] SVG Landmark Saliency & Skeleton Field S(x,y)\n(Identifies continuous 1D ridge paths)", color='#7ee787', fontsize=11, fontweight='bold')
    axes[1, 0].axis('off')

    # [2, 2] 2D Tangent Vector Flow Field (Quiver)
    axes[1, 1].imshow(img_np, alpha=0.6)
    step = 16
    H_s, W_s = saliency_np.shape
    y_coords, x_coords = np.mgrid[0:H_s:step, 0:W_s:step]
    u = tangent_np[0, ::step, ::step]
    v = tangent_np[1, ::step, ::step]
    sal_sub = saliency_np[::step, ::step]
    
    # Scale coordinates to 1024x1024 display space
    scale_factor = 1024.0 / float(H_s)
    axes[1, 1].quiver(x_coords * scale_factor, y_coords * scale_factor, u, -v, color='#00ffff',
                     scale=20, width=0.005, alpha=0.9)
    axes[1, 1].set_title("[6] 2D Tangent Flow Field T(x,y) (Vector Quiver)\n(Unit vector arrows pointing along tissue ridges)", color='#7ee787', fontsize=11, fontweight='bold')
    axes[1, 1].axis('off')

    # [2, 3] HSV Vector Direction Wheel
    axes[1, 2].imshow(hsv_vector_map)
    axes[1, 2].set_title("[7] HSV Vector Direction Wheel\n(Color Hue = Tangent Angle theta, Value = Saliency)", color='#7ee787', fontsize=11, fontweight='bold')
    axes[1, 2].axis('off')

    # [2, 4] Extracted SVG Continuous Bézier Spline Primitives
    axes[1, 3].imshow(img_np, alpha=0.7)
    colors = ['#ff00ff', '#00ffcc', '#ffaa00', '#00ff66', '#ff3366']
    for idx, bz in enumerate(bezier_curves):
        c = colors[idx % len(colors)]
        curve = bz["curve"] * 1024.0 # (num_points, 2)
        p0 = bz["P0"] * 1024.0
        p3 = bz["P3"] * 1024.0
        c1 = bz["C1"] * 1024.0
        c2 = bz["C2"] * 1024.0

        # Draw continuous Bézier curve
        axes[1, 3].plot(curve[:, 0], curve[:, 1], color=c, linewidth=3.5, label=f"Bézier #{idx+1}")
        
        # Draw control points & tangent handles
        axes[1, 3].scatter([p0[0], p3[0]], [p0[1], p3[1]], color='white', s=40, zorder=5)
        axes[1, 3].scatter([c1[0], c2[0]], [c1[1], c2[1]], color=c, s=30, marker='s', zorder=5)
        axes[1, 3].plot([p0[0], c1[0]], [p0[1], c1[1]], color=c, linestyle=':', alpha=0.7)
        axes[1, 3].plot([p3[0], c2[0]], [p3[1], c2[1]], color=c, linestyle=':', alpha=0.7)

    axes[1, 3].set_title("[8] Parametric SVG Bézier Primitives B(t)\n(Smooth continuous splines P0, C1, C2, P3)", color='#7ee787', fontsize=11, fontweight='bold')
    axes[1, 3].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(args.output_path, dpi=200, bbox_inches='tight', facecolor='#0d1117')
    plt.close()

    print(f"🎉 Comparative visualization saved successfully to:\n   👉 {args.output_path}")

if __name__ == "__main__":
    main()
