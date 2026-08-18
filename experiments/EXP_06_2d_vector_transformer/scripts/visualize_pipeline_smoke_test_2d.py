import os
import sys
import argparse
import warnings

# Suppress harmless Matplotlib font & HF Hub warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", message=".*Glyph.*")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add experiment root to path
EXP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if EXP_DIR not in sys.path:
    sys.path.append(EXP_DIR)

from configs.exp06_config import config
from utils.dataset_2d import Surgical2DVectorDataset
from models.surgical_2d_vector_transformer import Surgical2DVectorTransformer

def parse_args():
    parser = argparse.ArgumentParser(description="EXP_06 2D Vector Transformer Visual Diagnostic Smoke Test")
    parser.add_argument("--dataset_dir", type=str, default=config.dataset_dir, help="Dataset directory")
    parser.add_argument("--sample_idx", type=int, default=0, help="Index of sample to visualize")
    parser.add_argument("--output", "--save_path", dest="output", type=str, default="pipeline_diagnostic_2d.png", help="Output PNG path")
    return parser.parse_args()

def main():
    args = parse_args()
    print("=" * 70)
    print("🔬 Running EXP_06 Direct 2D Vector Transformer Visual Diagnostic Smoke Test")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Execution Device: {device}")

    dataset = Surgical2DVectorDataset(args.dataset_dir, mode="train")
    if len(dataset) == 0:
        print("⚠️ Dataset directory is empty or not found.")
        return

    # Check if requested sample has annotations; if not, search for first valid annotated frame
    sample = dataset[args.sample_idx]
    if sample["valid_mask"].sum() == 0:
        print(f"ℹ️ Sample index {args.sample_idx} has 0 annotations. Searching dataset for first annotated frame...")
        found_idx = None
        for i in range(len(dataset)):
            s_test = dataset[i]
            if s_test["valid_mask"].sum() > 0:
                sample = s_test
                found_idx = i
                print(f"🎯 Found annotated sample at index {i}!")
                break
        if found_idx is None:
            print("⚠️ No annotated samples found in dataset directory.")

    image = sample["image"].unsqueeze(0).to(device) # (1, 3, 1024, 1024)
    target_classes = sample["target_classes"].unsqueeze(0).to(device)
    target_polylines = sample["target_polylines"].unsqueeze(0).to(device)
    target_masks = sample["target_masks"].unsqueeze(0).to(device)
    valid_mask = sample["valid_mask"].unsqueeze(0).to(device)

    targets = {
        "target_classes": target_classes,
        "target_polylines": target_polylines,
        "target_masks": target_masks,
        "valid_mask": valid_mask
    }

    model = Surgical2DVectorTransformer(config).to(device)
    model.eval()

    with torch.no_grad():
        # Forward pass
        fused_features = model.backbone(image)
        decoder_outputs = model.decoder(fused_features)
        loss, loss_dict = model.loss_suite(
            outputs_cls=decoder_outputs["aux_cls"],
            outputs_polylines=decoder_outputs["aux_polylines"],
            outputs_masks=decoder_outputs["aux_masks"],
            targets=targets
        )

    # Denormalize Image
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    img_np = image[0].permute(1, 2, 0).cpu().numpy()
    img_rgb = np.clip(img_np * std + mean, 0.0, 1.0)

    # Prepare Plot Canvas (3x3 Diagnostic Grid)
    fig = plt.figure(figsize=(18, 14), facecolor='#0d1117')
    plt.suptitle("Surgical Direct 2D Vector Space Transformer (EXP_06) — Pipeline Diagnostic Verification", 
                 color='white', fontsize=16, fontweight='bold', y=0.98)

    # 1. Laparoscopic RGB + All Active GT Contours
    ax1 = fig.add_subplot(3, 3, 1)
    ax1.set_facecolor('#161b22')
    ax1.imshow(img_rgb)
    active_gts = [i for i in range(valid_mask.shape[1]) if valid_mask[0, i] > 0]
    colors_gt = ['#00ffcc', '#00ff88', '#00e1ff', '#33ffaa']
    for idx, g in enumerate(active_gts):
        gt_m = target_masks[0, g].cpu().numpy()
        if gt_m.sum() > 0:
            ax1.contour(gt_m, levels=[0.5], colors=[colors_gt[idx % len(colors_gt)]], linewidths=3)
    ax1.set_title("1. Laparoscopic RGB & GT Landmark Contours", color='white', fontsize=12, fontweight='bold')
    ax1.axis('off')

    # 2. Multi-Scale Stride-4 Feature Heatmap (256x256)
    ax2 = fig.add_subplot(3, 3, 2)
    ax2.set_facecolor('#161b22')
    feat_s4_mag = fused_features[0][0].norm(dim=0).cpu().numpy()
    ax2.imshow(feat_s4_mag, cmap='viridis')
    ax2.set_title("2. Stride-4 Visual Features (256x256 = 65k tokens)", color='white', fontsize=12, fontweight='bold')
    ax2.axis('off')

    # 3. 2D Sinusoidal Positional Encoding PE_2D Waves
    ax3 = fig.add_subplot(3, 3, 3)
    ax3.set_facecolor('#161b22')
    pe_slice = model.backbone.pe_2d(fused_features[0])[0, 0:3].permute(1, 2, 0).cpu().numpy()
    pe_norm = (pe_slice - pe_slice.min()) / (pe_slice.max() - pe_slice.min() + 1e-6)
    ax3.imshow(pe_norm)
    ax3.set_title("3. PE_2D Frequency Waves (128 Channels)", color='white', fontsize=12, fontweight='bold')
    ax3.axis('off')

    # 4. Learned Query Initial Priors (Layer 0)
    ax4 = fig.add_subplot(3, 3, 4)
    ax4.set_facecolor('#161b22')
    ax4.imshow(img_rgb)
    query_priors = model.decoder.query_polylines.detach().cpu().numpy() # (N, K, 2)
    for q in range(query_priors.shape[0]):
        qp = query_priors[q] * 1024.0
        ax4.plot(qp[:, 0], qp[:, 1], linestyle='--', linewidth=2, alpha=0.8, label=f"Query {q+1}" if q < 3 else None)
    ax4.set_title("4. Learned Query Embeddings Priors (Layer 0)", color='white', fontsize=12, fontweight='bold')
    ax4.axis('off')
    ax4.legend(loc="lower left", facecolor='#161b22', labelcolor='white')

    # 5. Decoder Layer 1 Mask Attention
    ax5 = fig.add_subplot(3, 3, 5)
    ax5.set_facecolor('#161b22')
    mask_l1 = torch.sigmoid(decoder_outputs["aux_masks"][0][0, 0]).cpu().numpy()
    im5 = ax5.imshow(mask_l1, cmap='magma')
    ax5.set_title("5. Decoder Layer 1 Dot-Product Mask (M_1)", color='white', fontsize=12, fontweight='bold')
    fig.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
    ax5.axis('off')

    # 6. Final Layer 6 Mask Attention (MLP(q) • F_stride4)
    ax6 = fig.add_subplot(3, 3, 6)
    ax6.set_facecolor('#161b22')
    mask_l6 = torch.sigmoid(decoder_outputs["aux_masks"][-1][0, 0]).cpu().numpy()
    im6 = ax6.imshow(mask_l6, cmap='magma')
    if target_masks[0, 0].sum() > 0:
        ax6.contour(target_masks[0, 0].cpu().numpy(), levels=[0.5], colors=['#00ffcc'], linewidths=2.5)
    ax6.set_title("6. Final Layer 6 Mask (MLP(q) • F_stride4)", color='white', fontsize=12, fontweight='bold')
    fig.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)
    ax6.axis('off')

    # 7. Layer-by-Layer 2D Polyline Refinement (L0 -> L2 -> L6)
    ax7 = fig.add_subplot(3, 3, 7)
    ax7.set_facecolor('#161b22')
    ax7.imshow(img_rgb)
    
    # Find best-matched query slot for primary GT landmark
    best_q = 0
    if len(active_gts) > 0:
        primary_gt_p = target_polylines[0, active_gts[0]] # (K, 2)
        gt_poly = primary_gt_p.cpu().numpy() * 1024.0
        ax7.plot(gt_poly[:, 0], gt_poly[:, 1], color='#00ffcc', linewidth=4, label="Target GT 2D")

        best_l1 = float('inf')
        for q in range(decoder_outputs["pred_polylines"].shape[1]):
            pred_p_q = decoder_outputs["pred_polylines"][0, q]
            l1_diff = torch.abs(pred_p_q - primary_gt_p).mean().item()
            if l1_diff < best_l1:
                best_l1 = l1_diff
                best_q = q

    l0_poly = query_priors[best_q] * 1024.0
    l2_poly = decoder_outputs["aux_polylines"][2][0, best_q].detach().cpu().numpy() * 1024.0
    l6_poly = decoder_outputs["aux_polylines"][-1][0, best_q].detach().cpu().numpy() * 1024.0

    ax7.plot(l0_poly[:, 0], l0_poly[:, 1], color='#ff9900', linestyle=':', linewidth=2.5, label="L0 (Learned Prior)")
    ax7.plot(l2_poly[:, 0], l2_poly[:, 1], color='#ff00ff', linestyle='--', linewidth=2.5, label="L2 Refined")
    ax7.plot(l6_poly[:, 0], l6_poly[:, 1], color='#00e1ff', linewidth=3.5, label="L6 Final Output")
    ax7.set_title("7. Layer-by-Layer 2D Curve Snapping (L0->L2->L6)", color='white', fontsize=12, fontweight='bold')
    ax7.axis('off')
    ax7.legend(loc="lower left", facecolor='#161b22', labelcolor='white')

    # 8. 2D Vector Predictions Overlay
    ax8 = fig.add_subplot(3, 3, 8)
    ax8.set_facecolor('#161b22')
    ax8.imshow(img_rgb)
    for idx, g in enumerate(active_gts):
        gt_m = target_masks[0, g].cpu().numpy()
        gt_p_norm = target_polylines[0, g]
        if gt_m.sum() > 0:
            ax8.contour(gt_m, levels=[0.5], colors=[colors_gt[idx % len(colors_gt)]], linewidths=2.5)

        # Match best predicted query for this landmark
        best_g_q = 0
        best_g_l1 = float('inf')
        for q in range(decoder_outputs["pred_polylines"].shape[1]):
            pred_q = decoder_outputs["pred_polylines"][0, q]
            l1_d = torch.abs(pred_q - gt_p_norm).mean().item()
            if l1_d < best_g_l1:
                best_g_l1 = l1_d
                best_g_q = q

        pred_p = decoder_outputs["pred_polylines"][0, best_g_q].detach().cpu().numpy() * 1024.0
        ax8.plot(pred_p[:, 0], pred_p[:, 1], color='#ff00ff', linewidth=3, linestyle='--', marker='o', markersize=3, label="Predicted Polyline" if idx == 0 else None)
    ax8.set_title("8. Final 2D Vector Landmark Predictions", color='white', fontsize=12, fontweight='bold')
    ax8.axis('off')
    ax8.legend(loc="lower left", facecolor='#161b22', labelcolor='white')

    # 9. Deep Supervision Loss Breakdown
    ax9 = fig.add_subplot(3, 3, 9)
    ax9.set_facecolor('#161b22')
    loss_names = list(loss_dict.keys())
    loss_values = [loss_dict[k] for k in loss_names]
    bar_colors = ['#ff6b6b', '#4dabf7', '#51cf66']
    bars = ax9.bar(loss_names, loss_values, color=bar_colors[:len(loss_names)], width=0.5)
    for bar, val in zip(bars, loss_values):
        ax9.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{val:.2f}",
                 ha='center', va='bottom', color='white', fontsize=11, fontweight='bold')
    ax9.set_title("9. Deep Supervision 2D Loss (L_cls, L_pos, L_mask)", color='white', fontsize=12, fontweight='bold')
    ax9.tick_params(colors='white')
    ax9.set_ylim([0, max(loss_values) * 1.35 if len(loss_values) > 0 and max(loss_values) > 0 else 3.0])
    ax9.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.output, dpi=140, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close('all')
    print(f"🎉 EXP_06 2D Diagnostic smoke test image generated successfully at: '{args.output}'")

if __name__ == "__main__":
    main()
