import os
import sys
import math
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add EXP_05 path
EXP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if EXP_DIR not in sys.path:
    sys.path.append(EXP_DIR)

from configs.exp05_config import config
from models.surgical_3d_vector_transformer import Surgical3DVectorTransformer
from utils.dataset_3d import Surgical3DVectorDataset

def parse_args():
    parser = argparse.ArgumentParser(description="EXP_05 Pipeline Diagnostic Visualizer for Colab/Kaggle")
    parser.add_argument("--dataset_dir", type=str, default=None, help="Path to real surgical dataset (/content/L3D)")
    parser.add_argument("--sample_idx", type=int, default=0, help="Index of sample image to inspect")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained checkpoint (.pth)")
    parser.add_argument("--output", type=str, default="exp05_pipeline_diagnostic_visualization.png", help="Output PNG path")
    parser.add_argument("--wandb", action="store_true", help="Log visual figure to Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="Surgical_AI_3D_Vector", help="W&B project name")
    return parser.parse_args()

def main():
    args = parse_args()
    print("=" * 80)
    print("🔬 EXP_05 MONOCULAR 3D VECTOR TRANSFORMER — PIPELINE VISUAL SMOKE TEST")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Execution Device: {device}")

    # 1. Load Real or Synthetic Sample
    if args.dataset_dir and os.path.exists(args.dataset_dir):
        print(f"📁 Loading real surgical sample from dataset: {args.dataset_dir}")
        dataset = Surgical3DVectorDataset(dataset_dir=args.dataset_dir, mode="val")
        if len(dataset) == 0:
            dataset = Surgical3DVectorDataset(dataset_dir=args.dataset_dir, mode="train")
        sample_idx = min(args.sample_idx, len(dataset) - 1)
        sample = dataset[sample_idx]

        img_tensor = sample["image"].unsqueeze(0).to(device) # (1, 3, 1024, 1024)
        depth_tensor = sample["depth"].unsqueeze(0).to(device) # (1, 1, 1024, 1024)
        targets = {
            "target_classes": sample["target_classes"].unsqueeze(0).to(device),
            "target_polylines": sample["target_polylines"].unsqueeze(0).to(device),
            "target_masks": sample["target_masks"].unsqueeze(0).to(device),
            "valid_mask": sample["valid_mask"].unsqueeze(0).to(device)
        }
        
        # Denormalize image for plotting
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
        img_np = sample["image"].permute(1, 2, 0).cpu().numpy()
        img_rgb = np.clip(img_np * std + mean, 0.0, 1.0)
        depth_map = sample["depth"][0].cpu().numpy()
        
        # Extract first valid ground-truth 3D polyline & mask
        valid_indices = torch.where(sample["valid_mask"])[0]
        if len(valid_indices) > 0:
            first_idx = valid_indices[0].item()
            pts_3d_gt = sample["target_polylines"][first_idx].cpu().numpy() # (K, 3)
            gt_mask_2d = sample["target_masks"][first_idx].cpu().numpy()   # (1024, 1024)
        else:
            pts_3d_gt = np.zeros((config.num_points, 3))
            gt_mask_2d = np.zeros((1024, 1024))
    else:
        print("⚠️ No dataset_dir provided. Running with synthetic mathematical surgical sample.")
        from visualize_smoke_test import create_synthetic_surgical_sample
        img_rgb, depth_map, _, pts_3d_gt, gt_mask_2d, img_tensor, depth_tensor, targets = create_synthetic_surgical_sample()
        img_tensor = img_tensor.to(device)
        depth_tensor = depth_tensor.to(device)
        for k in targets:
            targets[k] = targets[k].to(device)

    # 2. Instantiate Model & Load Checkpoint (if provided)
    model = Surgical3DVectorTransformer(config).to(device)
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"📥 Loading weights from checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)
        print("✅ Checkpoint loaded successfully!")
    model.eval()

    # 3. Intermediate Activations Extraction
    with torch.no_grad():
        # Step A: Backbone Feature & Pinhole Depth Lifting
        fused_features = model.backbone(img_tensor, depth_tensor)
        xyz_unprojected = model.backbone._unproject_depth_to_3d(depth_tensor, 128, 128) # (1, 3, 128, 128)
        pe_3d_feat = model.backbone.pe_3d_module(xyz_unprojected)                      # (1, 192, 128, 128)

        # Step B: Proposal Head Initial Anchors
        initial_anchors = model.proposal_head(fused_features[1])                      # (1, 10, 20, 3)

        # Step C: Transformer Decoder Layers Progression
        stride_idx = getattr(config, "decoder_stride_idx", 1)
        outputs_cls, outputs_polylines, outputs_masks = model.decoder(fused_features, initial_anchors, stride_idx=stride_idx)

        # Step D: Criterion & Loss Computation
        loss, loss_dict = model.criterion(outputs_cls, outputs_polylines, outputs_masks, targets)

    print(f"✅ Forward Pass Complete | Total Loss: {loss.item():.4f}")
    print(f"   • Loss Breakdown: {loss_dict}")

    # 4. Render Rich 3x3 Diagnostic Visualization
    fig = plt.figure(figsize=(24, 18), facecolor='#0d1117')
    plt.suptitle("Surgical Monocular 3D Vector Space Transformer (EXP_05) — Pipeline Diagnostic Verification", 
                 fontsize=18, fontweight='bold', color='white', y=0.98)

    # 1. RGB & GT Mask Contour
    ax1 = fig.add_subplot(3, 3, 1)
    ax1.set_facecolor('#161b22')
    ax1.imshow(img_rgb)
    if gt_mask_2d.sum() > 0:
        ax1.contour(gt_mask_2d, levels=[0.5], colors=['#00ffcc'], linewidths=3)
    ax1.set_title("1. Laparoscopic RGB & GT Landmark Contour", color='white', fontsize=13, fontweight='bold')
    ax1.axis('off')

    # 2. Monocular Depth Map
    ax2 = fig.add_subplot(3, 3, 2)
    ax2.set_facecolor('#161b22')
    im2 = ax2.imshow(depth_map, cmap='inferno')
    ax2.set_title("2. Depth Anything V2 Map (Relative Depth)", color='white', fontsize=13, fontweight='bold')
    cb2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cb2.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cb2.ax.axes, 'yticklabels'), color='white')
    ax2.axis('off')

    # 3. Canonical 3D Pinhole Point Cloud & GT Curve
    ax3 = fig.add_subplot(3, 3, 3, projection='3d')
    ax3.set_facecolor('#0d1117')
    xyz_np = xyz_unprojected[0].cpu().numpy()
    sub = 4
    xs = xyz_np[0, ::sub, ::sub].flatten()
    ys = xyz_np[1, ::sub, ::sub].flatten()
    zs = xyz_np[2, ::sub, ::sub].flatten()
    ax3.scatter(xs, ys, zs, c=zs, cmap='viridis', s=2, alpha=0.4)
    if pts_3d_gt.sum() != 0:
        ax3.plot(pts_3d_gt[:, 0], pts_3d_gt[:, 1], pts_3d_gt[:, 2], color='#ff0055', linewidth=4, label="GT 3D Curve")
        ax3.scatter(pts_3d_gt[:, 0], pts_3d_gt[:, 1], pts_3d_gt[:, 2], color='#00ffcc', s=40)
    ax3.set_title("3. Canonical 3D Pinhole Surface (X,Y,Z in [-1, 1]^3)", color='white', fontsize=13, fontweight='bold')
    ax3.tick_params(colors='white')
    ax3.legend(loc="upper left", facecolor='#161b22', labelcolor='white')

    # 4. 3D Positional Encoding PE_3D Wave Slices
    ax4 = fig.add_subplot(3, 3, 4)
    ax4.set_facecolor('#161b22')
    pe_slice = pe_3d_feat[0, 0:3].permute(1, 2, 0).cpu().numpy()
    pe_norm = (pe_slice - pe_slice.min()) / (pe_slice.max() - pe_slice.min() + 1e-6)
    ax4.imshow(pe_norm)
    ax4.set_title("4. PE_3D Positional Encoding Frequency Waves (192 Ch)", color='white', fontsize=13, fontweight='bold')
    ax4.axis('off')

    # 5. Dynamic 3D Proposal Anchors (Layer 0)
    ax5 = fig.add_subplot(3, 3, 5, projection='3d')
    ax5.set_facecolor('#0d1117')
    anchors_np = initial_anchors[0].cpu().numpy()
    if pts_3d_gt.sum() != 0:
        ax5.plot(pts_3d_gt[:, 0], pts_3d_gt[:, 1], pts_3d_gt[:, 2], color='#00ffcc', linewidth=4, label="Target GT 3D")
    for i in range(min(5, anchors_np.shape[0])):
        ax5.plot(anchors_np[i, :, 0], anchors_np[i, :, 1], anchors_np[i, :, 2], 
                 linestyle='--', linewidth=2, alpha=0.8, label=f"Anchor Query {i+1}" if i < 2 else None)
    ax5.set_title("5. Dynamic 3D Proposal Anchors (Layer 0)", color='white', fontsize=13, fontweight='bold')
    ax5.tick_params(colors='white')
    ax5.legend(loc="upper left", facecolor='#161b22', labelcolor='white')

    # 6. Decoder Layer 1 Dot-Product Mask
    ax6 = fig.add_subplot(3, 3, 6)
    ax6.set_facecolor('#161b22')
    mask_l1 = torch.sigmoid(outputs_masks[0][0, 0]).cpu().numpy()
    im6 = ax6.imshow(mask_l1, cmap='magma')
    ax6.set_title("6. Decoder Layer 1 Mask Attention (M_1)", color='white', fontsize=13, fontweight='bold')
    cb6 = fig.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)
    cb6.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cb6.ax.axes, 'yticklabels'), color='white')
    ax6.axis('off')

    # 7. Final Layer 6 Mask vs GT Stroke
    ax7 = fig.add_subplot(3, 3, 7)
    ax7.set_facecolor('#161b22')
    mask_l6 = torch.sigmoid(outputs_masks[-1][0, 0]).cpu().numpy()
    im7 = ax7.imshow(mask_l6, cmap='viridis')
    if gt_mask_2d.sum() > 0:
        ax7.contour(gt_mask_2d, levels=[0.5], colors=['#ff0055'], linewidths=2)
    ax7.set_title("7. Final Layer 6 Mask (MLP(q) • F_stride4)", color='white', fontsize=13, fontweight='bold')
    cb7 = fig.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)
    cb7.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cb7.ax.axes, 'yticklabels'), color='white')
    ax7.axis('off')

    # 8. Layer-by-Layer 3D Curve Progression (L0 -> L2 -> L6)
    ax8 = fig.add_subplot(3, 3, 8, projection='3d')
    ax8.set_facecolor('#0d1117')
    if pts_3d_gt.sum() != 0:
        ax8.plot(pts_3d_gt[:, 0], pts_3d_gt[:, 1], pts_3d_gt[:, 2], color='#00ffcc', linewidth=5, label="Target GT 3D")
    p0 = anchors_np[0]
    ax8.plot(p0[:, 0], p0[:, 1], p0[:, 2], color='#ffa500', linestyle=':', linewidth=2, label="L0 (Anchor)")
    p2 = outputs_polylines[1][0, 0].cpu().numpy()
    ax8.plot(p2[:, 0], p2[:, 1], p2[:, 2], color='#ff00ff', linestyle='--', linewidth=2, label="L2 Refined")
    p6 = outputs_polylines[-1][0, 0].cpu().numpy()
    ax8.plot(p6[:, 0], p6[:, 1], p6[:, 2], color='#00ffff', linewidth=3, label="L6 Final Output")
    ax8.set_title("8. Layer-by-Layer 3D Curve Refinement", color='white', fontsize=13, fontweight='bold')
    ax8.tick_params(colors='white')
    ax8.legend(loc="upper left", facecolor='#161b22', labelcolor='white')

    # 9. Loss Breakdown Bar Chart
    ax9 = fig.add_subplot(3, 3, 9)
    ax9.set_facecolor('#161b22')
    loss_names = ['l_cls', 'l_pos', 'l_tan', 'l_curv', 'l_mask']
    loss_vals = [loss_dict['l_cls'], loss_dict['l_pos'], loss_dict['l_tan'], loss_dict['l_curv'], loss_dict['l_mask']]
    colors = ['#ff7b72', '#79c0ff', '#d2a8ff', '#ffa657', '#56d364']
    bars = ax9.bar(loss_names, loss_vals, color=colors, edgecolor='white', linewidth=1.2)
    for bar, v in zip(bars, loss_vals):
        ax9.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f"{v:.2f}", 
                 ha='center', va='bottom', color='white', fontweight='bold')
    ax9.set_title("9. Deep Supervision Loss Breakdown", color='white', fontsize=13, fontweight='bold')
    ax9.tick_params(colors='white')
    ax9.set_ylim(0, max(loss_vals) * 1.35)
    ax9.grid(axis='y', linestyle='--', alpha=0.3, color='gray')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) if os.path.dirname(args.output) else ".", exist_ok=True)
    plt.savefig(args.output, dpi=180, facecolor=fig.get_facecolor(), edgecolor='none')
    print(f"🎉 Visualization saved successfully to: {args.output}")

    # Optional W&B Image Logging
    if args.wandb:
        try:
            import wandb
            api_key = os.environ.get("WANDB_API_KEY", "83f4544a22543e319c6009abceaac90b634c68a3")
            if api_key:
                wandb.login(key=api_key)
            wandb.init(project=args.wandb_project, name="Diagnostic_Visual_Smoke_Test")
            wandb.log({"diagnostic_visualization": wandb.Image(args.output)})
            print("📊 Diagnostic visualization logged directly to Weights & Biases dashboard!")
        except Exception as e:
            print(f"⚠️ Could not log to wandb: {e}")

    plt.close()
    print("=" * 80)

if __name__ == "__main__":
    main()
