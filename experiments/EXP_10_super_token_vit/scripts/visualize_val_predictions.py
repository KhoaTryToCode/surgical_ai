import os
import sys
import argparse
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt

exp_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if exp_root not in sys.path:
    sys.path.insert(0, exp_root)

from configs.exp10_config import EXP10Config, resolve_dataset_dir
from models.super_token_vit import SuperTokenGeometricViT
from utils.dataset_super_token import SuperTokenLandmarkDataset
from scripts.train_super_token_vit import render_curves_to_eval_mask, compute_dice_score


CLASS_COLORS = {
    0: (0, 255, 0),      # Ridge: Green
    1: (255, 120, 0),    # Silhouette: Cyan/Blue
    2: (0, 255, 255),    # Falciform: Yellow
    3: (255, 0, 255)     # Gallbladder: Magenta
}


def draw_splines_on_rgb(img_rgb_uint8, ctrl_points_np, exist_probs_np, exist_thresh=0.35, K=6):
    overlay = img_rgb_uint8.copy()
    C = ctrl_points_np.shape[0]
    S = img_rgb_uint8.shape[0]

    from scipy.special import comb
    t_vals = np.linspace(0.0, 1.0, 100)
    deg = K - 1

    for c in range(C):
        if exist_probs_np[c] < exist_thresh:
            continue
        color_bgr = CLASS_COLORS.get(c, (0, 255, 0))
        pts = ctrl_points_np[c]
        
        dense_curve = np.zeros((100, 2), dtype=np.float32)
        for i in range(K):
            c_val = comb(deg, i) * ((1.0 - t_vals) ** (deg - i)) * (t_vals ** i)
            dense_curve += np.outer(c_val, pts[i])
            
        pts_pix = np.clip(np.round(dense_curve * S), 0, S - 1).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(overlay, [pts_pix], isClosed=False, color=color_bgr, thickness=3, lineType=cv2.LINE_AA)
        
        # Draw control points as circles
        for k in range(K):
            kx, ky = int(pts[k, 0] * S), int(pts[k, 1] * S)
            cv2.circle(overlay, (kx, ky), radius=4, color=(255, 255, 255), thickness=-1)
            cv2.circle(overlay, (kx, ky), radius=5, color=color_bgr, thickness=1)

    return overlay


def generate_diagnostic_figure(img_rgb_norm, depth_norm, pred_mask_512, gt_mask_512, attn_heatmaps, ctrl_points, exist_probs, frame_name, dice_score, K=6):
    # Denormalize RGB
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    rgb = np.clip((img_rgb_norm * std + mean) * 255.0, 0, 255).astype(np.uint8)

    # 1. Overlay on RGB
    rgb_splines = draw_splines_on_rgb(rgb, ctrl_points, exist_probs, exist_thresh=0.35, K=K)

    # 2. GT vs Pred Mask Comparison (Cyan = GT, Red = Pred, White = Overlap)
    pred_combined = (pred_mask_512.sum(axis=0) > 0).astype(np.float32)
    gt_combined = (gt_mask_512.sum(axis=0) > 0).astype(np.float32)
    cmp_canvas = np.zeros((512, 512, 3), dtype=np.uint8)
    cmp_canvas[:, :, 1] = (gt_combined * 255).astype(np.uint8)  # Green GT
    cmp_canvas[:, :, 2] = (gt_combined * 255).astype(np.uint8)  # Blue GT (Cyan)
    cmp_canvas[:, :, 0] = (pred_combined * 255).astype(np.uint8) # Red Pred
    # Overlap becomes Red + Cyan = White/Yellowish

    # 3. Attention Heatmap Composite
    attn_sum = attn_heatmaps.sum(axis=0)  # (32, 32)
    attn_res = cv2.resize(attn_sum, (512, 512), interpolation=cv2.INTER_LINEAR)
    attn_res = (attn_res / (attn_res.max() + 1e-6) * 255).astype(np.uint8)
    attn_heatmap_colored = cv2.applyColorMap(attn_res, cv2.COLORMAP_JET)

    # 4. Depth Map Overlay
    depth_res = (np.clip(depth_norm * 0.25 + 0.5, 0, 1) * 255).astype(np.uint8)
    depth_rgb = cv2.cvtColor(depth_res, cv2.COLOR_GRAY2RGB)
    depth_splines = draw_splines_on_rgb(depth_rgb, ctrl_points, exist_probs, exist_thresh=0.35, K=K)

    # Create 4-panel figure
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), dpi=150)
    fig.patch.set_facecolor("#121212")

    titles = [
        f"RGB + Continuous 6-Point Spline",
        f"GT (Cyan) vs Pred (Red) | Dice: {dice_score*100:.1f}%",
        f"Super-Token Patch Attention (32×32)",
        f"Depth Map + Anatomical Trajectory"
    ]
    panels = [rgb_splines, cmp_canvas, attn_heatmap_colored, depth_splines]

    for ax, title, panel in zip(axes, titles, panels):
        ax.imshow(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB) if panel.ndim == 3 else panel)
        ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=8)
        ax.axis("off")

    plt.suptitle(f"EXP_10 Super-Token Geometric ViT | Frame: {frame_name}", color="yellow", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Visualize EXP_10 Predictions")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/EXP_10/best_model.pth", help="Checkpoint path")
    parser.add_argument("--dataset_dir", type=str, default=resolve_dataset_dir(), help="Dataset path")
    parser.add_argument("--output_dir", type=str, default="outputs/EXP_10/val_visualizations", help="Output directory")
    parser.add_argument("--max_samples", type=int, default=122, help="Number of validation frames to visualize")
    parser.add_argument("--device", type=str, default="", help="Device override")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return

    checkpoint = torch.load(args.checkpoint, map_location=device)
    saved_cfg = checkpoint.get("config", {})
    backbone_name = saved_cfg.get("backbone", "vit_base_patch16_224")
    num_ctrl_points = saved_cfg.get("num_ctrl_points", 6)
    use_depth = saved_cfg.get("use_depth", True)
    in_chans = 4 if use_depth else 3

    model = SuperTokenGeometricViT(
        backbone_name=backbone_name,
        in_chans=in_chans,
        pretrained=False,
        image_size=512,
        patch_size=16,
        num_classes=4,
        num_ctrl_points=num_ctrl_points,
        embed_dim=768,
        hidden_dim=512,
        render_size=128
    ).to(device)

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    print(f"✅ Loaded checkpoint from {args.checkpoint}")

    val_dataset = SuperTokenLandmarkDataset(
        dataset_dir=args.dataset_dir,
        mode="val",
        image_size=512,
        patch_size=16,
        num_ctrl_points=num_ctrl_points,
        render_size=128,
        use_depth=use_depth
    )

    print(f"🎨 Generating 4-panel diagnostic visualizations for up to {min(args.max_samples, len(val_dataset))} frames...")

    count = 0
    with torch.no_grad():
        for idx in range(min(args.max_samples, len(val_dataset))):
            sample = val_dataset[idx]
            img = sample["image"].unsqueeze(0).to(device)
            v_eval_masks = sample["target_eval_masks"].numpy()  # (4, 512, 512)
            img_path = sample["img_path"]
            frame_name = os.path.basename(img_path)

            preds = model(img)
            ctrl_points_np = preds["ctrl_points"][0].cpu().numpy()     # (4, K, 2)
            exist_probs_np = preds["exist_probs"][0].cpu().numpy()     # (4,)
            attn_heatmaps = preds["attn_heatmaps"][0].cpu().numpy()    # (4, 32, 32)

            pred_mask_512 = render_curves_to_eval_mask(
                ctrl_points_np,
                exist_probs_np,
                exist_thresh=0.35,
                canvas_size=512,
                stroke_px=2
            )

            # Compute combined Dice
            dice = compute_dice_score(pred_mask_512.sum(axis=0) > 0, v_eval_masks.sum(axis=0) > 0)

            # Extract RGB and Depth for plotting
            img_rgb_norm = sample["image"][:3].permute(1, 2, 0).numpy()
            depth_norm = sample["image"][3].numpy() if use_depth else np.zeros((512, 512), dtype=np.float32)

            fig = generate_diagnostic_figure(
                img_rgb_norm,
                depth_norm,
                pred_mask_512,
                v_eval_masks,
                attn_heatmaps,
                ctrl_points_np,
                exist_probs_np,
                frame_name,
                dice,
                K=num_ctrl_points
            )

            out_path = os.path.join(args.output_dir, f"{os.path.splitext(frame_name)[0]}_exp10_diag.png")
            fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
            plt.close(fig)
            count += 1

            if count % 10 == 0 or count == min(args.max_samples, len(val_dataset)):
                print(f"  🖼️ Rendered {count}/{min(args.max_samples, len(val_dataset))} figures...")

    print(f"\n✅ All diagnostic figures successfully saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
