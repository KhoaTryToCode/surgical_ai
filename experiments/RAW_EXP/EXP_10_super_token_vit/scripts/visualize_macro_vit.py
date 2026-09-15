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
from models.macro_patch_vit import MacroPatchViT
from models.macro_merger import merge_macro_beziers_to_image, compute_batch_dice
from utils.dataset_macro_vit import MacroPatchLandmarkDataset


CLASS_COLORS = {
    1: (0, 255, 0),      # Ridge: Green
    2: (255, 120, 0),    # Silhouette: Cyan/Blue
    3: (0, 255, 255),    # Falciform: Yellow
    4: (255, 0, 255)     # Gallbladder: Magenta
}


def draw_macro_splines_on_rgb(img_rgb_uint8, active_patches):
    overlay = img_rgb_uint8.copy()
    for patch in active_patches:
        cls_id = patch["class_id"]
        color = CLASS_COLORS.get(cls_id, (0, 255, 0))
        pts = patch["global_pts"]  # (20, 2)
        pts_pix = np.round(pts).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(overlay, [pts_pix], isClosed=False, color=color, thickness=3, lineType=cv2.LINE_AA)
        
        # Endpoint circles
        p_start = (int(pts[0, 0]), int(pts[0, 1]))
        p_end = (int(pts[-1, 0]), int(pts[-1, 1]))
        cv2.circle(overlay, p_start, radius=3, color=(255, 255, 255), thickness=-1)
        cv2.circle(overlay, p_end, radius=3, color=color, thickness=-1)
    return overlay


def generate_diagnostic_figure(img_rgb_norm, depth_norm, render_res, gt_mask_512, frame_name, dice_score, macro_patch_size=64):
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    rgb = np.clip((img_rgb_norm * std + mean) * 255.0, 0, 255).astype(np.uint8)

    active_patches = render_res["active_patches"]
    pred_pixel_masks = render_res["pixel_masks"]

    # 1. Spline Overlay on RGB
    rgb_splines = draw_macro_splines_on_rgb(rgb, active_patches)

    # 2. GT vs Pred Comparison (Cyan = GT, Red = Pred)
    pred_comb = (pred_pixel_masks.sum(axis=0) > 0.1).astype(np.float32)
    gt_comb = (gt_mask_512.sum(axis=0) > 0.1).astype(np.float32)
    cmp_canvas = np.zeros((512, 512, 3), dtype=np.uint8)
    cmp_canvas[:, :, 1] = (gt_comb * 255).astype(np.uint8)  # Green GT
    cmp_canvas[:, :, 2] = (gt_comb * 255).astype(np.uint8)  # Blue GT (Cyan)
    cmp_canvas[:, :, 0] = (pred_comb * 255).astype(np.uint8) # Red Pred

    # 3. Macro-Grid Activation Map (8x8 grid visualizer)
    grid_canvas = rgb.copy() // 2
    G = 512 // macro_patch_size
    for r in range(G + 1):
        cv2.line(grid_canvas, (0, r * macro_patch_size), (512, r * macro_patch_size), (80, 80, 80), 1)
    for c in range(G + 1):
        cv2.line(grid_canvas, (c * macro_patch_size, 0), (c * macro_patch_size, 512), (80, 80, 80), 1)
    for patch in active_patches:
        r, c = patch["row"], patch["col"]
        cls_id = patch["class_id"]
        color = CLASS_COLORS.get(cls_id, (0, 255, 0))
        cv2.rectangle(
            grid_canvas,
            (c * macro_patch_size + 2, r * macro_patch_size + 2),
            ((c + 1) * macro_patch_size - 2, (r + 1) * macro_patch_size - 2),
            color,
            2
        )
    # Draw splines on top of grid
    grid_canvas = draw_macro_splines_on_rgb(grid_canvas, active_patches)

    # 4. Depth Map Overlay
    depth_res = (np.clip(depth_norm * 0.25 + 0.5, 0, 1) * 255).astype(np.uint8)
    depth_rgb = cv2.cvtColor(depth_res, cv2.COLOR_GRAY2RGB)
    depth_splines = draw_macro_splines_on_rgb(depth_rgb, active_patches)

    # 4-panel figure
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), dpi=150)
    fig.patch.set_facecolor("#121212")

    titles = [
        f"RGB + Macro-Splines (64px)",
        f"GT (Cyan) vs Pred (Red) | Dice: {dice_score*100:.1f}%",
        f"8×8 Macro-Grid Activations ({len(active_patches)} Active)",
        f"Depth Map + Anatomical Trajectory"
    ]
    panels = [rgb_splines, cmp_canvas, grid_canvas, depth_splines]

    for ax, title, panel in zip(axes, titles, panels):
        ax.imshow(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB) if panel.ndim == 3 else panel)
        ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=8)
        ax.axis("off")

    plt.suptitle(f"EXP_10 Macro-Patch ViT (Way A) | Frame: {frame_name}", color="yellow", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Visualize EXP_10 Macro-Patch Predictions")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/EXP_10/best_model.pth", help="Checkpoint path")
    parser.add_argument("--dataset_dir", type=str, default=resolve_dataset_dir(), help="Dataset path")
    parser.add_argument("--output_dir", type=str, default="outputs/EXP_10/val_visualizations", help="Output directory")
    parser.add_argument("--max_samples", type=int, default=122, help="Number of validation frames to visualize")
    parser.add_argument("--thresh", type=float, default=0.30, help="Confidence threshold")
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
    print(f"✅ Loaded checkpoint from {args.checkpoint}")

    val_dataset = MacroPatchLandmarkDataset(
        dataset_dir=args.dataset_dir,
        mode="val",
        image_size=512,
        macro_patch_size=macro_patch_size,
        use_depth=use_depth
    )

    print(f"🎨 Generating 4-panel diagnostic visualizations for up to {min(args.max_samples, len(val_dataset))} frames...")

    count = 0
    with torch.no_grad():
        for idx in range(min(args.max_samples, len(val_dataset))):
            sample = val_dataset[idx]
            img = sample["image"].unsqueeze(0).to(device)
            v_eval_masks = sample["target_masks"].numpy()
            img_path = sample["img_path"]
            frame_name = os.path.basename(img_path)

            preds = model(img)
            macro_logits_np = preds["macro_logits"][0].cpu().numpy()
            macro_beziers_np = preds["macro_beziers"][0].cpu().numpy()

            render_res = merge_macro_beziers_to_image(
                macro_logits_np,
                macro_beziers_np,
                macro_patch_size=macro_patch_size,
                image_size=512,
                confidence_thresh=args.thresh
            )

            dice = compute_batch_dice(render_res["combined_mask"], v_eval_masks.sum(axis=0) > 0.1)

            img_rgb_norm = sample["image"][:3].permute(1, 2, 0).numpy()
            depth_norm = sample["image"][3].numpy() if use_depth else np.zeros((512, 512), dtype=np.float32)

            fig = generate_diagnostic_figure(
                img_rgb_norm,
                depth_norm,
                render_res,
                v_eval_masks,
                frame_name,
                dice,
                macro_patch_size=macro_patch_size
            )

            out_path = os.path.join(args.output_dir, f"{os.path.splitext(frame_name)[0]}_macro_diag.png")
            fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
            plt.close(fig)
            count += 1

            if count % 10 == 0 or count == min(args.max_samples, len(val_dataset)):
                print(f"  🖼️ Rendered {count}/{min(args.max_samples, len(val_dataset))} figures...")

    print(f"\n✅ All diagnostic figures successfully saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
