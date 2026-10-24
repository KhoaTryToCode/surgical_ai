import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch

# Ensure experiment root is in sys.path
exp_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if exp_root not in sys.path:
    sys.path.insert(0, exp_root)

from configs.exp09_config import config
from models.patch_vector_vit import PatchBezierViT
from models.patch_losses import PatchBezierLoss
from models.patch_merger import merge_patch_beziers_to_image
from utils.dataset_patch_vit import PatchBezierLandmarkDataset


def run_visual_smoke_test():
    print("=" * 70)
    print("🚀 [EXP_09] Running Patch-Bézier ViT Visual Smoke Test...")
    print("=" * 70)

    # 1. Dataset Loading
    dataset = PatchBezierLandmarkDataset(
        dataset_dir=config.dataset_dir,
        mode="train",
        image_size=config.image_size,
        patch_size=config.patch_size,
        spline_step_px=config.spline_step_px,
        stroke_thickness=config.stroke_thickness
    )
    
    sample = dataset[0]
    img_tensor = sample["image"]          # (3, 512, 512)
    tgt_classes = sample["target_classes"] # (32, 32)
    tgt_beziers = sample["target_beziers"] # (32, 32, 4, 2)
    act_mask = sample["active_mask"]       # (32, 32)
    tgt_masks = sample["target_masks"]     # (4, 512, 512)
    
    num_active = act_mask.sum().item()
    print(f"✅ Loaded sample. Active landmark patches: {num_active} / 1024 ({num_active/1024*100:.1f}%)")
    
    # 2. Denormalize Image for Visualization (use first 3 RGB channels)
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img_np = img_tensor[:3].numpy() * std + mean
    img_np = np.clip(img_np.transpose(1, 2, 0), 0.0, 1.0)
    
    # 3. Merging: Reconstruct Final Image from Ground Truth Patch Béziers
    merged_gt_rgb = merge_patch_beziers_to_image(
        patch_classes=tgt_classes.numpy(),
        patch_beziers=tgt_beziers.numpy(),
        patch_size=config.patch_size,
        img_size=config.image_size,
        stroke_thickness=config.stroke_thickness
    )
    print(f"✅ Ground truth Bézier curves successfully merged into (512, 512, 3) canvas.")
    
    # 4. Model Forward & Backward Pass Verification
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"⚙️ Testing PatchBezierViT (in_chans={config.in_chans}) forward & backward on device: {device}...")
    
    model = PatchBezierViT(
        backbone_name=config.backbone_name,
        in_chans=config.in_chans,
        pretrained=False,  # Use offline initialized weights for smoke test
        image_size=config.image_size,
        patch_size=config.patch_size,
        num_classes=config.num_classes,
        embed_dim=config.embed_dim
    ).to(device)
    
    batch_img = img_tensor.unsqueeze(0).to(device)  # (1, 4, 512, 512)
    pred_dict = model(batch_img)
    
    print(f"✅ Forward pass complete:")
    print(f"   - patch_logits:  {pred_dict['patch_logits'].shape}")
    print(f"   - patch_beziers: {pred_dict['patch_beziers'].shape}")
    
    criterion = PatchBezierLoss(
        lambda_cls=config.lambda_cls,
        lambda_ctrl=config.lambda_ctrl,
        lambda_sample=config.lambda_sample,
        lambda_tan=config.lambda_tan,
        lambda_cont=config.lambda_cont
    )
    
    loss_dict = criterion(
        pred_dict=pred_dict,
        target_classes=tgt_classes.unsqueeze(0).to(device),
        target_beziers=tgt_beziers.unsqueeze(0).to(device),
        active_mask=act_mask.unsqueeze(0).to(device)
    )
    
    total_loss = loss_dict["loss"]
    total_loss.backward()
    print(f"✅ Loss computed & backward pass passed:")
    print(f"   - Total Loss:   {total_loss.item():.4f}")
    print(f"   - Loss Cls:     {loss_dict['loss_cls'].item():.4f}")
    print(f"   - Loss Ctrl:    {loss_dict['loss_ctrl'].item():.4f}")
    print(f"   - Loss Sample:  {loss_dict['loss_sample'].item():.4f}")
    print(f"   - Loss Tan:     {loss_dict['loss_tan'].item():.4f}")
    print(f"   - Loss Cont:    {loss_dict['loss_cont'].item():.4f}")
    
    # 5. Build Multi-Panel Diagnostic Plot
    output_dir = os.path.join(exp_root, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    out_fig_path = os.path.join(output_dir, "smoke_test_patch_vit.png")
    
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    
    # Panel 1: Original Image
    axes[0].imshow(img_np)
    axes[0].set_title(f"Panel 1: Input Surgical Frame ({config.image_size}×{config.image_size})", fontsize=12)
    axes[0].axis("off")
    
    # Panel 2: Patch Grid Overlay & Bézier Control Points
    axes[1].imshow(img_np)
    P = config.patch_size
    # Draw faint patch grid lines every 16 px
    for g in range(0, config.image_size + 1, P * 4):  # Every 4 patches for visual clarity
        axes[1].axvline(g, color='white', alpha=0.15, linewidth=0.5)
        axes[1].axhline(g, color='white', alpha=0.15, linewidth=0.5)
        
    tgt_b_np = tgt_beziers.numpy()
    act_np = act_mask.numpy()
    for r in range(config.grid_size):
        for c in range(config.grid_size):
            if act_np[r, c]:
                # Draw active patch bounding box
                rect = plt.Rectangle((c * P, r * P), P, P, fill=True, color='red', alpha=0.25)
                axes[1].add_patch(rect)
                
                # Global control points
                ctrl_local = tgt_b_np[r, c]  # (4, 2)
                ctrl_global = np.array([c * P, r * P]) + ctrl_local * float(P)
                # Plot P0 and P3 (endpoints) in lime, P1 and P2 (handles) in orange
                axes[1].plot(ctrl_global[[0, 3], 0], ctrl_global[[0, 3], 1], 'o', color='lime', markersize=3)
                axes[1].plot(ctrl_global[[1, 2], 0], ctrl_global[[1, 2], 1], 'x', color='orange', markersize=3)
                axes[1].plot(ctrl_global[:, 0], ctrl_global[:, 1], ':', color='yellow', linewidth=0.8, alpha=0.7)
                
    axes[1].set_title(f"Panel 2: Active 16×16 Patches ({num_active}) & Bézier Handles", fontsize=12)
    axes[1].axis("off")
    
    # Panel 3: Merged Final Image
    axes[2].imshow(merged_gt_rgb)
    axes[2].set_title("Panel 3: Reconstructed Landmark Image (Merged Béziers)", fontsize=12)
    axes[2].axis("off")
    
    # Panel 4: Surgical Image + Merged Bézier Overlay
    overlay = img_np.copy()
    mask_bool = merged_gt_rgb.sum(axis=-1) > 0
    overlay[mask_bool] = merged_gt_rgb[mask_bool] / 255.0 * 0.8 + overlay[mask_bool] * 0.2
    axes[3].imshow(overlay)
    axes[3].set_title("Panel 4: High-Precision Anatomical Overlay", fontsize=12)
    axes[3].axis("off")
    
    plt.tight_layout()
    plt.savefig(out_fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"🎉 [EXP_09] Visual Smoke Test succeeded! Saved figure to:\n   {out_fig_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_visual_smoke_test()
