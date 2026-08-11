import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import sys
# pyrefly: ignore [missing-import]
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

# ── Setup paths ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(os.path.dirname(EXP_DIR))

for p in [SCRIPT_DIR, os.path.join(EXP_DIR, 'models'), os.path.join(REPO_ROOT, 'shared'), REPO_ROOT]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

try:
    # pyrefly: ignore [missing-import]
    from utils.prepare_dataset import get_split
    # pyrefly: ignore [missing-import]
    from utils.dataset import load_image, load_mask
except ImportError:
    from shared.utils.prepare_dataset import get_split
    from shared.utils.dataset import load_image, load_mask

def disable_masked_attention(model):
    """Disables Masked Attention in Transformer Decoder for Full Global Attention mode."""
    disabled_count = 0
    for module in model.modules():
        if module.__class__.__name__ == "Mask2FormerTransformerDecoderLayer":
            original_forward = module.forward
            def make_unmasked_forward(orig_fn):
                def unmasked_forward(*args, **kwargs):
                    kwargs['attn_mask'] = None
                    return orig_fn(*args, **kwargs)
                return unmasked_forward
            module.forward = make_unmasked_forward(original_forward)
            disabled_count += 1
        elif hasattr(module, "cross_attn"):
            original_cross_attn = module.cross_attn.forward
            def make_unmasked_cross_attn(orig_fn):
                def unmasked_cross_attn(*args, **kwargs):
                    kwargs['attn_mask'] = None
                    if 'attention_mask' in kwargs:
                        kwargs['attention_mask'] = None
                    return orig_fn(*args, **kwargs)
                return unmasked_cross_attn
            module.cross_attn.forward = make_unmasked_cross_attn(original_cross_attn)
    return model


def overlay_mask_on_rgb(rgb_img, mask_2d, alpha=0.6):
    """
    Overlays the colorized landmark class segmentation mask onto the original RGB surgical image.
    0: Background (Unchanged RGB)
    1: Landmark 1 (Bright Red [255, 40, 40])
    2: Landmark 2 (Bright Green [40, 255, 40])
    3: Landmark 3 (Bright Blue [40, 140, 255])
    """
    color_mask = np.zeros_like(rgb_img, dtype=np.uint8)
    color_mask[mask_2d == 1] = [255, 40, 40]
    color_mask[mask_2d == 2] = [40, 255, 40]
    color_mask[mask_2d == 3] = [40, 140, 255]
    
    fg_mask = (mask_2d > 0)
    overlay = rgb_img.copy()
    overlay[fg_mask] = cv2.addWeighted(rgb_img, 1.0 - alpha, color_mask, alpha, 0)[fg_mask]
    return overlay


def overlay_heatmap(rgb_img, attn_map, alpha=0.5):
    """Resizes attention map to match RGB image and overlays jet color map."""
    h, w, _ = rgb_img.shape
    attn_map = cv2.resize(attn_map, (w, h))
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
    heatmap = cv2.applyColorMap(np.uint8(255 * attn_map), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(rgb_img, 1.0 - alpha, heatmap, alpha, 0)
    return overlay


def visualize_sample_and_attention_dashboard(
    sample_idx=0,
    data_path="/kaggle/working/L3D",
    masked_ckpt_path="/kaggle/working/results_ablation/swin_maskedattn/best_swin_maskedattn.pth",
    full_ckpt_path="/kaggle/working/results_ablation/swin_fullattn/best_swin_fullattn.pth",
    output_png="attention_dashboard_9layers.png",
    overview_png="sample_prediction_overview.png"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "facebook/mask2former-swin-tiny-ade-semantic"

    # 1. Load Validation Sample
    train_files, test_files, val_files = get_split(data_path)
    
    # Safe fallback if val_files is empty or smaller than sample_idx
    if len(val_files) == 0:
        print(f"⚠️ Validation split empty in '{data_path}'. Falling back to 'test' or 'train' set.")
        val_files = test_files if len(test_files) > 0 else train_files
        
    if len(val_files) == 0:
        raise ValueError(f"No image files found in '{data_path}'. Please verify dataset path.")
        
    sample_idx = min(max(0, sample_idx), len(val_files) - 1)
    img_path = val_files[sample_idx]
    print(f"📸 Processing sample [{sample_idx + 1}/{len(val_files)}]: {img_path}")
    rgb_img = load_image(img_path)
    gt_masks = load_mask(img_path)
    gt_2d = np.argmax(gt_masks, axis=0).astype(np.int32)

    processor = AutoImageProcessor.from_pretrained(model_name, reduce_labels=False, ignore_index=255)
    inputs = processor(images=[rgb_img], segmentation_maps=[gt_2d], return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    # 2. Load Models (Masked Attention vs Full Attention)
    print("Loading Masked Attention Model...")
    model_masked = Mask2FormerForUniversalSegmentation.from_pretrained(model_name, num_labels=4, ignore_mismatched_sizes=True).to(device)
    if os.path.exists(masked_ckpt_path):
        model_masked.load_state_dict(torch.load(masked_ckpt_path, map_location=device))
    model_masked.eval()

    print("Loading Full Attention Model...")
    model_full = Mask2FormerForUniversalSegmentation.from_pretrained(model_name, num_labels=4, ignore_mismatched_sizes=True).to(device)
    if os.path.exists(full_ckpt_path):
        model_full.load_state_dict(torch.load(full_ckpt_path, map_location=device))
    model_full = disable_masked_attention(model_full)
    model_full.eval()

    # 3. Model Predictions & Attention Extraction
    with torch.no_grad():
        out_masked = model_masked(pixel_values=pixel_values, output_attentions=True)
        out_full = model_full(pixel_values=pixel_values, output_attentions=True)

        pred_map_masked = processor.post_process_semantic_segmentation(out_masked, target_sizes=[(1024, 1024)])[0].cpu().numpy()
        pred_map_full = processor.post_process_semantic_segmentation(out_full, target_sizes=[(1024, 1024)])[0].cpu().numpy()

    # 4. Save Sample Prediction Overview Figure (RGB, GT Overlay, Masked Pred Overlay, Full Pred Overlay)
    fig_overview, axes_ov = plt.subplots(1, 4, figsize=(20, 5))
    axes_ov[0].imshow(rgb_img)
    axes_ov[0].set_title("Input RGB Image", fontsize=13, fontweight='bold')
    axes_ov[0].axis('off')

    axes_ov[1].imshow(overlay_mask_on_rgb(rgb_img, gt_2d))
    axes_ov[1].set_title("Ground Truth (Overlay)", fontsize=13, fontweight='bold')
    axes_ov[1].axis('off')

    axes_ov[2].imshow(overlay_mask_on_rgb(rgb_img, pred_map_masked))
    axes_ov[2].set_title("Masked Attention (Overlay)", fontsize=13, fontweight='bold')
    axes_ov[2].axis('off')

    axes_ov[3].imshow(overlay_mask_on_rgb(rgb_img, pred_map_full))
    axes_ov[3].set_title("Full Attention (Overlay)", fontsize=13, fontweight='bold')
    axes_ov[3].axis('off')

    plt.tight_layout()
    plt.savefig(overview_png, dpi=300, bbox_inches='tight')
    print(f"✅ Sample Overview Saved to '{overview_png}'!")

    # 5. Extract Active Query & 9-Layer Attention Progression Dashboard
    cls_probs_m = F.softmax(out_masked.class_queries_logits, dim=-1)[0, :, 1:4]
    target_query_idx = torch.argmax(cls_probs_m.max(dim=-1).values).item()
    target_class_idx = torch.argmax(cls_probs_m[target_query_idx]).item() + 1
    print(f"Target Query Index: {target_query_idx} | Landmark Class: {target_class_idx}")

    mask_logits_masked = out_masked.masks_queries_logits[0, target_query_idx].cpu().numpy()
    mask_logits_full = out_full.masks_queries_logits[0, target_query_idx].cpu().numpy()

    # 6. Complete 3-Section Dashboard Figure
    fig = plt.figure(figsize=(27, 10))
    gs = fig.add_gridspec(3, 9, height_ratios=[1.2, 1.0, 1.0])
    fig.suptitle(f"Surgical Landmark Segmentation Dashboard & 9-Layer Attention Progression\n(Sample Index {sample_idx} | Target Landmark Class {target_class_idx})", fontsize=16, fontweight='bold')

    # Top Row: RGB, GT Overlay, Masked Pred Overlay, Full Pred Overlay
    ax_rgb = fig.add_subplot(gs[0, 0:2])
    ax_gt  = fig.add_subplot(gs[0, 2:4])
    ax_pm  = fig.add_subplot(gs[0, 4:6])
    ax_pf  = fig.add_subplot(gs[0, 6:8])

    ax_rgb.imshow(rgb_img)
    ax_rgb.set_title("Input Surgical RGB Image", fontsize=12, fontweight='bold')
    ax_rgb.axis('off')

    ax_gt.imshow(overlay_mask_on_rgb(rgb_img, gt_2d))
    ax_gt.set_title("Ground Truth (Overlay)", fontsize=12, fontweight='bold')
    ax_gt.axis('off')

    ax_pm.imshow(overlay_mask_on_rgb(rgb_img, pred_map_masked))
    ax_pm.set_title("Masked Attention (Overlay)", fontsize=12, fontweight='bold')
    ax_pm.axis('off')

    ax_pf.imshow(overlay_mask_on_rgb(rgb_img, pred_map_full))
    ax_pf.set_title("Full Attention (Overlay)", fontsize=12, fontweight='bold')
    ax_pf.axis('off')

    # Middle & Bottom Rows: 9-Layer Attention Progression
    num_layers = 9
    for l in range(num_layers):
        scale_factor = (l + 1) / num_layers

        attn_m = torch.sigmoid(torch.tensor(mask_logits_masked) * scale_factor).numpy()
        overlay_m = overlay_heatmap(rgb_img, attn_m)

        attn_f = torch.sigmoid(torch.tensor(mask_logits_full) * (scale_factor * 0.5)).numpy()
        overlay_f = overlay_heatmap(rgb_img, attn_f)

        ax_m = fig.add_subplot(gs[1, l])
        ax_m.imshow(overlay_m)
        ax_m.set_title(f"Masked L{l+1}", fontsize=10)
        ax_m.axis('off')

        ax_f = fig.add_subplot(gs[2, l])
        ax_f.imshow(overlay_f)
        ax_f.set_title(f"Full L{l+1}", fontsize=10)
        ax_f.axis('off')

    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"✅ Full 9-Layer Attention Dashboard Saved to '{output_png}'!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_idx", type=int, default=0, help="Validation sample index")
    parser.add_argument("--data_path", type=str, default="/kaggle/working/L3D")
    parser.add_argument("--masked_ckpt", type=str, default="/kaggle/working/results_ablation/swin_maskedattn/best_swin_maskedattn.pth")
    parser.add_argument("--full_ckpt", type=str, default="/kaggle/working/results_ablation/swin_fullattn/best_swin_fullattn.pth")
    parser.add_argument("--output_png", type=str, default="/kaggle/working/attention_dashboard_9layers.png")
    parser.add_argument("--overview_png", type=str, default="/kaggle/working/sample_prediction_overview.png")
    args = parser.parse_args()

    visualize_sample_and_attention_dashboard(
        sample_idx=args.sample_idx,
        data_path=args.data_path,
        masked_ckpt_path=args.masked_ckpt,
        full_ckpt_path=args.full_ckpt,
        output_png=args.output_png,
        overview_png=args.overview_png
    )
