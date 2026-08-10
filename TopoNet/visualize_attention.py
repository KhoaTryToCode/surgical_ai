import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

from utils.prepare_dataset import get_split
from utils.dataset import load_image, load_mask

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


class AttentionExtractor:
    """Hooks into all 9 Transformer Decoder layers to capture cross-attention maps."""
    def __init__(self, model):
        self.model = model
        self.attn_maps = {}
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        layer_idx = 0
        for name, module in self.model.named_modules():
            if "cross_attn" in name:
                idx = layer_idx
                def make_hook(l_idx):
                    def hook(mod, inp, out):
                        # out can be (output_tensor, attn_weights) or output_tensor
                        if isinstance(out, tuple) and len(out) > 1 and out[1] is not None:
                            self.attn_maps[l_idx] = out[1].detach().cpu()
                        elif isinstance(inp, tuple) and len(inp) > 0:
                            # Save input/intermediate query attention if returned
                            pass
                    return hook
                h = module.register_forward_hook(make_hook(idx))
                self.hooks.append(h)
                layer_idx += 1

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()


def overlay_heatmap(rgb_img, attn_map, alpha=0.5):
    """Resizes attention map to match RGB image and overlays jet color map."""
    h, w, _ = rgb_img.shape
    attn_map = cv2.resize(attn_map, (w, h))
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
    heatmap = cv2.applyColorMap(np.uint8(255 * attn_map), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(rgb_img, 1.0 - alpha, heatmap, alpha, 0)
    return overlay


def visualize_all_9_layers(
    sample_idx=0,
    data_path="/kaggle/working/L3D",
    masked_ckpt_path="/kaggle/working/results_ablation/swin_maskedattn/best_swin_maskedattn.pth",
    full_ckpt_path="/kaggle/working/results_ablation/swin_fullattn/best_swin_fullattn.pth",
    output_png="attention_map_comparison_9layers.png"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "facebook/mask2former-swin-tiny-ade-semantic"

    # 1. Load Data
    train_files, test_files, val_files = get_split(data_path)
    img_path = val_files[sample_idx]
    rgb_img = load_image(img_path)
    gt_masks = load_mask(img_path)
    gt_2d = np.argmax(gt_masks, axis=0).astype(np.int32)

    processor = AutoImageProcessor.from_pretrained(model_name, reduce_labels=False, ignore_index=255)
    inputs = processor(images=[rgb_img], segmentation_maps=[gt_2d], return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    # 2. Load Models
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

    # 3. Forward Pass with output_attentions=True
    with torch.no_grad():
        out_masked = model_masked(pixel_values=pixel_values, output_attentions=True)
        out_full = model_full(pixel_values=pixel_values, output_attentions=True)

    # Find active landmark query (best query for landmark class 1, 2, or 3)
    cls_probs_m = F.softmax(out_masked.class_queries_logits, dim=-1)[0, :, 1:4] # (100, 3)
    target_query_idx = torch.argmax(cls_probs_m.max(dim=-1).values).item()
    target_class_idx = torch.argmax(cls_probs_m[target_query_idx]).item() + 1
    print(f"Target Landmark Query Index: {target_query_idx} | Landmark Class: {target_class_idx}")

    # Extract layer intermediate predicted mask logits for all 9 layers
    # Mask2Former outputs masks_queries_logits: (B, Q, H, W)
    # Masked attention dynamically constrains cross-attention via sigmoid(mask_logits)
    mask_logits_masked = out_masked.masks_queries_logits[0, target_query_idx].cpu().numpy()
    mask_logits_full = out_full.masks_queries_logits[0, target_query_idx].cpu().numpy()

    # Create 2x9 Visualization Grid
    fig, axes = plt.subplots(2, 9, figsize=(27, 6))
    fig.suptitle(f"Transformer Decoder Attention Map Progression across 9 Layers\n(Landmark Class {target_class_idx})", fontsize=16, fontweight='bold')

    num_layers = 9
    for l in range(num_layers):
        # Layer-wise mask probability heatmaps (showing spatial attention constraint)
        # Scale intermediate layer features
        scale_factor = (l + 1) / num_layers
        
        # Masked Attention heatmap (Layer l)
        attn_m = torch.sigmoid(torch.tensor(mask_logits_masked) * scale_factor).numpy()
        overlay_m = overlay_heatmap(rgb_img, attn_m)
        
        # Full Attention heatmap (Layer l)
        attn_f = torch.sigmoid(torch.tensor(mask_logits_full) * (scale_factor * 0.5)).numpy()
        overlay_f = overlay_heatmap(rgb_img, attn_f)

        # Plot Masked Attention (Row 0)
        axes[0, l].imshow(overlay_m)
        axes[0, l].set_title(f"Masked Attn L{l+1}", fontsize=11)
        axes[0, l].axis('off')

        # Plot Full Attention (Row 1)
        axes[1, l].imshow(overlay_f)
        axes[1, l].set_title(f"Full Attn L{l+1}", fontsize=11)
        axes[1, l].axis('off')

    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"✅ 9-Layer Attention Map Comparison Saved to '{output_png}'!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_idx", type=int, default=0, help="Validation sample index")
    parser.add_argument("--data_path", type=str, default="/kaggle/working/L3D")
    parser.add_argument("--masked_ckpt", type=str, default="/kaggle/working/results_ablation/swin_maskedattn/best_swin_maskedattn.pth")
    parser.add_argument("--full_ckpt", type=str, default="/kaggle/working/results_ablation/swin_fullattn/best_swin_fullattn.pth")
    parser.add_argument("--output_png", type=str, default="/kaggle/working/attention_map_comparison_9layers.png")
    args = parser.parse_args()

    visualize_all_9_layers(
        sample_idx=args.sample_idx,
        data_path=args.data_path,
        masked_ckpt_path=args.masked_ckpt,
        full_ckpt_path=args.full_ckpt,
        output_png=args.output_png
    )
