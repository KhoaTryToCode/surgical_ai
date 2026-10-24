import cv2
import numpy as np
import torch
try:
    from models.bezier_utils import get_bernstein_matrix_numpy, sample_cubic_bezier_numpy
except ImportError:
    from .bezier_utils import get_bernstein_matrix_numpy, sample_cubic_bezier_numpy


CLASS_COLORS_BGR = {
    1: (255, 100, 0),    # Ridge (Cyan / Blueish)
    2: (0, 255, 255),    # Silhouette (Yellow)
    3: (0, 255, 0),      # Ligament (Green)
    4: (255, 0, 255)     # Gallbladder (Magenta)
}

CLASS_COLORS_RGB = {
    1: (0, 100, 255),    # Ridge
    2: (255, 255, 0),    # Silhouette
    3: (0, 255, 0),      # Ligament
    4: (255, 0, 255)     # Gallbladder
}


def merge_patch_beziers_to_image(
    patch_classes: np.ndarray,
    patch_beziers: np.ndarray,
    patch_size: int = 16,
    img_size: int = 512,
    threshold: float = 0.25,
    stroke_thickness: int = 2,
    num_samples_per_curve: int = 12,
    return_class_masks: bool = False,
    top_k_fallback: int = 0
):
    """
    Merges local patch Bézier predictions into a high-resolution landmark image
    using Global Coordinate Shift + Anti-Aliased Vector Rasterization.
    
    Args:
        patch_classes: (G, G) class IDs in 0..num_classes, OR (G, G, C+1) probability distribution.
        patch_beziers: (G, G, 4, 2) local control points in [0, 1]^2.
        patch_size: 16 pixels per patch.
        img_size: 512 pixels.
        threshold: Minimum probability threshold for active patches (default: 0.25).
        stroke_thickness: Rasterized line width in pixels.
        num_samples_per_curve: Number of continuous interpolation points along each cubic Bézier arc.
        return_class_masks: If True, also returns (C, H, W) binary masks per landmark class.
        top_k_fallback: If active count is 0, activate top-K most confident patches (useful for early epoch diagnostics).
        
    Returns:
        canvas_rgb: (img_size, img_size, 3) uint8 RGB image of rendered curves.
        (optional) class_masks: (num_classes, img_size, img_size) float32 binary masks.
    """
    canvas_rgb = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    
    # Check if probabilities or discrete classes
    if patch_classes.ndim == 3:
        # (G, G, C+1)
        probs = patch_classes
        cls_map = np.argmax(probs, axis=-1)  # (G, G)
        conf_map = np.max(probs[:, :, 1:], axis=-1) if probs.shape[-1] > 1 else np.zeros_like(cls_map)
        active = (cls_map > 0) & (conf_map >= threshold)
        
        # If no patch exceeds threshold (e.g. early training), activate top-k most confident patches
        if active.sum() == 0 and top_k_fallback > 0:
            flat_conf = conf_map.flatten()
            sorted_indices = np.argsort(flat_conf)[-top_k_fallback:]
            # Filter out near-zero noise
            valid_indices = [idx for idx in sorted_indices if flat_conf[idx] > 0.05]
            if len(valid_indices) > 0:
                active_flat = np.zeros_like(flat_conf, dtype=bool)
                active_flat[valid_indices] = True
                active = active_flat.reshape(cls_map.shape)
    else:
        cls_map = patch_classes.astype(np.int32)
        active = (cls_map > 0)

    grid_h, grid_w = cls_map.shape[:2]
    M_basis = get_bernstein_matrix_numpy(num_samples_per_curve)  # (S, 4)
    
    num_classes = 4
    if return_class_masks:
        class_masks = np.zeros((num_classes, img_size, img_size), dtype=np.float32)

    for r in range(grid_h):
        for c in range(grid_w):
            if not active[r, c]:
                continue
                
            cls_id = int(cls_map[r, c])
            local_ctrl = patch_beziers[r, c]  # (4, 2) in [0, 1]^2
            
            # 1. Global Coordinate Shift
            offset = np.array([c * patch_size, r * patch_size], dtype=np.float32)
            global_ctrl = offset + local_ctrl * float(patch_size)  # (4, 2) in [0, 512]
            
            # 2. Evaluate Continuous Cubic Bézier Arc Points
            curve_pts = (M_basis @ global_ctrl)  # (S, 2)
            curve_pts_int = np.clip(np.round(curve_pts), 0, img_size - 1).astype(np.int32).reshape((-1, 1, 2))
            
            # 3. Draw Anti-Aliased Line Stroke
            color = CLASS_COLORS_RGB.get(cls_id, (255, 255, 255))
            cv2.polylines(canvas_rgb, [curve_pts_int], isClosed=False, color=color, thickness=stroke_thickness, lineType=cv2.LINE_AA)
            
            if return_class_masks and (1 <= cls_id <= num_classes):
                cv2.polylines(class_masks[cls_id - 1], [curve_pts_int], isClosed=False, color=1.0, thickness=stroke_thickness, lineType=cv2.LINE_AA)

    if return_class_masks:
        return canvas_rgb, class_masks
    return canvas_rgb


def extract_global_beziers(
    patch_classes: np.ndarray,
    patch_beziers: np.ndarray,
    patch_size: int = 16,
    threshold: float = 0.5
) -> list:
    """
    Extracts all active predicted Bézier curves in global image pixel coordinates.
    Useful for SVG export, TopoNet metric evaluation, and Chamfer distance calculation.
    """
    if patch_classes.ndim == 3:
        cls_map = np.argmax(patch_classes, axis=-1)
        conf_map = np.max(patch_classes[:, :, 1:], axis=-1)
        active = (cls_map > 0) & (conf_map >= threshold)
    else:
        cls_map = patch_classes.astype(np.int32)
        active = (cls_map > 0)
        
    grid_h, grid_w = cls_map.shape[:2]
    curves = []
    
    for r in range(grid_h):
        for c in range(grid_w):
            if not active[r, c]:
                continue
            cls_id = int(cls_map[r, c])
            local_ctrl = patch_beziers[r, c]
            offset = np.array([c * patch_size, r * patch_size], dtype=np.float32)
            global_ctrl = offset + local_ctrl * float(patch_size)
            
            curves.append({
                "patch_row": r,
                "patch_col": c,
                "class_id": cls_id,
                "control_points_global": global_ctrl
            })
            
    return curves


def batch_vector_to_pixel_masks(
    patch_logits: torch.Tensor,
    patch_beziers: torch.Tensor,
    patch_size: int = 16,
    img_size: int = 512,
    threshold: float = 0.5,
    stroke_thickness: int = 2,
    num_classes: int = 4
) -> np.ndarray:
    """
    Converts a batch of patch Bézier vectors to multi-class pixel raster masks.
    
    Args:
        patch_logits: (B, G, G, C+1) or (B, num_patches, C+1) torch.Tensor
        patch_beziers: (B, G, G, 4, 2) or (B, num_patches, 4, 2) torch.Tensor
        patch_size: pixels per patch (e.g. 16)
        img_size: image resolution (e.g. 512)
        threshold: confidence threshold for landmark activation
        stroke_thickness: line thickness in pixels (e.g. 2)
        num_classes: number of landmark classes (4)
        
    Returns:
        batch_masks: (B, num_classes, img_size, img_size) float32 numpy array
    """
    B = patch_logits.shape[0]
    grid_size = img_size // patch_size
    
    if patch_logits.ndim == 3:
        patch_logits = patch_logits.view(B, grid_size, grid_size, -1)
    if patch_beziers.ndim == 4 and patch_beziers.shape[1] != grid_size:
        patch_beziers = patch_beziers.view(B, grid_size, grid_size, 4, 2)
        
    probs = torch.softmax(patch_logits, dim=-1).detach().cpu().numpy()
    beziers = patch_beziers.detach().cpu().numpy()
    
    batch_masks = np.zeros((B, num_classes, img_size, img_size), dtype=np.float32)
    for b in range(B):
        _, class_masks = merge_patch_beziers_to_image(
            patch_classes=probs[b],
            patch_beziers=beziers[b],
            patch_size=patch_size,
            img_size=img_size,
            threshold=threshold,
            stroke_thickness=stroke_thickness,
            return_class_masks=True
        )
        batch_masks[b] = class_masks
        
    return batch_masks


def compute_batch_dice(
    pred_masks: np.ndarray,
    target_masks: np.ndarray,
    eps: float = 1e-6,
    active_only: bool = True
) -> dict:
    """
    Computes Dice similarity metrics between predicted and target pixel masks.
    
    Args:
        pred_masks: (B, C, H, W) numpy array
        target_masks: (B, C, H, W) numpy array or torch.Tensor
        eps: numerical epsilon
        active_only: compute per-class dice only on classes present in GT or Pred
        
    Returns:
        dict with:
            binary_dice: float (all landmark classes combined)
            macro_class_dice: float (macro average across landmark classes)
            class_dice: dict of {1: ridge_dice, 2: sil_dice, 3: lig_dice, 4: gall_dice}
    """
    if isinstance(target_masks, torch.Tensor):
        target_masks = target_masks.detach().cpu().numpy()
        
    B, C, H, W = pred_masks.shape
    sample_binary_dices = []
    class_dices = {c: [] for c in range(1, C + 1)}
    
    for b in range(B):
        # 1. Binary Dice (any landmark vs background)
        pred_bin = (pred_masks[b].sum(axis=0) > 0).astype(np.float32)
        tgt_bin = (target_masks[b].sum(axis=0) > 0).astype(np.float32)
        
        inter_bin = np.sum(pred_bin * tgt_bin)
        total_bin = np.sum(pred_bin) + np.sum(tgt_bin)
        
        if total_bin < eps:
            sample_binary_dices.append(1.0)
        else:
            sample_binary_dices.append(float((2.0 * inter_bin + eps) / (total_bin + eps)))
            
        # 2. Per-class Dice
        for c in range(C):
            cls_id = c + 1
            p_c = pred_masks[b, c]
            t_c = target_masks[b, c]
            total_c = np.sum(p_c) + np.sum(t_c)
            if total_c < eps:
                if not active_only:
                    class_dices[cls_id].append(1.0)
            else:
                inter_c = np.sum(p_c * t_c)
                dice_c = float((2.0 * inter_c + eps) / (total_c + eps))
                class_dices[cls_id].append(dice_c)
                
    mean_binary_dice = float(np.mean(sample_binary_dices)) if sample_binary_dices else 0.0
    mean_class_scores = [np.mean(v) for v in class_dices.values() if len(v) > 0]
    macro_class_dice = float(np.mean(mean_class_scores)) if mean_class_scores else mean_binary_dice
    
    return {
        "binary_dice": mean_binary_dice,
        "macro_class_dice": macro_class_dice,
        "class_dice": {k: (float(np.mean(v)) if v else 0.0) for k, v in class_dices.items()}
    }


def render_epoch_diagnostic_figure(
    img_tensor: torch.Tensor,
    pred_logits: torch.Tensor,
    pred_beziers: torch.Tensor,
    tgt_classes: torch.Tensor,
    tgt_beziers: torch.Tensor,
    epoch: int,
    patch_size: int = 16,
    img_size: int = 512,
    threshold: float = 0.25,
    top_k_fallback: int = 30
) -> np.ndarray:
    """
    Renders a 3-panel side-by-side diagnostic image (uint8 RGB):
    [Input Surgical Frame] | [Ground Truth Béziers] | [Model Prediction Overlay]
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    # 1. Denormalize input frame (use first 3 RGB channels)
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img_rgb = img_tensor[:3].detach().cpu().numpy() * std + mean
    img_rgb = np.clip(img_rgb.transpose(1, 2, 0), 0.0, 1.0)
    
    # 2. Render Ground Truth Canvas
    gt_canvas = merge_patch_beziers_to_image(
        patch_classes=tgt_classes.detach().cpu().numpy(),
        patch_beziers=tgt_beziers.detach().cpu().numpy(),
        patch_size=patch_size,
        img_size=img_size,
        stroke_thickness=2
    )
    
    # 3. Render Model Prediction Canvas
    # Reshape if flat
    grid_size = img_size // patch_size
    if pred_logits.ndim == 3 and pred_logits.shape[1] != grid_size:
        pred_logits = pred_logits.view(1, grid_size, grid_size, -1)
    if pred_beziers.ndim == 4 and pred_beziers.shape[1] != grid_size:
        pred_beziers = pred_beziers.view(1, grid_size, grid_size, 4, 2)
        
    probs = torch.softmax(pred_logits, dim=-1).detach().cpu().numpy().squeeze(0)  # (G, G, C+1)
    beziers = pred_beziers.detach().cpu().numpy().squeeze(0)                      # (G, G, 4, 2)
    
    max_conf = float(np.max(probs[:, :, 1:]))
    cls_map = np.argmax(probs, axis=-1)
    active_thresh = int(np.sum((cls_map > 0) & (np.max(probs[:, :, 1:], axis=-1) >= threshold)))
    
    pred_canvas = merge_patch_beziers_to_image(
        patch_classes=probs,
        patch_beziers=beziers,
        patch_size=patch_size,
        img_size=img_size,
        threshold=threshold,
        stroke_thickness=2,
        top_k_fallback=top_k_fallback
    )
    
    # 4. Create Overlays
    gt_overlay = img_rgb.copy()
    gt_mask = gt_canvas.sum(axis=-1) > 0
    gt_overlay[gt_mask] = gt_canvas[gt_mask] / 255.0 * 0.8 + gt_overlay[gt_mask] * 0.2
    
    pred_overlay = img_rgb.copy()
    pred_mask = pred_canvas.sum(axis=-1) > 0
    num_pred_px = int(pred_mask.sum())
    pred_overlay[pred_mask] = pred_canvas[pred_mask] / 255.0 * 0.8 + pred_overlay[pred_mask] * 0.2
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(img_rgb)
    axes[0].set_title("Input Surgical Frame", fontsize=12)
    axes[0].axis("off")
    
    axes[1].imshow(gt_overlay)
    axes[1].set_title("Ground Truth Béziers", fontsize=12)
    axes[1].axis("off")
    
    status_text = f"Pred Epoch {epoch:02d} | MaxConf: {max_conf:.3f} | Active>Thresh: {active_thresh} | RenderedPx: {num_pred_px}"
    axes[2].imshow(pred_overlay)
    axes[2].set_title(status_text, fontsize=12)
    axes[2].axis("off")
    
    plt.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    rgb_out = rgba[:, :, :3].copy()
    plt.close(fig)
    
    return rgb_out
