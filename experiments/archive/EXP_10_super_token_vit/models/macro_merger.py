import numpy as np
import cv2
import torch


def sample_cubic_bezier_np(ctrl_pts: np.ndarray, num_samples: int = 20) -> np.ndarray:
    """
    Evaluates cubic Bézier curve at uniform t in [0, 1] using Bernstein polynomials.
    """
    t = np.linspace(0.0, 1.0, num_samples, dtype=np.float32)
    b0 = ((1.0 - t) ** 3)[:, None]
    b1 = (3.0 * (1.0 - t) ** 2 * t)[:, None]
    b2 = (3.0 * (1.0 - t) * (t ** 2))[:, None]
    b3 = (t ** 3)[:, None]
    curve = b0 * ctrl_pts[0] + b1 * ctrl_pts[1] + b2 * ctrl_pts[2] + b3 * ctrl_pts[3]
    return curve.astype(np.float32)


def merge_macro_beziers_to_image(
    macro_logits_np: np.ndarray,
    macro_beziers_np: np.ndarray,
    macro_patch_size: int = 64,
    image_size: int = 512,
    num_classes: int = 4,
    confidence_thresh: float = 0.30,
    stroke_thickness: int = 2,
    num_samples_per_bezier: int = 20
) -> dict:
    """
    Renders predicted macro-patch Bézier curves into multi-class raster masks.
    
    Args:
        macro_logits_np:  (8, 8, num_classes + 1)
        macro_beziers_np: (8, 8, 4, 2) in local [0, 1]^2
        macro_patch_size: 64 px
        image_size:       512 px
        
    Returns:
        dict containing:
            pixel_masks: (C, 512, 512) binary masks per class
            combined_mask: (512, 512) single binary mask
            active_patches: list of dicts with (r, c, class_id, conf, global_pts)
    """
    grid_h, grid_w = macro_logits_np.shape[:2]
    P = macro_patch_size
    S = image_size
    
    pixel_masks = np.zeros((num_classes, S, S), dtype=np.float32)
    
    # Softmax probabilities
    exp_logits = np.exp(macro_logits_np - np.max(macro_logits_np, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    
    active_patches = []
    
    for r in range(grid_h):
        for c in range(grid_w):
            patch_prob = probs[r, c]
            cls_id = int(np.argmax(patch_prob))
            conf = float(patch_prob[cls_id])
            
            # Skip background (0) or low confidence
            if cls_id == 0 or conf < confidence_thresh:
                continue
                
            local_ctrl = macro_beziers_np[r, c]  # (4, 2) in [0, 1]^2
            
            # Sample dense points along cubic Bézier
            sampled_local = sample_cubic_bezier_np(local_ctrl, num_samples=num_samples_per_bezier)
            
            # Global coordinate shift: c*64 + x*64, r*64 + y*64
            global_pts = np.zeros_like(sampled_local)
            global_pts[:, 0] = c * P + sampled_local[:, 0] * P
            global_pts[:, 1] = r * P + sampled_local[:, 1] * P
            global_pts = np.clip(global_pts, 0, S - 1)
            
            pts_pix = np.round(global_pts).astype(np.int32).reshape((-1, 1, 2))
            
            # Render with anti-aliasing
            class_idx = cls_id - 1
            cv2.polylines(
                pixel_masks[class_idx],
                [pts_pix],
                isClosed=False,
                color=1.0,
                thickness=stroke_thickness,
                lineType=cv2.LINE_AA
            )
            
            active_patches.append({
                "row": r,
                "col": c,
                "class_id": cls_id,
                "confidence": conf,
                "global_pts": global_pts
            })
            
    combined_mask = (pixel_masks.sum(axis=0) > 0.1).astype(np.float32)
    return {
        "pixel_masks": pixel_masks,
        "combined_mask": combined_mask,
        "active_patches": active_patches
    }


def compute_batch_dice(pred_masks_np: np.ndarray, gt_masks_np: np.ndarray, eps: float = 1e-6) -> float:
    """
    Computes macro Dice score between predicted masks and ground truth masks.
    """
    pred_bin = (pred_masks_np > 0.5).astype(np.float32)
    gt_bin = (gt_masks_np > 0.5).astype(np.float32)
    
    intersection = np.sum(pred_bin * gt_bin)
    cardinality = np.sum(pred_bin) + np.sum(gt_bin)
    
    if cardinality == 0:
        return 1.0  # Both empty: true negative
        
    return float((2.0 * intersection + eps) / (cardinality + eps))
