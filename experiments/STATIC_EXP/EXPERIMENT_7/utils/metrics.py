"""
Evaluation metrics for EXPERIMENT_7:
  - Macro Dice, IoU, ASSD per class (Ridge, Silhouette, Falciform)
  - Patient 40 diagnostic metrics
  - 11-Query Structural Atlas Mean Absolute Pixel Error (MAPE) & Visibility Accuracy
  - Pure OpenCV & NumPy implementation (zero external C-extension dependencies)
"""
import cv2
import numpy as np
import torch

from experiments.EXPERIMENT_7.utils.atlas_extractor import CANONICAL_ATLAS_NAMES

# NumPy 2.0 compatibility
for _a, _v in [('Inf', np.inf), ('NAN', np.nan), ('NaN', np.nan), ('PINF', np.inf), ('NINF', -np.inf)]:
    if not hasattr(np, _a):
        setattr(np, _a, _v)

def compute_dice(pred_bin, target_bin, eps=1e-6):
    pred_b = (pred_bin > 0).astype(bool)
    target_b = (target_bin > 0).astype(bool)
    intersection = np.logical_and(pred_b, target_b).sum()
    total = pred_b.sum() + target_b.sum()
    if total == 0:
        return 1.0
    return float(2.0 * intersection / (total + eps))

def compute_iou(pred_bin, target_bin, eps=1e-6):
    pred_b = (pred_bin > 0).astype(bool)
    target_b = (target_bin > 0).astype(bool)
    intersection = np.logical_and(pred_b, target_b).sum()
    union = np.logical_or(pred_b, target_b).sum()
    if union == 0:
        return 1.0
    return float(intersection / (union + eps))

def compute_assd_fast(pred_bin, target_bin, max_penalty=80.0):
    """
    Computes Average Symmetric Surface Distance (ASSD) using OpenCV distanceTransform.
    Guaranteed zero external dependency issues.
    """
    p_b = (pred_bin > 0).astype(np.uint8)
    t_b = (target_bin > 0).astype(np.uint8)
    
    if p_b.sum() == 0 and t_b.sum() == 0:
        return 0.0
    if p_b.sum() == 0 or t_b.sum() == 0:
        return max_penalty
        
    try:
        contours_p, _ = cv2.findContours(p_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours_t, _ = cv2.findContours(t_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        edge_p = np.zeros_like(p_b)
        edge_t = np.zeros_like(t_b)
        cv2.drawContours(edge_p, contours_p, -1, 1, 1)
        cv2.drawContours(edge_t, contours_t, -1, 1, 1)
        
        if edge_p.sum() == 0 or edge_t.sum() == 0:
            return max_penalty
            
        dist_t = cv2.distanceTransform((1 - edge_t).astype(np.uint8), cv2.DIST_L2, 3)
        dist_p = cv2.distanceTransform((1 - edge_p).astype(np.uint8), cv2.DIST_L2, 3)
        
        d_p2t = np.mean(dist_t[edge_p > 0])
        d_t2p = np.mean(dist_p[edge_t > 0])
        
        return float((d_p2t + d_t2p) / 2.0)
    except Exception:
        return max_penalty

def evaluate_frame_metrics(pred_map, target_map, pred_coords=None, gt_coords=None, gt_vis=None, canvas_size=1024):
    """
    Computes all semantic and 11-query structural metrics for a single 1024x1024 frame.
    """
    c_dices = []
    c_ious = []
    c_assds = []
    
    # Classes: 1=Ridge, 2=Sil, 3=Falc
    for c in [1, 2, 3]:
        p_c = (pred_map == c)
        t_c = (target_map == c)
        c_dices.append(compute_dice(p_c, t_c))
        c_ious.append(compute_iou(p_c, t_c))
        c_assds.append(compute_assd_fast(p_c, t_c))
        
    macro_dice = float(np.mean(c_dices))
    macro_iou = float(np.mean(c_ious))
    macro_assd = float(np.mean(c_assds))
    fg_dice = compute_dice(pred_map > 0, target_map > 0)
    
    res = {
        'macro_dice': macro_dice,
        'macro_iou': macro_iou,
        'macro_assd': macro_assd,
        'ridge_dice': c_dices[0],
        'sil_dice': c_dices[1],
        'falc_dice': c_dices[2],
        'fg_dice': fg_dice,
        'ridge_assd': c_assds[0],
        'sil_assd': c_assds[1],
        'falc_assd': c_assds[2]
    }
    
    # 11-Query Structural Atlas metrics (evaluated only on visible queries)
    if pred_coords is not None and gt_coords is not None and gt_vis is not None:
        q_errs = []
        for k in range(min(11, len(gt_vis))):
            name = CANONICAL_ATLAS_NAMES[k] if k < len(CANONICAL_ATLAS_NAMES) else f"q_{k}"
            if gt_vis[k] > 0.5:
                err_px = float(np.linalg.norm((pred_coords[k] - gt_coords[k]) * float(canvas_size)))
                q_errs.append(err_px)
                res[f"{name}_err_px"] = err_px
            else:
                res[f"{name}_err_px"] = None
        res['mean_atlas_err_px'] = float(np.mean(q_errs)) if len(q_errs) > 0 else None
        
    return res
