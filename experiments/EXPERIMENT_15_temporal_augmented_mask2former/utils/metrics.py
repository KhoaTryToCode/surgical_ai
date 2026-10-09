"""
Evaluation metrics for EXPERIMENT_15:
  - Macro Dice, IoU, ASSD per class (Ridge, Silhouette, Falciform)
  - Temporal Boundary Jitter metric across consecutive clip frames
  - Pure OpenCV & NumPy implementation (zero external C-extension dependencies)
"""
import cv2
import numpy as np

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
            
        dt_target = cv2.distanceTransform(1 - edge_t, cv2.DIST_L2, 3)
        dt_pred = cv2.distanceTransform(1 - edge_p, cv2.DIST_L2, 3)
        
        d_p_to_t = dt_target[edge_p > 0]
        d_t_to_p = dt_pred[edge_t > 0]
        
        assd = float((d_p_to_t.mean() + d_t_to_p.mean()) / 2.0)
        return min(assd, max_penalty)
    except Exception:
        return max_penalty

def evaluate_frame_metrics(pred_map, gt_map, num_classes=3):
    """
    Evaluates 3 surgical classes: 1: Ridge, 2: Silhouette, 3: Falciform Ligament.
    """
    results = {}
    class_names = {1: 'ridge', 2: 'silhouette', 3: 'falciform'}
    
    dice_list, iou_list, assd_list = [], [], []
    
    for c_idx in range(1, num_classes + 1):
        name = class_names[c_idx]
        p_bin = (pred_map == c_idx)
        t_bin = (gt_map == c_idx)
        
        d = compute_dice(p_bin, t_bin)
        iou = compute_iou(p_bin, t_bin)
        assd = compute_assd_fast(p_bin, t_bin)
        
        results[f'dice_{name}'] = d
        results[f'iou_{name}'] = iou
        results[f'assd_{name}'] = assd
        
        dice_list.append(d)
        iou_list.append(iou)
        assd_list.append(assd)
        
    results['macro_dice'] = float(np.mean(dice_list))
    results['macro_iou'] = float(np.mean(iou_list))
    results['macro_assd'] = float(np.mean(assd_list))
    return results

def compute_clip_jitter(pred_maps_clip):
    """
    Computes average boundary difference between consecutive predictions in a clip (T >= 2).
    """
    T = len(pred_maps_clip)
    if T < 2:
        return 0.0
    jitters = []
    for t in range(1, T):
        p_curr = (pred_maps_clip[t] > 0).astype(np.uint8)
        p_prev = (pred_maps_clip[t-1] > 0).astype(np.uint8)
        jitters.append(compute_assd_fast(p_curr, p_prev, max_penalty=50.0))
    return float(np.mean(jitters))
