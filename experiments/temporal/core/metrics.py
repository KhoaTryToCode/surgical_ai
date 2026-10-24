"""
Standardized Evaluation Metrics for Laparoscopic Landmark Video Segmentation.
Exact mathematical parity with TopoNet benchmark and EXP 15:
- Mutually exclusive argmax assignment at threshold 0.5
- Macro Dice, Micro Dice, Per-class Dice (Ridge, Silhouette, Ligament)
- ASSD (Average Symmetric Surface Distance) and HD95 (Hausdorff Distance 95%) via EDT
"""

import numpy as np
import cv2
from typing import Dict, Tuple, Union


def compute_dice(pred_bin: np.ndarray, target_bin: np.ndarray) -> float:
    """
    Computes Sørensen-Dice coefficient between binary masks.
    """
    p = (pred_bin > 0).astype(np.float32)
    t = (target_bin > 0).astype(np.float32)
    intersection = float(np.sum(p * t))
    total = float(np.sum(p) + np.sum(t))
    if total == 0:
        return 1.0  # Both empty -> perfect agreement
    if np.sum(p) == 0 or np.sum(t) == 0:
        return 0.0  # One empty, other not
    return float(2.0 * intersection / total)


def compute_iou(pred_bin: np.ndarray, target_bin: np.ndarray) -> float:
    """
    Computes Intersection-over-Union (Jaccard Index).
    """
    p = (pred_bin > 0).astype(np.float32)
    t = (target_bin > 0).astype(np.float32)
    intersection = float(np.sum(p * t))
    union = float(np.sum((p + t) > 0))
    if union == 0:
        return 1.0
    return float(intersection / union)


def compute_assd_hd95(
    pred_bin: np.ndarray,
    target_bin: np.ndarray,
    max_penalty: float = 80.0
) -> Tuple[float, float]:
    """
    Computes Average Symmetric Surface Distance (ASSD) and Hausdorff Distance 95% (HD95)
    using Euclidean Distance Transforms.
    """
    p = (pred_bin > 0).astype(np.uint8)
    t = (target_bin > 0).astype(np.uint8)

    p_sum = np.sum(p)
    t_sum = np.sum(t)

    if p_sum == 0 and t_sum == 0:
        return 0.0, 0.0
    if p_sum == 0 or t_sum == 0:
        return max_penalty, max_penalty

    cnts_p, _ = cv2.findContours(p, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts_t, _ = cv2.findContours(t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    edge_p = np.zeros_like(p)
    edge_t = np.zeros_like(t)

    cv2.drawContours(edge_p, cnts_p, -1, 1, 1)
    cv2.drawContours(edge_t, cnts_t, -1, 1, 1)

    pts_p = edge_p > 0
    pts_t = edge_t > 0

    if not np.any(pts_p) or not np.any(pts_t):
        return max_penalty, max_penalty

    dt_target = cv2.distanceTransform(1 - edge_t, cv2.DIST_L2, 3)
    dt_pred = cv2.distanceTransform(1 - edge_p, cv2.DIST_L2, 3)

    d_p_to_t = dt_target[pts_p]
    d_t_to_p = dt_pred[pts_t]

    if len(d_p_to_t) == 0 or len(d_t_to_p) == 0:
        return max_penalty, max_penalty

    assd = float((np.mean(d_p_to_t) + np.mean(d_t_to_p)) / 2.0)
    all_dists = np.concatenate([d_p_to_t, d_t_to_p])
    hd95 = float(np.percentile(all_dists, 95))

    return assd, hd95


def evaluate_frame_metrics(
    pred_sem_seg: np.ndarray,
    gt_input: np.ndarray,
    num_classes: int = 3,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluates comprehensive metrics for a single laparoscopic frame
    using TopoNet's exact mutually exclusive argmax formulation.

    Args:
        pred_sem_seg: (3, H, W) continuous probability map from model
        gt_input: (H, W) discrete class map in {0, 1, 2, 3} or (3, H, W) one-hot binary masks
        num_classes: number of landmark classes (default: 3)
        threshold: minimum probability required for foreground (default: 0.5)

    Returns:
        res: dictionary containing:
            - micro_dice: TopoNet's exact flattened foreground formula: 2*sum(pred*gt)/(sum(pred)+sum(gt))
            - macro_dice: arithmetic unweighted mean across landmark classes
            - per-class dice, iou, assd, hd95 for Ridge, Silhouette, Ligament
    """
    res = {}
    class_names = ['ridge', 'silhouette', 'ligament']

    # 1. Convert pred_sem_seg to mutually exclusive 2D class map using argmax
    max_prob = np.max(pred_sem_seg, axis=0)
    argmax_cls = np.argmax(pred_sem_seg, axis=0) + 1  # Classes 1, 2, 3
    pred_map = np.where(max_prob >= threshold, argmax_cls, 0).astype(np.uint8)

    # 2. Parse ground truth to discrete 2D class map if needed
    if gt_input.ndim == 3:
        gt_map = np.zeros(gt_input.shape[1:], dtype=np.uint8)
        for c in range(min(num_classes, gt_input.shape[0])):
            gt_map[gt_input[c] > 0.5] = (c + 1)
    else:
        gt_map = gt_input.astype(np.uint8)

    # 3. Micro Dice: TopoNet's flattened foreground formulation
    pred_fg = (pred_map > 0).astype(np.float32)
    gt_fg = (gt_map > 0).astype(np.float32)
    fg_inter = float(np.sum(pred_fg * gt_fg))
    fg_total = float(np.sum(pred_fg) + np.sum(gt_fg))
    res['micro_dice'] = (2.0 * fg_inter / fg_total) * 100.0 if fg_total > 0 else 100.0

    # 4. Per-class metrics
    class_dices = []
    class_ious = []
    class_assds = []
    class_hd95s = []

    for c_idx, c_name in enumerate(class_names):
        cls_val = c_idx + 1
        p_c = (pred_map == cls_val).astype(np.uint8)
        g_c = (gt_map == cls_val).astype(np.uint8)

        d = compute_dice(p_c, g_c) * 100.0
        iou = compute_iou(p_c, g_c) * 100.0
        assd, hd95 = compute_assd_hd95(p_c, g_c)

        res[f'{c_name}_dice'] = d
        res[f'{c_name}_iou'] = iou
        res[f'{c_name}_assd'] = assd
        res[f'{c_name}_hd95'] = hd95

        class_dices.append(d)
        class_ious.append(iou)
        class_assds.append(assd)
        class_hd95s.append(hd95)

    # 5. Macro Averages (unweighted arithmetic mean across landmark classes)
    res['macro_dice'] = float(np.mean(class_dices))
    res['macro_iou'] = float(np.mean(class_ious))
    res['macro_assd'] = float(np.mean(class_assds))
    res['macro_hd95'] = float(np.mean(class_hd95s))

    return res
