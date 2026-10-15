import numpy as np
import cv2


def compute_dice(pred_bin, target_bin):
    """
    Computes Sørensen-Dice coefficient between binary masks.
    Handles empty masks consistently.
    """
    p = (pred_bin > 0).astype(np.float32)
    t = (target_bin > 0).astype(np.float32)
    intersection = float(np.sum(p * t))
    total = float(np.sum(p) + np.sum(t))
    if total == 0:
        return 1.0  # Both ground truth and prediction are empty
    if np.sum(p) == 0 or np.sum(t) == 0:
        return 0.0  # One is empty and the other is not
    return float(2.0 * intersection / total)


def compute_iou(pred_bin, target_bin):
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


def compute_assd_hd95(pred_bin, target_bin, max_penalty=80.0):
    """
    Computes Average Symmetric Surface Distance (ASSD) and Hausdorff Distance 95% (HD95)
    using Euclidean Distance Transforms.
    Guards against empty contours and numpy slice zero-division warnings.
    """
    p = (pred_bin > 0).astype(np.uint8)
    t = (target_bin > 0).astype(np.uint8)

    p_sum = np.sum(p)
    t_sum = np.sum(t)

    # Empty cases
    if p_sum == 0 and t_sum == 0:
        return 0.0, 0.0
    if p_sum == 0 or t_sum == 0:
        return max_penalty, max_penalty

    # Extract contours
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


def evaluate_frame_metrics(pred_sem_seg, gt_input, num_classes=3, threshold=0.5):
    """
    Evaluates comprehensive metrics for a single laparoscopic frame
    using TopoNet's exact mutually exclusive argmax formulation.

    Args:
        pred_sem_seg: (3, H, W) continuous probability map from Mask2Former
        gt_input: (H, W) discrete class map in {0, 1, 2, 3} or (3, H, W) one-hot channels
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
        # (3, H, W) binary channels
        gt_map = np.zeros(gt_input.shape[1:], dtype=np.uint8)
        for c in range(min(num_classes, gt_input.shape[0])):
            gt_map[gt_input[c] > 0.5] = (c + 1)
    else:
        gt_map = gt_input.astype(np.uint8)

    # 3. Create 4-class one-hot arrays (0: bg, 1: ridge, 2: silhouette, 3: ligament)
    pred_onehot = np.array([pred_map == i for i in range(4)]).astype(np.uint8)
    gt_onehot = np.array([gt_map == i for i in range(4)]).astype(np.uint8)

    # 4. TopoNet Exact Micro-Dice (line 55 of TopoNet test.py)
    smooth = 1e-5
    inter = float(np.sum(pred_onehot[1:] * gt_onehot[1:]))
    total_px = float(np.sum(pred_onehot[1:]) + np.sum(gt_onehot[1:]))
    micro_dice = float((2.0 * inter + smooth) / (total_px + smooth))
    micro_iou = float(micro_dice / (2.0 - micro_dice))

    res['micro_dice'] = micro_dice
    res['micro_iou'] = micro_iou

    # 5. Per-class metrics and Macro-Dice
    dices, ious, assds, hd95s = [], [], [], []
    for c in range(1, min(num_classes + 1, 4)):
        c_name = class_names[c - 1]
        p_c = pred_onehot[c]
        t_c = gt_onehot[c]

        d = compute_dice(p_c, t_c)
        iou = compute_iou(p_c, t_c)
        assd, hd95 = compute_assd_hd95(p_c, t_c)

        res[f'dice_{c_name}'] = d
        res[f'iou_{c_name}'] = iou
        res[f'assd_{c_name}'] = assd
        res[f'hd95_{c_name}'] = hd95

        dices.append(d)
        ious.append(iou)
        assds.append(assd)
        hd95s.append(hd95)

    res['macro_dice'] = float(np.mean(dices))
    res['macro_iou'] = float(np.mean(ious))
    res['mean_assd'] = float(np.mean(assds))
    res['mean_hd95'] = float(np.mean(hd95s))

    return res

