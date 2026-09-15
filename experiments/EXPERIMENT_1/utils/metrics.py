import numpy as np
import cv2
import torch


def compute_dice_iou(pred_binary, gt_binary):
    """
    Computes Dice Similarity Coefficient and IoU for binary 1D or 2D arrays.
    """
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    pred_sum = pred_binary.sum()
    gt_sum = gt_binary.sum()
    total_sum = pred_sum + gt_sum

    if total_sum == 0:
        return 1.0, 1.0  # Perfect match on empty ground truth
    if pred_sum == 0 or gt_sum == 0:
        return 0.0, 0.0

    dice = (2.0 * intersection) / (total_sum + 1e-7)
    iou = intersection / (pred_sum + gt_sum - intersection + 1e-7)
    return float(dice), float(iou)


def compute_assd(pred_mask, gt_mask, fallback=80.0):
    """
    Computes Average Symmetric Surface Distance (ASSD) in pixels.
    Uses surface_distance / medpy if available, or robust OpenCV Euclidean distance transform fallback.
    """
    pred_mask = (pred_mask > 0).astype(np.uint8)
    gt_mask = (gt_mask > 0).astype(np.uint8)

    if pred_mask.sum() == 0 or gt_mask.sum() == 0:
        return float(fallback)

    # Try surface_distance / medpy first
    try:
        from surface_distance import metrics
        sd = metrics.compute_surface_distances(gt_mask.astype(bool), pred_mask.astype(bool), (1.0, 1.0))
        assd_val = metrics.compute_average_surface_distance(sd)[1]
        if np.isnan(assd_val) or assd_val > 500:
            return float(fallback)
        return float(assd_val)
    except Exception:
        pass

    try:
        import medpy.metric
        return float(medpy.metric.assd(pred_mask, gt_mask))
    except Exception:
        pass

    # Pure OpenCV Euclidean Distance Transform Fallback
    # Extract 1-pixel boundary contours
    contours_pred, _ = cv2.findContours(pred_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours_gt, _ = cv2.findContours(gt_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    border_pred = np.zeros_like(pred_mask)
    border_gt = np.zeros_like(gt_mask)

    cv2.drawContours(border_pred, contours_pred, -1, 1, 1)
    cv2.drawContours(border_gt, contours_gt, -1, 1, 1)

    # Distance to GT surface
    dist_to_gt = cv2.distanceTransform(1 - border_gt, cv2.DIST_L2, 5)
    # Distance to Pred surface
    dist_to_pred = cv2.distanceTransform(1 - border_pred, cv2.DIST_L2, 5)

    dist_pred_to_gt = dist_to_gt[border_pred == 1]
    dist_gt_to_pred = dist_to_pred[border_gt == 1]

    if len(dist_pred_to_gt) == 0 or len(dist_gt_to_pred) == 0:
        return float(fallback)

    assd_val = (dist_pred_to_gt.mean() + dist_gt_to_pred.mean()) / 2.0
    return float(min(assd_val, fallback))


def evaluate_batch(pred_logits, gt_masks):
    """
    Evaluates a batch of multi-class predictions against ground truth.
    pred_logits: Tensor of shape (B, 4, H, W)
    gt_masks: Tensor of shape (B, 4, H, W) where masks are one-hot (0: BG, 1: Ridge, 2: Sil, 3: Falc)
    
    Returns list of metric dicts per sample.
    """
    pred_classes = torch.argmax(pred_logits, dim=1).detach().cpu().numpy()  # (B, H, W)
    gt_classes = torch.argmax(gt_masks, dim=1).detach().cpu().numpy()        # (B, H, W)

    batch_metrics = []

    for b in range(pred_classes.shape[0]):
        p_map = pred_classes[b]
        g_map = gt_classes[b]

        # Per-class foreground metrics (1: Ridge, 2: Silhouette, 3: Falciform)
        class_dices = []
        class_ious = []
        class_assds = []

        for c, name in enumerate(['ridge', 'silhouette', 'falciform'], start=1):
            p_c = (p_map == c)
            g_c = (g_map == c)

            d, iou = compute_dice_iou(p_c, g_c)
            class_dices.append(d)
            class_ious.append(iou)

            if g_c.sum() > 0:
                assd_c = compute_assd(p_c, g_c)
                class_assds.append(assd_c)

        macro_dice = float(np.mean(class_dices))
        macro_iou = float(np.mean(class_ious))
        macro_assd = float(np.mean(class_assds)) if len(class_assds) > 0 else 80.0

        # Overall flattened foreground metric (exact repos/TopoNet/test.py standard)
        p_fg = (p_map > 0)
        g_fg = (g_map > 0)
        fg_dice, fg_iou = compute_dice_iou(p_fg, g_fg)
        fg_assd = compute_assd(p_fg, g_fg)

        batch_metrics.append({
            'macro_dice': macro_dice,
            'macro_iou': macro_iou,
            'macro_assd': macro_assd,
            'fg_dice': fg_dice,
            'fg_iou': fg_iou,
            'fg_assd': fg_assd,
            'ridge_dice': class_dices[0],
            'sil_dice': class_dices[1],
            'falc_dice': class_dices[2],
            'ridge_iou': class_ious[0],
            'sil_iou': class_ious[1],
            'falc_iou': class_ious[2],
        })

    return batch_metrics
