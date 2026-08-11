"""
Pixel-to-Vector Converter.

Converts pixel-based segmentation masks (from TopoNet, Mask2Former, etc.)
into ordered polyline control points for vector-metric evaluation.

Pipeline per class:
1. Extract binary mask for the class
2. Skeletonize → 1-pixel-wide skeleton
3. Trace connected components as ordered point sequences
4. Resample to K equidistant points, normalize to [0, 1]
"""

import numpy as np
import cv2

try:
    from skimage.morphology import skeletonize
except ImportError:
    skeletonize = None

from utils.vector_dataset import resample_polyline


def _trace_skeleton_component(skeleton_mask):
    """
    Trace a connected skeleton component into an ordered point sequence.

    Uses a simple greedy walk from an endpoint (or any start pixel if no
    endpoint exists), following 8-connected neighbors.

    Args:
        skeleton_mask: (H, W) binary mask of a single connected component.

    Returns:
        points: (M, 2) array of ordered [x, y] coordinates.
    """
    # Find all skeleton pixels
    ys, xs = np.nonzero(skeleton_mask)
    if len(xs) == 0:
        return np.array([]).reshape(0, 2)

    # Build adjacency via 8-connectivity
    pixel_set = set(zip(xs.tolist(), ys.tolist()))

    def neighbors(x, y):
        nbrs = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                if (x + dx, y + dy) in pixel_set:
                    nbrs.append((x + dx, y + dy))
        return nbrs

    # Find an endpoint (pixel with exactly 1 neighbor) to start
    start = None
    for x, y in pixel_set:
        n_nbrs = len(neighbors(x, y))
        if n_nbrs == 1:
            start = (x, y)
            break

    if start is None:
        # No endpoint (cycle) — just pick any pixel
        start = (xs[0], ys[0])

    # Greedy walk
    ordered = [start]
    visited = {start}

    while True:
        cx, cy = ordered[-1]
        nbrs = neighbors(cx, cy)
        unvisited = [n for n in nbrs if n not in visited]

        if not unvisited:
            break

        # Pick the nearest unvisited neighbor (prefer straight lines)
        next_pt = unvisited[0]
        ordered.append(next_pt)
        visited.add(next_pt)

    return np.array(ordered, dtype=np.float64)  # (M, 2) as [x, y]


def mask_to_polylines(mask, class_id, K=20):
    """
    Convert a single-class binary mask to ordered polylines.

    Args:
        mask: (H, W) uint8 mask where pixels == class_id are foreground.
        class_id: the class value to extract.
        K: number of resampled points per polyline.

    Returns:
        polylines: list of (K, 2) numpy arrays in pixel coordinates.
    """
    if skeletonize is None:
        raise ImportError("scikit-image is required. Install with: pip install scikit-image")

    # Binary mask for this class
    binary = (mask == class_id).astype(np.uint8)

    if binary.sum() == 0:
        return []

    # Skeletonize
    skel = skeletonize(binary > 0).astype(np.uint8)

    if skel.sum() == 0:
        return []

    # Find connected components
    num_labels, labels = cv2.connectedComponents(skel, connectivity=8)

    polylines = []
    for label_id in range(1, num_labels):
        component = (labels == label_id).astype(np.uint8)
        points = _trace_skeleton_component(component)

        if len(points) < 2:
            continue

        resampled = resample_polyline(points, K)
        polylines.append(resampled)

    return polylines


def prediction_mask_to_vectors(pred_mask, K=20, normalize=True):
    """
    Convert a full multi-class prediction mask to polyline vectors.

    Args:
        pred_mask: (H, W) uint8 mask with class indices {0, 1, 2, 3}.
        K: points per polyline.
        normalize: if True, normalize coordinates to [0, 1].

    Returns:
        all_polylines: list of (K, 2) arrays
        all_classes: list of int class labels {1, 2, 3}
    """
    H, W = pred_mask.shape
    all_polylines = []
    all_classes = []

    for cls in [1, 2, 3]:
        polylines = mask_to_polylines(pred_mask, cls, K)
        for pl in polylines:
            if normalize:
                pl_norm = pl.copy()
                pl_norm[:, 0] /= W
                pl_norm[:, 1] /= H
                all_polylines.append(pl_norm)
            else:
                all_polylines.append(pl)
            all_classes.append(cls)

    return all_polylines, all_classes
