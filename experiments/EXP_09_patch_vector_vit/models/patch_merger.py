import cv2
import numpy as np
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
    threshold: float = 0.5,
    stroke_thickness: int = 2,
    num_samples_per_curve: int = 12,
    return_class_masks: bool = False
):
    """
    Merges local patch Bézier predictions into a high-resolution landmark image
    using Global Coordinate Shift + Anti-Aliased Vector Rasterization.
    
    Args:
        patch_classes: (G, G) class IDs in 0..num_classes, OR (G, G, C+1) probability distribution.
        patch_beziers: (G, G, 4, 2) local control points in [0, 1]^2.
        patch_size: 16 pixels per patch.
        img_size: 512 pixels.
        threshold: Minimum probability threshold if patch_classes is a probability distribution.
        stroke_thickness: Rasterized line width in pixels.
        num_samples_per_curve: Number of continuous interpolation points along each cubic Bézier arc.
        return_class_masks: If True, also returns (C, H, W) binary masks per landmark class.
        
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
