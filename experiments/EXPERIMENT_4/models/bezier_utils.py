import numpy as np
import cv2
import torch

def bernstein_eval_torch(control_points, num_samples=10):
    """
    Evaluates Bezier curves using Bernstein basis.
    
    Args:
        control_points: Tensor shape (N, 4, 2)
        num_samples: int
        
    Returns:
        Tensor shape (N, num_samples, 2)
    """
    N = control_points.shape[0]
    t = torch.linspace(0, 1, num_samples, device=control_points.device, dtype=control_points.dtype)
    
    B0 = (1 - t)**3
    B1 = 3 * (1 - t)**2 * t
    B2 = 3 * (1 - t) * t**2
    B3 = t**3
    
    P0 = control_points[:, 0, :]
    P1 = control_points[:, 1, :]
    P2 = control_points[:, 2, :]
    P3 = control_points[:, 3, :]
    
    B0 = B0.view(1, num_samples, 1)
    B1 = B1.view(1, num_samples, 1)
    B2 = B2.view(1, num_samples, 1)
    B3 = B3.view(1, num_samples, 1)
    
    P0 = P0.view(N, 1, 2)
    P1 = P1.view(N, 1, 2)
    P2 = P2.view(N, 1, 2)
    P3 = P3.view(N, 1, 2)
    
    curve = B0 * P0 + B1 * P1 + B2 * P2 + B3 * P3
    return curve

def resample_polyline(pts, spacing=5.0):
    """
    Resamples a polyline at given spacing.
    
    Args:
        pts: list of [x,y] pairs
        spacing: float
        
    Returns:
        np.ndarray shape (M, 2)
    """
    if len(pts) < 2:
        return np.array(pts, dtype=np.float32)
    
    pts = np.array(pts, dtype=np.float32)
    resampled = [pts[0]]
    
    current_pt = pts[0]
    next_idx = 1
    
    while next_idx < len(pts):
        dist_to_next = np.linalg.norm(pts[next_idx] - current_pt)
        if dist_to_next < 1e-6:
            next_idx += 1
            continue
            
        if dist_to_next >= spacing:
            t = spacing / dist_to_next
            new_pt = current_pt + t * (pts[next_idx] - current_pt)
            resampled.append(new_pt)
            current_pt = new_pt
        else:
            current_pt = pts[next_idx]
            next_idx += 1
            
    if len(resampled) < 2:
        return pts
    return np.array(resampled, dtype=np.float32)

def fit_bezier_to_patch(pts_in_canvas, patch_bbox):
    """
    Fits a Bezier curve to points inside a patch.
    
    Args:
        pts_in_canvas: list/array of (x, y) points in canvas coords
        patch_bbox: [x_min, y_min, x_max, y_max]
        
    Returns:
        np.ndarray shape (4, 2) normalized to [0,1]^2
    """
    pts = np.array(pts_in_canvas, dtype=np.float32)
    x_min, y_min, x_max, y_max = patch_bbox
    patch_w = x_max - x_min
    patch_h = y_max - y_min
    
    local_pts = np.zeros_like(pts)
    local_pts[:, 0] = (pts[:, 0] - x_min) / max(patch_w, 1e-5)
    local_pts[:, 1] = (pts[:, 1] - y_min) / max(patch_h, 1e-5)
    
    if len(local_pts) < 2:
        if len(local_pts) == 1:
            p = local_pts[0]
            cp = np.array([p, p, p, p], dtype=np.float32)
            return np.clip(cp, 0, 1)
        return np.zeros((4, 2), dtype=np.float32)
        
    diffs = np.diff(local_pts, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    cum_dists = np.concatenate(([0], np.cumsum(dists)))
    total_len = cum_dists[-1]
    
    if total_len < 1e-6:
        p0 = local_pts[0]
        p3 = local_pts[-1]
        cp = np.array([p0, p0, p3, p3], dtype=np.float32)
        return np.clip(cp, 0, 1)
        
    t = cum_dists / total_len
    
    P0 = local_pts[0]
    P3 = local_pts[-1]

    if len(local_pts) < 3:
        P1 = P0 + (P3 - P0) * (1.0 / 3.0)
        P2 = P0 + (P3 - P0) * (2.0 / 3.0)
        control_points = np.array([P0, P1, P2, P3], dtype=np.float32)
        return np.clip(control_points, 0, 1)

    B0 = (1 - t)**3
    B1 = 3 * (1 - t)**2 * t
    B2 = 3 * (1 - t) * t**2
    B3 = t**3

    A = np.column_stack((B1, B2))

    RHS_x = local_pts[:, 0] - B0 * P0[0] - B3 * P3[0]
    RHS_y = local_pts[:, 1] - B0 * P0[1] - B3 * P3[1]
    RHS = np.column_stack((RHS_x, RHS_y))

    try:
        if np.linalg.matrix_rank(A) < 2:
            P1 = P0 + (P3 - P0) * (1.0 / 3.0)
            P2 = P0 + (P3 - P0) * (2.0 / 3.0)
        else:
            result = np.linalg.lstsq(A, RHS, rcond=None)[0]
            P1 = result[0]
            P2 = result[1]
    except Exception:
        P1 = P0 + (P3 - P0) * (1.0 / 3.0)
        P2 = P0 + (P3 - P0) * (2.0 / 3.0)

    control_points = np.array([P0, P1, P2, P3], dtype=np.float32)
    return np.clip(control_points, 0, 1)

def rasterize_bezier_predictions(pred_class_np, pred_bezier_np, grid_size=8, canvas_size=1024, stroke_width=35):
    """
    Rasterizes predicted Bezier curves into a semantic class map.
    
    Args:
        pred_class_np: np.ndarray (B, grid_size*grid_size)
        pred_bezier_np: np.ndarray (B, grid_size*grid_size, 4, 2)
        grid_size: int
        canvas_size: int
        stroke_width: int
        
    Returns:
        np.ndarray (B, canvas_size, canvas_size)
    """
    B = pred_class_np.shape[0]
    canvas_maps = np.zeros((B, canvas_size, canvas_size), dtype=np.int32)
    patch_size_px = canvas_size // grid_size
    
    t = np.linspace(0, 1, 50)
    B0 = (1 - t)**3
    B1 = 3 * (1 - t)**2 * t
    B2 = 3 * (1 - t) * t**2
    B3 = t**3
    
    for b in range(B):
        for i in range(grid_size * grid_size):
            cls_id = int(pred_class_np[b, i])
            if cls_id > 0:
                r = i // grid_size
                c = i % grid_size
                patch_x_min = c * patch_size_px
                patch_y_min = r * patch_size_px
                
                cps = pred_bezier_np[b, i]
                
                P0_x, P0_y = cps[0]
                P1_x, P1_y = cps[1]
                P2_x, P2_y = cps[2]
                P3_x, P3_y = cps[3]
                
                curve_x = B0 * P0_x + B1 * P1_x + B2 * P2_x + B3 * P3_x
                curve_y = B0 * P0_y + B1 * P1_y + B2 * P2_y + B3 * P3_y
                
                global_x = patch_x_min + curve_x * patch_size_px
                global_y = patch_y_min + curve_y * patch_size_px
                
                pts = np.column_stack((global_x, global_y)).astype(np.int32)
                pts = pts.reshape((-1, 1, 2))
                
                cv2.polylines(canvas_maps[b], [pts], isClosed=False, color=cls_id, thickness=stroke_width, lineType=cv2.LINE_AA)
                
    return canvas_maps
