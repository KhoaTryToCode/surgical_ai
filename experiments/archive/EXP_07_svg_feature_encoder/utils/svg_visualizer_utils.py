import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

def feature_pca_rgb(feature_tensor: torch.Tensor) -> np.ndarray:
    """
    Projects high-dimensional feature map (C, H, W) to (H, W, 3) RGB via PCA.
    """
    if isinstance(feature_tensor, torch.Tensor):
        feat = feature_tensor.detach().cpu().float().numpy()
    else:
        feat = feature_tensor

    if feat.ndim == 4:
        feat = feat[0] # (C, H, W)

    C, H, W = feat.shape
    feat_flat = feat.reshape(C, -1).T # (H*W, C)

    # Standardize
    feat_mean = np.mean(feat_flat, axis=0, keepdims=True)
    feat_std = np.std(feat_flat, axis=0, keepdims=True) + 1e-5
    feat_norm = (feat_flat - feat_mean) / feat_std

    # SVD for top-3 principal components
    try:
        _, _, Vt = np.linalg.svd(feat_norm, full_matrices=False)
        pca_proj = np.dot(feat_norm, Vt[:3, :].T) # (H*W, 3)
    except Exception:
        pca_proj = feat_flat[:, :3]

    # Normalize to [0, 1] per channel
    rgb = np.zeros((H * W, 3), dtype=np.float32)
    for c in range(3):
        min_v = np.percentile(pca_proj[:, c], 1)
        max_v = np.percentile(pca_proj[:, c], 99)
        rgb[:, c] = np.clip((pca_proj[:, c] - min_v) / (max_v - min_v + 1e-5), 0.0, 1.0)

    return rgb.reshape(H, W, 3)

def vector_field_to_hsv(tangent_field: torch.Tensor, saliency_field: torch.Tensor = None) -> np.ndarray:
    """
    Converts 2D Vector Field (2, H, W) into an HSV Color Wheel representation:
      - Angle theta -> Hue (0..360)
      - Magnitude r * Saliency -> Saturation & Value
    Returns: (H, W, 3) RGB image
    """
    if isinstance(tangent_field, torch.Tensor):
        tf = tangent_field.detach().cpu().float().numpy()
    else:
        tf = tangent_field

    if tf.ndim == 4:
        tf = tf[0]

    tx = tf[0] # (H, W)
    ty = tf[1]

    angle = (np.arctan2(ty, tx) + np.pi) / (2.0 * np.pi) # in [0, 1]
    mag = np.sqrt(tx ** 2 + ty ** 2)
    mag = np.clip(mag / (np.max(mag) + 1e-5), 0.0, 1.0)

    if saliency_field is not None:
        if isinstance(saliency_field, torch.Tensor):
            sal = saliency_field.detach().cpu().float().numpy()
        else:
            sal = saliency_field
        if sal.ndim == 4:
            sal = sal[0, 0]
        elif sal.ndim == 3:
            sal = sal[0]
        sal = cv2.resize(sal, (tx.shape[1], tx.shape[0]))
        mag = mag * np.clip(sal * 1.5, 0.0, 1.0)

    hsv = np.zeros((tx.shape[0], tx.shape[1], 3), dtype=np.float32)
    hsv[..., 0] = angle * 180.0 # OpenCV Hue: 0..180
    hsv[..., 1] = np.clip(mag * 255.0, 0.0, 255.0)
    hsv[..., 2] = np.clip(mag * 255.0, 0.0, 255.0)

    bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return rgb

def create_synthetic_surgical_frame(h: int = 1024, w: int = 1024) -> tuple:
    """
    Generates a realistic synthetic laparoscopic liver frame with:
      - Liver reddish-brown tissue texture
      - Falciform ligament (vertical midline ridge)
      - Anterior liver ridge (curved horizontal boundary)
      - Specular laparoscope lighting
    """
    # Base peritoneal cavity / dark background
    canvas = np.zeros((h, w, 3), dtype=np.float32)
    canvas[:, :] = [0.15, 0.08, 0.06]

    # Liver lobe shape (large curved polygon)
    pts_liver = np.array([
        [int(w * 0.1), int(h * 0.9)],
        [int(w * 0.2), int(h * 0.35)],
        [int(w * 0.5), int(h * 0.22)],
        [int(w * 0.85), int(h * 0.38)],
        [int(w * 0.92), int(h * 0.85)],
        [int(w * 0.5), int(h * 0.95)]
    ], dtype=np.int32)
    
    cv2.fillPoly(canvas, [pts_liver], color=[0.62, 0.22, 0.16])

    # Add organic tissue noise
    noise = np.random.normal(0.0, 0.03, (h, w, 3)).astype(np.float32)
    canvas = np.clip(canvas + noise, 0.0, 1.0)

    # Anterior Ridge (Curved bright contour along top liver border)
    ridge_pts = []
    for x in np.linspace(w * 0.2, w * 0.85, 50):
        y = h * (0.35 - 0.13 * np.sin((x - w * 0.2) / (w * 0.65) * np.pi))
        ridge_pts.append([int(x), int(y)])
    ridge_arr = np.array(ridge_pts, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(canvas, [ridge_arr], isClosed=False, color=[0.85, 0.75, 0.55], thickness=6)

    # Falciform Ligament (Vertical translucent fibrous line)
    falc_pts = []
    for y in np.linspace(h * 0.25, h * 0.85, 40):
        x = w * (0.50 + 0.03 * np.sin((y - h * 0.25) / (h * 0.6) * np.pi))
        falc_pts.append([int(x), int(y)])
    falc_arr = np.array(falc_pts, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(canvas, [falc_arr], isClosed=False, color=[0.90, 0.88, 0.82], thickness=8)

    # Specular lighting highlight
    cv2.circle(canvas, (int(w * 0.45), int(h * 0.38)), int(w * 0.12), color=[0.95, 0.70, 0.65], thickness=-1)
    canvas = cv2.GaussianBlur(canvas, (15, 15), 0)

    return np.clip(canvas, 0.0, 1.0)
