"""
Heatmap generation and coordinate decoding utilities for EXPERIMENT_6.
Features:
  - Generates 2D Gaussian heatmaps for visible landmark keypoints.
  - Automatically produces all-zero planes for absent / occluded keypoints.
  - Decodes predicted peak coordinates (sub-grid argmax) and existence scores from heatmaps.
"""
import torch
import numpy as np


def generate_gaussian_heatmaps(coords_norm, visibility, map_size=64, sigma=2.0):
    """
    Generates ground-truth Gaussian heatmaps.

    Args:
        coords_norm (Tensor or np.ndarray): Shape (B, 4, 2) or (4, 2) in normalized [0, 1]^2.
        visibility (Tensor or np.ndarray): Shape (B, 4) or (4,) with values in {0.0, 1.0}.
        map_size (int): Spatial grid resolution (e.g. 64 for stride-16 features).
        sigma (float): Standard deviation of Gaussian peak in grid pixels.

    Returns:
        heatmaps (Tensor): Shape (B, 4, map_size, map_size) or (4, map_size, map_size), float32 in [0, 1].
    """
    is_batch = (len(coords_norm.shape) == 3)
    if not is_batch:
        coords_norm = coords_norm.unsqueeze(0) if isinstance(coords_norm, torch.Tensor) else np.expand_dims(coords_norm, 0)
        visibility = visibility.unsqueeze(0) if isinstance(visibility, torch.Tensor) else np.expand_dims(visibility, 0)

    if isinstance(coords_norm, torch.Tensor):
        coords_np = coords_norm.detach().cpu().numpy()
        vis_np = visibility.detach().cpu().numpy()
    else:
        coords_np = coords_norm
        vis_np = visibility

    B, K, _ = coords_np.shape
    heatmaps_np = np.zeros((B, K, map_size, map_size), dtype=np.float32)

    # Coordinate grid
    grid_y, grid_x = np.ogrid[:map_size, :map_size]

    for b in range(B):
        for k in range(K):
            if vis_np[b, k] > 0.5:
                cx = coords_np[b, k, 0] * float(map_size)
                cy = coords_np[b, k, 1] * float(map_size)

                # Skip if center is outside valid coordinate bounds
                if cx < 0 or cx >= map_size or cy < 0 or cy >= map_size:
                    continue

                # 2D Gaussian
                d2 = (grid_x - cx) ** 2 + (grid_y - cy) ** 2
                g = np.exp(-d2 / (2.0 * sigma * sigma))
                if g.max() > 0:
                    g = g / g.max()
                heatmaps_np[b, k] = g.astype(np.float32)
            else:
                # Absent point = clean zeros everywhere
                heatmaps_np[b, k] = 0.0

    heatmaps_tensor = torch.from_numpy(heatmaps_np).float()
    if not is_batch:
        heatmaps_tensor = heatmaps_tensor.squeeze(0)
    return heatmaps_tensor


def extract_peak_coords(pred_heatmaps, threshold=0.3):
    """
    Decodes predicted coordinates and existence flags from predicted sigmoid heatmaps.

    Args:
        pred_heatmaps (Tensor): Shape (B, K, H, W) in [0, 1].
        threshold (float): Existence confidence threshold.

    Returns:
        coords (Tensor): Shape (B, K, 2) normalized to [0, 1]^2.
        visibilities (Tensor): Shape (B, K) binary flags {0.0, 1.0}.
        confidences (Tensor): Shape (B, K) peak activation scores in [0, 1].
    """
    is_batch = (len(pred_heatmaps.shape) == 4)
    if not is_batch:
        pred_heatmaps = pred_heatmaps.unsqueeze(0)

    B, K, H, W = pred_heatmaps.shape
    device = pred_heatmaps.device

    # Flatten spatial dimensions: (B, K, H*W)
    flat = pred_heatmaps.view(B, K, -1)
    max_vals, max_indices = torch.max(flat, dim=-1)

    peak_y = (max_indices // W).float() / float(H)
    peak_x = (max_indices % W).float() / float(W)

    coords = torch.stack([peak_x, peak_y], dim=-1) # (B, K, 2)
    visibilities = (max_vals >= threshold).float()  # (B, K)
    confidences = max_vals                         # (B, K)

    if not is_batch:
        return coords.squeeze(0), visibilities.squeeze(0), confidences.squeeze(0)
    return coords, visibilities, confidences
