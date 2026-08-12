"""
Vector Landmark Dataset for Surgical-GeMap.

Extracts ordered 2D polyline control points directly from raw JSON/XML
annotations (no pixel rasterization during training). Each polyline is
resampled to K equidistant points normalized to [0, 1].

Also provides rasterized pixel masks for pixel-metric evaluation at
validation time.
"""

import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


# ──────────────────────────────────────────────
#  Polyline Resampling
# ──────────────────────────────────────────────

def resample_polyline(points: np.ndarray, K: int = 20) -> np.ndarray:
    """
    Resample an ordered polyline to K equidistant points via linear
    interpolation along arc-length parameterization.

    Args:
        points: (M, 2) array of ordered [x, y] coordinates (pixel space).
        K: number of output points.

    Returns:
        (K, 2) array of resampled points.
    """
    if len(points) < 2:
        # Degenerate: single point → repeat K times
        return np.tile(points[0], (K, 1))

    # Compute cumulative arc-length
    diffs = np.diff(points, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cum_lengths = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_length = cum_lengths[-1]

    if total_length < 1e-8:
        # All points are the same → repeat
        return np.tile(points[0], (K, 1))

    # Target arc-length positions (equidistant)
    target_lengths = np.linspace(0.0, total_length, K)

    # Interpolate x and y separately
    resampled = np.zeros((K, 2), dtype=np.float64)
    resampled[:, 0] = np.interp(target_lengths, cum_lengths, points[:, 0])
    resampled[:, 1] = np.interp(target_lengths, cum_lengths, points[:, 1])

    return resampled


# ──────────────────────────────────────────────
#  Annotation Loaders (Vector Mode)
# ──────────────────────────────────────────────

LABEL_MAP = {
    'ridge': 1, 'rigde': 1,  # typo in some annotations
    'silhouette': 2, 'sil': 2,
    'ligament': 3, 'lig': 3,
}


def _classify_label(label_str: str) -> int:
    """Map a label string to class index {1, 2, 3} or 0 for unknown."""
    label = label_str.strip().lower()
    # Direct match
    if label in LABEL_MAP:
        return LABEL_MAP[label]
    # Prefix match
    if label.startswith('r'):
        return 1
    if label.startswith('s'):
        return 2
    if label.startswith('l'):
        return 3
    return 0


def load_json_vectors(path: str, K: int = 20, default_h: int = 1080, default_w: int = 1920):
    """
    Extract polyline vectors from a JSON annotation file.

    Returns:
        polylines: list of (K, 2) np.ndarray in pixel coordinates
        classes: list of int class labels {1, 2, 3}
        img_h, img_w: original image dimensions
    """
    with open(path, 'r') as f:
        data = json.load(f)

    img_h = data.get('imageHeight', default_h)
    img_w = data.get('imageWidth', default_w)

    polylines = []
    classes = []

    for shape in data['shapes']:
        raw_points = np.array(shape['points'], dtype=np.float64)  # (M, 2)
        label = str(shape.get('label', '')).lower()
        cls = _classify_label(label)

        if cls == 0 or len(raw_points) < 2:
            continue

        resampled = resample_polyline(raw_points, K)
        polylines.append(resampled)
        classes.append(cls)

    return polylines, classes, img_h, img_w


def load_xml_vectors(path: str, K: int = 20, default_h: int = 1080, default_w: int = 1920):
    """
    Extract polyline vectors from an XML annotation file.

    Returns:
        polylines: list of (K, 2) np.ndarray in pixel coordinates
        classes: list of int class labels {1, 2, 3}
        img_h, img_w: image dimensions from fallback
    """
    tree = ET.parse(path)
    root = tree.getroot()

    img_h, img_w = default_h, default_w
    polylines = []
    classes = []

    for contour in root.findall('contour'):
        ctype_elem = contour.find('contourType')
        ctype = ctype_elem.text.strip() if ctype_elem is not None else ''

        if ctype == 'Ridge':
            cls = 1
        elif ctype == 'Silhouette':
            cls = 2
        else:
            cls = 3

        x_coords = [float(x) for x in contour.find('imagePoints/x').text.split(',')]
        y_coords = [float(y) for y in contour.find('imagePoints/y').text.split(',')]
        raw_points = np.array(list(zip(x_coords, y_coords)), dtype=np.float64)

        if len(raw_points) < 2:
            continue

        resampled = resample_polyline(raw_points, K)
        polylines.append(resampled)
        classes.append(cls)

    return polylines, classes, img_h, img_w


# ──────────────────────────────────────────────
#  Rasterization (for pixel-metric evaluation)
# ──────────────────────────────────────────────

def rasterize_polylines(polylines, classes, H=1024, W=1024,
                        orig_h=1080, orig_w=1920, thickness=35):
    """
    Rasterize polylines into a pixel mask matching TopoNet's convention.

    Args:
        polylines: list of (K, 2) arrays in pixel coordinates (original resolution).
        classes: list of int class labels.
        H, W: output mask resolution.
        orig_h, orig_w: original image resolution (for coordinate scaling).
        thickness: line thickness (35 to match dataset convention).

    Returns:
        mask: (H, W) uint8 array with class indices {0, 1, 2, 3}.
    """
    # Draw at original resolution first, then resize
    mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

    for pts, cls in zip(polylines, classes):
        int_pts = pts.astype(np.int32)
        for i in range(len(int_pts) - 1):
            cv2.line(mask, tuple(int_pts[i]), tuple(int_pts[i + 1]), int(cls), thickness)

    mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
    return mask


def rasterize_normalized_polylines(polylines_norm, classes, H=1024, W=1024,
                                   thickness=19):
    """
    Rasterize normalized [0,1] polylines into pixel mask.

    Args:
        polylines_norm: list of (K, 2) arrays in [0, 1] normalized coords.
        classes: list of int class labels.
        H, W: output resolution.
        thickness: line thickness.

    Returns:
        mask: (H, W) uint8 array with class indices.
    """
    mask = np.zeros((H, W), dtype=np.uint8)

    for pts_norm, cls in zip(polylines_norm, classes):
        # Denormalize to pixel coords at output resolution
        pts_pixel = pts_norm.copy()
        pts_pixel[:, 0] *= W
        pts_pixel[:, 1] *= H
        int_pts = pts_pixel.astype(np.int32)

        for i in range(len(int_pts) - 1):
            cv2.line(mask, tuple(int_pts[i]), tuple(int_pts[i + 1]), int(cls), thickness)

    return mask


# ──────────────────────────────────────────────
#  Dataset Class
# ──────────────────────────────────────────────

class VectorLandmarkDataset(Dataset):
    """
    Dataset that returns:
        - image: (3, H, W) tensor
        - polylines: (N, K, 2) tensor of normalized [0,1] control points
        - labels: (N,) tensor of class labels (0 = no-object padding)
        - num_instances: int, number of real (non-padded) instances
        - pixel_mask: (4, H, W) tensor — rasterized GT for pixel-metric eval
        - img_path: str — original image path
    """

    def __init__(self, file_names, N=30, K=20, num_pts=None, max_polylines=None,
                 img_size=1024, transform=None, mode='train'):
        self.file_names = file_names
        self.N = max_polylines if max_polylines is not None else N
        self.K = num_pts if num_pts is not None else K
        self.img_size = img_size
        self.mode = mode

        if transform is not None:
            self.transform = transform
        else:
            self.transform = T.Compose([
                T.ToTensor(),
            ])

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_path = str(self.file_names[idx])

        # ── Load image ──
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]
        img = cv2.resize(img, (self.img_size, self.img_size))
        img_tensor = self.transform(img)  # (3, H, W)

        # ── Load vector annotations ──
        base_path = img_path.replace('images', 'labels')
        json_path = os.path.splitext(base_path)[0] + '.json'
        xml_path = os.path.splitext(base_path)[0] + '.xml'

        if os.path.exists(json_path):
            polylines, classes, ann_h, ann_w = load_json_vectors(json_path, self.K, default_h=orig_h, default_w=orig_w)
        elif os.path.exists(xml_path):
            polylines, classes, ann_h, ann_w = load_xml_vectors(xml_path, self.K, default_h=orig_h, default_w=orig_w)
        else:
            polylines, classes = [], []
            ann_h, ann_w = orig_h, orig_w

        # ── Normalize polylines to [0, 1] ──
        normalized_polylines = []
        for pts in polylines:
            norm_pts = pts.copy()
            norm_pts[:, 0] /= ann_w   # x / width
            norm_pts[:, 1] /= ann_h   # y / height
            norm_pts = np.clip(norm_pts, 0.0, 1.0)
            normalized_polylines.append(norm_pts)

        # ── Pad to N instances ──
        num_instances = len(normalized_polylines)
        padded_polylines = np.zeros((self.N, self.K, 2), dtype=np.float32)
        padded_labels = np.zeros(self.N, dtype=np.int64)  # 0 = no-object

        for i in range(min(num_instances, self.N)):
            padded_polylines[i] = normalized_polylines[i].astype(np.float32)
            padded_labels[i] = classes[i]

        # ── Generate pixel mask for evaluation ──
        pixel_mask = rasterize_polylines(
            polylines, classes,
            H=self.img_size, W=self.img_size,
            orig_h=ann_h, orig_w=ann_w, thickness=35
        )
        # Convert to 4-channel binary mask (matching TopoNet format)
        masks = np.zeros((4, self.img_size, self.img_size), dtype=np.uint8)
        masks[0][pixel_mask == 0] = 255
        masks[1][pixel_mask == 1] = 255
        masks[2][pixel_mask == 2] = 255
        masks[3][pixel_mask == 3] = 255

        return (
            img_tensor,                                          # (3, H, W)
            torch.from_numpy(padded_polylines),                  # (N, K, 2)
            torch.from_numpy(padded_labels),                     # (N,)
            torch.tensor(num_instances, dtype=torch.long),       # scalar
            torch.from_numpy(masks).float() / 255.0,             # (4, H, W)
            img_path,
        )
