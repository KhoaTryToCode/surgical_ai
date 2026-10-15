import os
import glob
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


class TopoNetDataset(Dataset):
    """
    Robust L3D Dataset Reader for TopoNet with dynamic canvas sizing.
    Fixes the Patient 32 4K canvas truncation bug by reading imageHeight/imageWidth
    directly from the Label JSON.
    """
    def __init__(self, data_dir, depth_dir=None, transform=None, mode='train'):
        self.data_dir = data_dir
        self.depth_dir = depth_dir
        self.mode = mode
        
        # Resolve images directory flexibly (supports both nested 'images/' and flat folders)
        if os.path.exists(os.path.join(data_dir, 'images')):
            self.image_paths = sorted(glob.glob(os.path.join(data_dir, 'images', '*.[jJ][pP][gG]')) + 
                                     glob.glob(os.path.join(data_dir, 'images', '*.[pP][nN][gG]')))
        else:
            self.image_paths = sorted(glob.glob(os.path.join(data_dir, '*.[jJ][pP][gG]')) + 
                                     glob.glob(os.path.join(data_dir, '*.[pP][nN][gG]')))
            
        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No images found in dataset directory: {data_dir}")

        self.transform = transform if transform else self._default_transform

    @staticmethod
    def _default_transform(image, mask, depth):
        to_tensor = T.ToTensor()
        return to_tensor(image), to_tensor(mask), torch.from_numpy(depth).float()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = self.load_image(img_path)
        mask = self.load_mask(img_path)
        
        # Resolve precomputed depth map path
        fname = os.path.basename(img_path)
        fname_base = os.path.splitext(fname)[0]
        depth_path = None

        search_dirs = []
        if self.depth_dir:
            search_dirs.append(self.depth_dir)
        search_dirs.extend([
            os.path.join(self.data_dir, 'depth_anything_v2'),
            os.path.join(self.data_dir, 'depth_AdelaiDepth')
        ])

        for d in search_dirs:
            if d and os.path.exists(d):
                p_png = os.path.join(d, fname_base + '.png')
                if os.path.exists(p_png):
                    depth_path = p_png
                    break
                p_jpg = os.path.join(d, fname_base + '.jpg')
                if os.path.exists(p_jpg):
                    depth_path = p_jpg
                    break

        if depth_path:
            depth = self.load_depth(depth_path)
        else:
            depth = np.zeros((3, 1024, 1024), dtype=np.float32)

        # Transform expects mask in shape (H, W, C)
        image_t, mask_t, depth_t = self.transform(image, mask.transpose(1, 2, 0), depth)

        return image_t, depth_t, mask_t, fname

    @staticmethod
    def load_depth(path):
        """Loads precomputed depth map and normalizes to (3, 1024, 1024) float32 in [0, 1]."""
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            return np.zeros((3, 1024, 1024), dtype=np.float32)

        # Normalize based on integer depth bit-depth
        if depth.dtype == np.uint16:
            depth = depth.astype(np.float32) / 65535.0
        elif depth.dtype == np.uint8:
            depth = depth.astype(np.float32) / 255.0
        else:
            depth = depth.astype(np.float32)
            if depth.max() > 1.0:
                depth = depth / depth.max()

        depth = cv2.resize(depth, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        if depth.ndim == 2:
            depth = np.repeat(depth[:, :, None], 3, axis=-1)
        elif depth.ndim == 3 and depth.shape[2] == 1:
            depth = np.repeat(depth, 3, axis=-1)
        elif depth.ndim == 3 and depth.shape[2] == 4:
            depth = depth[:, :, :3]

        return depth.transpose(2, 0, 1).astype(np.float32)

    @staticmethod
    def load_image(path):
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Could not load image at {path}")
        img = cv2.resize(img, (1024, 1024))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @staticmethod
    def load_mask(img_path):
        """
        Dynamically renders ground truth mask from the corresponding JSON label file.
        Uses exact image dimensions from JSON metadata to prevent coordinate clipping.
        """
        # Resolve label path
        if 'images' in str(img_path):
            json_path = str(img_path).replace('images', 'labels')
        else:
            parent = os.path.dirname(img_path)
            json_path = os.path.join(os.path.dirname(parent), 'labels', os.path.basename(img_path))
            
        json_path = os.path.splitext(json_path)[0] + '.json'

        if not os.path.exists(json_path):
            # Fallback: check alongside image
            json_path = os.path.splitext(str(img_path))[0] + '.json'
            if not os.path.exists(json_path):
                raise FileNotFoundError(f"Label JSON not found for image: {img_path}")

        # Load JSON and extract true image canvas dimensions
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Dynamic canvas size: guarantees Patient 32 4K (2160x3840) is never truncated
        height = data.get('imageHeight', 1080)
        width = data.get('imageWidth', 1920)
        canvas = np.zeros((height, width), dtype=np.uint8)

        # Draw contours with thickness 35 (exact paper standard)
        for shape in data.get('shapes', []):
            points = shape.get('points', [])
            label = shape.get('label', '').lower().strip()

            # Class mapping: 1 = Ridge, 2 = Silhouette, 3 = Falciform Ligament
            if label.startswith('r'):
                color = 1
            elif label.startswith('s'):
                color = 2
            elif label.startswith('l'):
                color = 3
            else:
                color = 0

            for i in range(1, len(points)):
                pt1 = tuple(map(int, points[i - 1]))
                pt2 = tuple(map(int, points[i]))
                cv2.line(canvas, pt1, pt2, color, 35)

        # Resize to network input resolution (1024, 1024) using INTER_NEAREST
        canvas = cv2.resize(canvas, (1024, 1024), interpolation=cv2.INTER_NEAREST)

        # One-hot map into 4 channels: (0: BG, 1: Ridge, 2: Silhouette, 3: Falciform)
        masks = np.zeros((4, 1024, 1024), dtype=np.uint8)
        masks[0][canvas == 0] = 255
        masks[1][canvas == 1] = 255
        masks[2][canvas == 2] = 255
        masks[3][canvas == 3] = 255

        return masks
