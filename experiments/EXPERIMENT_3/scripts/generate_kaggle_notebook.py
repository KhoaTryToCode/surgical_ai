#!/usr/bin/env python3
"""
Generator script for EXPERIMENT_3 Kaggle Runner Notebook.
Produces a fully self-contained notebook reflecting all bug fixes and architectural specifications.
"""
import os
import json

def build_notebook():
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    def add_md(text):
        nb["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.strip().split("\n")]
        })

    def add_code(text):
        nb["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in text.strip().split("\n")]
        })

    # Cell 1: Header
    add_md("""# 🏥 EXPERIMENT_3: Mask2Former Backbone + Patch Bézier Decoder
### Dedicated Kaggle GPU Runner — L3D Laparoscopic Liver Landmark Detection
**Architecture:** HuggingFace `facebook/mask2former-swin-tiny-ade-semantic` Backbone + MSDeformAttn Pixel Decoder $\\rightarrow$ $8 \\times 8$ Spatial Patch Bézier Decoder.
**Output Archive:** `EXPERIMENT_3_PatchBezier_RESULTS.zip` in `/kaggle/working/`

---
### Key Features
- **Spatially Grounded Queries:** 64 macro-patch queries initialized from stride-16 features via adaptive pooling + 2D sinusoidal PE.
- **Continuous Bézier Parameterization:** Predicts 4 normalized control points in $[0, 1]^2$ per active patch.
- **Topological Continuity Loss:** Enforces $C^0$ boundary matching and $C^1$ tangent alignment across adjacent same-class patches starting at epoch 31.
- **Patient 32 4K Canvas Fix:** Dynamically preserves native resolution for ground truth line drawing at thickness 35 before resizing to $1024 \\times 1024$.
""")

    # Cell 2 & 3: Environment Diagnostics
    add_md("## Step 1: Environment & GPU Diagnostics\nVerify CUDA availability, GPU architecture, VRAM, and install required libraries.")
    add_code("""import os, sys
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Install dependencies silently
!pip install -q transformers surface-distance medpy > /dev/null 2>&1 || true

import numpy as np
# NumPy 2.0 monkeypatch for legacy surface_distance & medpy
for attr, val in [('Inf', np.inf), ('Infinity', np.inf), ('NAN', np.nan), ('NaN', np.nan), ('bool8', np.bool_), ('float_', np.float64)]:
    if not hasattr(np, attr):
        setattr(np, attr, val)

import torch
print('=' * 70)
print('🚀 GPU & ENVIRONMENT DIAGNOSTICS')
print('=' * 70)
print(f'Python Version : {sys.version.split()[0]}')
print(f'PyTorch Version: {torch.__version__}')
print(f'CUDA Available : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Active GPU     : {torch.cuda.get_device_name(0)}')
    print(f'Total VRAM     : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB')
else:
    print('⚠️ WARNING: GPU not detected! Enable GPU Accelerator in Kaggle sidebar.')
""")

    # Cell 4 & 5: Dataset Discovery
    add_md("## Step 2: Automatic Scored Dataset Discovery\nScans `/kaggle/input` using scored directory matching to locate Train, Val, and Test splits.")
    add_code("""import os, glob
from pathlib import Path

def find_dataset_split(split_keyword):
    candidates = []
    search_roots = ['/kaggle/input', '/kaggle/working', './data', '../data']
    for search_root in search_roots:
        if not os.path.exists(search_root):
            continue
        for root, dirs, _ in os.walk(search_root, followlinks=True):
            parts_lower = [p.lower() for p in Path(root).parts]
            if split_keyword.lower() in parts_lower and 'images' in parts_lower:
                score = 0
                if 'khoatrytopublish' in parts_lower: score += 50
                if 'l3d' in parts_lower or any('l3d' in p for p in parts_lower): score += 30
                if 'laparoscopic' in parts_lower: score += 20
                candidates.append((score, root))
            elif os.path.basename(root).lower() == split_keyword.lower():
                if 'images' in [d.lower() for d in dirs]:
                    candidates.append((10, os.path.join(root, 'images')))
                else:
                    candidates.append((5, root))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]

train_img_dir = find_dataset_split('train')
val_img_dir   = find_dataset_split('val')
test_img_dir  = find_dataset_split('test')

print('=' * 70)
print('📂 DATASET DISCOVERY RESULTS')
print('=' * 70)
print(f'Train Images: {train_img_dir}')
print(f'Val Images  : {val_img_dir}')
print(f'Test Images : {test_img_dir}')
assert train_img_dir, 'ERROR: Could not locate Train directory in /kaggle/input'
assert val_img_dir,   'ERROR: Could not locate Val directory in /kaggle/input'
""")

    # Cell 6 & 7: Dataset Loader & Bézier Target Fitting
    add_md("## Step 3: Dataset Loader & Bézier Curve Target Generation\n- Robust typo-tolerant label matching.\n- Patient 32 4K canvas dynamic resolution preservation.\n- Non-degenerate Bézier fitting (linear interpolation for <3 points).\n- Independent polyline parsing to prevent cross-curve distortion.")
    add_code("""import json, cv2
from torch.utils.data import Dataset, DataLoader

def resample_polyline(pts, spacing=5.0):
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
    pts = np.array(pts_in_canvas, dtype=np.float32)
    x_min, y_min, x_max, y_max = patch_bbox
    patch_w = max(x_max - x_min, 1e-5)
    patch_h = max(y_max - y_min, 1e-5)
    
    local_pts = np.zeros_like(pts)
    local_pts[:, 0] = (pts[:, 0] - x_min) / patch_w
    local_pts[:, 1] = (pts[:, 1] - y_min) / patch_h
    
    if len(local_pts) < 2:
        if len(local_pts) == 1:
            p = local_pts[0]
            return np.clip(np.array([p, p, p, p], dtype=np.float32), 0, 1)
        return np.zeros((4, 2), dtype=np.float32)
        
    diffs = np.diff(local_pts, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    cum_dists = np.concatenate(([0], np.cumsum(dists)))
    total_len = cum_dists[-1]
    
    P0 = local_pts[0]
    P3 = local_pts[-1]
    
    if total_len < 1e-6 or len(local_pts) < 3:
        P1 = P0 + (P3 - P0) * (1.0 / 3.0)
        P2 = P0 + (P3 - P0) * (2.0 / 3.0)
        return np.clip(np.array([P0, P1, P2, P3], dtype=np.float32), 0, 1)
        
    t = cum_dists / total_len
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
            P1, P2 = result[0], result[1]
    except Exception:
        P1 = P0 + (P3 - P0) * (1.0 / 3.0)
        P2 = P0 + (P3 - P0) * (2.0 / 3.0)
        
    return np.clip(np.array([P0, P1, P2, P3], dtype=np.float32), 0, 1)

class BezierPatchDataset(Dataset):
    def __init__(self, data_dir, grid_size=8, canvas_size=1024):
        self.data_dir = data_dir
        self.grid_size = grid_size
        self.canvas_size = canvas_size
        self.patch_size = canvas_size // grid_size
        self.num_patches = grid_size * grid_size
        
        image_exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        self.image_files = []
        for ext in image_exts:
            self.image_files.extend(glob.glob(os.path.join(data_dir, ext)))
            self.image_files.extend(glob.glob(os.path.join(data_dir, 'images', ext)))
        self.image_files = sorted(list(set(self.image_files)))
        
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        filename = os.path.basename(img_path)
        
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[0] != self.canvas_size or img.shape[1] != self.canvas_size:
            img = cv2.resize(img, (self.canvas_size, self.canvas_size), interpolation=cv2.INTER_LINEAR)
        img_norm = (img.astype(np.float32) / 255.0 - self.mean) / self.std
        pixel_values = torch.from_numpy(np.transpose(img_norm, (2, 0, 1))).float()
        
        gt_2d, target_class, target_bezier, active_mask = self._load_targets(img_path)
        return pixel_values, gt_2d, target_class, target_bezier, active_mask, filename

    def _load_targets(self, path):
        base_dir = os.path.dirname(path)
        name, _ = os.path.splitext(os.path.basename(path))
        candidates = [
            os.path.join(base_dir, name + '.json'),
            os.path.join(base_dir, '..', 'labels', name + '.json'),
            os.path.join(base_dir, 'labels', name + '.json'),
            os.path.join(base_dir.replace('images', 'labels'), name + '.json')
        ]
        json_path = next((c for c in candidates if os.path.exists(c)), None)
        
        gt_2d = torch.zeros((self.canvas_size, self.canvas_size), dtype=torch.int64)
        target_class = torch.zeros(self.num_patches, dtype=torch.int64)
        target_bezier = torch.zeros((self.num_patches, 4, 2), dtype=torch.float32)
        active_mask = torch.zeros(self.num_patches, dtype=torch.bool)
        
        if not json_path:
            return gt_2d, target_class, target_bezier, active_mask
            
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        orig_h = data.get('imageHeight', 1080)
        orig_w = data.get('imageWidth', 1920)
        canvas_raw = np.zeros((orig_h, orig_w), dtype=np.uint8)
        
        scale_x = self.canvas_size / float(orig_w)
        scale_y = self.canvas_size / float(orig_h)
        
        curves_by_class = {1: [], 2: [], 3: []}
        for shape in data.get('shapes', []):
            label = str(shape.get('label', '')).lower().strip()
            if label.startswith('r') or 'ridge' in label or 'rigde' in label:
                class_id = 1
            elif label.startswith('s') or 'sil' in label or 'margin' in label:
                class_id = 2
            elif label.startswith('f') or 'falc' in label or 'lig' in label or 'ligament' in label:
                class_id = 3
            else:
                continue
                
            points = shape.get('points', [])
            if len(points) < 2:
                continue
                
            for i in range(1, len(points)):
                pt1 = tuple(map(int, points[i - 1]))
                pt2 = tuple(map(int, points[i]))
                cv2.line(canvas_raw, pt1, pt2, int(class_id), thickness=35)
                
            scaled_line = np.array([[p[0] * scale_x, p[1] * scale_y] for p in points], dtype=np.float32)
            resampled = resample_polyline(scaled_line, spacing=5.0)
            if len(resampled) >= 2:
                curves_by_class[class_id].append(resampled)
                
        if canvas_raw.shape[0] != self.canvas_size or canvas_raw.shape[1] != self.canvas_size:
            gt_2d_np = cv2.resize(canvas_raw, (self.canvas_size, self.canvas_size), interpolation=cv2.INTER_NEAREST)
        else:
            gt_2d_np = canvas_raw
            
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                patch_idx = r * self.grid_size + c
                x_min = c * self.patch_size
                y_min = r * self.patch_size
                x_max = (c + 1) * self.patch_size
                y_max = (r + 1) * self.patch_size
                patch_bbox = [x_min, y_min, x_max, y_max]
                
                best_curve_pts = None
                best_class_id = 0
                max_pts_count = 0
                
                for cid in [1, 2, 3]:
                    for curve in curves_by_class[cid]:
                        inside = (
                            (curve[:, 0] >= x_min) & (curve[:, 0] < x_max) &
                            (curve[:, 1] >= y_min) & (curve[:, 1] < y_max)
                        )
                        pts_inside = curve[inside]
                        if len(pts_inside) > max_pts_count:
                            max_pts_count = len(pts_inside)
                            best_curve_pts = pts_inside
                            best_class_id = cid
                            
                if best_curve_pts is not None and max_pts_count >= 2:
                    bezier_ctrl = fit_bezier_to_patch(best_curve_pts, patch_bbox)
                    target_class[patch_idx] = best_class_id
                    target_bezier[patch_idx] = torch.from_numpy(bezier_ctrl).float()
                    active_mask[patch_idx] = True
                    
        return torch.from_numpy(gt_2d_np.astype(np.int64)), target_class, target_bezier, active_mask

train_dataset = BezierPatchDataset(train_img_dir)
val_dataset   = BezierPatchDataset(val_img_dir)
test_dataset  = BezierPatchDataset(test_img_dir) if test_img_dir else None

print(f'Train Dataset: {len(train_dataset)} frames')
print(f'Val Dataset  : {len(val_dataset)} frames')
print(f'Test Dataset : {len(test_dataset) if test_dataset else 0} frames')
""")

    # Cell 8 & 9: Model Architecture
    add_md("## Step 4: Model Architecture (Swin-Tiny Backbone + Bézier Patch Decoder)\n- Loads pretrained `facebook/mask2former-swin-tiny-ade-semantic` `pixel_level_module`.\n- Preserves MSDeformAttn multi-scale 256-dim feature extraction via `output_hidden_states=True`.\n- 8×8 Content-aware query initialization with 2D sinusoidal positional encodings.\n- 6 TransformerDecoderLayers with full cross-attention cycling across multi-scale feature pyramids.")
    add_code("""import torch.nn as nn
import torch.nn.functional as F
from transformers import Mask2FormerForUniversalSegmentation

def bernstein_eval_torch(control_points, num_samples=10):
    N = control_points.shape[0]
    t = torch.linspace(0, 1, num_samples, device=control_points.device, dtype=control_points.dtype)
    B0 = ((1 - t)**3).view(1, num_samples, 1)
    B1 = (3 * (1 - t)**2 * t).view(1, num_samples, 1)
    B2 = (3 * (1 - t) * t**2).view(1, num_samples, 1)
    B3 = (t**3).view(1, num_samples, 1)
    
    P0 = control_points[:, 0:1, :]
    P1 = control_points[:, 1:2, :]
    P2 = control_points[:, 2:3, :]
    P3 = control_points[:, 3:4, :]
    return B0 * P0 + B1 * P1 + B2 * P2 + B3 * P3

def build_2d_sinusoidal_pe(grid_size, embed_dim):
    pe = torch.zeros(grid_size * grid_size, embed_dim)
    d_model = embed_dim // 2
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
    for r in range(grid_size):
        for c in range(grid_size):
            pos_idx = r * grid_size + c
            pe[pos_idx, 0:d_model:2]   = torch.sin(r * div_term)
            pe[pos_idx, 1:d_model:2]   = torch.cos(r * div_term)
            pe[pos_idx, d_model::2]    = torch.sin(c * div_term)
            pe[pos_idx, d_model+1::2]  = torch.cos(c * div_term)
    return pe.unsqueeze(0)

class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, embed_dim)
        )

    def forward(self, queries, context):
        q_norm = self.norm1(queries)
        sa_out, _ = self.self_attn(q_norm, q_norm, q_norm)
        queries = queries + sa_out
        
        q_norm2 = self.norm2(queries)
        ca_out, _ = self.cross_attn(q_norm2, context, context)
        queries = queries + ca_out
        
        q_norm3 = self.norm3(queries)
        ffn_out = self.ffn(q_norm3)
        queries = queries + ffn_out
        return queries

class BezierPatchDecoder(nn.Module):
    def __init__(self, embed_dim=256, grid_size=8, num_classes=4, num_decoder_layers=6, num_heads=8):
        super().__init__()
        self.grid_size = grid_size
        self.num_patches = grid_size * grid_size
        self.register_buffer('pos_enc', build_2d_sinusoidal_pe(grid_size, embed_dim))
        
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(embed_dim=embed_dim, num_heads=num_heads)
            for _ in range(num_decoder_layers)
        ])
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.class_head = nn.Linear(embed_dim, num_classes)
        self.bezier_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 8),
            nn.Sigmoid()
        )

    def forward(self, multi_scale_features):
        B = multi_scale_features[0].size(0)
        f_medium = multi_scale_features[1] # stride-16 features
        
        f_pooled = F.adaptive_avg_pool2d(f_medium, (self.grid_size, self.grid_size))
        queries = f_pooled.flatten(2).transpose(1, 2)
        queries = self.query_proj(queries) + self.pos_enc.to(queries.device)
        
        for i, layer in enumerate(self.decoder_layers):
            f_cur = multi_scale_features[i % 3]
            context = f_cur.flatten(2).transpose(1, 2)
            queries = layer(queries, context)
            
        pred_class = self.class_head(queries)
        pred_bezier = self.bezier_head(queries).view(B, self.num_patches, 4, 2)
        return pred_class, pred_bezier

class BezierPatchModel(nn.Module):
    def __init__(self, grid_size=8, num_classes=4, embed_dim=256, num_decoder_layers=6):
        super().__init__()
        hf_model = Mask2FormerForUniversalSegmentation.from_pretrained(
            'facebook/mask2former-swin-tiny-ade-semantic',
            ignore_mismatched_sizes=True
        )
        self.pixel_level_module = hf_model.model.pixel_level_module
        del hf_model
        
        self.bezier_decoder = BezierPatchDecoder(
            embed_dim=embed_dim,
            grid_size=grid_size,
            num_classes=num_classes,
            num_decoder_layers=num_decoder_layers
        )

    def forward(self, pixel_values):
        pixel_level_outputs = self.pixel_level_module(pixel_values, output_hidden_states=True)
        multi_scale_features = list(pixel_level_outputs.decoder_hidden_states)
        pred_class, pred_bezier = self.bezier_decoder(multi_scale_features)
        return pred_class, pred_bezier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BezierPatchModel(grid_size=8, num_classes=4).to(device)

total_params = sum(p.numel() for p in model.parameters())
train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total parameters    : {total_params:,}')
print(f'Trainable parameters: {train_params:,}')
""")

    # Cell 10 & 11: Loss Formulation
    add_md("## Step 5: Loss Formulation (Phase 1 & Phase 2 Continuity)\n- Multi-class Focal Loss on all 64 patches.\n- Smooth L1 control point loss on active patches.\n- Bernstein sampled point loss ($N=10$) along continuous curve.\n- Phase 2 (epoch $\\ge 31$): Bidirectional $C^0$ boundary gap & $C^1$ cosine tangent alignment across adjacent same-class patches.")
    add_code("""class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, ignore_index=-1):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        ce = F.cross_entropy(pred, target, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        return torch.mean(focal_weight * ce)

class BezierPatchLoss(nn.Module):
    def __init__(self, lambda_cls=2.0, lambda_ctrl=5.0, lambda_sample=2.0,
                 lambda_cont=1.0, lambda_tan=0.5, continuity_phase_epoch=31,
                 num_sample_pts=10, ctrl_beta=0.02):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_ctrl = lambda_ctrl
        self.lambda_sample = lambda_sample
        self.lambda_cont = lambda_cont
        self.lambda_tan = lambda_tan
        self.continuity_phase_epoch = continuity_phase_epoch
        self.num_sample_pts = num_sample_pts
        self.ctrl_beta = ctrl_beta
        self.focal_loss = FocalLoss(gamma=2.0, alpha=0.25)

    def forward(self, pred_class, pred_bezier, target_class, target_bezier, active_mask, epoch=1):
        B, G = pred_class.shape[:2]
        
        # Classification Focal Loss
        L_cls = self.focal_loss(pred_class.view(B*G, 4), target_class.view(B*G))
        
        # Control Point Smooth L1
        if active_mask.any():
            L_ctrl = F.smooth_l1_loss(pred_bezier[active_mask], target_bezier[active_mask], beta=self.ctrl_beta)
            pred_pts = bernstein_eval_torch(pred_bezier[active_mask], self.num_sample_pts)
            gt_pts   = bernstein_eval_torch(target_bezier[active_mask], self.num_sample_pts)
            L_sample = F.l1_loss(pred_pts, gt_pts)
        else:
            L_ctrl = pred_bezier.sum() * 0.0
            L_sample = pred_bezier.sum() * 0.0
            
        total_loss = self.lambda_cls * L_cls + self.lambda_ctrl * L_ctrl + self.lambda_sample * L_sample
        
        L_cont_val = 0.0
        L_tan_val  = 0.0
        
        if epoch >= self.continuity_phase_epoch:
            grid_size = int(G ** 0.5)
            cont_losses = []
            tan_losses  = []
            
            for b in range(B):
                active_b = active_mask[b]
                bezier_b = pred_bezier[b]
                class_b  = target_class[b]
                
                # Horizontal neighbors
                for r in range(grid_size):
                    for c in range(grid_size - 1):
                        idx_left  = r * grid_size + c
                        idx_right = r * grid_size + c + 1
                        if active_b[idx_left] and active_b[idx_right] and (class_b[idx_left] == class_b[idx_right]) and (class_b[idx_left] > 0):
                            P0_l = torch.stack([c + bezier_b[idx_left, 0, 0], r + bezier_b[idx_left, 0, 1]])
                            P3_l = torch.stack([c + bezier_b[idx_left, 3, 0], r + bezier_b[idx_left, 3, 1]])
                            P0_r = torch.stack([(c + 1) + bezier_b[idx_right, 0, 0], r + bezier_b[idx_right, 0, 1]])
                            P3_r = torch.stack([(c + 1) + bezier_b[idx_right, 3, 0], r + bezier_b[idx_right, 3, 1]])
                            
                            gap_fwd = (P3_l - P0_r).pow(2).sum()
                            gap_rev = (P0_l - P3_r).pow(2).sum()
                            cont_losses.append(torch.minimum(gap_fwd, gap_rev))
                            
                            if gap_fwd <= gap_rev:
                                exit_tan  = bezier_b[idx_left, 3]  - bezier_b[idx_left, 2]
                                entry_tan = bezier_b[idx_right, 1] - bezier_b[idx_right, 0]
                            else:
                                exit_tan  = bezier_b[idx_right, 3]  - bezier_b[idx_right, 2]
                                entry_tan = bezier_b[idx_left, 1] - bezier_b[idx_left, 0]
                            cos_sim = F.cosine_similarity(exit_tan.unsqueeze(0), entry_tan.unsqueeze(0)).squeeze(0)
                            tan_losses.append(1.0 - cos_sim)
                            
                # Vertical neighbors
                for r in range(grid_size - 1):
                    for c in range(grid_size):
                        idx_top    = r * grid_size + c
                        idx_bottom = (r + 1) * grid_size + c
                        if active_b[idx_top] and active_b[idx_bottom] and (class_b[idx_top] == class_b[idx_bottom]) and (class_b[idx_top] > 0):
                            P0_t = torch.stack([c + bezier_b[idx_top, 0, 0], r + bezier_b[idx_top, 0, 1]])
                            P3_t = torch.stack([c + bezier_b[idx_top, 3, 0], r + bezier_b[idx_top, 3, 1]])
                            P0_b = torch.stack([c + bezier_b[idx_bottom, 0, 0], (r + 1) + bezier_b[idx_bottom, 0, 1]])
                            P3_b = torch.stack([c + bezier_b[idx_bottom, 3, 0], (r + 1) + bezier_b[idx_bottom, 3, 1]])
                            
                            gap_v_fwd = (P3_t - P0_b).pow(2).sum()
                            gap_v_rev = (P0_t - P3_b).pow(2).sum()
                            cont_losses.append(torch.minimum(gap_v_fwd, gap_v_rev))
                            
                            if gap_v_fwd <= gap_v_rev:
                                exit_tan  = bezier_b[idx_top, 3]  - bezier_b[idx_top, 2]
                                entry_tan = bezier_b[idx_bottom, 1] - bezier_b[idx_bottom, 0]
                            else:
                                exit_tan  = bezier_b[idx_bottom, 3]  - bezier_b[idx_bottom, 2]
                                entry_tan = bezier_b[idx_top, 1] - bezier_b[idx_top, 0]
                            cos_sim = F.cosine_similarity(exit_tan.unsqueeze(0), entry_tan.unsqueeze(0)).squeeze(0)
                            tan_losses.append(1.0 - cos_sim)
                            
            if cont_losses:
                L_cont = torch.stack(cont_losses).mean()
                L_cont_val = L_cont.item()
                total_loss = total_loss + self.lambda_cont * L_cont
                
            if tan_losses:
                L_tan = torch.stack(tan_losses).mean()
                L_tan_val = L_tan.item()
                total_loss = total_loss + self.lambda_tan * L_tan
                
        loss_dict = {
            'loss':        total_loss.item(),
            'cls_loss':    L_cls.item(),
            'ctrl_loss':   L_ctrl.item() if active_mask.any() else 0.0,
            'sample_loss': L_sample.item() if active_mask.any() else 0.0,
            'cont_loss':   L_cont_val,
            'tan_loss':    L_tan_val,
        }
        return total_loss, loss_dict
""")

    # Cell 12 & 13: Evaluation Metrics & Rasterization
    add_md("## Step 6: Evaluation Metrics & Rasterization\n- Rasterizes predicted Bézier curves to 2D class masks with anti-aliasing (`thickness=35`).\n- Computes Macro Dice, Macro IoU, Per-class Dice, and ASSD with OpenCV distance transform fallback.\n- Generates 4-panel diagnostic images for Patient 40 validation frames.")
    add_code("""import time
import pandas as pd

def rasterize_bezier_predictions(pred_class_np, pred_bezier_np, grid_size=8, canvas_size=1024, stroke_width=35):
    B = pred_class_np.shape[0]
    canvas_maps = np.zeros((B, canvas_size, canvas_size), dtype=np.uint8)
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
                r, c = i // grid_size, i % grid_size
                patch_x_min = c * patch_size_px
                patch_y_min = r * patch_size_px
                cps = pred_bezier_np[b, i]
                
                curve_x = B0 * cps[0, 0] + B1 * cps[1, 0] + B2 * cps[2, 0] + B3 * cps[3, 0]
                curve_y = B0 * cps[0, 1] + B1 * cps[1, 1] + B2 * cps[2, 1] + B3 * cps[3, 1]
                
                global_x = patch_x_min + curve_x * patch_size_px
                global_y = patch_y_min + curve_y * patch_size_px
                
                pts = np.column_stack((global_x, global_y)).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(canvas_maps[b], [pts], isClosed=False, color=cls_id, thickness=stroke_width, lineType=cv2.LINE_AA)
    return canvas_maps

def compute_dice_iou(pred_bin, gt_bin):
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    sum_area = pred_bin.sum() + gt_bin.sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    if sum_area == 0:
        return 1.0, 1.0
    if pred_bin.sum() == 0 or gt_bin.sum() == 0:
        return 0.0, 0.0
    dice = 2.0 * intersection / sum_area
    iou = intersection / union if union > 0 else 0.0
    return float(dice), float(iou)

def compute_assd(pred_mask, gt_mask, fallback=80.0):
    if pred_mask.sum() == 0 or gt_mask.sum() == 0:
        return fallback
    try:
        from surface_distance import metrics as sd_metrics
        dist = sd_metrics.compute_surface_distances(gt_mask.astype(bool), pred_mask.astype(bool), spacing_mm=(1.0, 1.0))
        assd = sd_metrics.compute_average_surface_distance(dist)
        return float((assd[0] + assd[1]) / 2.0)
    except Exception:
        try:
            contours_p, _ = cv2.findContours(pred_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours_g, _ = cv2.findContours(gt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            edge_p = np.zeros_like(pred_mask, dtype=np.uint8)
            edge_g = np.zeros_like(gt_mask, dtype=np.uint8)
            cv2.drawContours(edge_p, contours_p, -1, 1, 1)
            cv2.drawContours(edge_g, contours_g, -1, 1, 1)
            if edge_p.sum() == 0 or edge_g.sum() == 0:
                return fallback
            dist_g = cv2.distanceTransform((1 - edge_g).astype(np.uint8), cv2.DIST_L2, 3)
            dist_p = cv2.distanceTransform((1 - edge_p).astype(np.uint8), cv2.DIST_L2, 3)
            return float((np.mean(dist_g[edge_p > 0]) + np.mean(dist_p[edge_g > 0])) / 2.0)
        except Exception:
            return fallback

def compute_frame_metrics(pred_map, gt_2d):
    rd, riou = compute_dice_iou(pred_map == 1, gt_2d == 1)
    sd, siou = compute_dice_iou(pred_map == 2, gt_2d == 2)
    fd, fiou = compute_dice_iou(pred_map == 3, gt_2d == 3)
    fgd, fgiou = compute_dice_iou(pred_map > 0, gt_2d > 0)
    
    rassd = compute_assd(pred_map == 1, gt_2d == 1)
    sassd = compute_assd(pred_map == 2, gt_2d == 2)
    fassd = compute_assd(pred_map == 3, gt_2d == 3)
    
    return {
        'macro_dice': (rd + sd + fd) / 3.0,
        'macro_iou':  (riou + siou + fiou) / 3.0,
        'macro_assd': (rassd + sassd + fassd) / 3.0,
        'ridge_dice': rd, 'ridge_iou': riou, 'ridge_assd': rassd,
        'sil_dice': sd,   'sil_iou': siou,   'sil_assd': sassd,
        'falc_dice': fd,  'falc_iou': fiou,  'falc_assd': fassd,
        'fg_dice': fgd,   'fg_iou': fgiou
    }

def _render_patient40_panel(img_tensor, gt_2d, pred_map, fname, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img  = (img_tensor * std + mean).clamp(0, 1).numpy()
    img_bgr = cv2.cvtColor((np.transpose(img, (1, 2, 0)) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    
    colors = {1: (0, 0, 255), 2: (0, 255, 0), 3: (255, 0, 0)}
    def apply_overlay(base, mask):
        res = base.copy()
        for cid, color in colors.items():
            res[mask == cid] = cv2.addWeighted(res[mask == cid], 0.4, np.full_like(res[mask == cid], color), 0.6, 0)
        return res
        
    gt_vis = apply_overlay(img_bgr, gt_2d)
    pred_vis = apply_overlay(img_bgr, pred_map)
    
    err = np.zeros_like(img_bgr)
    err[(pred_map == gt_2d) & (gt_2d > 0)] = (0, 255, 0) # TP
    err[(pred_map != gt_2d) & (pred_map > 0)] = (0, 0, 255) # FP
    err[(pred_map != gt_2d) & (gt_2d > 0)] = (255, 0, 0) # FN
    
    montage = np.vstack([np.hstack([img_bgr, gt_vis]), np.hstack([pred_vis, err])])
    cv2.imwrite(os.path.join(out_dir, f"{Path(fname).stem}_diag.png"), montage)

def run_evaluation(model, dataloader, device, split_name='Val', save_patient40_dir=None):
    model.eval()
    records, latencies = [], []
    with torch.no_grad():
        for pixel_values, gt_2d_batch, _, _, _, filenames in dataloader:
            pixel_values = pixel_values.to(device)
            t0 = time.time()
            pred_class, pred_bezier = model(pixel_values)
            if device.type == 'cuda': torch.cuda.synchronize()
            latencies.append((time.time() - t0) / pixel_values.shape[0])
            
            pred_cls_np = pred_class.argmax(dim=-1).cpu().numpy()
            pred_bez_np = pred_bezier.cpu().numpy()
            pred_maps = rasterize_bezier_predictions(pred_cls_np, pred_bez_np)
            gt_maps = gt_2d_batch.numpy()
            
            for b, fname in enumerate(filenames):
                m = compute_frame_metrics(pred_maps[b], gt_maps[b])
                fname_base = os.path.basename(str(fname))
                is_p40 = 'Patient_40' in fname_base or '_40_' in fname_base
                m['filename'] = fname_base
                m['is_patient40'] = is_p40
                records.append(m)
                if save_patient40_dir and is_p40:
                    _render_patient40_panel(pixel_values[b].cpu(), gt_maps[b], pred_maps[b], fname_base, save_patient40_dir)
                    
    df = pd.DataFrame(records)
    p40_df = df[df['is_patient40']]
    mean_lat = np.mean(latencies[3:]) if len(latencies) > 3 else (np.mean(latencies) if latencies else 0.0)
    summary = {
        'split': split_name,
        'frames': len(records),
        'macro_dice': float(df['macro_dice'].mean()),
        'macro_iou':  float(df['macro_iou'].mean()),
        'macro_assd': float(df['macro_assd'].mean()),
        'ridge_dice': float(df['ridge_dice'].mean()),
        'sil_dice':   float(df['sil_dice'].mean()),
        'falc_dice':  float(df['falc_dice'].mean()),
        'fg_dice':    float(df['fg_dice'].mean()),
        'patient_40_dice': float(p40_df['macro_dice'].mean()) if len(p40_df) > 0 else 0.0,
        'patient_40_assd': float(p40_df['macro_assd'].mean()) if len(p40_df) > 0 else 80.0,
        'fps': float(1.0 / mean_lat) if mean_lat > 0 else 0.0
    }
    return summary, df
""")

    # Cell 14 & 15: Training Setup
    add_md("## Step 7: Training Configuration")
    add_code("""EPOCHS             = 60
BATCH_SIZE         = 1
ACCUMULATION_STEPS = 4   # Effective batch size = 4
LEARNING_RATE      = 8e-5
WEIGHT_DECAY       = 3e-5
PHASE2_EPOCH       = 31  # Activates continuity + tangent loss
SAVE_DIR           = '/kaggle/working/EXP3_PatchBezier_results'
ZIP_NAME           = 'EXPERIMENT_3_PatchBezier_RESULTS.zip'
os.makedirs(SAVE_DIR, exist_ok=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True) if test_dataset else None

criterion  = BezierPatchLoss(lambda_cls=2.0, lambda_ctrl=5.0, lambda_sample=2.0,
                              lambda_cont=1.0, lambda_tan=0.5, continuity_phase_epoch=PHASE2_EPOCH)
optimizer  = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
scaler     = torch.amp.GradScaler('cuda')

print('✓ Training configuration initialized:')
print(f'  Epochs: {EPOCHS} | Micro Batch: {BATCH_SIZE} | Accumulation: {ACCUMULATION_STEPS} | Effective Batch: {BATCH_SIZE*ACCUMULATION_STEPS}')
print(f'  Optimizer: AdamW (lr={LEARNING_RATE}, wd={WEIGHT_DECAY}) | Cosine Annealing')
print(f'  Phase 2 Continuity activates at epoch {PHASE2_EPOCH}')
""")

    # Cell 16 & 17: Training Loop
    add_md("## Step 8: Training Loop (60 Epochs)")
    add_code("""from tqdm.auto import tqdm

best_val_dice = 0.0
training_log = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    accum_loss = {'cls': 0.0, 'ctrl': 0.0, 'sample': 0.0, 'cont': 0.0, 'tan': 0.0}
    optimizer.zero_grad()
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch:02d}/{EPOCHS:02d} [Train]')
    for step, (pixel_values, _, target_class, target_bezier, active_mask, _) in enumerate(pbar):
        pixel_values  = pixel_values.to(device)
        target_class  = target_class.to(device)
        target_bezier = target_bezier.to(device)
        active_mask   = active_mask.to(device)
        
        with torch.amp.autocast('cuda'):
            pred_class, pred_bezier = model(pixel_values)
            loss, loss_dict = criterion(pred_class, pred_bezier, target_class, target_bezier, active_mask, epoch)
            loss_scaled = loss / ACCUMULATION_STEPS
            
        scaler.scale(loss_scaled).backward()
        
        if (step + 1) % ACCUMULATION_STEPS == 0 or (step + 1) == len(train_loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
        epoch_loss += loss_dict['loss']
        accum_loss['cls']  += loss_dict['cls_loss']
        accum_loss['ctrl'] += loss_dict['ctrl_loss']
        accum_loss['cont'] += loss_dict['cont_loss']
        
        pbar.set_postfix({
            'loss': f"{epoch_loss / (step + 1):.4f}",
            'cls':  f"{accum_loss['cls'] / (step + 1):.4f}",
            'ctrl': f"{accum_loss['ctrl'] / (step + 1):.4f}",
            'phase': 'P2' if epoch >= PHASE2_EPOCH else 'P1'
        })
        
    scheduler.step()
    
    val_summary, _ = run_evaluation(model, val_loader, device, split_name=f'Val-Ep{epoch}')
    cur_val_dice = val_summary['macro_dice']
    
    print(f"\\nEpoch {epoch:02d}: TrainLoss={epoch_loss/len(train_loader):.4f} | Val MacroDice={cur_val_dice:.4f} | Ridge={val_summary['ridge_dice']:.4f} | Sil={val_summary['sil_dice']:.4f} | Falc={val_summary['falc_dice']:.4f} | ASSD={val_summary['macro_assd']:.2f}px")
    
    if cur_val_dice > best_val_dice:
        best_val_dice = cur_val_dice
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_dice': best_val_dice,
            'grid_size': 8
        }, os.path.join(SAVE_DIR, 'best_model.pth'))
        print(f"   ⭐ New best model saved! (Val MacroDice: {best_val_dice:.4f})")
        
    training_log.append({
        'epoch': epoch,
        'train_loss': epoch_loss / len(train_loader),
        'val_macro_dice': cur_val_dice,
        'val_macro_assd': val_summary['macro_assd'],
        'val_ridge_dice': val_summary['ridge_dice'],
        'val_sil_dice':   val_summary['sil_dice'],
        'val_falc_dice':  val_summary['falc_dice'],
        'val_p40_dice':   val_summary['patient_40_dice'],
        'lr': scheduler.get_last_lr()[0]
    })

pd.DataFrame(training_log).to_csv(os.path.join(SAVE_DIR, 'training_log.csv'), index=False)
print('✓ Training complete. Training curves logged.')
""")

    # Cell 18 & 19: Final Benchmark Evaluation
    add_md("## Step 9: Final Benchmark Evaluation\nReloads the best checkpoint, executes full validation with Patient 40 diagnostic rendering, and runs test evaluation.")
    add_code("""checkpoint = torch.load(os.path.join(SAVE_DIR, 'best_model.pth'), map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
print(f"✓ Loaded best model from epoch {checkpoint['epoch']} with Val MacroDice {checkpoint['best_val_dice']:.4f}")

patient40_dir = os.path.join(SAVE_DIR, 'patient_40_diagnostics')
final_val_summary, final_val_df = run_evaluation(model, val_loader, device, split_name='Val-Final', save_patient40_dir=patient40_dir)
final_val_df.to_csv(os.path.join(SAVE_DIR, 'val_per_frame_metrics.csv'), index=False)

final_test_summary = {}
if test_loader:
    final_test_summary, final_test_df = run_evaluation(model, test_loader, device, split_name='Test-Final')
    final_test_df.to_csv(os.path.join(SAVE_DIR, 'test_per_frame_metrics.csv'), index=False)

metrics_summary = {'val': final_val_summary, 'test': final_test_summary}
with open(os.path.join(SAVE_DIR, 'metrics_summary.json'), 'w') as f:
    json.dump(metrics_summary, f, indent=4)

print('\\n' + '='*65)
print('FINAL BENCHMARK RESULTS — EXPERIMENT_3 (Patch Bézier Decoder)')
print('='*65)
print(f"| Metric            | Validation (122 frames) | Test (109 frames) |")
print(f"|-------------------|-------------------------|-------------------|")
print(f"| Macro Dice        | {final_val_summary.get('macro_dice', 0):.4f}                  | {final_test_summary.get('macro_dice', 0):.4f}            |")
print(f"| Macro ASSD (px)   | {final_val_summary.get('macro_assd', 0):.2f} px               | {final_test_summary.get('macro_assd', 0):.2f} px         |")
print(f"| Macro IoU         | {final_val_summary.get('macro_iou', 0):.4f}                  | {final_test_summary.get('macro_iou', 0):.4f}            |")
print(f"| Ridge Dice        | {final_val_summary.get('ridge_dice', 0):.4f}                  | {final_test_summary.get('ridge_dice', 0):.4f}            |")
print(f"| Silhouette Dice   | {final_val_summary.get('sil_dice', 0):.4f}                  | {final_test_summary.get('sil_dice', 0):.4f}            |")
print(f"| Falciform Dice    | {final_val_summary.get('falc_dice', 0):.4f}                  | {final_test_summary.get('falc_dice', 0):.4f}            |")
print(f"| Patient 40 Dice   | {final_val_summary.get('patient_40_dice', 0):.4f}                  | —                 |")
print(f"| Inference FPS     | {final_val_summary.get('fps', 0):.2f} FPS                 | —                 |")
print('='*65)
""")

    # Cell 20 & 21: Package Archive
    add_md("## Step 10: Package Results Archive")
    add_code("""import zipfile
zip_path = os.path.join('/kaggle/working', ZIP_NAME)
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, _, files in os.walk(SAVE_DIR):
        for file in files:
            if file.endswith('.zip'): continue
            fp = os.path.join(root, file)
            zipf.write(fp, os.path.relpath(fp, SAVE_DIR))

print(f'✅ Results packaged successfully: {zip_path}')
print(f'   Archive Size: {os.path.getsize(zip_path) / (1024*1024):.2f} MB')
print(f'   Download from Kaggle sidebar → Output → {ZIP_NAME}')
""")

    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../notebooks/EXP3_PatchBezier_Kaggle.ipynb'))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(nb, f, indent=2)
    print(f'Successfully generated {out_path}')

if __name__ == '__main__':
    build_notebook()
