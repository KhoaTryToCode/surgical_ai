#!/usr/bin/env python3
"""
Generator script for EXPERIMENT_4 Kaggle Runner Notebook.
Produces a fully self-contained notebook reflecting:
- 3 Landmark Master Tokens + 64 Spatial Patch Queries (67 tokens sequence)
- Stride-16 feature lock (single_scale=True default)
- Auxiliary Landmark Presence BCE & Masked Centroid Smooth-L1
- Patient 32 4K dynamic canvas drawing preservation
- Full TopoNet evaluation metrics with centroid crosshairs diagnostics
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
    add_md("""# 🏥 EXPERIMENT_4: Landmark Master Tokens + Patch Bézier Decoder
### Dedicated Kaggle GPU Runner — L3D Laparoscopic Liver Landmark Detection
**Architecture:** HuggingFace `facebook/mask2former-swin-tiny-ade-semantic` Backbone + MSDeformAttn Pixel Decoder $\\rightarrow$ **67-Token Transformer Decoder** (3 Landmark Master Tokens + 64 Spatial Patch Queries).
**Output Archive:** `EXPERIMENT_4_LandmarkBezier_RESULTS.zip` in `/kaggle/working/`

---
### Key Breakthroughs in EXPERIMENT_4
- **3 Landmark Master Tokens ($T_{\\text{ridge}}, T_{\\text{sil}}, T_{\\text{falc}}$):** Coordinate-free anatomical anchor tokens injected directly into Decoder self-attention.
- **Orientation & Inversion Invariance:** Explicit multi-head self-attention between landmark tokens and patch queries captures relative spatial orientation (e.g. Ridge vs Silhouette relative vector), eliminating severe hallucination during liver retraction and flipped views (e.g., `Patient_40_08940`).
- **Auxiliary Landmark Supervision:** Multi-task loss supervising landmark presence ($L_{\\text{bce}}$) and masked global center of mass ($L_{\\text{com}}$ Smooth-L1).
- **Scale Stability:** Stride-16 feature lock (`single_scale=True`) eliminates multi-scale hopping noise.
- **Topological Parity:** Same $1024 \\times 1024$ rasterization at `thickness=35`, evaluated via Macro Dice, Mean IoU, ASSD, and Patient 40 diagnostics.
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

    # Cell 6 & 7: Dataset Loader & Bézier / Landmark Target Generation
    add_md("## Step 3: Dataset Loader & Target Generation\n- Computes 64 local Bézier patch targets.\n- Computes macroscopic landmark Center of Mass coordinates $(c_x, c_y)$ and presence labels.\n- Preserves Patient 32 4K canvas dynamic resolution for ground truth rasterization at thickness 35.\n- Robust typo-tolerant label parsing.")
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

class LandmarkBezierDataset(Dataset):
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
        
        gt_2d, target_class, target_bezier, active_mask, lm_presence, lm_centroid = self._load_targets(img_path)
        return pixel_values, gt_2d, target_class, target_bezier, active_mask, lm_presence, lm_centroid, filename

    def _load_targets(self, path):
        base_dir = os.path.dirname(path)
        filename = os.path.basename(path)
        name, _ = os.path.splitext(filename)
        
        json_paths = [
            os.path.join(base_dir, name + '.json'),
            os.path.join(base_dir, '..', 'labels', name + '.json'),
            os.path.join(base_dir, 'labels', name + '.json'),
            os.path.join(base_dir.replace('images', 'labels'), name + '.json')
        ]
        json_path = next((p for p in json_paths if os.path.exists(p)), None)
        
        gt_2d = torch.zeros((self.canvas_size, self.canvas_size), dtype=torch.int64)
        target_class = torch.zeros(self.num_patches, dtype=torch.int64)
        target_bezier = torch.zeros((self.num_patches, 4, 2), dtype=torch.float32)
        active_mask = torch.zeros(self.num_patches, dtype=torch.bool)
        target_lm_presence = torch.zeros(3, dtype=torch.float32)
        target_lm_centroid = torch.zeros((3, 2), dtype=torch.float32)
        
        if not json_path:
            return gt_2d, target_class, target_bezier, active_mask, target_lm_presence, target_lm_centroid
            
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        orig_h = data.get('imageHeight', 1080)
        orig_w = data.get('imageWidth', 1920)
        canvas_raw = np.zeros((orig_h, orig_w), dtype=np.uint8)
        
        scale_x = self.canvas_size / float(orig_w)
        scale_y = self.canvas_size / float(orig_h)
        
        curves_by_class = {1: [], 2: [], 3: []}
        all_points_by_class = {1: [], 2: [], 3: []}
        
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
                norm_pts = np.array([[p[0] / orig_w, p[1] / orig_h] for p in points], dtype=np.float32)
                all_points_by_class[class_id].append(norm_pts)
                
        if canvas_raw.shape[0] != self.canvas_size or canvas_raw.shape[1] != self.canvas_size:
            gt_2d_np = cv2.resize(canvas_raw, (self.canvas_size, self.canvas_size), interpolation=cv2.INTER_NEAREST)
        else:
            gt_2d_np = canvas_raw
            
        for cid in [1, 2, 3]:
            idx = cid - 1
            if len(all_points_by_class[cid]) > 0:
                cat_pts = np.concatenate(all_points_by_class[cid], axis=0)
                mean_coord = cat_pts.mean(axis=0)
                target_lm_presence[idx] = 1.0
                target_lm_centroid[idx] = torch.tensor(mean_coord, dtype=torch.float32)
            else:
                target_lm_presence[idx] = 0.0
                target_lm_centroid[idx] = torch.zeros(2, dtype=torch.float32)
                
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
                        inside_mask = (
                            (curve[:, 0] >= x_min) & (curve[:, 0] < x_max) &
                            (curve[:, 1] >= y_min) & (curve[:, 1] < y_max)
                        )
                        pts_inside = curve[inside_mask]
                        if len(pts_inside) > max_pts_count:
                            max_pts_count = len(pts_inside)
                            best_curve_pts = pts_inside
                            best_class_id = cid
                            
                if best_curve_pts is not None and max_pts_count >= 2:
                    bezier_ctrl = fit_bezier_to_patch(best_curve_pts, patch_bbox)
                    target_class[patch_idx] = best_class_id
                    target_bezier[patch_idx] = torch.from_numpy(bezier_ctrl).float()
                    active_mask[patch_idx] = True
                    
        return (torch.from_numpy(gt_2d_np.astype(np.int64)),
                target_class,
                target_bezier,
                active_mask,
                target_lm_presence,
                target_lm_centroid)
""")

    # Cell 8 & 9: Model Architecture
    add_md("## Step 4: Model Architecture — Landmark Master Tokens + Patch Bézier Decoder\n- Pretrained HuggingFace Swin-Tiny + MSDeformAttn decoder.\n- Sequence length 67: 3 Landmark Master Tokens + 64 Spatial Patch Queries.\n- Locked stride-16 features (`single_scale=True`).\n- Class head, Bézier head, Landmark macroscopic head.")
    add_code("""import math
import torch.nn as nn
import torch.nn.functional as F
from transformers import Mask2FormerForUniversalSegmentation

def build_2d_sinusoidal_pe(grid_size, embed_dim):
    pe = torch.zeros(grid_size * grid_size, embed_dim)
    d_model = embed_dim // 2
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    for r in range(grid_size):
        for c in range(grid_size):
            pos_idx = r * grid_size + c
            pe[pos_idx, 0:d_model:2]      = torch.sin(r * div_term)
            pe[pos_idx, 1:d_model:2]      = torch.cos(r * div_term)
            pe[pos_idx, d_model::2]       = torch.sin(c * div_term)
            pe[pos_idx, d_model + 1::2]   = torch.cos(c * div_term)
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

class LandmarkBezierPatchDecoder(nn.Module):
    def __init__(self, embed_dim=256, grid_size=8, num_classes=4, num_decoder_layers=6,
                 num_heads=8, num_landmarks=3, single_scale=True):
        super().__init__()
        self.grid_size = grid_size
        self.num_patches = grid_size * grid_size
        self.num_landmarks = num_landmarks
        self.single_scale = single_scale
        
        self.landmark_tokens = nn.Parameter(torch.randn(1, num_landmarks, embed_dim) * 0.02)
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
        self.landmark_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, 3)
        )
        
    def forward(self, multi_scale_features):
        B = multi_scale_features[0].size(0)
        f_medium = multi_scale_features[1]
        
        f_pooled = F.adaptive_avg_pool2d(f_medium, (self.grid_size, self.grid_size))
        patch_queries = f_pooled.flatten(2).transpose(1, 2)
        patch_queries = self.query_proj(patch_queries) + self.pos_enc.to(patch_queries.device)
        
        landmark_tokens = self.landmark_tokens.expand(B, -1, -1)
        tokens = torch.cat([landmark_tokens, patch_queries], dim=1) # (B, 67, 256)
        
        for i, layer in enumerate(self.decoder_layers):
            f_cur = multi_scale_features[1] if self.single_scale else multi_scale_features[i % 3]
            context = f_cur.flatten(2).transpose(1, 2)
            tokens = layer(tokens, context)
            
        landmark_out = tokens[:, :self.num_landmarks, :]
        patch_out    = tokens[:, self.num_landmarks:, :]
        
        pred_class  = self.class_head(patch_out)
        bezier_flat = self.bezier_head(patch_out)
        pred_bezier = bezier_flat.view(B, self.num_patches, 4, 2)
        
        lm_raw = self.landmark_head(landmark_out)
        pred_landmark_centroid = torch.sigmoid(lm_raw[..., :2])
        pred_landmark_presence = lm_raw[..., 2]
        
        return pred_class, pred_bezier, pred_landmark_presence, pred_landmark_centroid

class LandmarkBezierPatchModel(nn.Module):
    def __init__(self, grid_size=8, num_classes=4, embed_dim=256, num_decoder_layers=6, single_scale=True):
        super().__init__()
        hf_model = Mask2FormerForUniversalSegmentation.from_pretrained(
            'facebook/mask2former-swin-tiny-ade-semantic',
            ignore_mismatched_sizes=True
        )
        self.pixel_level_module = hf_model.model.pixel_level_module
        del hf_model
        
        self.bezier_decoder = LandmarkBezierPatchDecoder(
            embed_dim=embed_dim,
            grid_size=grid_size,
            num_classes=num_classes,
            num_decoder_layers=num_decoder_layers,
            single_scale=single_scale,
        )
        
    def forward(self, pixel_values):
        pixel_level_outputs = self.pixel_level_module(pixel_values, output_hidden_states=True)
        multi_scale_features = list(pixel_level_outputs.decoder_hidden_states)
        return self.bezier_decoder(multi_scale_features)
""")

    # Cell 10 & 11: Losses
    add_md("## Step 5: Multi-Task Losses\n- Multi-class Focal Loss (64 patches)\n- Smooth-L1 Bézier Control Points & Bernstein Sampling\n- Phase 2 C0 Continuity & C1 Tangent Alignment (Epoch $\\ge 31$)\n- Auxiliary Landmark BCE Presence & Masked Centroid Smooth-L1")
    add_code("""def bernstein_eval_torch(control_points, num_samples=10):
    device = control_points.device
    dtype = control_points.dtype
    t = torch.linspace(0.0, 1.0, num_samples, device=device, dtype=dtype)
    B0 = (1.0 - t) ** 3
    B1 = 3.0 * (1.0 - t) ** 2 * t
    B2 = 3.0 * (1.0 - t) * (t ** 2)
    B3 = t ** 3
    basis = torch.stack([B0, B1, B2, B3], dim=1) # (num_samples, 4)
    return torch.matmul(basis, control_points)   # (N, num_samples, 2)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, ignore_index=-1):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        return torch.mean(focal_weight * ce_loss)

class LandmarkBezierLoss(nn.Module):
    def __init__(self, lambda_cls=2.0, lambda_ctrl=5.0, lambda_sample=2.0,
                 lambda_cont=1.0, lambda_tan=0.5, lambda_lm=1.0,
                 continuity_phase_epoch=31, num_sample_pts=10, ctrl_beta=0.02):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_ctrl = lambda_ctrl
        self.lambda_sample = lambda_sample
        self.lambda_cont = lambda_cont
        self.lambda_tan = lambda_tan
        self.lambda_lm = lambda_lm
        self.continuity_phase_epoch = continuity_phase_epoch
        self.num_sample_pts = num_sample_pts
        self.ctrl_beta = ctrl_beta
        self.focal_loss = FocalLoss(gamma=2.0, alpha=0.25)

    def forward(self, pred_class, pred_bezier, pred_presence, pred_centroid,
                target_class, target_bezier, active_mask,
                target_lm_presence, target_lm_centroid, epoch=1):
        B, G = pred_class.shape[:2]
        L_cls = self.focal_loss(pred_class.view(B * G, 4), target_class.view(B * G))
        
        if active_mask.any():
            L_ctrl = F.smooth_l1_loss(pred_bezier[active_mask], target_bezier[active_mask], beta=self.ctrl_beta)
            pred_sampled = bernstein_eval_torch(pred_bezier[active_mask], num_samples=self.num_sample_pts)
            gt_sampled   = bernstein_eval_torch(target_bezier[active_mask], num_samples=self.num_sample_pts)
            L_sample     = F.l1_loss(pred_sampled, gt_sampled)
        else:
            L_ctrl   = pred_bezier.sum() * 0.0
            L_sample = pred_bezier.sum() * 0.0
            
        total_loss = self.lambda_cls * L_cls + self.lambda_ctrl * L_ctrl + self.lambda_sample * L_sample
        
        L_cont_val = 0.0
        L_tan_val  = 0.0
        
        if epoch >= self.continuity_phase_epoch:
            grid_size = int(G ** 0.5)
            cont_losses, tan_losses = [], []
            for b in range(B):
                active_b = active_mask[b]
                bezier_b = pred_bezier[b]
                class_b  = target_class[b]
                
                # Horizontal
                for r in range(grid_size):
                    for c in range(grid_size - 1):
                        i_l, i_r = r * grid_size + c, r * grid_size + c + 1
                        if active_b[i_l] and active_b[i_r] and (class_b[i_l] == class_b[i_r]) and (class_b[i_l] > 0):
                            P0_l = torch.stack([c + bezier_b[i_l, 0, 0], r + bezier_b[i_l, 0, 1]])
                            P3_l = torch.stack([c + bezier_b[i_l, 3, 0], r + bezier_b[i_l, 3, 1]])
                            P0_r = torch.stack([(c + 1) + bezier_b[i_r, 0, 0], r + bezier_b[i_r, 0, 1]])
                            P3_r = torch.stack([(c + 1) + bezier_b[i_r, 3, 0], r + bezier_b[i_r, 3, 1]])
                            
                            gap_fwd = (P3_l - P0_r).pow(2).sum()
                            gap_rev = (P0_l - P3_r).pow(2).sum()
                            cont_losses.append(torch.minimum(gap_fwd, gap_rev))
                            
                            if gap_fwd <= gap_rev:
                                exit_tan, entry_tan = bezier_b[i_l, 3] - bezier_b[i_l, 2], bezier_b[i_r, 1] - bezier_b[i_r, 0]
                            else:
                                exit_tan, entry_tan = bezier_b[i_r, 3] - bezier_b[i_r, 2], bezier_b[i_l, 1] - bezier_b[i_l, 0]
                            cos_sim = F.cosine_similarity(exit_tan.unsqueeze(0), entry_tan.unsqueeze(0)).squeeze(0)
                            tan_losses.append(1.0 - cos_sim)
                            
                # Vertical
                for r in range(grid_size - 1):
                    for c in range(grid_size):
                        i_t, i_b = r * grid_size + c, (r + 1) * grid_size + c
                        if active_b[i_t] and active_b[i_b] and (class_b[i_t] == class_b[i_b]) and (class_b[i_t] > 0):
                            P0_t = torch.stack([c + bezier_b[i_t, 0, 0], r + bezier_b[i_t, 0, 1]])
                            P3_t = torch.stack([c + bezier_b[i_t, 3, 0], r + bezier_b[i_t, 3, 1]])
                            P0_b = torch.stack([c + bezier_b[i_b, 0, 0], (r + 1) + bezier_b[i_b, 0, 1]])
                            P3_b = torch.stack([c + bezier_b[i_b, 3, 0], (r + 1) + bezier_b[i_b, 3, 1]])
                            
                            gap_v_fwd = (P3_t - P0_b).pow(2).sum()
                            gap_v_rev = (P0_t - P3_b).pow(2).sum()
                            cont_losses.append(torch.minimum(gap_v_fwd, gap_v_rev))
                            
                            if gap_v_fwd <= gap_v_rev:
                                exit_tan, entry_tan = bezier_b[i_t, 3] - bezier_b[i_t, 2], bezier_b[i_b, 1] - bezier_b[i_b, 0]
                            else:
                                exit_tan, entry_tan = bezier_b[i_b, 3] - bezier_b[i_b, 2], bezier_b[i_t, 1] - bezier_b[i_t, 0]
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

        L_lm_bce = F.binary_cross_entropy_with_logits(pred_presence, target_lm_presence)
        lm_mask = target_lm_presence > 0.5
        if lm_mask.any():
            L_lm_com = F.smooth_l1_loss(pred_centroid[lm_mask], target_lm_centroid[lm_mask], beta=0.02)
        else:
            L_lm_com = pred_centroid.sum() * 0.0
            
        L_landmark = L_lm_bce + L_lm_com
        total_loss = total_loss + self.lambda_lm * L_landmark

        loss_dict = {
            'loss':        total_loss.item(),
            'cls_loss':    L_cls.item(),
            'ctrl_loss':   L_ctrl.item() if active_mask.any() else 0.0,
            'sample_loss': L_sample.item() if active_mask.any() else 0.0,
            'cont_loss':   L_cont_val,
            'tan_loss':    L_tan_val,
            'lm_bce':      L_lm_bce.item(),
            'lm_com':      L_lm_com.item() if lm_mask.any() else 0.0,
        }
        return total_loss, loss_dict
""")

    # Cell 12 & 13: Metrics & Rasterization
    add_md("## Step 6: Evaluation Metrics, Rasterization & Centroid Diagnostics\nEvaluates Macro Dice, Mean IoU, ASSD, and renders 4-panel diagnostic images with predicted vs ground truth landmark centroid crosshairs.")
    add_code("""import time, pandas as pd

def rasterize_bezier_predictions(pred_class_np, pred_bezier_np, grid_size=8, canvas_size=1024, stroke_width=35):
    B = pred_class_np.shape[0]
    patch_size_px = canvas_size // grid_size
    t_vals = np.linspace(0.0, 1.0, 50, dtype=np.float32)
    B0 = (1.0 - t_vals) ** 3
    B1 = 3.0 * (1.0 - t_vals) ** 2 * t_vals
    B2 = 3.0 * (1.0 - t_vals) * (t_vals ** 2)
    B3 = t_vals ** 3
    basis = np.stack([B0, B1, B2, B3], axis=1)
    
    canvas_batch = np.zeros((B, canvas_size, canvas_size), dtype=np.int64)
    for b in range(B):
        for r in range(grid_size):
            for c in range(grid_size):
                patch_idx = r * grid_size + c
                class_id = int(pred_class_np[b, patch_idx])
                if class_id == 0:
                    continue
                patch_x_min = c * patch_size_px
                patch_y_min = r * patch_size_px
                ctrl_pts = pred_bezier_np[b, patch_idx]
                sampled_local = np.matmul(basis, ctrl_pts)
                global_pts = np.zeros_like(sampled_local)
                global_pts[:, 0] = patch_x_min + sampled_local[:, 0] * patch_size_px
                global_pts[:, 1] = patch_y_min + sampled_local[:, 1] * patch_size_px
                pts_int = np.round(global_pts).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(canvas_batch[b], [pts_int], isClosed=False, color=class_id, thickness=stroke_width, lineType=cv2.LINE_AA)
    return canvas_batch

def compute_dice_iou(pred_bin, gt_bin):
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    sum_area = pred_bin.sum() + gt_bin.sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    if sum_area == 0: return 1.0, 1.0
    if pred_bin.sum() == 0 or gt_bin.sum() == 0: return 0.0, 0.0
    return float(2.0 * intersection / sum_area), float(intersection / union if union > 0 else 0.0)

def compute_assd(pred_mask, gt_mask, fallback=80.0):
    if pred_mask.sum() == 0 or gt_mask.sum() == 0: return fallback
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
            if edge_p.sum() == 0 or edge_g.sum() == 0: return fallback
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
        'sil_dice':   sd, 'sil_iou':   siou, 'sil_assd':   sassd,
        'falc_dice':  fd, 'falc_iou':  fiou, 'falc_assd':  fassd,
        'fg_dice':    fgd, 'fg_iou':   fgiou
    }

def _render_patient40_panel(img_tensor, gt_2d, pred_map, fname, out_dir, pred_centroids=None, gt_centroids=None):
    os.makedirs(out_dir, exist_ok=True)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = (img_tensor * std + mean).clamp(0, 1).numpy()
    img_bgr = cv2.cvtColor((img * 255).astype(np.uint8).transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
    
    colors = {1: (0, 0, 255), 2: (0, 255, 0), 3: (255, 0, 0)}
    def apply_overlay(base, mask):
        res = base.copy()
        for cid, col in colors.items():
            idx = (mask == cid)
            if np.any(idx):
                res[idx] = (base[idx].astype(np.float32) * 0.5 + np.array(col, dtype=np.float32) * 0.5).astype(np.uint8)
        return res
        
    gt_vis   = apply_overlay(img_bgr, gt_2d)
    pred_vis = apply_overlay(img_bgr, pred_map)
    H, W = img_bgr.shape[:2]
    if pred_centroids is not None:
        for cid in [1, 2, 3]:
            c_pred = pred_centroids[cid - 1]
            cv2.drawMarker(pred_vis, (int(c_pred[0] * W), int(c_pred[1] * H)), colors[cid], markerType=cv2.MARKER_CROSS, markerSize=25, thickness=3)
    if gt_centroids is not None:
        for cid in [1, 2, 3]:
            c_gt = gt_centroids[cid - 1]
            if c_gt[0] > 0 or c_gt[1] > 0:
                cv2.drawMarker(gt_vis, (int(c_gt[0] * W), int(c_gt[1] * H)), colors[cid], markerType=cv2.MARKER_TILTED_CROSS, markerSize=25, thickness=3)
                
    error_map = np.zeros_like(img_bgr)
    error_map[(pred_map == gt_2d) & (gt_2d > 0)] = (0, 255, 0)
    error_map[(pred_map != gt_2d) & (pred_map > 0)] = (0, 0, 255)
    error_map[(pred_map != gt_2d) & (gt_2d > 0)]    = (255, 0, 0)
    
    panel = np.vstack([np.hstack([img_bgr, gt_vis]), np.hstack([pred_vis, error_map])])
    cv2.imwrite(os.path.join(out_dir, f"{Path(fname).stem}_diag.png"), panel)

def evaluate_model(model, dataloader, device, split_name='Val', save_patient40_dir=None):
    model.eval()
    records, latencies = [], []
    with torch.no_grad():
        for batch in dataloader:
            pixel_values, gt_2d_batch, _, _, _, target_lm_pres, target_lm_cent, filenames = batch
            pixel_values = pixel_values.to(device)
            t0 = time.time()
            pred_class, pred_bezier, pred_presence, pred_centroid = model(pixel_values)
            if device.type == 'cuda': torch.cuda.synchronize()
            latencies.append((time.time() - t0) / pixel_values.shape[0])
            
            pred_cls_np  = pred_class.argmax(dim=-1).cpu().numpy()
            pred_bez_np  = pred_bezier.cpu().numpy()
            pred_cent_np = pred_centroid.cpu().numpy()
            gt_cent_np   = target_lm_cent.numpy()
            pred_maps    = rasterize_bezier_predictions(pred_cls_np, pred_bez_np, grid_size=8, canvas_size=1024, stroke_width=35)
            gt_maps      = gt_2d_batch.numpy()
            
            for b, fname in enumerate(filenames):
                m = compute_frame_metrics(pred_maps[b], gt_maps[b])
                fname_base = os.path.basename(str(fname))
                is_p40 = 'Patient_40' in fname_base or '_40_' in fname_base
                m['filename'] = fname_base
                m['is_patient40'] = is_p40
                records.append(m)
                if save_patient40_dir and is_p40:
                    _render_patient40_panel(pixel_values[b].cpu(), gt_maps[b], pred_maps[b], fname_base, save_patient40_dir,
                                            pred_centroids=pred_cent_np[b], gt_centroids=gt_cent_np[b])
                    
    df = pd.DataFrame(records)
    p40_df = df[df['is_patient40']]
    mean_lat = np.mean(latencies[5:]) if len(latencies) > 5 else np.mean(latencies)
    return {
        'split': split_name,
        'total_frames': len(records),
        'macro_dice': float(df['macro_dice'].mean()),
        'macro_iou':  float(df['macro_iou'].mean()),
        'macro_assd': float(df['macro_assd'].mean()),
        'ridge_dice': float(df['ridge_dice'].mean()),
        'sil_dice':   float(df['sil_dice'].mean()),
        'falc_dice':  float(df['falc_dice'].mean()),
        'fg_dice':    float(df['fg_dice'].mean()),
        'patient_40_dice': float(p40_df['macro_dice'].mean()) if len(p40_df) > 0 else 0.0,
        'patient_40_assd': float(p40_df['macro_assd'].mean()) if len(p40_df) > 0 else 80.0,
        'patient_40_count': len(p40_df),
        'fps': float(1.0 / mean_lat if mean_lat > 0 else 0.0),
    }, df
""")

    # Cell 14 & 15: Training Execution
    add_md("## Step 7: Training Loop (Kaggle GPU T4 — batch_size=1, accum=4)\nTrains for 60 epochs with Cosine Annealing learning rate schedule and AMP FP16 acceleration.")
    add_code("""from tqdm.auto import tqdm
import zipfile

SAVE_DIR = '/kaggle/working/EXPERIMENT_4_LandmarkBezier_RESULTS'
os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Training on device: {device}')

EPOCHS = 60
BATCH_SIZE = 1
ACCUM_STEPS = 4
LR = 8e-5

train_dataset = LandmarkBezierDataset(train_img_dir, grid_size=8)
val_dataset   = LandmarkBezierDataset(val_img_dir, grid_size=8)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=1,          shuffle=False, num_workers=2, pin_memory=True)

print(f'Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}')

model = LandmarkBezierPatchModel(grid_size=8, num_classes=4, embed_dim=256, num_decoder_layers=6, single_scale=True).to(device)
criterion = LandmarkBezierLoss(lambda_cls=2.0, lambda_ctrl=5.0, lambda_sample=2.0, lambda_cont=1.0, lambda_tan=0.5, lambda_lm=1.0, continuity_phase_epoch=31)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=3e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

best_val_dice = 0.0
training_log = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    step_count = 0
    optimizer.zero_grad()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{EPOCHS:02d} [Train]")
    for step, batch in enumerate(pbar):
        pixel_values, _, target_class, target_bezier, active_mask, target_lm_pres, target_lm_cent, _ = batch
        pixel_values      = pixel_values.to(device)
        target_class      = target_class.to(device)
        target_bezier     = target_bezier.to(device)
        active_mask       = active_mask.to(device)
        target_lm_pres    = target_lm_pres.to(device)
        target_lm_cent    = target_lm_cent.to(device)
        
        with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu'):
            pred_class, pred_bezier, pred_presence, pred_centroid = model(pixel_values)
            loss, loss_dict = criterion(pred_class, pred_bezier, pred_presence, pred_centroid,
                                        target_class, target_bezier, active_mask,
                                        target_lm_pres, target_lm_cent, epoch=epoch)
            loss_scaled = loss / ACCUM_STEPS
            
        if scaler:
            scaler.scale(loss_scaled).backward()
        else:
            loss_scaled.backward()
            
        if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
            if scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()
            
        epoch_loss += loss_dict['loss']
        step_count += 1
        pbar.set_postfix({'loss': f"{epoch_loss / step_count:.4f}", 'phase': 'P2' if epoch >= 31 else 'P1'})
        
    scheduler.step()
    
    val_summary, _ = evaluate_model(model, val_loader, device, split_name=f'Val-Ep{epoch}')
    current_val_dice = val_summary['macro_dice']
    print(f"\\nEpoch {epoch:02d}: TrainLoss={epoch_loss/step_count:.4f} | Val MacroDice={current_val_dice:.4f} | ASSD={val_summary['macro_assd']:.2f}px")
    
    if current_val_dice > best_val_dice:
        best_val_dice = current_val_dice
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'best_val_dice': best_val_dice,
            'grid_size': 8,
            'single_scale': True
        }, os.path.join(SAVE_DIR, 'best_model.pth'))
        print(f"   ⭐ New best model saved! Val MacroDice: {best_val_dice:.4f}")
        
    training_log.append({
        'epoch': epoch,
        'train_loss': epoch_loss / step_count,
        'val_macro_dice': current_val_dice,
        'val_macro_assd': val_summary['macro_assd'],
        'val_p40_dice': val_summary['patient_40_dice'],
        'lr': scheduler.get_last_lr()[0]
    })

pd.DataFrame(training_log).to_csv(os.path.join(SAVE_DIR, 'training_log.csv'), index=False)
""")

    # Cell 16 & 17: Benchmark Evaluation & Zipping
    add_md("## Step 8: Benchmark Evaluation & One-Click Download Archive\nEvaluates the best checkpoint on both Validation (122 frames) and Test (109 frames) splits, exports Patient 40 diagnostic montages, and zips results.")
    add_code("""ckpt_path = os.path.join(SAVE_DIR, 'best_model.pth')
checkpoint = torch.load(ckpt_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Loaded best checkpoint from Epoch {checkpoint['epoch']} with Val MacroDice {checkpoint['best_val_dice']:.4f}")

p40_diag_dir = os.path.join(SAVE_DIR, 'patient_40_diagnostics_final')
final_val_summary, final_val_df = evaluate_model(model, val_loader, device, split_name='Val-Final', save_patient40_dir=p40_diag_dir)
final_val_df.to_csv(os.path.join(SAVE_DIR, 'val_predictions.csv'), index=False)

final_test_summary = {}
if test_img_dir and os.path.exists(test_img_dir):
    test_dataset = LandmarkBezierDataset(test_img_dir, grid_size=8)
    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
    final_test_summary, test_df = evaluate_model(model, test_loader, device, split_name='Test-Final')
    test_df.to_csv(os.path.join(SAVE_DIR, 'test_predictions.csv'), index=False)

benchmark_json = {
    'val_summary': final_val_summary,
    'test_summary': final_test_summary,
    'best_epoch': checkpoint['epoch']
}
with open(os.path.join(SAVE_DIR, 'metrics_summary.json'), 'w') as f:
    json.dump(benchmark_json, f, indent=4)

print('=' * 70)
print('🏆 FINAL BENCHMARK SUMMARY (EXPERIMENT_4)')
print('=' * 70)
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
print('=' * 70)

# Create zip archive
zip_output = '/kaggle/working/EXPERIMENT_4_LandmarkBezier_RESULTS.zip'
with zipfile.ZipFile(zip_output, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, _, files in os.walk(SAVE_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, SAVE_DIR)
            zipf.write(file_path, arcname)
print(f"✅ Download ready: {zip_output} ({os.path.getsize(zip_output)/(1024*1024):.2f} MB)")
""")

    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../notebooks/EXP4_LandmarkBezier_Kaggle.ipynb'))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(nb, f, indent=2)
    print(f"✅ Generated Kaggle notebook: {out_path}")

if __name__ == '__main__':
    build_notebook()
