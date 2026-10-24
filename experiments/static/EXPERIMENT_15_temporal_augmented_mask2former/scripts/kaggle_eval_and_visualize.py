import os
import sys
import glob
import re
import json
import time
import zipfile
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ==============================================================================
# 0. USER CONFIGURATION & ENVIRONMENT SETUP
# ==============================================================================
# Set your model checkpoint path here:
CHECKPOINT_PATH = "/kaggle/working/best_temporal_m2f.pth"

# If CHECKPOINT_PATH does not exist, auto-search in /kaggle/input and local dirs
def resolve_checkpoint_path(explicit_path):
    if os.path.exists(explicit_path):
        return explicit_path
    candidates = [
        "best_temporal_m2f.pth",
        "/kaggle/working/best_temporal_m2f.pth",
        "checkpoints/exp15_temporal_m2f/best_temporal_m2f.pth",
        "../checkpoints/exp15_temporal_m2f/best_temporal_m2f.pth",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    if os.path.exists('/kaggle/input'):
        for root, _, files in os.walk('/kaggle/input'):
            for f in files:
                if f.endswith('.pth') and ('temporal' in f.lower() or 'm2f' in f.lower() or 'best' in f.lower()):
                    return os.path.join(root, f)
    raise FileNotFoundError(f"Could not find checkpoint at '{explicit_path}' or any standard paths.")

# Install transformers if missing
try:
    import transformers
except ImportError:
    os.system("pip install -q transformers")
    import transformers

from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"🚀 Active Compute Device: {device}")
if torch.cuda.is_available():
    print(f"🎮 GPU Device Name: {torch.cuda.get_device_name(0)}")
    print(f"💾 Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

# ==============================================================================
# 1. DATASET RESOLUTION
# ==============================================================================
def resolve_split_dir(split_name):
    split_lower = split_name.lower()
    candidates = {
        'val': [
            '/kaggle/input/l3d-val/Val',
            '/kaggle/input/l3d-val/val',
            '/kaggle/input/datasets/khoatrytopublish/l3d-val/Val',
            '/kaggle/input/datasets/khoatrytopublish/l3d-val/val',
            '/kaggle/input/laparoscopic-liver-dataset/L3D/val',
            '/kaggle/input/l3d-dataset/L3D/val',
            'data/L3D/val',
            '../data/L3D/val',
            '../../data/L3D/val',
            '/Users/khoale/Downloads/Surgical AI/data/L3D/val'
        ],
        'test': [
            '/kaggle/input/l3d-test/Test',
            '/kaggle/input/l3d-test/test',
            '/kaggle/input/datasets/khoatrytopublish/l3d-test/Test',
            '/kaggle/input/datasets/khoatrytopublish/l3d-test/test',
            '/kaggle/input/laparoscopic-liver-dataset/L3D/test',
            '/kaggle/input/l3d-dataset/L3D/test',
            'data/L3D/test',
            '../data/L3D/test',
            '../../data/L3D/test',
            '/Users/khoale/Downloads/Surgical AI/data/L3D/test'
        ]
    }
    for path in candidates.get(split_lower, []):
        if os.path.exists(path):
            return path
    if os.path.exists('/kaggle/input'):
        for root, dirs, _ in os.walk('/kaggle/input', followlinks=True):
            bname = os.path.basename(root).lower()
            if bname == split_lower or (split_lower in bname):
                if any(d.lower() == 'labels' for d in dirs):
                    return root
    raise FileNotFoundError(f"Could not resolve directory for split '{split_name}' in Kaggle or local environments.")

VAL_DIR = resolve_split_dir('val')
TEST_DIR = resolve_split_dir('test')
print(f"📁 Verified Val Directory:  {VAL_DIR}")
print(f"📁 Verified Test Directory: {TEST_DIR}")

IMAGE_SIZE = 1024
STROKE_WIDTH = 35
CLIP_LEN = 3
MAX_GAP = 240
NUM_CLASSES = 3
CLASS_NAMES = ['Ridge', 'Silhouette', 'Ligament']

# Landmark Visualization Palette (RGB)
# Class 1: Ridge (Crimson Red)
# Class 2: Silhouette (Emerald Green)
# Class 3: Falciform Ligament (Vibrant Blue / Cyan)
COLOR_PALETTE = {
    0: (0, 0, 0),         # Background
    1: (255, 50, 50),     # Ridge: Red
    2: (50, 255, 50),     # Silhouette: Green
    3: (50, 150, 255)     # Ligament: Sky Blue
}

# ==============================================================================
# 2. DATASET DEFINITION (TOPONET TWO-STEP RASTERIZATION)
# ==============================================================================
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

class PatientTemporalL3DDataset(Dataset):
    def __init__(self, split_dir, clip_len=3, max_gap=240, image_size=1024, stroke_width=35):
        self.split_dir = split_dir
        self.clip_len = clip_len
        self.max_gap = max_gap
        self.image_size = image_size
        self.stroke_width = stroke_width
        self.clips = self._index_temporal_clips()

    def _index_temporal_clips(self):
        labels_dir = None
        for cand in ['labels', 'Labels', 'LABELS']:
            p = os.path.join(self.split_dir, cand)
            if os.path.exists(p):
                labels_dir = p
                break
        if labels_dir is None:
            labels_dir = os.path.join(self.split_dir, 'labels')

        images_dir = None
        for cand in ['images', 'Images', 'IMAGES']:
            p = os.path.join(self.split_dir, cand)
            if os.path.exists(p):
                images_dir = p
                break
        if images_dir is None:
            images_dir = os.path.join(self.split_dir, 'images')

        json_files = sorted(glob.glob(os.path.join(labels_dir, '*.json')))
        patient_dict = defaultdict(list)

        for jf in json_files:
            stem = Path(jf).stem
            m = re.match(r'(Patient_\d+)_(\d+)', stem)
            if not m:
                continue
            pat, fid = m.group(1), int(m.group(2))
            img_path = os.path.join(images_dir, f"{stem}.png")
            if not os.path.exists(img_path):
                img_path = os.path.join(images_dir, f"{stem}.jpg")
            if os.path.exists(img_path):
                patient_dict[pat].append((fid, jf, img_path))

        clips = []
        for pat in sorted(patient_dict.keys()):
            frames = patient_dict[pat]
            frames.sort(key=lambda x: x[0])

            for i in range(len(frames)):
                target = frames[i]
                if i >= 1 and (frames[i][0] - frames[i-1][0]) <= self.max_gap:
                    c1 = frames[i-1]
                    if i >= 2 and (frames[i-1][0] - frames[i-2][0]) <= self.max_gap:
                        c0 = frames[i-2]
                    else:
                        c0 = c1
                    window = [c0, c1, target]
                else:
                    window = [target, target, target]

                clips.append({
                    'patient': pat,
                    'frame_ids': [w[0] for w in window],
                    'json_paths': [w[1] for w in window],
                    'img_paths': [w[2] for w in window],
                    'target_idx': self.clip_len - 1
                })
        return clips

    def __len__(self):
        return len(self.clips)

    def _load_single_frame(self, img_path, json_path):
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to load image: {img_path}")
        orig_h, orig_w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if orig_w != self.image_size or orig_h != self.image_size:
            img_resized = cv2.resize(img_rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        else:
            img_resized = img_rgb

        norm_img = (img_resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        pixel_tensor = torch.from_numpy(norm_img).permute(2, 0, 1).float()

        # TopoNet two-step native canvas rasterization
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        canvas_h = data.get('imageHeight', orig_h)
        canvas_w = data.get('imageWidth', orig_w)
        if any(x in json_path for x in ['_31', '_36', '_25', '_29']):
            canvas_h, canvas_w = 2160, 3840
        elif canvas_h < 100 or canvas_w < 100:
            canvas_h, canvas_w = 1080, 1920

        canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

        shapes = data.get('shapes', [])
        for shape in shapes:
            lbl = str(shape.get('label', '')).lower().strip()
            if lbl.startswith('r'):
                color = 1
            elif lbl.startswith('s'):
                color = 2
            elif lbl.startswith('l') or 'falciform' in lbl:
                color = 3
            else:
                color = 0

            if color > 0:
                pts = shape.get('points', [])
                for i in range(1, len(pts)):
                    pt1 = tuple(map(int, pts[i - 1]))
                    pt2 = tuple(map(int, pts[i]))
                    cv2.line(canvas, pt1, pt2, color, int(self.stroke_width))

        gt_map = cv2.resize(canvas, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        mask_channels = np.stack([gt_map == 1, gt_map == 2, gt_map == 3], axis=0).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_channels)
        gt_map_tensor = torch.from_numpy(gt_map.astype(np.int64))

        return pixel_tensor, mask_tensor, gt_map_tensor, img_resized

    def __getitem__(self, idx):
        clip_info = self.clips[idx]
        pixel_tensors, mask_tensors, gt_map_tensors = [], [], []
        raw_images = []
        for img_p, json_p in zip(clip_info['img_paths'], clip_info['json_paths']):
            pt, mt, gm, raw_img = self._load_single_frame(img_p, json_p)
            pixel_tensors.append(pt)
            mask_tensors.append(mt)
            gt_map_tensors.append(gm)
            raw_images.append(raw_img)

        return {
            'pixel_values': torch.stack(pixel_tensors, dim=0),
            'mask_labels': torch.stack(mask_tensors, dim=0),
            'target_masks': mask_tensors[-1],
            'gt_map': gt_map_tensors[-1],
            'raw_img': raw_images[-1],
            'patient': clip_info['patient'],
            'frame_ids': clip_info['frame_ids']
        }

# ==============================================================================
# 3. METRICS EVALUATION (TOPONET COMPATIBLE)
# ==============================================================================
def compute_dice(pred_bin, target_bin):
    p = (pred_bin > 0).astype(np.float32)
    t = (target_bin > 0).astype(np.float32)
    intersection = float(np.sum(p * t))
    total = float(np.sum(p) + np.sum(t))
    if total == 0:
        return 1.0
    if np.sum(p) == 0 or np.sum(t) == 0:
        return 0.0
    return float(2.0 * intersection / total)

def compute_iou(pred_bin, target_bin):
    p = (pred_bin > 0).astype(np.float32)
    t = (target_bin > 0).astype(np.float32)
    intersection = float(np.sum(p * t))
    union = float(np.sum((p + t) > 0))
    if union == 0:
        return 1.0
    return float(intersection / union)

def compute_assd_hd95(pred_bin, target_bin, max_penalty=80.0):
    p = (pred_bin > 0).astype(np.uint8)
    t = (target_bin > 0).astype(np.uint8)
    if np.sum(p) == 0 and np.sum(t) == 0:
        return 0.0, 0.0
    if np.sum(p) == 0 or np.sum(t) == 0:
        return max_penalty, max_penalty

    cnts_p, _ = cv2.findContours(p, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts_t, _ = cv2.findContours(t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    edge_p, edge_t = np.zeros_like(p), np.zeros_like(t)
    cv2.drawContours(edge_p, cnts_p, -1, 1, 1)
    cv2.drawContours(edge_t, cnts_t, -1, 1, 1)

    pts_p, pts_t = edge_p > 0, edge_t > 0
    if not np.any(pts_p) or not np.any(pts_t):
        return max_penalty, max_penalty

    dt_target = cv2.distanceTransform(1 - edge_t, cv2.DIST_L2, 3)
    dt_pred = cv2.distanceTransform(1 - edge_p, cv2.DIST_L2, 3)

    d_p_to_t = dt_target[pts_p]
    d_t_to_p = dt_pred[pts_t]
    if len(d_p_to_t) == 0 or len(d_t_to_p) == 0:
        return max_penalty, max_penalty

    assd = float((np.mean(d_p_to_t) + np.mean(d_t_to_p)) / 2.0)
    all_dists = np.concatenate([d_p_to_t, d_t_to_p])
    hd95 = float(np.percentile(all_dists, 95))
    return assd, hd95

def evaluate_frame_metrics(pred_sem_seg, gt_input, num_classes=3, threshold=0.5):
    res = {}
    class_names = ['ridge', 'silhouette', 'ligament']

    max_prob = np.max(pred_sem_seg, axis=0)
    argmax_cls = np.argmax(pred_sem_seg, axis=0) + 1
    pred_map = np.where(max_prob >= threshold, argmax_cls, 0).astype(np.uint8)

    if gt_input.ndim == 3:
        gt_map = np.zeros(gt_input.shape[1:], dtype=np.uint8)
        for c in range(min(num_classes, gt_input.shape[0])):
            gt_map[gt_input[c] > 0.5] = (c + 1)
    else:
        gt_map = gt_input.astype(np.uint8)

    pred_onehot = np.array([pred_map == i for i in range(4)]).astype(np.uint8)
    gt_onehot = np.array([gt_map == i for i in range(4)]).astype(np.uint8)

    smooth = 1e-5
    inter = float(np.sum(pred_onehot[1:] * gt_onehot[1:]))
    total_px = float(np.sum(pred_onehot[1:]) + np.sum(gt_onehot[1:]))
    micro_dice = float((2.0 * inter + smooth) / (total_px + smooth))
    micro_iou = float(micro_dice / (2.0 - micro_dice))

    res['micro_dice'] = micro_dice
    res['micro_iou'] = micro_iou

    dices, ious, assds, hd95s = [], [], [], []
    for c in range(1, min(num_classes + 1, 4)):
        c_name = class_names[c - 1]
        p_c = pred_onehot[c]
        t_c = gt_onehot[c]

        d = compute_dice(p_c, t_c)
        iou = compute_iou(p_c, t_c)
        assd, hd95 = compute_assd_hd95(p_c, t_c)

        res[f'dice_{c_name}'] = d
        res[f'iou_{c_name}'] = iou
        res[f'assd_{c_name}'] = assd
        res[f'hd95_{c_name}'] = hd95

        dices.append(d)
        ious.append(iou)
        assds.append(assd)
        hd95s.append(hd95)

    res['macro_dice'] = float(np.mean(dices))
    res['macro_iou'] = float(np.mean(ious))
    res['mean_assd'] = float(np.mean(assds))
    res['mean_hd95'] = float(np.mean(hd95s))
    res['pred_map'] = pred_map
    res['gt_map'] = gt_map
    return res

# ==============================================================================
# 4. MODEL ARCHITECTURE (OPTION B: INTERLEAVED PROTOTYPE CROSS-ATTENTION)
# ==============================================================================
class LandmarkPrototypePooler(nn.Module):
    def __init__(self, num_classes=3, hidden_dim=256):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.empty_tokens = nn.Parameter(torch.randn(num_classes, hidden_dim) * 0.02)
        self.class_embeddings = nn.Parameter(torch.randn(num_classes, hidden_dim) * 0.02)
        self.time_embeddings = nn.Parameter(torch.randn(2, hidden_dim) * 0.02)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def pool_single_frame(self, feature_map, mask, time_idx=0):
        B, C, H_f, W_f = feature_map.shape
        _, K, H_m, W_m = mask.shape
        if (H_m, W_m) != (H_f, W_f):
            mask = F.interpolate(mask.float(), size=(H_f, W_f), mode='bilinear', align_corners=False)

        prototypes = []
        for c in range(self.num_classes):
            m_c = mask[:, c:c+1, :, :]
            m_sum = m_c.sum(dim=(2, 3), keepdim=True)
            weighted_feat = (feature_map * m_c).sum(dim=(2, 3))
            proto_c = weighted_feat / (m_sum.squeeze(-1).squeeze(-1) + 1e-6)

            is_present = (m_sum.squeeze(-1).squeeze(-1) > 1.0).float()
            empty_c = self.empty_tokens[c].unsqueeze(0).expand(B, -1)
            proto_c = is_present * proto_c + (1.0 - is_present) * empty_c
            proto_c = proto_c + self.class_embeddings[c].unsqueeze(0) + self.time_embeddings[time_idx].unsqueeze(0)
            prototypes.append(proto_c)

        stacked = torch.stack(prototypes, dim=1)
        return self.norm(self.proj(stacked))

    def forward(self, past_feature_maps, past_masks):
        all_protos = []
        for time_idx, (f_map, m) in enumerate(zip(past_feature_maps, past_masks)):
            proto = self.pool_single_frame(f_map, m, time_idx=min(time_idx, 1))
            all_protos.append(proto)
        return torch.cat(all_protos, dim=1)

class TemporalQueryInterleavedBlock(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=8, dropout=0.1, gate_init=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.gate = nn.Parameter(torch.tensor([gate_init], dtype=torch.float32))
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, past_prototypes):
        if past_prototypes is None:
            return queries
        attn_out, _ = self.cross_attn(query=queries, key=past_prototypes, value=past_prototypes)
        return self.norm(queries + self.gate * self.dropout(attn_out))

class InterleavedDecoderLayerWrapper(nn.Module):
    def __init__(self, base_layer, temporal_block):
        super().__init__()
        self.base_layer = base_layer
        self.temporal_block = temporal_block
        self.past_prototypes = None

    def forward(self, hidden_states, *args, **kwargs):
        if self.past_prototypes is not None:
            q_b = hidden_states.permute(1, 0, 2)
            q_steered = self.temporal_block(q_b, self.past_prototypes)
            hidden_states = q_steered.permute(1, 0, 2)
        return self.base_layer(hidden_states, *args, **kwargs)

class TemporalMask2Former(nn.Module):
    def __init__(self, base_checkpoint="facebook/mask2former-swin-tiny-ade-semantic", num_classes=3, hidden_dim=256, temporal_gate_init=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        try:
            self.base_model = Mask2FormerForUniversalSegmentation.from_pretrained(
                base_checkpoint,
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
        except Exception:
            cfg = Mask2FormerConfig(num_labels=num_classes, hidden_dim=hidden_dim, num_queries=100)
            self.base_model = Mask2FormerForUniversalSegmentation(cfg)

        self.prototype_pooler = LandmarkPrototypePooler(num_classes=num_classes, hidden_dim=hidden_dim)
        self.decoder_wrappers = nn.ModuleList()
        decoder_layers = self.base_model.model.transformer_module.decoder.layers
        for i in range(len(decoder_layers)):
            base_l = decoder_layers[i]
            t_blk = TemporalQueryInterleavedBlock(hidden_dim=hidden_dim, gate_init=temporal_gate_init)
            wrapper = InterleavedDecoderLayerWrapper(base_l, t_blk)
            decoder_layers[i] = wrapper
            self.decoder_wrappers.append(wrapper)

    def _predict_mask_from_pix_decoder(self, pixel_decoder_output):
        transformer_out = self.base_model.model.transformer_module(
            multi_scale_features=pixel_decoder_output.decoder_hidden_states,
            mask_features=pixel_decoder_output.decoder_last_hidden_state
        )
        class_logits = self.base_model.class_predictor(transformer_out.last_hidden_state)
        mask_logits = transformer_out.masks_queries_logits[-1]

        mask_probs = mask_logits.sigmoid()
        class_probs = class_logits.softmax(dim=-1)[..., :self.num_classes]
        sem_seg = torch.einsum("bqc, bqhw -> bchw", class_probs, mask_probs)
        return (sem_seg > 0.5).float()

    def forward(self, pixel_values, mask_labels=None, return_dict=None):
        B, T, C, H, W = pixel_values.shape
        c0 = pixel_values[:, 0]
        c1 = pixel_values[:, 1]
        target = pixel_values[:, 2]

        with torch.no_grad():
            pix_0 = self.base_model.model.pixel_level_module(c0)
            feat_0 = pix_0.decoder_last_hidden_state

            pix_1 = self.base_model.model.pixel_level_module(c1)
            feat_1 = pix_1.decoder_last_hidden_state

            m0 = self._predict_mask_from_pix_decoder(pix_0)
            m1 = self._predict_mask_from_pix_decoder(pix_1)

            K_past = self.prototype_pooler([feat_0, feat_1], [m0, m1])

        for wrapper in self.decoder_wrappers:
            wrapper.past_prototypes = K_past

        return self.base_model(pixel_values=target, return_dict=return_dict)

# ==============================================================================
# 5. VISUAL INSPECTION GENERATOR (4-PANEL COLOR-CODED MONTAGE)
# ==============================================================================
def render_class_map_rgb(class_map):
    h, w = class_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c, col in COLOR_PALETTE.items():
        if c > 0:
            rgb[class_map == c] = col
    return rgb

def create_blended_overlay(raw_img_rgb, pred_map, gt_map, alpha=0.45):
    overlay = raw_img_rgb.copy()
    
    # Ground Truth drawn with subtle translucent fill
    gt_colored = render_class_map_rgb(gt_map)
    gt_mask = gt_map > 0
    overlay[gt_mask] = (overlay[gt_mask] * (1.0 - alpha) + gt_colored[gt_mask] * alpha).astype(np.uint8)

    # Predicted landmarks drawn with vibrant outlines
    pred_colored = render_class_map_rgb(pred_map)
    pred_mask = pred_map > 0
    overlay[pred_mask] = (overlay[pred_mask] * (1.0 - alpha) + pred_colored[pred_mask] * alpha).astype(np.uint8)

    # Draw 1px contours for predictions to ensure sharp landmark boundaries
    for c, col in COLOR_PALETTE.items():
        if c > 0:
            p_bin = (pred_map == c).astype(np.uint8)
            cnts, _ = cv2.findContours(p_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(overlay, cnts, -1, col, 2)
            
            # Ground truth drawn as dotted/dashed white inner core
            g_bin = (gt_map == c).astype(np.uint8)
            g_cnts, _ = cv2.findContours(g_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(overlay, g_cnts, -1, (255, 255, 255), 1)
            
    return overlay

def generate_4panel_strip(raw_img_rgb, gt_map, pred_map, metrics_dict, patient, fid, target_size=512):
    # Downscale panels for storage efficiency
    raw_panel = cv2.resize(raw_img_rgb, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    gt_panel = cv2.resize(render_class_map_rgb(gt_map), (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    pred_panel = cv2.resize(render_class_map_rgb(pred_map), (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    overlay_panel = cv2.resize(create_blended_overlay(raw_img_rgb, pred_map, gt_map), (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    # Draw Banner Titles on each panel
    banner_h = 36
    def add_banner(img, title, color=(255, 255, 255)):
        header = np.zeros((banner_h, target_size, 3), dtype=np.uint8)
        cv2.putText(header, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
        return np.vstack([header, img])

    p1 = add_banner(raw_panel, "1. Laparoscopy RGB", (200, 200, 200))
    p2 = add_banner(gt_panel, "2. Ground Truth", (100, 255, 100))
    p3 = add_banner(pred_panel, "3. Model Prediction", (100, 200, 255))
    p4 = add_banner(overlay_panel, "4. Anatomical Overlay", (255, 200, 100))

    strip = np.hstack([p1, p2, p3, p4])

    # Top Status HUD Banner
    hud_h = 42
    hud = np.zeros((hud_h, target_size * 4, 3), dtype=np.uint8)
    hud_text = (
        f"Patient: {patient} | Frame: {fid:07d} | "
        f"Macro Dice: {metrics_dict['macro_dice']*100:.1f}% | "
        f"Micro Dice: {metrics_dict['micro_dice']*100:.1f}% | "
        f"ASSD: {metrics_dict['mean_assd']:.1f}px | "
        f"Ridge: {metrics_dict['dice_ridge']*100:.1f}% | "
        f"Silh: {metrics_dict['dice_silhouette']*100:.1f}% | "
        f"Lig: {metrics_dict['dice_ligament']*100:.1f}%"
    )
    cv2.putText(hud, hud_text, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)

    # Legend sub-bar
    legend_h = 28
    legend = np.zeros((legend_h, target_size * 4, 3), dtype=np.uint8)
    cv2.putText(legend, "LEGEND: [RED] Liver Ridge   [GREEN] Liver Silhouette   [BLUE] Falciform Ligament   [WHITE CONTOUR] Ground Truth Reference", (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)

    final_image = np.vstack([hud, strip, legend])
    return final_image

# ==============================================================================
# 6. INFERENCE & VISUALIZATION PIPELINE
# ==============================================================================
def run_split_inference(model, dataset, split_name, output_base_dir, threshold=0.5):
    print(f"\n==================================================================")
    print(f"🎬 Running Full Inference & Visual Inspection on {split_name.upper()} ({len(dataset)} frames)...")
    print(f"==================================================================")

    vis_dir = os.path.join(output_base_dir, "visualizations", split_name)
    os.makedirs(vis_dir, exist_ok=True)

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    records = []

    model.eval()
    start_time = time.time()

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            pixel_values = batch['pixel_values'].to(device)
            target_masks = batch['target_masks'].to(device)
            gt_map_t = batch['gt_map'][0].numpy()
            raw_img_np = batch['raw_img'][0].numpy()
            pat = batch['patient'][0]
            fids = batch['frame_ids']
            target_fid = fids[-1].item() if torch.is_tensor(fids[-1]) else fids[-1]

            # Forward pass (test distribution: model uses own predicted past masks)
            outputs = model(pixel_values)
            class_logits = outputs.class_queries_logits
            mask_logits = outputs.masks_queries_logits

            mask_probs = mask_logits.sigmoid()
            class_probs = class_logits.softmax(dim=-1)[..., :NUM_CLASSES]
            sem_seg = torch.einsum('bqc, bqhw -> bchw', class_probs, mask_probs)

            sem_seg_full = nn.functional.interpolate(
                sem_seg,
                size=(target_masks.shape[-2], target_masks.shape[-1]),
                mode='bilinear',
                align_corners=False
            )

            pred_prob = sem_seg_full[0].cpu().numpy()
            m = evaluate_frame_metrics(pred_prob, gt_map_t, num_classes=NUM_CLASSES, threshold=threshold)

            m_record = {k: v for k, v in m.items() if k not in ['pred_map', 'gt_map']}
            m_record['patient'] = pat
            m_record['frame_id'] = target_fid
            records.append(m_record)

            # Generate 4-panel visual inspection strip
            vis_strip = generate_4panel_strip(
                raw_img_np,
                m['gt_map'],
                m['pred_map'],
                m_record,
                pat,
                target_fid,
                target_size=512
            )

            # Save PNG
            out_filename = f"{pat}_{target_fid:07d}_macro{int(m_record['macro_dice']*100):02d}.png"
            out_path = os.path.join(vis_dir, out_filename)
            cv2.imwrite(out_path, cv2.cvtColor(vis_strip, cv2.COLOR_RGB2BGR))

            if (idx + 1) % 25 == 0 or (idx + 1) == len(dataset):
                print(f"  [{split_name.upper()} Progress] {idx+1}/{len(dataset)} frames processed & visualized ({time.time()-start_time:.1f}s)")

    df = pd.DataFrame(records)
    csv_path = os.path.join(output_base_dir, f"{split_name}_per_frame_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"💾 Per-frame metrics exported to: {csv_path}")

    # Summary
    summary = {
        'split': split_name,
        'num_frames': len(df),
        'macro_dice': float(df['macro_dice'].mean()),
        'micro_dice': float(df['micro_dice'].mean()),
        'macro_iou': float(df['macro_iou'].mean()),
        'mean_assd': float(df['mean_assd'].dropna().mean()),
        'dice_ridge': float(df['dice_ridge'].mean()),
        'dice_silhouette': float(df['dice_silhouette'].mean()),
        'dice_ligament': float(df['dice_ligament'].mean())
    }

    print("\n" + "=" * 65)
    print(f"📊 {split_name.upper()} SPLIT RESULTS SUMMARY")
    print("=" * 65)
    print(f"Total Evaluated Frames:  {summary['num_frames']}")
    print(f"Overall Micro-Dice (TopoNet benchmark standard): {summary['micro_dice']*100:.2f}%")
    print(f"Overall Macro-Dice (BCRNet benchmark standard):  {summary['macro_dice']*100:.2f}%")
    print(f"Overall Mean ASSD:                                {summary['mean_assd']:.2f} px")
    print("-" * 65)
    print(f"  • Ridge Dice:      {summary['dice_ridge']*100:.2f}%")
    print(f"  • Silhouette Dice: {summary['dice_silhouette']*100:.2f}%")
    print(f"  • Ligament Dice:   {summary['dice_ligament']*100:.2f}%")
    print("-" * 65)
    print("Per-Patient Breakdown:")
    for p, grp in df.groupby('patient'):
        print(f"  [{p:10s}] Frames: {len(grp):2d} | Macro: {grp['macro_dice'].mean()*100:.2f}% | Micro: {grp['micro_dice'].mean()*100:.2f}% | ASSD: {grp['mean_assd'].dropna().mean():.2f}px")
    print("=" * 65)

    return summary, df

# ==============================================================================
# 7. MAIN EXECUTION CONTROLLER
# ==============================================================================
def main():
    checkpoint_file = resolve_checkpoint_path(CHECKPOINT_PATH)
    print(f"\n📦 Loading Trained Weights from: {checkpoint_file}")
    
    ckpt = torch.load(checkpoint_file, map_location=device)
    print(f"  • Trained Epoch: {ckpt.get('epoch', 'N/A')}")
    if 'val_summary' in ckpt:
        vs = ckpt['val_summary']
        print(f"  • Recorded Best Val Macro: {vs.get('macro_dice', 0)*100:.2f}% (Micro: {vs.get('micro_dice', 0)*100:.2f}%)")

    # Instantiate Model
    print("🏗️ Initializing TemporalMask2Former Architecture...")
    model = TemporalMask2Former(
        base_checkpoint="facebook/mask2former-swin-tiny-ade-semantic",
        num_classes=NUM_CLASSES,
        hidden_dim=256,
        temporal_gate_init=0.1
    ).to(device)

    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print("✅ Model weights loaded successfully!")

    # Output Directory Setup
    output_base_dir = "/kaggle/working" if os.path.exists("/kaggle/working") else "inference_results"
    os.makedirs(output_base_dir, exist_ok=True)

    # 1. Evaluate Validation Split (122 frames)
    val_dataset = PatientTemporalL3DDataset(VAL_DIR, clip_len=CLIP_LEN, max_gap=MAX_GAP, image_size=IMAGE_SIZE, stroke_width=STROKE_WIDTH)
    val_summary, val_df = run_split_inference(model, val_dataset, "val", output_base_dir, threshold=0.5)

    # 2. Evaluate Test Split (109 frames)
    test_dataset = PatientTemporalL3DDataset(TEST_DIR, clip_len=CLIP_LEN, max_gap=MAX_GAP, image_size=IMAGE_SIZE, stroke_width=STROKE_WIDTH)
    test_summary, test_df = run_split_inference(model, test_dataset, "test", output_base_dir, threshold=0.5)

    # Export combined summary JSON
    combined_summary = {
        'checkpoint': checkpoint_file,
        'val': val_summary,
        'test': test_summary
    }
    summary_path = os.path.join(output_base_dir, "exp15_full_benchmark_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(combined_summary, f, indent=2)

    # Zip all visualizations for 1-click download from Kaggle
    zip_path = os.path.join(output_base_dir, "exp15_visual_inspections.zip")
    print(f"\n📦 Packaging all 231 visual inspection images into {zip_path}...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(os.path.join(output_base_dir, "visualizations")):
            for f in files:
                if f.endswith('.png'):
                    abs_p = os.path.join(root, f)
                    rel_p = os.path.relpath(abs_p, output_base_dir)
                    zipf.write(abs_p, rel_p)
    print(f"🎉 Complete! You can download '{zip_path}' directly from Kaggle's Output tab!")

if __name__ == '__main__':
    main()
