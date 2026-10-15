import os
import json
from pathlib import Path


def create_kaggle_notebook():
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            },
            "accelerator": "GPU",
            "gpuClass": "standard"
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    def add_cell(cell_type, source):
        nb["cells"].append({
            "cell_type": cell_type,
            "metadata": {},
            "source": source if isinstance(source, list) else source.splitlines(keepends=True)
        })

    # Cell 0: Header Markdown
    add_cell("markdown", """# Experiment 15: Temporal Mask2Former (Option B: Interleaved Prototype Conditioning)
## Laparoscopic Liver Landmark Detection (L3D Benchmark)

### Executive Architecture & Hypotheses
1. **Teacher Forcing in Training:** Uses 100% Ground Truth (GT) past masks to pool clean landmark prototypes ($P_{\\text{Ridge}}, P_{\\text{Silhouette}}, P_{\\text{Ligament}}$), enabling stable, noise-free learning of landmark deformation dynamics.
2. **Distribution Shift Testing:** Evaluates on Validation (122 clips) and Test (109 clips) using the model's own detached predictions for past frames to directly measure if test-distribution shift impacts performance.
3. **Option B Interleaved Conditioning:** In each of the 9 decoder layers, target queries cross-attend to the 6 past landmark prototypes before performing standard masked cross-attention.
4. **SOTA Benchmark Parity:** Evaluates 100% of the 109-frame Test set across all 4 patients (Patients 21, 31, 41, 51) and logs both **Micro-Dice** (TopoNet: 65.19%) and **Macro-Dice** (BCRNet: 69.57%).
""")

    # Cell 1: Environment & Imports
    add_cell("code", """# --- Cell 1: Environment & Package Setup ---
import os
import sys
import glob
import re
import json
import time
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Install transformers if absent
try:
    import transformers
except ImportError:
    !pip install -q transformers
    import transformers

from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation

# Set deterministic seed
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Active Compute Device: {device}")
if torch.cuda.is_available():
    print(f"🎮 GPU Device Name: {torch.cuda.get_device_name(0)}")
    print(f"💾 Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
""")

    # Cell 2: Dataset Path Resolution & Config
    add_cell("code", """# --- Cell 2: Dataset Path Resolution & Hyperparameters ---
def resolve_split_dir(split_name):
    split_lower = split_name.lower()
    candidates = {
        'train': [
            '/kaggle/input/l3d-train/Train',
            '/kaggle/input/l3d-train/train',
            '/kaggle/input/datasets/khoatrytopublish/l3d-train/Train',
            '/kaggle/input/datasets/khoatrytopublish/l3d-train/train',
            '/kaggle/input/laparoscopic-liver-landmarks/train',
            '/kaggle/input/laparoscopic-liver-dataset/L3D/train',
            '/kaggle/input/l3d-dataset/L3D/train',
            'data/L3D/train',
            '../data/L3D/train',
            '../../data/L3D/train',
            '/Users/khoale/Downloads/Surgical AI/data/L3D/train'
        ],
        'val': [
            '/kaggle/input/l3d-val/Val',
            '/kaggle/input/l3d-val/val',
            '/kaggle/input/datasets/khoatrytopublish/l3d-val/Val',
            '/kaggle/input/datasets/khoatrytopublish/l3d-val/val',
            '/kaggle/input/laparoscopic-liver-landmarks/val',
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
            '/kaggle/input/laparoscopic-liver-landmarks/test',
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

TRAIN_DIR = resolve_split_dir('train')
VAL_DIR = resolve_split_dir('val')
TEST_DIR = resolve_split_dir('test')

print(f"📁 Verified Train Directory: {TRAIN_DIR}")
print(f"📁 Verified Val Directory:   {VAL_DIR}")
print(f"📁 Verified Test Directory:  {TEST_DIR}")

# Exact benchmark configurations
IMAGE_SIZE = 1024
STROKE_WIDTH = 35
CLIP_LEN = 3
MAX_GAP = 240
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 4
NUM_EPOCHS = 30
LR_BACKBONE = 1e-5
LR_DECODER = 1e-4
WEIGHT_DECAY = 1e-4

# Exact annotation mapping matching L3D JSONs
CLASS_MAP = {
    'liver_ridge': 1,
    'ridge': 1,
    'rigde': 1,  # Official annotator typo in L3D benchmark (1661 shapes)
    'liver_silhouette': 2,
    'silhouette': 2,
    'falciform_ligament': 3,
    'falciform': 3,
    'ligament': 3  # Official annotator string in L3D benchmark (1101 shapes)
}
CLASS_NAMES = ['ridge', 'silhouette', 'ligament']
NUM_CLASSES = 3
""")

    # Cell 3: Dataset Definition with 100% Causal Coverage
    add_cell("code", r"""# --- Cell 3: PatientTemporalL3DDataset (100% Benchmark Coverage) ---
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
            frames.sort(key=lambda x: x[0])  # Sort chronologically

            for i in range(len(frames)):
                target = frames[i]

                # Strictly causal context padding
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

        # TopoNet exact two-step rasterization
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

        return pixel_tensor, mask_tensor, gt_map_tensor

    def __getitem__(self, idx):
        clip_info = self.clips[idx]
        pixel_tensors, mask_tensors, gt_map_tensors = [], [], []
        for img_p, json_p in zip(clip_info['img_paths'], clip_info['json_paths']):
            pt, mt, gm = self._load_single_frame(img_p, json_p)
            pixel_tensors.append(pt)
            mask_tensors.append(mt)
            gt_map_tensors.append(gm)

        return {
            'pixel_values': torch.stack(pixel_tensors, dim=0),
            'mask_labels': torch.stack(mask_tensors, dim=0),
            'target_masks': mask_tensors[-1],
            'gt_map': gt_map_tensors[-1],
            'patient': clip_info['patient'],
            'frame_ids': clip_info['frame_ids']
        }
""")

    # Cell 4: Metrics (Dual Macro/Micro Dice, ASSD, HD95)
    add_cell("code", """# --- Cell 4: Evaluation Metrics (Dual Macro/Micro Dice & Distances) ---
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
    p_sum = np.sum(p)
    t_sum = np.sum(t)
    if p_sum == 0 and t_sum == 0:
        return 0.0, 0.0
    if p_sum == 0 or t_sum == 0:
        return max_penalty, max_penalty

    cnts_p, _ = cv2.findContours(p, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts_t, _ = cv2.findContours(t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    edge_p = np.zeros_like(p)
    edge_t = np.zeros_like(t)
    cv2.drawContours(edge_p, cnts_p, -1, 1, 1)
    cv2.drawContours(edge_t, cnts_t, -1, 1, 1)

    pts_p = edge_p > 0
    pts_t = edge_t > 0
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

    # 1. Convert pred_sem_seg to mutually exclusive 2D class map using argmax
    max_prob = np.max(pred_sem_seg, axis=0)
    argmax_cls = np.argmax(pred_sem_seg, axis=0) + 1  # Classes 1, 2, 3
    pred_map = np.where(max_prob >= threshold, argmax_cls, 0).astype(np.uint8)

    # 2. Parse ground truth to discrete 2D class map if needed
    if gt_input.ndim == 3:
        gt_map = np.zeros(gt_input.shape[1:], dtype=np.uint8)
        for c in range(min(num_classes, gt_input.shape[0])):
            gt_map[gt_input[c] > 0.5] = (c + 1)
    else:
        gt_map = gt_input.astype(np.uint8)

    # 3. Create 4-class one-hot arrays (0: bg, 1: ridge, 2: silhouette, 3: ligament)
    pred_onehot = np.array([pred_map == i for i in range(4)]).astype(np.uint8)
    gt_onehot = np.array([gt_map == i for i in range(4)]).astype(np.uint8)

    # 4. TopoNet Exact Micro-Dice (flattened foreground one-hot intersection)
    smooth = 1e-5
    inter = float(np.sum(pred_onehot[1:] * gt_onehot[1:]))
    total_px = float(np.sum(pred_onehot[1:]) + np.sum(gt_onehot[1:]))
    micro_dice = float((2.0 * inter + smooth) / (total_px + smooth))
    micro_iou = float(micro_dice / (2.0 - micro_dice))

    res['micro_dice'] = micro_dice
    res['micro_iou'] = micro_iou

    # 5. Per-class metrics and Macro-Dice
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

    return res
""")

    # Cell 5: Architecture (LandmarkPrototypePooler, InterleavedWrapper, TemporalMask2Former)
    add_cell("code", """# --- Cell 5: Model Architecture (Option B: Interleaved Decoder) ---
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
        return torch.cat(all_protos, dim=1)  # (B, 6, 256)


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
        return (sem_seg > 0.4).float()

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

            if self.training and mask_labels is not None:
                # 100% Teacher Forcing with Ground Truth past masks
                m0 = mask_labels[:, 0]
                m1 = mask_labels[:, 1]
            else:
                # Inference: model predicted past masks
                m0 = self._predict_mask_from_pix_decoder(pix_0)
                m1 = self._predict_mask_from_pix_decoder(pix_1)

            K_past = self.prototype_pooler([feat_0, feat_1], [m0, m1])

        for wrapper in self.decoder_wrappers:
            wrapper.past_prototypes = K_past

        batch_mask_labels = None
        batch_class_labels = None
        if mask_labels is not None:
            target_masks = mask_labels[:, -1]
            batch_mask_labels = []
            batch_class_labels = []
            for b in range(B):
                masks_b = []
                classes_b = []
                for c in range(self.num_classes):
                    if target_masks[b, c].sum() > 0:
                        masks_b.append(target_masks[b, c])
                        classes_b.append(c)
                if len(masks_b) == 0:
                    masks_b.append(torch.zeros_like(target_masks[b, 0]))
                    classes_b.append(0)
                batch_mask_labels.append(torch.stack(masks_b))
                batch_class_labels.append(torch.tensor(classes_b, dtype=torch.long, device=target_masks.device))

        try:
            outputs = self.base_model(
                pixel_values=target,
                mask_labels=batch_mask_labels,
                class_labels=batch_class_labels,
                return_dict=return_dict
            )
        finally:
            for wrapper in self.decoder_wrappers:
                wrapper.past_prototypes = None

        return outputs
""")

    # Cell 6: DataLoaders
    add_cell("code", """# --- Cell 6: DataLoaders Creation & Inspection ---
train_dataset = PatientTemporalL3DDataset(TRAIN_DIR, clip_len=CLIP_LEN, max_gap=MAX_GAP, image_size=IMAGE_SIZE, stroke_width=STROKE_WIDTH)
val_dataset = PatientTemporalL3DDataset(VAL_DIR, clip_len=CLIP_LEN, max_gap=MAX_GAP, image_size=IMAGE_SIZE, stroke_width=STROKE_WIDTH)
test_dataset = PatientTemporalL3DDataset(TEST_DIR, clip_len=CLIP_LEN, max_gap=MAX_GAP, image_size=IMAGE_SIZE, stroke_width=STROKE_WIDTH)

print(f"📊 Dataset Clustered Clips (100% Causal Coverage):")
print(f"  • Train: {len(train_dataset)} clips (Expected: 921)")
print(f"  • Val:   {len(val_dataset)} clips (Expected: 122)")
print(f"  • Test:  {len(test_dataset)} clips (Expected: 109)")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
""")

    # Cell 7: Model & Optimizer Setup
    add_cell("code", """# --- Cell 7: Model Instantiation & Differential Optimizer ---
print("📦 Building TemporalMask2Former with Swin-Tiny Backbone...")
model = TemporalMask2Former(
    base_checkpoint="facebook/mask2former-swin-tiny-ade-semantic",
    num_classes=NUM_CLASSES,
    hidden_dim=256,
    temporal_gate_init=0.1
).to(device)

backbone_params = []
decoder_params = []
for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    if 'pixel_level_module.encoder' in name:
        backbone_params.append(param)
    else:
        decoder_params.append(param)

optimizer = AdamW([
    {'params': backbone_params, 'lr': LR_BACKBONE},
    {'params': decoder_params, 'lr': LR_DECODER}
], weight_decay=WEIGHT_DECAY)

scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

print(f"✅ Optimizer Initialized: {len(backbone_params)} backbone tensors (lr={LR_BACKBONE}), {len(decoder_params)} decoder tensors (lr={LR_DECODER})")
""")

    # Cell 8: Validation Loop Helper
    add_cell("code", """# --- Cell 8: Validation Function ---
def run_val(model, loader, threshold=0.5):
    model.eval()
    all_metrics = []
    p40_macro_dices = []
    records = []

    with torch.no_grad():
        for batch in loader:
            pixel_values = batch['pixel_values'].to(device)
            target_masks = batch['target_masks'].to(device)
            pat = batch['patient'][0]
            fids = batch['frame_ids']
            target_fid = fids[-1].item() if torch.is_tensor(fids[-1]) else fids[-1]

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
            gt_in = batch['gt_map'][0].cpu().numpy() if 'gt_map' in batch else target_masks[0].cpu().numpy()

            m = evaluate_frame_metrics(pred_prob, gt_in, num_classes=NUM_CLASSES, threshold=threshold)
            m['patient'] = pat
            m['frame_id'] = target_fid
            all_metrics.append(m)
            records.append(m)

            if pat == 'Patient_40':
                p40_macro_dices.append(m['macro_dice'])

    summary = {
        'macro_dice': float(np.mean([m['macro_dice'] for m in all_metrics])),
        'micro_dice': float(np.mean([m['micro_dice'] for m in all_metrics])),
        'macro_iou': float(np.mean([m['macro_iou'] for m in all_metrics])),
        'mean_assd': float(np.nanmean([m['mean_assd'] for m in all_metrics])),
        'p40_macro_dice': float(np.mean(p40_macro_dices)) if len(p40_macro_dices) > 0 else 0.0,
        'dice_ridge': float(np.mean([m['dice_ridge'] for m in all_metrics])),
        'dice_silhouette': float(np.mean([m['dice_silhouette'] for m in all_metrics])),
        'dice_ligament': float(np.mean([m['dice_ligament'] for m in all_metrics]))
    }
    return summary, pd.DataFrame(records)
""")

    # Cell 9: Training Loop (30 Epochs)
    add_cell("code", """# --- Cell 9: 30-Epoch Training Loop with Teacher Forcing ---
print(f"🚀 Starting 30-Epoch Training (1024x1024 | Stroke 35px | Micro-batch 1 | Accum 4)...")

best_val_macro = 0.0
best_model_path = "/kaggle/working/best_temporal_m2f.pth" if os.path.exists("/kaggle/working") else "best_temporal_m2f.pth"

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    start_time = time.time()
    optimizer.zero_grad()

    for step, batch in enumerate(train_loader):
        pixel_values = batch['pixel_values'].to(device)
        mask_labels = batch['mask_labels'].to(device)  # 100% GT for Teacher Forcing!

        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            outputs = model(pixel_values, mask_labels=mask_labels)
            loss = outputs.loss / GRAD_ACCUM_STEPS

        scaler.scale(loss).backward()

        if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        epoch_loss += loss.item() * GRAD_ACCUM_STEPS

    scheduler.step()
    train_loss = epoch_loss / len(train_loader)
    elapsed = time.time() - start_time

    val_summary, _ = run_val(model, val_loader)

    print(
        f"Epoch [{epoch:02d}/{NUM_EPOCHS:02d}] "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Macro: {val_summary['macro_dice']*100:.2f}% | "
        f"Val Micro: {val_summary['micro_dice']*100:.2f}% | "
        f"ASSD: {val_summary['mean_assd']:.2f}px | "
        f"P40 Macro: {val_summary['p40_macro_dice']*100:.2f}% "
        f"({elapsed:.1f}s)"
    )

    if val_summary['macro_dice'] > best_val_macro:
        best_val_macro = val_summary['macro_dice']
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_summary': val_summary
        }, best_model_path)
        print(f"  ⭐ New best checkpoint saved: Macro Dice {best_val_macro*100:.2f}% (Micro Dice {val_summary['micro_dice']*100:.2f}%)")

print(f"🏁 Training Complete! Best Val Macro Dice: {best_val_macro*100:.2f}%")
""")

    # Cell 10: Official Test Benchmark Evaluation
    add_cell("code", r"""# --- Cell 10: Official Test Set Benchmark Evaluation ---
print(f"🔍 Loading best model for Official Test Set Evaluation...")
checkpoint = torch.load(best_model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

test_summary, test_df = run_val(model, test_loader)

# Export results
test_csv_path = "/kaggle/working/test_per_frame_metrics.csv" if os.path.exists("/kaggle/working") else "test_per_frame_metrics.csv"
test_json_path = "/kaggle/working/test_summary_metrics.json" if os.path.exists("/kaggle/working") else "test_summary_metrics.json"

test_df.to_csv(test_csv_path, index=False)
with open(test_json_path, 'w') as f:
    json.dump(test_summary, f, indent=2)

print("\n" + "=" * 65)
print("📊 OFFICIAL L3D TEST BENCHMARK RESULTS")
print("=" * 65)
print(f"Total Evaluated Test Frames: {len(test_df)} (100% of benchmark)")
print(f"Overall Micro-Dice (TopoNet benchmark standard): {test_summary['micro_dice']*100:.2f}%")
print(f"Overall Macro-Dice (BCRNet benchmark standard):  {test_summary['macro_dice']*100:.2f}%")
print(f"Overall Mean ASSD:                                {test_summary['mean_assd']:.2f} px")
print("-" * 65)
print(f"  • Ridge Dice:      {test_summary['dice_ridge']*100:.2f}%")
print(f"  • Silhouette Dice: {test_summary['dice_silhouette']*100:.2f}%")
print(f"  • Ligament Dice:   {test_summary['dice_ligament']*100:.2f}%")
print("-" * 65)
print("Per-Patient Breakdown:")
for pat, group in test_df.groupby('patient'):
    print(f"  [{pat:10s}] Frames: {len(group):2d} | Macro: {group['macro_dice'].mean()*100:.2f}% | Micro: {group['micro_dice'].mean()*100:.2f}% | ASSD: {group['mean_assd'].dropna().mean():.2f}px")
print("=" * 65 + "\n")
""")

    notebook_dir = Path("experiments/EXPERIMENT_15_temporal_augmented_mask2former/notebooks")
    notebook_dir.mkdir(parents=True, exist_ok=True)
    nb_path = notebook_dir / "EXP15_Temporal_Mask2Former_Kaggle.ipynb"

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

    print(f"🎉 Generated Kaggle Notebook with {len(nb['cells'])} cells at: {nb_path}")


if __name__ == "__main__":
    create_kaggle_notebook()
