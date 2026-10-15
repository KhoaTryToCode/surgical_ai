"""
Builds the standalone, fully self-contained Kaggle notebook for Experiment 17 (MinVIS).
The resulting notebook runs on a Kaggle T4 GPU with zero external repository dependencies.
Parameterized by `FOLD_IDX = 0` at the very top.
"""

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

    # Read canonical 5-fold split JSON to embed directly
    splits_json_path = Path(__file__).resolve().parent.parent.parent.parent / "_config" / "5fold_patient_splits.json"
    with open(splits_json_path, 'r') as f:
        splits_content = f.read().strip()

    # Cell 0: Header Markdown
    add_cell("markdown", """# Experiment 17: MinVIS (Minimal Video Instance Segmentation)
## Laparoscopic Liver Landmark Detection — 5-Fold Cross Validation

### Scientific Principles & Design:
1. **Zero Video Architecture Overhead**: Standard Mask2Former (Swin-Tiny, ADE20K pretraining) trained on single images.
2. **Online Hungarian Matching at Inference**: Queries are tracked across video frames in post-processing by computing cosine distances on the query embedding outputs ($E_t$).
3. **Exact SOTA Parity**: Identical EXP 15 dataset rasterization (35px canvas stroke width, `INTER_NEAREST` downsampling) and TopoNet mutually exclusive argmax evaluation.
4. **Frozen Test Set**: Patients 21, 31, 41, 51 (109 frames) evaluated after training to benchmark against TopoNet (65.19% Micro Dice).
""")

    # Cell 1: Configuration & Fold Selector
    add_cell("code", """# ==============================================================================
# FOLD SELECTOR (Change to 0, 1, 2, 3, or 4 to run different folds in parallel)
# ==============================================================================
FOLD_IDX = 0

# Training Hyperparameters
EPOCHS = 40
BATCH_SIZE = 4
LR_DECODER = 1.0e-4
LR_BACKBONE = 1.0e-5
WEIGHT_DECAY = 0.05
STROKE_WIDTH = 35
IMAGE_SIZE = 1024
CLIP_LEN = 3

print(f"🎯 Selected Fold: {FOLD_IDX} | Epochs: {EPOCHS} | Batch Size: {BATCH_SIZE}")
""")

    # Cell 2: Embedded Canonical 5-Fold Splits
    add_cell("code", f"""# Canonical 5-Fold Cross-Validation Partition (Locked SHA-256: 18d76bcd...)
import json

SPLITS_CONFIG = json.loads('''{splits_content}''')

fold_data = SPLITS_CONFIG['folds'][f'fold_{{FOLD_IDX}}']
TRAIN_PATIENTS = fold_data['train_patients']
VAL_PATIENTS = fold_data['val_patients']
FROZEN_TEST_PATIENTS = SPLITS_CONFIG['metadata']['frozen_test_patients']

print(f"👥 Fold {{FOLD_IDX}} -> Train Patients ({{len(TRAIN_PATIENTS)}}): {{TRAIN_PATIENTS[:5]}}...")
print(f"👥 Fold {{FOLD_IDX}} -> Val Patients ({{len(VAL_PATIENTS)}}): {{VAL_PATIENTS}}")
print(f"🔒 Frozen Test Patients ({{len(FROZEN_TEST_PATIENTS)}}): {{FROZEN_TEST_PATIENTS}}")
""")

    # Cell 3: Environment Setup & Dataset Resolver
    add_cell("code", """import os
import sys
import glob
import re
import time
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.optimize import linear_sum_assignment

# Install transformers if not present
try:
    import transformers
except ImportError:
    !pip install -q transformers==4.44.2

from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ Using Execution Device: {DEVICE}")

def resolve_split_dir(split_name, candidate_root=None):
    split_lower = split_name.lower()
    if candidate_root and os.path.exists(candidate_root):
        if any(os.path.exists(os.path.join(candidate_root, d)) for d in ['labels', 'Labels', 'LABELS']):
            return candidate_root
        try:
            for sub in os.listdir(candidate_root):
                if sub.lower() == split_lower:
                    full_p = os.path.join(candidate_root, sub)
                    if os.path.isdir(full_p) and any(os.path.exists(os.path.join(full_p, d)) for d in ['labels', 'Labels', 'LABELS']):
                        return full_p
        except OSError:
            pass

    candidates = {
        'train': [
            '/kaggle/input/l3d-train/Train',
            '/kaggle/input/l3d-train/train',
            '/kaggle/input/datasets/khoatrytopublish/l3d-train/Train',
            '/kaggle/input/datasets/khoatrytopublish/l3d-train/train',
            '/kaggle/input/laparoscopic-liver-landmarks/train',
            '/kaggle/input/laparoscopic-liver-landmarks/Train',
            '/kaggle/input/laparoscopic-liver-dataset/L3D/train',
            '/kaggle/input/laparoscopic-liver-dataset/L3D/Train',
            '/kaggle/input/l3d-dataset/L3D/train',
            '/kaggle/input/l3d-dataset/L3D/Train',
            'data/L3D/Train',
            'data/L3D/train',
            '../data/L3D/Train',
            '../data/L3D/train',
            '../../data/L3D/Train',
            '../../data/L3D/train',
        ],
        'val': [
            '/kaggle/input/l3d-val/Val',
            '/kaggle/input/l3d-val/val',
            '/kaggle/input/datasets/khoatrytopublish/l3d-val/Val',
            '/kaggle/input/datasets/khoatrytopublish/l3d-val/val',
            '/kaggle/input/laparoscopic-liver-landmarks/val',
            '/kaggle/input/laparoscopic-liver-landmarks/Val',
            '/kaggle/input/laparoscopic-liver-dataset/L3D/val',
            '/kaggle/input/laparoscopic-liver-dataset/L3D/Val',
            '/kaggle/input/l3d-dataset/L3D/val',
            '/kaggle/input/l3d-dataset/L3D/Val',
            'data/L3D/Val',
            'data/L3D/val',
            '../data/L3D/Val',
            '../data/L3D/val',
            '../../data/L3D/Val',
            '../../data/L3D/val',
        ],
        'test': [
            '/kaggle/input/l3d-test/Test',
            '/kaggle/input/l3d-test/test',
            '/kaggle/input/datasets/khoatrytopublish/l3d-test/Test',
            '/kaggle/input/datasets/khoatrytopublish/l3d-test/test',
            '/kaggle/input/laparoscopic-liver-landmarks/test',
            '/kaggle/input/laparoscopic-liver-landmarks/Test',
            '/kaggle/input/laparoscopic-liver-dataset/L3D/test',
            '/kaggle/input/laparoscopic-liver-dataset/L3D/Test',
            '/kaggle/input/l3d-dataset/L3D/test',
            '/kaggle/input/l3d-dataset/L3D/Test',
            'data/L3D/Test',
            'data/L3D/test',
            '../data/L3D/Test',
            '../data/L3D/test',
            '../../data/L3D/Test',
            '../../data/L3D/test',
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

    raise FileNotFoundError(f"Could not resolve directory for split '{split_name}' across Kaggle or local environments.")

def resolve_all_split_dirs(candidate_root=None):
    splits = {}
    for s in ['train', 'val', 'test']:
        try:
            splits[s] = resolve_split_dir(s, candidate_root=candidate_root)
        except FileNotFoundError:
            pass
    return splits

SPLIT_DIRS_MAP = resolve_all_split_dirs()
ALL_SPLIT_DIRS = list(SPLIT_DIRS_MAP.values())
print("📁 Resolved L3D Dataset Splits:")
for k, v in SPLIT_DIRS_MAP.items():
    print(f"   [{k.upper()}]: {v}")
""")

    # Cell 4: Exact EXP 15 Dataset Loader
    add_cell("code", r"""IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def rasterize_l3d_mask(json_path, orig_shape, target_size=1024, stroke_width=35):
    orig_h, orig_w = orig_shape
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    canvas_h = data.get('imageHeight', orig_h)
    canvas_w = data.get('imageWidth', orig_w)
    if any(x in json_path for x in ['_31', '_36', '_25', '_29']):
        canvas_h, canvas_w = 2160, 3840
    elif canvas_h < 100 or canvas_w < 100:
        canvas_h, canvas_w = 1080, 1920

    canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    for shape in data.get('shapes', []):
        lbl = str(shape.get('label', '')).lower().strip()
        color = 1 if lbl.startswith('r') else (2 if lbl.startswith('s') else (3 if (lbl.startswith('l') or 'falciform' in lbl) else 0))
        if color > 0:
            pts = shape.get('points', [])
            for i in range(1, len(pts)):
                pt1 = tuple(map(int, pts[i - 1]))
                pt2 = tuple(map(int, pts[i]))
                cv2.line(canvas, pt1, pt2, color, int(stroke_width))

    gt_map = cv2.resize(canvas, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    mask_channels = np.stack([gt_map == 1, gt_map == 2, gt_map == 3], axis=0).astype(np.float32)
    return mask_channels, gt_map.astype(np.int64)

class SingleFrameL3DDataset(Dataset):
    def __init__(self, patient_ids, split_dirs=None, data_root=None, image_size=1024, stroke_width=35):
        self.patient_ids = set(patient_ids)
        self.image_size = image_size
        self.stroke_width = stroke_width
        if split_dirs is not None:
            self.split_dirs = [s for s in split_dirs if os.path.exists(s)]
        elif 'ALL_SPLIT_DIRS' in globals():
            self.split_dirs = ALL_SPLIT_DIRS
        else:
            self.split_dirs = list(resolve_all_split_dirs(data_root).values())
        self.samples = self._discover(self.split_dirs)

    def _discover(self, split_dirs):
        samples = []
        for s_dir in split_dirs:
            if not os.path.exists(s_dir):
                continue
            labels_dir, images_dir = None, None
            for cand in ['labels', 'Labels', 'LABELS']:
                p = os.path.join(s_dir, cand)
                if os.path.exists(p):
                    labels_dir = p
                    break
            for cand in ['images', 'Images', 'IMAGES']:
                p = os.path.join(s_dir, cand)
                if os.path.exists(p):
                    images_dir = p
                    break
            if labels_dir is None or images_dir is None:
                continue

            for jf in sorted(glob.glob(os.path.join(labels_dir, '*.json'))):
                m = re.match(r'(Patient_\d+)_(\d+)', Path(jf).stem)
                if not m or m.group(1) not in self.patient_ids:
                    continue
                stem = Path(jf).stem
                ip = os.path.join(images_dir, f"{stem}.png")
                if not os.path.exists(ip):
                    ip = os.path.join(images_dir, f"{stem}.jpg")
                if os.path.exists(ip):
                    samples.append((m.group(1), int(m.group(2)), ip, jf))
        samples.sort(key=lambda x: (x[0], x[1]))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pat, fid, img_path, json_path = self.samples[idx]
        img_bgr = cv2.imread(img_path)
        orig_h, orig_w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_res = cv2.resize(img_rgb, (self.image_size, self.image_size)) if (orig_w != self.image_size or orig_h != self.image_size) else img_rgb

        masks, gt_map = rasterize_l3d_mask(json_path, (orig_h, orig_w), self.image_size, self.stroke_width)
        norm = (img_res.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        return {
            'pixel_values': torch.from_numpy(norm).permute(2, 0, 1).float(),
            'target_masks': torch.from_numpy(masks).float(),
            'gt_map': torch.from_numpy(gt_map).long(),
            'patient': pat, 'frame_id': fid
        }

class PatientVideoL3DDataset(Dataset):
    def __init__(self, patient_ids, split_dirs=None, data_root=None, image_size=1024, stroke_width=35, clip_len=3, max_gap=240):
        self.patient_ids = set(patient_ids)
        self.image_size = image_size
        self.stroke_width = stroke_width
        self.clip_len = clip_len
        self.max_gap = max_gap
        if split_dirs is not None:
            self.split_dirs = [s for s in split_dirs if os.path.exists(s)]
        elif 'ALL_SPLIT_DIRS' in globals():
            self.split_dirs = ALL_SPLIT_DIRS
        else:
            self.split_dirs = list(resolve_all_split_dirs(data_root).values())
        self.clips = self._index_clips(self.split_dirs)

    def _index_clips(self, split_dirs):
        patient_dict = defaultdict(list)
        for s_dir in split_dirs:
            if not os.path.exists(s_dir):
                continue
            labels_dir, images_dir = None, None
            for cand in ['labels', 'Labels', 'LABELS']:
                p = os.path.join(s_dir, cand)
                if os.path.exists(p):
                    labels_dir = p
                    break
            for cand in ['images', 'Images', 'IMAGES']:
                p = os.path.join(s_dir, cand)
                if os.path.exists(p):
                    images_dir = p
                    break
            if labels_dir is None or images_dir is None:
                continue

            for jf in sorted(glob.glob(os.path.join(labels_dir, '*.json'))):
                m = re.match(r'(Patient_\d+)_(\d+)', Path(jf).stem)
                if not m or m.group(1) not in self.patient_ids:
                    continue
                stem = Path(jf).stem
                ip = os.path.join(images_dir, f"{stem}.png")
                if not os.path.exists(ip):
                    ip = os.path.join(images_dir, f"{stem}.jpg")
                if os.path.exists(ip):
                    patient_dict[m.group(1)].append((int(m.group(2)), jf, ip))

        clips = []
        for pat in sorted(patient_dict.keys()):
            frames = sorted(patient_dict[pat], key=lambda x: x[0])
            for i in range(len(frames)):
                tgt = frames[i]
                if i >= 1 and (frames[i][0] - frames[i-1][0]) <= self.max_gap:
                    c1 = frames[i-1]
                    c0 = frames[i-2] if (i >= 2 and (frames[i-1][0] - frames[i-2][0]) <= self.max_gap) else c1
                    window = [c0, c1, tgt]
                else:
                    window = [tgt, tgt, tgt]
                clips.append({'patient': pat, 'paths': [w[2] for w in window], 'jsons': [w[1] for w in window], 'ids': [w[0] for w in window]})
        return clips

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        c = self.clips[idx]
        pts, mts = [], []
        for ip, jp in zip(c['paths'], c['jsons']):
            img = cv2.imread(ip)
            h, w = img.shape[:2]
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = cv2.resize(rgb, (self.image_size, self.image_size)) if (w != self.image_size or h != self.image_size) else rgb
            masks, _ = rasterize_l3d_mask(jp, (h, w), self.image_size, self.stroke_width)
            norm = (res.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
            pts.append(torch.from_numpy(norm).permute(2, 0, 1).float())
            mts.append(torch.from_numpy(masks).float())

        return {
            'pixel_values': torch.stack(pts, dim=0),      # (T, 3, H, W)
            'target_masks': mts[-1],                       # (3, H, W)
            'patient': c['patient'],
            'frame_ids': c['ids']
        }
""")

    # Cell 5: Standard TopoNet Metrics
    add_cell("code", """def compute_dice(p, t):
    inter, total = float(np.sum((p > 0) * (t > 0))), float(np.sum(p > 0) + np.sum(t > 0))
    return 1.0 if total == 0 else (0.0 if np.sum(p > 0) == 0 or np.sum(t > 0) == 0 else 2.0 * inter / total)

def compute_iou(p, t):
    inter, union = float(np.sum((p > 0) * (t > 0))), float(np.sum((p > 0) | (t > 0)))
    return 1.0 if union == 0 else inter / union

def compute_assd_hd95(p, t, max_penalty=80.0):
    p_b, t_b = (p > 0).astype(np.uint8), (t > 0).astype(np.uint8)
    if np.sum(p_b) == 0 and np.sum(t_b) == 0:
        return 0.0, 0.0
    if np.sum(p_b) == 0 or np.sum(t_b) == 0:
        return max_penalty, max_penalty

    cnts_p, _ = cv2.findContours(p_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts_t, _ = cv2.findContours(t_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    edge_p, edge_t = np.zeros_like(p_b), np.zeros_like(t_b)
    cv2.drawContours(edge_p, cnts_p, -1, 1, 1)
    cv2.drawContours(edge_t, cnts_t, -1, 1, 1)

    pts_p, pts_t = edge_p > 0, edge_t > 0
    if not np.any(pts_p) or not np.any(pts_t):
        return max_penalty, max_penalty

    dt_target, dt_pred = cv2.distanceTransform(1 - edge_t, cv2.DIST_L2, 3), cv2.distanceTransform(1 - edge_p, cv2.DIST_L2, 3)
    d_p_to_t, d_t_to_p = dt_target[pts_p], dt_pred[pts_t]
    if len(d_p_to_t) == 0 or len(d_t_to_p) == 0:
        return max_penalty, max_penalty

    assd = float((np.mean(d_p_to_t) + np.mean(d_t_to_p)) / 2.0)
    hd95 = float(np.percentile(np.concatenate([d_p_to_t, d_t_to_p]), 95))
    return assd, hd95

def evaluate_frame_metrics(pred_sem_seg, gt_input, threshold=0.5):
    max_prob = np.max(pred_sem_seg, axis=0)
    argmax_cls = np.argmax(pred_sem_seg, axis=0) + 1
    pred_map = np.where(max_prob >= threshold, argmax_cls, 0).astype(np.uint8)

    if gt_input.ndim == 3:
        gt_map = np.zeros(gt_input.shape[1:], dtype=np.uint8)
        for c in range(gt_input.shape[0]):
            gt_map[gt_input[c] > 0.5] = (c + 1)
    else:
        gt_map = gt_input.astype(np.uint8)

    fg_inter, fg_total = float(np.sum((pred_map > 0) * (gt_map > 0))), float(np.sum(pred_map > 0) + np.sum(gt_map > 0))
    micro_dice = (2.0 * fg_inter / fg_total * 100.0) if fg_total > 0 else 100.0

    dices, ious, assds, hd95s = [], [], [], []
    res = {'micro_dice': micro_dice}
    for c_idx, c_name in enumerate(['ridge', 'silhouette', 'ligament']):
        p_c, g_c = (pred_map == (c_idx + 1)).astype(np.uint8), (gt_map == (c_idx + 1)).astype(np.uint8)
        d, iou = compute_dice(p_c, g_c) * 100.0, compute_iou(p_c, g_c) * 100.0
        assd, hd95 = compute_assd_hd95(p_c, g_c)
        res[f'{c_name}_dice'] = d
        res[f'{c_name}_assd'] = assd
        res[f'{c_name}_hd95'] = hd95
        dices.append(d); ious.append(iou); assds.append(assd); hd95s.append(hd95)

    res['macro_dice'] = float(np.mean(dices))
    res['macro_iou'] = float(np.mean(ious))
    res['macro_assd'] = float(np.mean(assds))
    res['macro_hd95'] = float(np.mean(hd95s))
    return res
""")

    # Cell 6: MinVIS Model & Online Tracker
    add_cell("code", """class MinVISMask2Former(nn.Module):
    def __init__(self, base_checkpoint="facebook/mask2former-swin-tiny-ade-semantic", num_classes=3, hidden_dim=256):
        super().__init__()
        self.num_classes = num_classes
        try:
            self.base_model = Mask2FormerForUniversalSegmentation.from_pretrained(
                base_checkpoint, num_labels=num_classes, ignore_mismatched_sizes=True, output_hidden_states=True
            )
        except Exception:
            cfg = Mask2FormerConfig(num_queries=100, num_labels=num_classes, output_hidden_states=True)
            self.base_model = Mask2FormerForUniversalSegmentation(cfg)
        self.base_model.config.output_hidden_states = True
        self.query_norm = nn.LayerNorm(hidden_dim)

    def prepare_targets(self, target_masks):
        bs, num_cls, h, w = target_masks.shape
        mask_labels, class_labels = [], []
        for b in range(bs):
            b_masks, b_classes = [], []
            for c in range(num_cls):
                m = target_masks[b, c]
                if m.sum() > 0:
                    b_masks.append(m)
                    b_classes.append(c)
            if len(b_masks) == 0:
                b_masks.append(torch.zeros((h, w), device=target_masks.device))
                b_classes.append(0)
            mask_labels.append(torch.stack(b_masks, dim=0))
            class_labels.append(torch.tensor(b_classes, dtype=torch.long, device=target_masks.device))
        return mask_labels, class_labels

    def forward(self, pixel_values, target_masks=None):
        orig_shape = pixel_values.shape
        is_video = (pixel_values.ndim == 5)
        x = pixel_values.view(-1, orig_shape[-3], orig_shape[-2], orig_shape[-1]) if is_video else pixel_values
        h, w = x.shape[-2:]

        loss = None
        if target_masks is not None and self.training:
            ml, cl = self.prepare_targets(target_masks)
            outputs = self.base_model(pixel_values=x, mask_labels=ml, class_labels=cl, output_hidden_states=True)
            loss = outputs.loss
        else:
            outputs = self.base_model(pixel_values=x, output_hidden_states=True)

        class_logits = outputs.class_queries_logits
        mask_logits = outputs.masks_queries_logits
        raw_hidden = outputs.transformer_decoder_last_hidden_state
        if raw_hidden.shape[0] == 100 and raw_hidden.shape[1] != 100:
            query_embds = raw_hidden.permute(1, 0, 2)
        else:
            query_embds = raw_hidden
        query_embds = self.query_norm(query_embds)

        mask_probs = F.interpolate(mask_logits.sigmoid(), size=(h, w), mode='bilinear', align_corners=False)
        class_probs = class_logits.softmax(dim=-1)[..., :self.num_classes]
        sem_seg = torch.einsum('bqc, bqhw -> bchw', class_probs, mask_probs)

        return {
            'sem_seg': sem_seg,
            'class_queries_logits': class_logits,
            'masks_queries_logits': mask_logits,
            'query_embeddings': query_embds,
            'loss': loss
        }

class MinVISTracker:
    def __init__(self, num_classes=3):
        self.num_classes = num_classes

    @staticmethod
    def match_from_embds(tgt_embds, cur_embds):
        cur_norm = cur_embds / (cur_embds.norm(dim=1, keepdim=True) + 1e-6)
        tgt_norm = tgt_embds / (tgt_embds.norm(dim=1, keepdim=True) + 1e-6)
        cos_sim = torch.mm(cur_norm, tgt_norm.transpose(0, 1))
        cost = (1.0 - cos_sim).detach().cpu().numpy()
        _, col_ind = linear_sum_assignment(cost.T)
        return col_ind

    def track_sequence(self, query_embds, class_logits, mask_probs):
        t, q = query_embds.shape[:2]
        tracked_logits, tracked_masks, tracked_embds = [class_logits[0]], [mask_probs[0]], [query_embds[0]]
        for i in range(1, t):
            indices = self.match_from_embds(tracked_embds[-1], query_embds[i])
            idx_t = torch.from_numpy(indices).to(query_embds.device)
            tracked_logits.append(class_logits[i][idx_t, :])
            tracked_masks.append(mask_probs[i][idx_t, :, :])
            tracked_embds.append(query_embds[i][idx_t, :])

        t_logits = torch.stack(tracked_logits, dim=0)
        t_masks = torch.stack(tracked_masks, dim=0)
        class_probs = t_logits.softmax(dim=-1)[..., :self.num_classes]
        tracked_sem_seg = torch.einsum('tqc, tqhw -> tchw', class_probs, t_masks)
        return tracked_sem_seg
""")

    # Cell 7: Training & Validation Loop
    add_cell("code", """def evaluate_split(model, tracker, loader):
    model.eval()
    all_metrics = []
    with torch.no_grad():
        for batch in loader:
            pv = batch['pixel_values'][0].to(DEVICE)
            tm = batch['target_masks'][0].cpu().numpy()
            h, w = pv.shape[-2:]

            out = model(pv)
            mask_probs = F.interpolate(out['masks_queries_logits'].sigmoid(), size=(h, w), mode='bilinear', align_corners=False)
            res = tracker.track_sequence(out['query_embeddings'], out['class_queries_logits'], mask_probs)
            tracked_sem = res[0] if isinstance(res, (tuple, list)) else res
            pred_target = tracked_sem[-1].cpu().numpy()

            all_metrics.append(evaluate_frame_metrics(pred_target, tm))

    summary = {k: float(np.mean([m[k] for m in all_metrics])) for k in all_metrics[0].keys()}
    return summary

# Instantiate Datasets & Loaders
train_ds = SingleFrameL3DDataset(TRAIN_PATIENTS, split_dirs=ALL_SPLIT_DIRS, image_size=IMAGE_SIZE, stroke_width=STROKE_WIDTH)
val_ds = PatientVideoL3DDataset(VAL_PATIENTS, split_dirs=ALL_SPLIT_DIRS, image_size=IMAGE_SIZE, stroke_width=STROKE_WIDTH, clip_len=CLIP_LEN)
test_ds = PatientVideoL3DDataset(FROZEN_TEST_PATIENTS, split_dirs=ALL_SPLIT_DIRS, image_size=IMAGE_SIZE, stroke_width=STROKE_WIDTH, clip_len=CLIP_LEN)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)

print(f"📦 Train Batches: {len(train_loader)} | Val Clips: {len(val_loader)} | Frozen Test Clips: {len(test_loader)}")

model = MinVISMask2Former().to(DEVICE)
tracker = MinVISTracker(num_classes=3)

backbone_params, decoder_params = [], []
for name, param in model.named_parameters():
    if param.requires_grad:
        (backbone_params if ("pixel_level_module" in name or "backbone" in name) else decoder_params).append(param)

optimizer = AdamW([{'params': decoder_params, 'lr': LR_DECODER}, {'params': backbone_params, 'lr': LR_BACKBONE}], weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1.0e-6)
scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == 'cuda'))

best_val_dice = -1.0
best_checkpoint_path = f"best_minvis_fold{FOLD_IDX}.pth"

print(f"🚀 Starting MinVIS Training on Fold {FOLD_IDX} ({EPOCHS} epochs)...")
start_time = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    for batch in train_loader:
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(DEVICE.type == 'cuda')):
            out = model(batch['pixel_values'].to(DEVICE), target_masks=batch['target_masks'].to(DEVICE))
            loss = out['loss']

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()

    scheduler.step()
    train_loss = epoch_loss / max(1, len(train_loader))

    val_metrics = evaluate_split(model, tracker, val_loader)
    val_dice = val_metrics['macro_dice']

    if val_dice > best_val_dice:
        best_val_dice = val_dice
        torch.save(model.state_dict(), best_checkpoint_path)

    elapsed = time.time() - start_time
    print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] Loss: {train_loss:.4f} | "
          f"Val Macro Dice: {val_dice:.2f}% (Ridge: {val_metrics['ridge_dice']:.1f}%, Silh: {val_metrics['silhouette_dice']:.1f}%, Lig: {val_metrics['ligament_dice']:.1f}%) | "
          f"ASSD: {val_metrics['macro_assd']:.2f}px | Best: {best_val_dice:.2f}% | Elapsed: {elapsed/60:.1f}m")
""")

    # Cell 8: Final Frozen Test Evaluation & JSON Export
    add_cell("code", """print(f"🏁 Training Complete! Loading Best Weights ({best_val_dice:.2f}% Dice)...")
model.load_state_dict(torch.load(best_checkpoint_path, map_location=DEVICE))

best_val_metrics = evaluate_split(model, tracker, val_loader)
frozen_test_metrics = evaluate_split(model, tracker, test_loader)

print("\n" + "=" * 60)
print(f"🏆 BEST FOLD {FOLD_IDX} VALIDATION METRICS:")
print(f"  Macro Dice:  {best_val_metrics['macro_dice']:.2f}% | Micro Dice: {best_val_metrics['micro_dice']:.2f}%")
print(f"  Ridge Dice:  {best_val_metrics['ridge_dice']:.2f}% | Silhouette: {best_val_metrics['silhouette_dice']:.2f}% | Ligament: {best_val_metrics['ligament_dice']:.2f}%")
print(f"  Macro ASSD:  {best_val_metrics['macro_assd']:.2f}px | Macro HD95: {best_val_metrics['macro_hd95']:.2f}px")
print("=" * 60)

print("\n" + "=" * 60)
print("🔒 FROZEN TEST SET METRICS (PATIENTS 21, 31, 41, 51):")
print(f"  Macro Dice:  {frozen_test_metrics['macro_dice']:.2f}% | Micro Dice: {frozen_test_metrics['micro_dice']:.2f}%")
print(f"  Ridge Dice:  {frozen_test_metrics['ridge_dice']:.2f}% | Silhouette: {frozen_test_metrics['silhouette_dice']:.2f}% | Ligament: {frozen_test_metrics['ligament_dice']:.2f}%")
print(f"  Macro ASSD:  {frozen_test_metrics['macro_assd']:.2f}px | Macro HD95: {frozen_test_metrics['macro_hd95']:.2f}px")
print("=" * 60)

# Export standard JSON result files for easy aggregation
with open(f"results_fold_{FOLD_IDX}.json", "w") as f:
    json.dump({
        "experiment": "EXPERIMENT_17_minvis",
        "fold_idx": FOLD_IDX,
        "best_val_metrics": best_val_metrics,
        "val_patients": VAL_PATIENTS
    }, f, indent=2)

with open(f"test_results_fold_{FOLD_IDX}.json", "w") as f:
    json.dump({
        "experiment": "EXPERIMENT_17_minvis",
        "fold_idx": FOLD_IDX,
        "frozen_test_patients": FROZEN_TEST_PATIENTS,
        "test_metrics": frozen_test_metrics
    }, f, indent=2)

print(f"💾 Successfully exported 'results_fold_{FOLD_IDX}.json' and 'test_results_fold_{FOLD_IDX}.json'!")
""")

    out_dir = Path(__file__).resolve().parent.parent / "notebooks"
    os.makedirs(out_dir, exist_ok=True)
    out_file = out_dir / "EXP17_MinVIS_Kaggle.ipynb"

    with open(out_file, "w") as f:
        json.dump(nb, f, indent=2)

    print(f"✅ Kaggle notebook generated successfully: '{out_file}'")


if __name__ == "__main__":
    create_kaggle_notebook()
