"""
Standalone Inference & Diagnostic Script for EXPERIMENT_14 (Stratified Junction-Steered Mask2Former).
Identifies and renders the Top 20 Worst Frames from both the Validation and Test splits.
Can be executed locally or pasted directly into a single Kaggle notebook cell.
"""
import os
import sys
import glob
import json
import time
import zipfile
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# NumPy 2.0 compatibility
for _a, _v in [('Inf', np.inf), ('NAN', np.nan), ('NaN', np.nan), ('PINF', np.inf), ('NINF', -np.inf)]:
    if not hasattr(np, _a):
        setattr(np, _a, _v)

# ==============================================================================
# 1. Configuration & Path Resolution
# ==============================================================================
# Set this to your checkpoint directory or the direct path to best_model.pth:
CHECKPOINT_INPUT = "/kaggle/working/results_exp14"  # Or "/kaggle/input/your-dataset/best_model.pth"
OUTPUT_DIR = "/kaggle/working/diagnostics_worst20"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ Using compute device: {device}")
if device.type == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

def resolve_checkpoint_file(target_path):
    if os.path.isfile(target_path):
        return os.path.abspath(target_path)
    if os.path.isdir(target_path):
        candidates = [
            os.path.join(target_path, "best_model.pth"),
            os.path.join(target_path, "results", "best_model.pth"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                return os.path.abspath(c)
        # Search for any .pth file
        pths = glob.glob(os.path.join(target_path, "**", "*.pth"), recursive=True)
        if len(pths) > 0:
            return os.path.abspath(pths[0])
    raise FileNotFoundError(f"Could not locate best_model.pth in: {target_path}")

def find_split_path(split_name, explicit_path=None):
    if explicit_path and os.path.exists(explicit_path):
        return os.path.abspath(explicit_path)
    target = split_name.lower()
    candidates = [
        f"/kaggle/input/datasets/khoatrytopublish/l3d-{target}/{split_name}",
        f"/kaggle/input/datasets/khoatrytopublish/l3d_{target}/{split_name}",
        f"/kaggle/input/datasets/khoatrytopublish/l3d{target}/{split_name}",
        f"/kaggle/input/l3d-{target}/{split_name}",
        f"/kaggle/input/l3d_{target}/{split_name}",
        f"/kaggle/input/l3d - {split_name}/{split_name}",
        f"/kaggle/input/l3d - {target}/{split_name}",
        f"/kaggle/input/l3d-{target}",
        f"/kaggle/input/l3d/{split_name}",
        f"/kaggle/input/laparoscopic-liver-landmark-dataset/L3D/{split_name}",
        f"/kaggle/input/l3d-dataset/L3D/{split_name}",
        f"/data/khoalq/data/L3D/{split_name}",
        f"data/L3D/{split_name}",
        f"../data/L3D/{split_name}",
        f"../../data/L3D/{split_name}",
    ]
    for c in candidates:
        if os.path.exists(c) and os.path.exists(os.path.join(c, "labels")):
            return os.path.abspath(c)
    if os.path.exists("/kaggle/input"):
        for m in glob.glob(f"/kaggle/input/**/{split_name}", recursive=True):
            if os.path.isdir(m) and os.path.exists(os.path.join(m, "labels")):
                return os.path.abspath(m)
        for m in glob.glob(f"/kaggle/input/**/{target}", recursive=True):
            if os.path.isdir(m) and os.path.exists(os.path.join(m, "labels")):
                return os.path.abspath(m)
        for p in glob.glob("/kaggle/input/**/labels", recursive=True):
            parent = os.path.dirname(p)
            if target in parent.lower():
                return os.path.abspath(parent)
    raise RuntimeError(f"Could not locate {split_name} split directory! Check your dataset inputs.")

# ==============================================================================
# 2. Architecture Definitions
# ==============================================================================
from transformers import Mask2FormerForUniversalSegmentation

class JunctionDecoderLayer(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, context):
        q_norm = self.norm1(queries)
        q_self, _ = self.self_attn(q_norm, q_norm, q_norm)
        queries = queries + self.dropout(q_self)
        q_norm = self.norm2(queries)
        q_cross, _ = self.cross_attn(query=q_norm, key=context, value=context)
        queries = queries + self.dropout(q_cross)
        queries = queries + self.ffn(self.norm3(queries))
        return queries

class JunctionAnchorHead(nn.Module):
    def __init__(self, embed_dim=256, num_layers=2, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.junction_queries = nn.Parameter(torch.randn(1, 4, embed_dim) * 0.02)
        self.layers = nn.ModuleList([
            JunctionDecoderLayer(embed_dim=embed_dim, num_heads=num_heads)
            for _ in range(num_layers)
        ])
        self.coord_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, 2)
        )
        self.vis_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        priors = torch.tensor([
            [0.50, 0.20],
            [0.50, 0.70],
            [0.85, 0.40],
            [0.15, 0.60]
        ], dtype=torch.float32)
        prior_logits = torch.log(priors / (1.0 - priors + 1e-6))
        self.register_buffer('prior_logits', prior_logits)

    def forward(self, feature_map):
        B, C, H, W = feature_map.shape
        context = feature_map.flatten(2).permute(0, 2, 1)
        queries = self.junction_queries.repeat(B, 1, 1)
        for layer in self.layers:
            queries = layer(queries, context)
        raw_coords = self.coord_head(queries)
        pred_coords = torch.sigmoid(raw_coords + self.prior_logits.unsqueeze(0))
        pred_vis = self.vis_head(queries).squeeze(-1)
        return pred_coords, pred_vis, queries

class JunctionQuerySteering(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.proj_q = nn.Linear(embed_dim, embed_dim)
        self.proj_k = nn.Linear(embed_dim, embed_dim)
        self.proj_v = nn.Linear(embed_dim, embed_dim)
        self.gate = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, junction_features, junction_coords=None):
        q = self.proj_q(queries)
        k = self.proj_k(junction_features)
        v = self.proj_v(junction_features)
        delta_q, attn_weights = self.cross_attn(query=q, key=k, value=v)
        steered_queries = self.norm(queries + self.gate * self.dropout(delta_q))
        return steered_queries, attn_weights

class JunctionSteeredMask2Former(nn.Module):
    def __init__(self, model_name="facebook/mask2former-swin-tiny-ade-semantic", num_labels=4):
        super().__init__()
        self.m2f = Mask2FormerForUniversalSegmentation.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )
        embed_dim = 256
        self.junction_head = JunctionAnchorHead(embed_dim=embed_dim, num_layers=2, num_heads=8)
        self.query_steering = JunctionQuerySteering(embed_dim=embed_dim, num_heads=8)

    def forward_transformer_decoder(self, multi_scale_features, mask_features, steered_query_feat):
        tm = self.m2f.model.transformer_module
        multi_stage_features = []
        multi_stage_positional_embeddings = []
        size_list = []
        for i in range(tm.num_feature_levels):
            size_list.append(multi_scale_features[i].shape[-2:])
            multi_stage_positional_embeddings.append(
                tm.position_embedder(
                    multi_scale_features[i].shape, multi_scale_features[i].device, multi_scale_features[i].dtype, None
                ).flatten(2)
            )
            multi_stage_features.append(
                tm.input_projections[i](multi_scale_features[i]).flatten(2)
                + tm.level_embed.weight[i][None, :, None]
            )
            multi_stage_positional_embeddings[-1] = multi_stage_positional_embeddings[-1].permute(2, 0, 1)
            multi_stage_features[-1] = multi_stage_features[-1].permute(2, 0, 1)

        _, batch_size, _ = multi_stage_features[0].shape
        query_embeddings = tm.queries_embedder.weight.unsqueeze(1).repeat(1, batch_size, 1)
        query_features = steered_query_feat.permute(1, 0, 2)

        decoder_output = tm.decoder(
            inputs_embeds=query_features,
            multi_stage_positional_embeddings=multi_stage_positional_embeddings,
            pixel_embeddings=mask_features,
            encoder_hidden_states=multi_stage_features,
            query_position_embeddings=query_embeddings,
            feature_size_list=size_list,
            output_hidden_states=False,
            output_attentions=False,
            return_dict=True,
        )
        return decoder_output

    def forward(self, pixel_values):
        B = pixel_values.shape[0]
        pixel_level_outputs = self.m2f.model.pixel_level_module(pixel_values, output_hidden_states=True)
        multi_scale_features = list(pixel_level_outputs.decoder_hidden_states)
        mask_features = pixel_level_outputs.decoder_last_hidden_state
        stride_16_feat = multi_scale_features[1]
        pred_j_coords, pred_j_vis, j_features = self.junction_head(stride_16_feat)

        tm = self.m2f.model.transformer_module
        base_query_feat = tm.queries_features.weight.unsqueeze(0).repeat(B, 1, 1)
        steered_query_feat, j_attn_weights = self.query_steering(base_query_feat, j_features, pred_j_coords)

        decoder_output = self.forward_transformer_decoder(multi_scale_features, mask_features, steered_query_feat)
        class_queries_logits = self.m2f.class_predictor(decoder_output.last_hidden_state)
        masks_queries_logits = decoder_output.masks_queries_logits[-1]

        return {
            'masks_queries_logits': masks_queries_logits,
            'class_queries_logits': class_queries_logits,
            'pred_junction_coords': pred_j_coords,
            'pred_junction_vis': pred_j_vis,
            'junction_attn_weights': j_attn_weights
        }

# ==============================================================================
# 3. Dataset & Preprocessing
# ==============================================================================
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def extract_gt_junctions_local(data, orig_w, orig_h, canvas_size=1024, threshold_px=100.0):
    sx = float(canvas_size) / float(orig_w)
    sy = float(canvas_size) / float(orig_h)
    r_curves, s_curves, f_curves = [], [], []
    for shape in data.get('shapes', []):
        lbl = str(shape.get('label', '')).lower().strip()
        pts = np.array(shape.get('points', []), dtype=np.float32)
        if len(pts) < 2:
            continue
        scaled = np.column_stack([pts[:, 0] * sx, pts[:, 1] * sy])
        if lbl.startswith('r') or 'ridge' in lbl or 'rigde' in lbl:
            r_curves.append(scaled)
        elif lbl.startswith('s') or 'sil' in lbl:
            s_curves.append(scaled)
        elif lbl.startswith('f') or 'falc' in lbl or 'lig' in lbl:
            f_curves.append(scaled)

    coords_px = np.zeros((4, 2), dtype=np.float32)
    visibility = np.zeros(4, dtype=np.float32)

    # 1. J_top
    best_d = 1e9
    j_top = None
    for fc in f_curves:
        for sc in s_curves:
            for f_pt in [fc[0], fc[-1]]:
                dists = np.linalg.norm(sc - f_pt, axis=1)
                min_idx = np.argmin(dists)
                if dists[min_idx] < best_d:
                    best_d = dists[min_idx]
                    j_top = (f_pt + sc[min_idx]) / 2.0
    if best_d <= threshold_px and j_top is not None:
        coords_px[0] = j_top
        visibility[0] = 1.0

    # 2. J_bottom
    best_d = 1e9
    j_bot = None
    for fc in f_curves:
        for rc in r_curves:
            for f_pt in [fc[0], fc[-1]]:
                dists = np.linalg.norm(rc - f_pt, axis=1)
                min_idx = np.argmin(dists)
                if dists[min_idx] < best_d:
                    best_d = dists[min_idx]
                    j_bot = (f_pt + rc[min_idx]) / 2.0
    if best_d <= threshold_px and j_bot is not None:
        coords_px[1] = j_bot
        visibility[1] = 1.0

    # 3. J_lat_right & Left
    best_dr, pt_r = 1e9, None
    best_dl, pt_l = 1e9, None
    for rc in r_curves:
        for sc in s_curves:
            for r_pt in [rc[0], rc[-1]]:
                d_s = np.linalg.norm(sc - r_pt, axis=1)
                idx = np.argmin(d_s)
                d = d_s[idx]
                pt = (r_pt + sc[idx]) / 2.0
                if pt[0] >= canvas_size * 0.45 and d < best_dr:
                    best_dr, pt_r = d, pt
                elif pt[0] < canvas_size * 0.45 and d < best_dl:
                    best_dl, pt_l = d, pt
    if best_dr <= threshold_px and pt_r is not None:
        coords_px[2] = pt_r
        visibility[2] = 1.0
    if best_dl <= threshold_px and pt_l is not None:
        coords_px[3] = pt_l
        visibility[3] = 1.0

    coords_norm = coords_px / float(canvas_size)
    return coords_norm, visibility

class StratifiedL3DDataset(Dataset):
    VAL_PATIENTS = {'patient_38', 'patient_32', 'patient_18'}

    def __init__(self, split='Val', image_size=1024, stroke_width=35):
        self.split = split
        self.image_size = image_size
        self.stroke_width = stroke_width

        if split == 'Test':
            self.test_dir = find_split_path('Test')
            self.items = []
            for jf in sorted(glob.glob(os.path.join(self.test_dir, 'labels', '*.json'))):
                stem = Path(jf).stem
                img_path = os.path.join(self.test_dir, 'images', f"{stem}.jpg")
                if not os.path.exists(img_path):
                    img_path = os.path.join(self.test_dir, 'images', f"{stem}.png")
                self.items.append((jf, img_path))
        else:
            self.train_dir = find_split_path('Train')
            self.val_dir = find_split_path('Val')
            all_files = []
            for sp_dir in [self.train_dir, self.val_dir]:
                for jf in glob.glob(os.path.join(sp_dir, 'labels', '*.json')):
                    stem = Path(jf).stem
                    img_path = os.path.join(sp_dir, 'images', f"{stem}.jpg")
                    if not os.path.exists(img_path):
                        img_path = os.path.join(sp_dir, 'images', f"{stem}.png")
                    all_files.append((jf, img_path))
            self.items = []
            for jf, img_path in sorted(all_files):
                stem = Path(jf).stem.lower()
                is_val = any(vp in stem for vp in self.VAL_PATIENTS)
                if split == 'Val' and is_val:
                    self.items.append((jf, img_path))
                elif split == 'Train' and not is_val:
                    self.items.append((jf, img_path))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        json_path, img_path = self.items[idx]
        stem = Path(json_path).stem
        img_bgr = cv2.imread(img_path)
        orig_h, orig_w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        norm_img = (img_resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        pixel_values = torch.from_numpy(norm_img).permute(2, 0, 1).float()

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        data_w = data.get('imageWidth', orig_w)
        data_h = data.get('imageHeight', orig_h)
        canvas_raw = np.zeros((data_h, data_w), dtype=np.uint8)

        for shape in data.get('shapes', []):
            lbl = str(shape.get('label', '')).lower().strip()
            pts = shape.get('points', [])
            if len(pts) < 2:
                continue
            if lbl.startswith('r') or 'ridge' in lbl or 'rigde' in lbl:
                color = 1
            elif lbl.startswith('s') or 'sil' in lbl:
                color = 2
            elif lbl.startswith('f') or 'falc' in lbl or 'lig' in lbl:
                color = 3
            else:
                color = 0

            if color > 0:
                for i in range(1, len(pts)):
                    pt1 = tuple(map(int, pts[i - 1]))
                    pt2 = tuple(map(int, pts[i]))
                    cv2.line(canvas_raw, pt1, pt2, color, self.stroke_width)

        mask = cv2.resize(canvas_raw, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        coords_norm, vis = extract_gt_junctions_local(data, data_w, data_h, canvas_size=self.image_size)

        return {
            'pixel_values': pixel_values,
            'mask': torch.from_numpy(mask).long(),
            'junction_coords': torch.from_numpy(coords_norm).float(),
            'junction_vis': torch.from_numpy(vis).float(),
            'filename': f"{stem}.jpg",
            'json_path': json_path
        }

# ==============================================================================
# 4. Metrics & Post-Processing
# ==============================================================================
def rasterize_class_map(masks_queries_logits, class_queries_logits, canvas_size=1024):
    masks = torch.nn.functional.interpolate(
        masks_queries_logits.float().unsqueeze(0),
        size=(canvas_size, canvas_size),
        mode='bilinear',
        align_corners=False
    ).squeeze(0).sigmoid()
    cls_probs = torch.softmax(class_queries_logits.float(), dim=-1)
    fg_cls_probs = cls_probs[:, :4]
    sem_probs = torch.einsum("qc,qhw->chw", fg_cls_probs, masks)
    pred_map = sem_probs.argmax(dim=0).cpu().numpy().astype(np.int64)
    max_prob = sem_probs.max(dim=0)[0].cpu().numpy()
    pred_map[max_prob < 0.25] = 0
    return pred_map

def compute_dice(pred_bin, target_bin, eps=1e-6):
    pred_b = (pred_bin > 0).astype(bool)
    target_b = (target_bin > 0).astype(bool)
    intersection = np.logical_and(pred_b, target_b).sum()
    total = pred_b.sum() + target_b.sum()
    if total == 0:
        return 1.0
    return float(2.0 * intersection / (total + eps))

def compute_assd_fast(pred_bin, target_bin, max_penalty=80.0):
    p_b = (pred_bin > 0).astype(np.uint8)
    t_b = (target_bin > 0).astype(np.uint8)
    if p_b.sum() == 0 and t_b.sum() == 0:
        return 0.0
    if p_b.sum() == 0 or t_b.sum() == 0:
        return max_penalty
    try:
        contours_p, _ = cv2.findContours(p_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours_t, _ = cv2.findContours(t_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        edge_p = np.zeros_like(p_b)
        edge_t = np.zeros_like(t_b)
        cv2.drawContours(edge_p, contours_p, -1, 1, 1)
        cv2.drawContours(edge_t, contours_t, -1, 1, 1)
        if edge_p.sum() == 0 or edge_t.sum() == 0:
            return max_penalty
        dist_t = cv2.distanceTransform((1 - edge_t).astype(np.uint8), cv2.DIST_L2, 3)
        dist_p = cv2.distanceTransform((1 - edge_p).astype(np.uint8), cv2.DIST_L2, 3)
        d_p2t = np.mean(dist_t[edge_p > 0])
        d_t2p = np.mean(dist_p[edge_t > 0])
        return float((d_p2t + d_t2p) / 2.0)
    except Exception:
        return max_penalty

COLOR_MAP_BGR = {
    1: (0, 220, 0),     # Ridge: Green
    2: (0, 0, 255),     # Silhouette: Red
    3: (255, 120, 0)    # Falciform: Sky Blue
}
JUNCTION_COLORS_BGR = {
    0: (0, 215, 255), # Top: Gold
    1: (255, 255, 0), # Bottom: Cyan
    2: (255, 0, 255), # Lat_Right: Magenta
    3: (0, 140, 255)  # Lat_Left: Orange
}

def render_diagnostic_montage(orig_rgb_norm, gt_mask, pred_map, pred_j_coords, gt_j_coords, gt_j_vis, out_path, metrics):
    img = (orig_rgb_norm.transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    H, W = bgr.shape[:2]

    # Panel 2: GT Overlay
    p2 = bgr.copy()
    for c_id in [1, 2, 3]:
        mask_c = (gt_mask == c_id)
        if mask_c.any():
            p2[mask_c] = (p2[mask_c] * 0.4 + np.array(COLOR_MAP_BGR[c_id]) * 0.6).astype(np.uint8)
    for k in range(4):
        if gt_j_vis[k] > 0.5:
            pt = (gt_j_coords[k] * float(W)).astype(int)
            cv2.circle(p2, tuple(pt), 14, (0, 0, 0), -1)
            cv2.circle(p2, tuple(pt), 10, JUNCTION_COLORS_BGR[k], -1)

    # Panel 3: Prediction Overlay
    p3 = bgr.copy()
    for c_id in [1, 2, 3]:
        mask_c = (pred_map == c_id)
        if mask_c.any():
            p3[mask_c] = (p3[mask_c] * 0.4 + np.array(COLOR_MAP_BGR[c_id]) * 0.6).astype(np.uint8)
    for k in range(4):
        pt = (pred_j_coords[k] * float(W)).astype(int)
        cv2.circle(p3, tuple(pt), 14, (0, 0, 0), -1)
        cv2.circle(p3, tuple(pt), 10, JUNCTION_COLORS_BGR[k], -1)

    # Panel 4: Error Map (Green=TP, Yellow=FP, Red=FN)
    p4 = bgr.copy()
    tp = (pred_map > 0) & (gt_mask > 0)
    fp = (pred_map > 0) & (gt_mask == 0)
    fn = (pred_map == 0) & (gt_mask > 0)
    p4[tp] = (p4[tp] * 0.3 + np.array([0, 255, 0]) * 0.7).astype(np.uint8)
    p4[fp] = (p4[fp] * 0.3 + np.array([0, 255, 255]) * 0.7).astype(np.uint8)
    p4[fn] = (p4[fn] * 0.3 + np.array([0, 0, 255]) * 0.7).astype(np.uint8)

    top_row = np.hstack([bgr, p2])
    bot_row = np.hstack([p3, p4])
    montage = np.vstack([top_row, bot_row])

    cv2.putText(montage, f"1. Input: {metrics['filename']}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(montage, "2. Ground Truth (Green=Ridge, Red=Sil, Blue=Falc)", (W + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(montage, f"3. Pred | Dice: {metrics['macro_dice']:.1%} (R:{metrics['ridge_dice']:.1%}, S:{metrics['sil_dice']:.1%}, F:{metrics['falc_dice']:.1%})", (20, H + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 200), 2)
    cv2.putText(montage, f"4. Error Map | ASSD: {metrics['macro_assd']:.1f}px (Green=TP, Yellow=FP, Red=FN)", (W + 20, H + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, montage, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return montage

# ==============================================================================
# 5. Full Split Evaluation Function
# ==============================================================================
def evaluate_split(model, split_name):
    dataset = StratifiedL3DDataset(split=split_name)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=2 if device.type == 'cuda' else 0)
    print(f"\n🔍 Running Inference on {split_name} Split ({len(dataset)} frames)...")

    results = []
    raw_samples = {}

    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            pixel_values = batch['pixel_values'].to(device)
            masks = batch['mask'].cpu().numpy()
            gt_j_coords = batch['junction_coords'].cpu().numpy()
            gt_j_vis = batch['junction_vis'].cpu().numpy()
            filenames = batch['filename']

            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                outputs = model(pixel_values)

            pred_masks_logits = outputs['masks_queries_logits']
            pred_cls_logits = outputs['class_queries_logits']
            pred_j_coords = outputs['pred_junction_coords'].float().cpu().numpy()

            for b in range(pixel_values.shape[0]):
                fname = filenames[b]
                pred_map = rasterize_class_map(pred_masks_logits[b], pred_cls_logits[b])
                gt_map = masks[b]

                d_r = compute_dice(pred_map == 1, gt_map == 1)
                d_s = compute_dice(pred_map == 2, gt_map == 2)
                d_f = compute_dice(pred_map == 3, gt_map == 3)
                m_dice = float(np.mean([d_r, d_s, d_f]))

                a_r = compute_assd_fast(pred_map == 1, gt_map == 1)
                a_s = compute_assd_fast(pred_map == 2, gt_map == 2)
                a_f = compute_assd_fast(pred_map == 3, gt_map == 3)
                m_assd = float(np.mean([a_r, a_s, a_f]))

                pat_parts = fname.split('_')
                pat_id = f"{pat_parts[0]}_{pat_parts[1]}" if len(pat_parts) >= 2 else "Unknown"

                # Diagnose error archetype
                diagnosis = "Good"
                if m_dice < 0.35:
                    if d_r < 0.2 and d_s < 0.2:
                        diagnosis = "Severe Inversion / Class Inversion"
                    elif d_f < 0.1:
                        diagnosis = "Complete Falciform Omission"
                    else:
                        diagnosis = "Severe Disconnection / False Negatives"
                elif d_f < 0.2:
                    diagnosis = "Missed Falciform Ligament"
                elif m_assd > 35.0:
                    diagnosis = "Large Contour Spatial Offset"
                elif m_dice < 0.55:
                    diagnosis = "Partial Class Confusion"

                record = {
                    'filename': fname,
                    'patient': pat_id,
                    'macro_dice': m_dice,
                    'ridge_dice': d_r,
                    'sil_dice': d_s,
                    'falc_dice': d_f,
                    'macro_assd': m_assd,
                    'ridge_assd': a_r,
                    'sil_assd': a_s,
                    'falc_assd': a_f,
                    'primary_diagnosis': diagnosis
                }
                results.append(record)

                raw_samples[fname] = {
                    'pixel_values': pixel_values[b].float().cpu().numpy(),
                    'gt_mask': gt_map,
                    'pred_map': pred_map,
                    'pred_j_coords': pred_j_coords[b],
                    'gt_j_coords': gt_j_coords[b],
                    'gt_j_vis': gt_j_vis[b],
                    'metrics': record
                }

    df = pd.DataFrame(results)
    print(f"   Split Summary: Macro Dice = {df['macro_dice'].mean():.2%} | ASSD = {df['macro_assd'].mean():.2f}px")
    return df, raw_samples

# ==============================================================================
# 6. Main Execution Pipeline
# ==============================================================================
def main():
    ckpt_file = resolve_checkpoint_file(CHECKPOINT_INPUT)
    print(f"📦 Loading best checkpoint: {ckpt_file}")
    model = JunctionSteeredMask2Former(num_labels=4)
    ckpt_data = torch.load(ckpt_file, map_location=device, weights_only=False)
    state_dict = ckpt_data['model_state_dict'] if 'model_state_dict' in ckpt_data else ckpt_data
    model.load_state_dict(state_dict)
    model.to(device).eval()
    print("✅ Model weights loaded successfully!")

    # 1. Run inference on Val & Test
    val_df, val_samples = evaluate_split(model, "Val")
    test_df, test_samples = evaluate_split(model, "Test")

    # 2. Extract Top 20 Worst Frames for each split
    val_worst20 = val_df.sort_values(by=['macro_dice', 'macro_assd'], ascending=[True, False]).head(20).copy()
    test_worst20 = test_df.sort_values(by=['macro_dice', 'macro_assd'], ascending=[True, False]).head(20).copy()

    val_worst20.to_csv(os.path.join(OUTPUT_DIR, "val_top20_worst.csv"), index=False)
    test_worst20.to_csv(os.path.join(OUTPUT_DIR, "test_top20_worst.csv"), index=False)

    print("\n" + "="*80)
    print("🔴 TOP 20 WORST FRAMES IN VALIDATION SPLIT (120 frames total):")
    print("="*80)
    print(val_worst20[['filename', 'patient', 'macro_dice', 'ridge_dice', 'sil_dice', 'falc_dice', 'macro_assd', 'primary_diagnosis']].to_string(index=False))

    print("\n" + "="*80)
    print("🔴 TOP 20 WORST FRAMES IN TEST SPLIT (109 frames total):")
    print("="*80)
    print(test_worst20[['filename', 'patient', 'macro_dice', 'ridge_dice', 'sil_dice', 'falc_dice', 'macro_assd', 'primary_diagnosis']].to_string(index=False))

    # 3. Render 4-Panel Diagnostic Montages
    print("\n🖼️ Rendering 4-Panel Diagnostic Montages for the Worst Frames...")
    val_diag_dir = os.path.join(OUTPUT_DIR, "val_worst_montages")
    test_diag_dir = os.path.join(OUTPUT_DIR, "test_worst_montages")

    for rank, (_, row) in enumerate(val_worst20.iterrows(), start=1):
        fname = row['filename']
        s = val_samples[fname]
        out_path = os.path.join(val_diag_dir, f"rank{rank:02d}_{fname}")
        render_diagnostic_montage(
            s['pixel_values'], s['gt_mask'], s['pred_map'],
            s['pred_j_coords'], s['gt_j_coords'], s['gt_j_vis'],
            out_path, s['metrics']
        )

    for rank, (_, row) in enumerate(test_worst20.iterrows(), start=1):
        fname = row['filename']
        s = test_samples[fname]
        out_path = os.path.join(test_diag_dir, f"rank{rank:02d}_{fname}")
        render_diagnostic_montage(
            s['pixel_values'], s['gt_mask'], s['pred_map'],
            s['pred_j_coords'], s['gt_j_coords'], s['gt_j_vis'],
            out_path, s['metrics']
        )

    # 4. Package into Zip
    zip_path = "/kaggle/working/worst_cases_exp14.zip" if os.path.exists("/kaggle") else os.path.join(OUTPUT_DIR, "worst_cases_exp14.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(OUTPUT_DIR):
            for file in files:
                if file.endswith('.zip'):
                    continue
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, OUTPUT_DIR)
                zipf.write(file_path, arcname)

    print(f"\n🎉 Finished! Download worst-case diagnostic package from: {zip_path}")

if __name__ == '__main__':
    main()
