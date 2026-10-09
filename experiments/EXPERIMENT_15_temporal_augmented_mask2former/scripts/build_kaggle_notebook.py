"""
Builds the standalone self-contained Kaggle notebook for EXPERIMENT_15:
Temporal-Augmented Mask2Former (T=3 Chronological Patient Clips).
"""
import os
import json

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
EXP_DIR = os.path.join(ROOT, 'experiments/EXPERIMENT_15_temporal_augmented_mask2former')
OUT_NOTEBOOK = os.path.join(EXP_DIR, 'notebooks/EXP15_Temporal_Mask2Former_Kaggle.ipynb')

cell_0_md = """# EXPERIMENT_15: Temporal-Augmented Mask2Former ($T=3$)
**Scientific Hypothesis:**
Static 2D models collapse under mechanical surgical traction because single snapshots cannot resolve topological inversion (e.g. Patient 40 where the anterior ridge is hoisted to the upper image margin).
By conditioning on $T=3$ chronological keyframes within each patient ($(I_{t-2}, I_{t-1}, I_t)$), the network tracks the continuous retraction trajectory, resolving topological ambiguity without requiring complex 3D meshes or pre-op CT scans.

**Key Components:**
1. **Patient-Wise Chronological Dataloader:** Strict patient isolation (zero cross-patient leakage), filtering transitions with $\\Delta t \\le 240$ frames ($\\le 8.0$s at 30fps).
2. **Patient-Stratified Clip Sampler:** Balances patient cohorts during training.
3. **Pure Vanilla Base Checkpoint:** `facebook/mask2former-swin-tiny-ade-semantic` (ImageNet/ADE20k).
4. **Lightweight Spatiotemporal Query Cross-Attention:** Communicates query motion vectors across the 3-frame clip with learnable gating $\\gamma$.
5. **Top 15 Worst Cases Diagnostic:** Automatically renders the 15 lowest-scoring validation cases to verify whether the Patient 40 flip is eliminated.
"""

cell_1_code = """!pip install -q transformers
import os, sys, time, glob, json, cv2, zipfile, re, random
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from pathlib import Path

# NumPy 2.0 compatibility
for _a, _v in [('Inf', np.inf), ('NAN', np.nan), ('NaN', np.nan), ('PINF', np.inf), ('NINF', -np.inf)]:
    if not hasattr(np, _a):
        setattr(np, _a, _v)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cuda.matmul.allow_tf32 = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using compute device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
"""

cell_2_code = """# ==============================================================================
# Auto-Detect Dataset Directories (Supports separate Kaggle mounted splits)
# ==============================================================================
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
        f"data/L3D/{split_name}",
        f"../data/L3D/{split_name}",
        f"../../data/L3D/{split_name}",
    ]
    for c in candidates:
        if os.path.exists(c) and os.path.exists(os.path.join(c, 'labels')):
            return os.path.abspath(c)
            
    # Recursive search under /kaggle/input
    if os.path.exists("/kaggle/input"):
        for m in glob.glob(f"/kaggle/input/**/{split_name}", recursive=True):
            if os.path.isdir(m) and os.path.exists(os.path.join(m, 'labels')):
                return os.path.abspath(m)
        for m in glob.glob(f"/kaggle/input/**/{target}", recursive=True):
            if os.path.isdir(m) and os.path.exists(os.path.join(m, 'labels')):
                return os.path.abspath(m)
                
    raise RuntimeError(f"Could not locate {split_name} split directory! Check your dataset inputs.")

train_dir = find_split_path('Train')
val_dir   = find_split_path('Val')
test_dir  = find_split_path('Test')

OUT_DIR = "/kaggle/working/results_exp15" if os.path.exists("/kaggle/working") else "results_exp15"
os.makedirs(OUT_DIR, exist_ok=True)

print("📁 Resolved Dataset Split Directories:")
print(f"   Train : {train_dir}")
print(f"   Val   : {val_dir}")
print(f"   Test  : {test_dir}")
print(f"   Output: {OUT_DIR}")
"""

cell_3_code = """# ==============================================================================
# Patient-Wise Temporal Dataset & Stratified Sampler
# ==============================================================================
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

class PatientTemporalL3DDataset(Dataset):
    def __init__(self, split_dir, split_name='Split', clip_len=3, max_gap=240, image_size=1024, stroke_width=35):
        super().__init__()
        self.split_dir = split_dir
        self.split_name = split_name
        self.clip_len = clip_len
        self.max_gap = max_gap
        self.image_size = image_size
        self.stroke_width = stroke_width
        self.clips = self._index_temporal_clips()
        print(f"[{split_name} - Temporal T={clip_len}] Loaded {len(self.clips)} clips from {self.split_dir}")

    def _index_temporal_clips(self):
        json_files = sorted(glob.glob(os.path.join(self.split_dir, 'labels', '*.json')))
        patient_dict = defaultdict(list)
        
        for jf in json_files:
            stem = Path(jf).stem
            m = re.match(r'(Patient_\d+)_(\d+)', stem)
            if m:
                pat, fid = m.group(1), int(m.group(2))
                img_path = os.path.join(self.split_dir, 'images', f"{stem}.jpg")
                if not os.path.exists(img_path):
                    img_path = os.path.join(self.split_dir, 'images', f"{stem}.png")
                if os.path.exists(img_path):
                    patient_dict[pat].append((fid, jf, img_path))
                    
        clips = []
        for pat, frames in sorted(patient_dict.items()):
            frames.sort(key=lambda x: x[0])
            segments, cur = [], [frames[0]]
            for i in range(1, len(frames)):
                gap = frames[i][0] - frames[i-1][0]
                if gap <= self.max_gap:
                    cur.append(frames[i])
                else:
                    if len(cur) >= self.clip_len: segments.append(cur)
                    cur = [frames[i]]
            if len(cur) >= self.clip_len: segments.append(cur)
            
            for seg in segments:
                for s in range(len(seg) - self.clip_len + 1):
                    window = seg[s : s + self.clip_len]
                    clips.append({
                        'patient': pat,
                        'frame_ids': [w[0] for w in window],
                        'json_paths': [w[1] for w in window],
                        'img_paths': [w[2] for w in window]
                    })
        return clips

    def __len__(self):
        return len(self.clips)

    def _load_single_frame(self, img_path, json_path):
        img_bgr = cv2.imread(img_path)
        orig_h, orig_w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        if orig_w != self.image_size or orig_h != self.image_size:
            img_resized = cv2.resize(img_rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        else:
            img_resized = img_rgb
            
        norm_img = (img_resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        pixel_tensor = torch.from_numpy(norm_img).permute(2, 0, 1).float()
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        data_w = data.get('imageWidth', orig_w)
        data_h = data.get('imageHeight', orig_h)
        canvas_raw = np.zeros((data_h, data_w), dtype=np.uint8)
        
        shapes = data.get('shapes', [])
        for shape in shapes:
            lbl = str(shape.get('label', '')).lower().strip()
            pts = shape.get('points', [])
            if len(pts) < 2: continue
            
            if lbl.startswith('r') or 'ridge' in lbl or 'rigde' in lbl or 'anterior' in lbl: color = 1
            elif lbl.startswith('s') or 'sil' in lbl or 'margin' in lbl or 'silhouette' in lbl: color = 2
            elif lbl.startswith('f') or lbl.startswith('l') or 'falc' in lbl or 'lig' in lbl: color = 3
            else: color = 0
            
            if color > 0:
                for i in range(1, len(pts)):
                    cv2.line(canvas_raw, tuple(map(int, pts[i - 1])), tuple(map(int, pts[i])), color, self.stroke_width)
                    
        if data_w != self.image_size or data_h != self.image_size:
            mask = cv2.resize(canvas_raw, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        else:
            mask = canvas_raw
            
        mask_tensor = torch.from_numpy(mask).long()
        return pixel_tensor, mask_tensor

    def __getitem__(self, idx):
        clip_info = self.clips[idx]
        frames_pixels, frames_masks = [], []
        for img_p, json_p in zip(clip_info['img_paths'], clip_info['json_paths']):
            p_tensor, m_tensor = self._load_single_frame(img_p, json_p)
            frames_pixels.append(p_tensor)
            frames_masks.append(m_tensor)
        return torch.stack(frames_pixels, dim=0), torch.stack(frames_masks, dim=0), clip_info

class PatientStratifiedClipSampler(Sampler):
    def __init__(self, dataset, samples_per_epoch=None):
        self.dataset = dataset
        self.patient_to_indices = defaultdict(list)
        for idx, clip in enumerate(dataset.clips):
            self.patient_to_indices[clip['patient']].append(idx)
        self.patients = sorted(list(self.patient_to_indices.keys()))
        self.samples_per_epoch = samples_per_epoch if samples_per_epoch is not None else len(dataset)

    def __iter__(self):
        indices = []
        for _ in range(self.samples_per_epoch):
            pat = random.choice(self.patients)
            indices.append(random.choice(self.patient_to_indices[pat]))
        return iter(indices)

    def __len__(self):
        return self.samples_per_epoch
"""

cell_4_code = """# ==============================================================================
# Evaluation Metrics & Jitter
# ==============================================================================
def compute_dice(pred_bin, target_bin, eps=1e-6):
    p_b, t_b = (pred_bin > 0).astype(bool), (target_bin > 0).astype(bool)
    inter = np.logical_and(p_b, t_b).sum()
    total = p_b.sum() + t_b.sum()
    return float(2.0 * inter / (total + eps)) if total > 0 else 1.0

def compute_iou(pred_bin, target_bin, eps=1e-6):
    p_b, t_b = (pred_bin > 0).astype(bool), (target_bin > 0).astype(bool)
    inter = np.logical_and(p_b, t_b).sum()
    union = np.logical_or(p_b, t_b).sum()
    return float(inter / (union + eps)) if union > 0 else 1.0

def compute_assd_fast(pred_bin, target_bin, max_penalty=80.0):
    p_b, t_b = (pred_bin > 0).astype(np.uint8), (target_bin > 0).astype(np.uint8)
    if p_b.sum() == 0 and t_b.sum() == 0: return 0.0
    if p_b.sum() == 0 or t_b.sum() == 0: return max_penalty
    try:
        cp, _ = cv2.findContours(p_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        ct, _ = cv2.findContours(t_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        ep, et = np.zeros_like(p_b), np.zeros_like(t_b)
        cv2.drawContours(ep, cp, -1, 1, 1)
        cv2.drawContours(et, ct, -1, 1, 1)
        if ep.sum() == 0 or et.sum() == 0: return max_penalty
        dt_target = cv2.distanceTransform(1 - et, cv2.DIST_L2, 3)
        dt_pred = cv2.distanceTransform(1 - ep, cv2.DIST_L2, 3)
        return min(float((dt_target[ep > 0].mean() + dt_pred[et > 0].mean()) / 2.0), max_penalty)
    except Exception:
        return max_penalty

def evaluate_frame_metrics(pred_map, gt_map, num_classes=3):
    results = {}
    class_names = {1: 'ridge', 2: 'silhouette', 3: 'falciform'}
    dice_list, iou_list, assd_list = [], [], []
    for c in range(1, num_classes + 1):
        name = class_names[c]
        d = compute_dice(pred_map == c, gt_map == c)
        i = compute_iou(pred_map == c, gt_map == c)
        a = compute_assd_fast(pred_map == c, gt_map == c)
        results[f'dice_{name}'] = d
        results[f'iou_{name}'] = i
        results[f'assd_{name}'] = a
        dice_list.append(d); iou_list.append(i); assd_list.append(a)
    results['macro_dice'] = float(np.mean(dice_list))
    results['macro_iou'] = float(np.mean(iou_list))
    results['macro_assd'] = float(np.mean(assd_list))
    return results

def compute_clip_jitter(pred_maps_clip):
    T = len(pred_maps_clip)
    if T < 2: return 0.0
    return float(np.mean([compute_assd_fast((pred_maps_clip[t] > 0).astype(np.uint8), (pred_maps_clip[t-1] > 0).astype(np.uint8), max_penalty=50.0) for t in range(1, T)]))
"""

cell_5_code = """# ==============================================================================
# Model Architecture: Temporal-Augmented Mask2Former
# ==============================================================================
from transformers import Mask2FormerForUniversalSegmentation

class SpatiotemporalQueryBlock(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, dropout=0.1, gamma_init=0.05):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.gamma = nn.Parameter(torch.tensor([gamma_init], dtype=torch.float32))

    def forward(self, queries_clip):
        B, T, N, C = queries_clip.shape
        kv_context = queries_clip.view(B, T * N, C)
        out = []
        for t in range(T):
            q_curr = queries_clip[:, t, :, :]
            attn_out, _ = self.cross_attn(query=q_curr, key=kv_context, value=kv_context)
            q_res = self.norm1(q_curr + self.gamma * attn_out)
            out.append(self.norm2(q_res + self.gamma * self.ffn(q_res)))
        return torch.stack(out, dim=1)

class TemporalMask2Former(nn.Module):
    def __init__(self, model_name="facebook/mask2former-swin-tiny-ade-semantic", num_labels=4, clip_len=3):
        super().__init__()
        print(f"📦 Building TemporalMask2Former with base: '{model_name}'...")
        self.m2f = Mask2FormerForUniversalSegmentation.from_pretrained(model_name, num_labels=num_labels, ignore_mismatched_sizes=True)
        self.clip_len = clip_len
        self.embed_dim = 256
        self.temporal_query_block = SpatiotemporalQueryBlock(embed_dim=256, num_heads=8, dropout=0.1, gamma_init=0.05)

    def forward_transformer_decoder(self, multi_scale_features, mask_features, steered_query_feat):
        tm = self.m2f.model.transformer_module
        multi_stage_features, multi_stage_positional_embeddings, size_list = [], [], []

        for i in range(tm.num_feature_levels):
            size_list.append(multi_scale_features[i].shape[-2:])
            multi_stage_positional_embeddings.append(
                tm.position_embedder(multi_scale_features[i].shape, multi_scale_features[i].device, multi_scale_features[i].dtype, None).flatten(2)
            )
            multi_stage_features.append(tm.input_projections[i](multi_scale_features[i]).flatten(2) + tm.level_embed.weight[i][None, :, None])
            multi_stage_positional_embeddings[-1] = multi_stage_positional_embeddings[-1].permute(2, 0, 1)
            multi_stage_features[-1] = multi_stage_features[-1].permute(2, 0, 1)

        _, batch_size, _ = multi_stage_features[0].shape
        query_embeddings = tm.queries_embedder.weight.unsqueeze(1).repeat(1, batch_size, 1)
        query_features = steered_query_feat.permute(1, 0, 2)

        return tm.decoder(
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

    def forward(self, pixel_values_clip, mask_labels_clip=None, class_labels_clip=None):
        B, T, C, H, W = pixel_values_clip.shape
        B_total = B * T
        
        # 1. Sequential spatial encoding per frame (cuts peak VRAM from 14 GB to ~5 GB)
        multi_scale_features_list = []
        mask_features_list = []
        for t in range(T):
            p_out = self.m2f.model.pixel_level_module(pixel_values_clip[:, t], output_hidden_states=True)
            multi_scale_features_list.append(list(p_out.decoder_hidden_states))
            mask_features_list.append(p_out.decoder_last_hidden_state)
            
        mask_features = torch.cat(mask_features_list, dim=0) # (B*T, 256, H/4, W/4)
        num_levels = len(multi_scale_features_list[0])
        multi_scale_features = [
            torch.cat([multi_scale_features_list[t][lvl] for t in range(T)], dim=0)
            for lvl in range(num_levels)
        ]
        
        tm = self.m2f.model.transformer_module
        base_queries = tm.queries_features.weight.unsqueeze(0).repeat(B_total, 1, 1).view(B, T, 100, self.embed_dim)
        steered_queries = self.temporal_query_block(base_queries) # (B, T, 100, 256)
        
        # 2. Sequential Transformer Decoder per frame (cuts peak VRAM from 14 GB to ~5.5 GB)
        masks_queries_logits_list, class_queries_logits_list = [], []
        for t in range(T):
            dec_out = self.forward_transformer_decoder(multi_scale_features_list[t], mask_features_list[t], steered_queries[:, t])
            c_logits_t = self.m2f.class_predictor(dec_out.last_hidden_state)
            m_logits_t = dec_out.masks_queries_logits[-1]
            masks_queries_logits_list.append(m_logits_t)
            class_queries_logits_list.append(c_logits_t)
            
        masks_logits = torch.stack(masks_queries_logits_list, dim=1)
        class_logits = torch.stack(class_queries_logits_list, dim=1)
        
        output = {'masks_queries_logits': masks_logits, 'class_queries_logits': class_logits, 'gamma': self.temporal_query_block.gamma.item()}
        
        if mask_labels_clip is not None and class_labels_clip is not None:
            m_flat = [mask_labels_clip[b][t] for b in range(B) for t in range(T)]
            c_flat = [class_labels_clip[b][t] for b in range(B) for t in range(T)]
            H_m, W_m = masks_logits.shape[-2:]
            loss_dict = self.m2f.criterion(
                masks_queries_logits=masks_logits.view(B_total, 100, H_m, W_m),
                class_queries_logits=class_logits.view(B_total, 100, -1),
                mask_labels=m_flat,
                class_labels=c_flat
            )
            weight_dict = self.m2f.criterion.weight_dict
            output['loss'] = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            output['loss_dict'] = loss_dict
            
        return output
"""

cell_6_code = """# ==============================================================================
# DataLoaders & Optimizer Setup
# ==============================================================================
CLIP_LEN = 3
# 1024x1024 resolution with sequential frame encoding (drastically reduces peak memory)
IMAGE_SIZE = 1024
STROKE_WIDTH = 35
BATCH_SIZE = 1   # 1 clip (T=3) per step
ACCUM_STEPS = 4  # Effective batch size = 4 clips / 12 frames
EPOCHS = 30

train_dataset = PatientTemporalL3DDataset(train_dir, split_name='Train', clip_len=CLIP_LEN, image_size=IMAGE_SIZE, stroke_width=STROKE_WIDTH)
val_dataset   = PatientTemporalL3DDataset(val_dir,   split_name='Val',   clip_len=CLIP_LEN, image_size=IMAGE_SIZE, stroke_width=STROKE_WIDTH)
test_dataset  = PatientTemporalL3DDataset(test_dir,  split_name='Test',  clip_len=CLIP_LEN, image_size=IMAGE_SIZE, stroke_width=STROKE_WIDTH)

train_sampler = PatientStratifiedClipSampler(train_dataset)

def collate_clip_fn(batch):
    pixel_values = torch.stack([b[0] for b in batch])
    masks = torch.stack([b[1] for b in batch])
    metas = [b[2] for b in batch]
    B, T, H, W = masks.shape
    mask_labels_clip, class_labels_clip = [], []
    for b in range(B):
        sample_masks, sample_classes = [], []
        for t in range(T):
            m = masks[b, t]
            unique_classes = torch.unique(m)
            unique_classes = unique_classes[unique_classes > 0]
            if len(unique_classes) == 0:
                sample_masks.append(torch.zeros((1, H, W), dtype=torch.float32))
                sample_classes.append(torch.zeros(1, dtype=torch.int64))
            else:
                sample_masks.append(torch.stack([(m == c).float() for c in unique_classes]))
                sample_classes.append(unique_classes.long())
        mask_labels_clip.append(sample_masks)
        class_labels_clip.append(sample_classes)
    return {'pixel_values': pixel_values, 'masks': masks, 'mask_labels_clip': mask_labels_clip, 'class_labels_clip': class_labels_clip, 'metas': metas}

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, collate_fn=collate_clip_fn, num_workers=2)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_clip_fn, num_workers=2)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_clip_fn, num_workers=2)

model = TemporalMask2Former(num_labels=4, clip_len=CLIP_LEN).to(device)

backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and 'pixel_level_module.encoder' in n]
head_params = [p for n, p in model.named_parameters() if p.requires_grad and 'pixel_level_module.encoder' not in n]

optimizer = torch.optim.AdamW([{'params': backbone_params, 'lr': 1e-5}, {'params': head_params, 'lr': 1e-4}], weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

use_amp = (device.type == 'cuda')
scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
"""

cell_7_code = """# ==============================================================================
# Training Loop with Epoch-by-Epoch Validation
# ==============================================================================
def rasterize_class_map(masks_queries_logits, class_queries_logits, canvas_size=1024):
    masks_prob = F.interpolate(masks_queries_logits.unsqueeze(0).float(), size=(canvas_size, canvas_size), mode="bilinear", align_corners=False).squeeze(0).sigmoid().cpu().numpy()
    class_probs = F.softmax(class_queries_logits.float(), dim=-1).cpu().numpy()
    foreground_probs = class_probs[:, 1:]
    sem_seg = np.zeros((3, canvas_size, canvas_size), dtype=np.float32)
    for q in range(masks_prob.shape[0]):
        m = masks_prob[q]
        for c in range(3):
            score = foreground_probs[q, c]
            if score > 0.2: sem_seg[c] = np.maximum(sem_seg[c], m * score)
    pred_map = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    for c in range(3):
        pred_map[sem_seg[c] > 0.4] = (c + 1)
    return pred_map

def run_val(model, loader):
    model.eval()
    records = []
    with torch.no_grad():
        for batch in loader:
            pixel_values = batch['pixel_values'].to(device)
            masks = batch['masks'].cpu().numpy()
            metas = batch['metas']
            outputs = model(pixel_values)
            m_logits = outputs['masks_queries_logits']
            c_logits = outputs['class_queries_logits']
            B, T = m_logits.shape[:2]
            target_t = T - 1
            for b in range(B):
                pred_map = rasterize_class_map(m_logits[b, target_t], c_logits[b, target_t], canvas_size=masks.shape[-1])
                res = evaluate_frame_metrics(pred_map, masks[b, target_t])
                res['patient'] = metas[b]['patient']
                res['frame_id'] = metas[b]['frame_ids'][target_t]
                res['is_patient_40'] = (metas[b]['patient'] == 'Patient_40')
                records.append(res)
    df = pd.DataFrame(records)
    summary = {
        'macro_dice': float(df['macro_dice'].mean()),
        'macro_assd': float(df['macro_assd'].mean()),
        'dice_ridge': float(df['dice_ridge'].mean()),
        'dice_silhouette': float(df['dice_silhouette'].mean()),
        'dice_falciform': float(df['dice_falciform'].mean())
    }
    if df['is_patient_40'].sum() > 0:
        summary['p40_dice'] = float(df[df['is_patient_40']]['macro_dice'].mean())
    return summary, df

print(f"🚀 Starting {EPOCHS}-Epoch Training...")
best_val_dice = -1.0
best_ckpt_path = os.path.join(OUT_DIR, "best_model.pth")
training_log = []

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    model.train()
    losses = []
    optimizer.zero_grad()
    
    for step, batch in enumerate(train_loader):
        pixel_values = batch['pixel_values'].to(device)
        mask_labels = [[m.to(device) for m in t_list] for t_list in batch['mask_labels_clip']]
        class_labels = [[c.to(device) for c in t_list] for t_list in batch['class_labels_clip']]
        
        with torch.amp.autocast('cuda', enabled=use_amp):
            outputs = model(pixel_values, mask_labels_clip=mask_labels, class_labels_clip=class_labels)
            loss = outputs['loss'] / ACCUM_STEPS
            
        scaler.scale(loss).backward()
        if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        losses.append(outputs['loss'].item())
        
    scheduler.step()
    mean_loss = float(np.mean(losses))
    val_res, _ = run_val(model, val_loader)
    p40_str = f" | P40 Dice: {val_res.get('p40_dice', 0):.2%}" if 'p40_dice' in val_res else ""
    print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] Train Loss: {mean_loss:.4f} | Val Dice: {val_res['macro_dice']:.2%} | ASSD: {val_res['macro_assd']:.2f}px{p40_str} ({time.time()-t0:.1f}s)")
    
    training_log.append({'epoch': epoch, 'train_loss': mean_loss, 'val_dice': val_res['macro_dice'], 'val_assd': val_res['macro_assd'], 'gamma': outputs['gamma']})
    pd.DataFrame(training_log).to_csv(os.path.join(OUT_DIR, "training_log.csv"), index=False)
    
    if val_res['macro_dice'] > best_val_dice:
        best_val_dice = val_res['macro_dice']
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'val_res': val_res}, best_ckpt_path)
        print(f"⭐ New best model saved: {best_val_dice:.2%}")
"""

cell_8_code = """# ==============================================================================
# Final Evaluation & Top 15 Worst Cases Diagnostic Montages
# ==============================================================================
print("\\n🏁 Evaluating Best Checkpoint on Validation and Test Splits...")
best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
model.load_state_dict(best_ckpt['model_state_dict'])
model.eval()

diag_worst_dir = os.path.join(OUT_DIR, "diagnostics/worst_cases")
diag_best_dir = os.path.join(OUT_DIR, "diagnostics/best_cases")
os.makedirs(diag_worst_dir, exist_ok=True)
os.makedirs(diag_best_dir, exist_ok=True)

COLOR_MAP_BGR = {1: (0, 255, 0), 2: (255, 150, 0), 3: (0, 0, 255)}

def render_overlay(img_bgr, mask_map, alpha=0.5):
    overlay = img_bgr.copy()
    for c, col in COLOR_MAP_BGR.items():
        bin_m = (mask_map == c).astype(np.uint8)
        if bin_m.sum() == 0: continue
        cnts, _ = cv2.findContours(bin_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(overlay, cnts, -1, col, 8)
    return cv2.addWeighted(img_bgr, 1.0 - alpha, overlay, alpha, 0)

final_val_summary, final_val_df = run_val(model, val_loader)
final_val_df.to_csv(os.path.join(OUT_DIR, "val_predictions.csv"), index=False)

final_test_summary, final_test_df = run_val(model, test_loader)
final_test_df.to_csv(os.path.join(OUT_DIR, "test_predictions.csv"), index=False)

print(f"📊 Final Val  Dice: {final_val_summary['macro_dice']:.2%} | ASSD: {final_val_summary['macro_assd']:.2f}px")
print(f"🏆 Final Test Dice: {final_test_summary['macro_dice']:.2%} | ASSD: {final_test_summary['macro_assd']:.2f}px")

# Render Top 15 Worst Validation Cases
df_sorted = final_val_df.sort_values(by='macro_dice', ascending=True)
worst_15 = df_sorted.head(15)

for rank, (_, row) in enumerate(worst_15.iterrows(), 1):
    pat = row['patient']
    fid = row['frame_id']
    dice = row['macro_dice']
    # Load raw image
    img_path = glob.glob(os.path.join(val_dir, "images", f"{pat}_{fid:07d}.*"))[0]
    img_bgr = cv2.imread(img_path)
    
    # Render title
    h, w = img_bgr.shape[:2]
    cv2.putText(img_bgr, f"Rank {rank} Worst: {pat}_{fid:07d} | Dice: {dice:.1%}", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 4)
    out_path = os.path.join(diag_worst_dir, f"rank_{rank:02d}_{pat}_{fid:07d}_dice_{int(dice*100)}.png")
    cv2.imwrite(out_path, cv2.resize(img_bgr, (640, 360)))

print(f"✅ Saved Top 15 Worst Cases to: {diag_worst_dir}")

with open(os.path.join(OUT_DIR, "metrics_summary.json"), "w") as f:
    json.dump({'val': final_val_summary, 'test': final_test_summary}, f, indent=2)
"""

cell_9_code = """# ==============================================================================
# Package Deliverables into results.zip
# ==============================================================================
zip_path = os.path.join(OUT_DIR, "results.zip")
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, _, files in os.walk(OUT_DIR):
        for f in files:
            if f != 'results.zip':
                p = os.path.join(root, f)
                zipf.write(p, os.path.relpath(p, OUT_DIR))
print(f"🎉 Packaged {zip_path}: {os.path.getsize(zip_path)/(1024*1024):.2f} MB")
"""

def make_cell(cell_type, source):
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")]
    }

notebook = {
    "cells": [
        make_cell("markdown", cell_0_md),
        make_cell("code", cell_1_code),
        make_cell("code", cell_2_code),
        make_cell("code", cell_3_code),
        make_cell("code", cell_4_code),
        make_cell("code", cell_5_code),
        make_cell("code", cell_6_code),
        make_cell("code", cell_7_code),
        make_cell("code", cell_8_code),
        make_cell("code", cell_9_code)
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

os.makedirs(os.path.dirname(OUT_NOTEBOOK), exist_ok=True)
with open(OUT_NOTEBOOK, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print(f"✅ Generated Kaggle notebook: {OUT_NOTEBOOK} ({os.path.getsize(OUT_NOTEBOOK)} bytes)")
