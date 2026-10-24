"""
Builds the standalone self-contained Kaggle notebook for EXPERIMENT_14.
Directly reads verified modules from experiments/EXPERIMENT_14 to eliminate syntax errors.
"""
import os
import json

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
EXP_DIR = os.path.join(ROOT, 'experiments/EXPERIMENT_14')
OUT_NOTEBOOK = os.path.join(EXP_DIR, 'notebooks/EXP14_Stratified_JunctionSteered_Mask2Former_Kaggle.ipynb')

cell_0_md = """# EXPERIMENT_14: Stratified Junction-Steered Mask2Former (Option 1: Patient 40 <-> Patient 38 Swap)
**Scientific Objective:**
Balance anatomical pose distribution between Train and Val by swapping Patient 40 (101 frames, 15 flipped) into Train, and Patient 38 (99 frames, 6 flipped) into Val.
Evaluate whether this enables the model to learn extreme grasper elevation mechanics during training, and test generalization on the frozen Test set (Patient 41).
"""

cell_1_code = """!pip install -q transformers
import os, sys, time, glob, json, cv2, zipfile
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
        f"/data/khoalq/data/L3D/{split_name}",
        f"data/L3D/{split_name}",
        f"../data/L3D/{split_name}",
        f"../../data/L3D/{split_name}",
    ]
    for c in candidates:
        if os.path.exists(c) and os.path.exists(os.path.join(c, "labels")):
            return os.path.abspath(c)
            
    # Recursive search under /kaggle/input
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

TRAIN_DIR = find_split_path("Train")
VAL_DIR = find_split_path("Val")
TEST_DIR = find_split_path("Test")
if os.path.exists("/kaggle"):
    OUT_DIR = "/kaggle/working/results_exp14"
else:
    OUT_DIR = os.path.abspath("experiments/EXPERIMENT_14/results")
os.makedirs(OUT_DIR, exist_ok=True)

print("📁 Resolved Dataset Split Directories:")
print(f"   Train : {TRAIN_DIR}")
print(f"   Val   : {VAL_DIR}")
print(f"   Test  : {TEST_DIR}")
print(f"   Output: {OUT_DIR}")
"""

def clean_for_notebook(code_text):
    clean_lines = []
    for line in code_text.splitlines():
        if '__file__' in line:
            continue
        if '_WORKSPACE_ROOT' in line:
            continue
        if 'sys.path.insert' in line:
            continue
        if 'from experiments.EXPERIMENT_14' in line:
            continue
        clean_lines.append(line)
    return '\n'.join(clean_lines)

# Read utility files directly
with open(os.path.join(EXP_DIR, 'utils/junction_extractor.py'), 'r', encoding='utf-8') as f:
    cell_3_code = "# ==============================================================================\n# 1. Deterministic Anatomical Junction Extractor\n# ==============================================================================\n" + clean_for_notebook(f.read())

with open(os.path.join(EXP_DIR, 'utils/metrics.py'), 'r', encoding='utf-8') as f:
    cell_4_code = "# ==============================================================================\n# 2. Evaluation Metrics (NumPy 2.0 & Pure OpenCV)\n# ==============================================================================\n" + clean_for_notebook(f.read())

with open(os.path.join(EXP_DIR, 'utils/dataset.py'), 'r', encoding='utf-8') as f:
    cell_5_code = "# ==============================================================================\n# 3. Stratified L3D Dataset Reader (Option 1 Re-stratification)\n# ==============================================================================\n" + clean_for_notebook(f.read())

with open(os.path.join(EXP_DIR, 'models/junction_head.py'), 'r', encoding='utf-8') as f:
    cell_6_code = "# ==============================================================================\n# 4. Junction Anchor Head Architecture\n# ==============================================================================\n" + clean_for_notebook(f.read())

with open(os.path.join(EXP_DIR, 'models/junction_steered_mask2former.py'), 'r', encoding='utf-8') as f:
    cell_7_code = "# ==============================================================================\n# 5. Junction-Steered Mask2Former Architecture\n# ==============================================================================\n" + clean_for_notebook(f.read())

with open(os.path.join(EXP_DIR, 'models/losses.py'), 'r', encoding='utf-8') as f:
    cell_8_code = "# ==============================================================================\n# 6. Multi-Task Loss Function\n# ==============================================================================\n" + clean_for_notebook(f.read())

cell_9_code = """# ==============================================================================
# 7. Post-Processing & Evaluation Metrics
# ==============================================================================
COLOR_MAP_RGB = {
    0: (0, 0, 0),       # BG
    1: (34, 197, 94),   # Ridge: Green
    2: (239, 68, 68),   # Silhouette: Red
    3: (59, 130, 246)   # Falciform: Blue
}

JUNCTION_COLORS_BGR = {
    0: (0, 215, 255), # Top: Gold
    1: (255, 255, 0), # Bottom: Cyan
    2: (255, 0, 255), # Lat_Right: Magenta
    3: (0, 140, 255)  # Lat_Left: Orange
}

def rasterize_class_map(masks_queries_logits, class_queries_logits, canvas_size=1024):
    \"\"\"
    Standard Mask2Former post-processing: Argmax over class probabilities * sigmoid mask probabilities.
    Matches EXPERIMENT_5 implementation.
    \"\"\"
    masks_queries_logits = masks_queries_logits.float()
    class_queries_logits = class_queries_logits.float()
    
    # 1. Resize mask logits to canvas_size
    masks = torch.nn.functional.interpolate(
        masks_queries_logits.unsqueeze(0),
        size=(canvas_size, canvas_size),
        mode='bilinear',
        align_corners=False
    ).squeeze(0).sigmoid() # (100, H, W)
    
    # 2. Query classification probabilities
    cls_probs = torch.softmax(class_queries_logits, dim=-1) # (100, 5), last is BG
    
    # Exclude background class (index 4)
    fg_cls_probs = cls_probs[:, :4] # (100, 4)
    
    # Multiply: sem_probs = sum_q (cls_prob * mask_prob)
    sem_probs = torch.einsum('qc,qhw->chw', fg_cls_probs, masks) # (4, H, W)
    
    # Thresholding & argmax in NumPy
    pred_map = sem_probs.argmax(dim=0).cpu().numpy().astype(np.int64) # (H, W)
    max_prob = sem_probs.max(dim=0)[0].cpu().numpy()
    pred_map[max_prob < 0.25] = 0
    
    return pred_map

def render_frame_diagnostic(orig_rgb_norm, gt_mask, pred_map, pred_j_coords, gt_j_coords, gt_j_vis, out_path, metrics):
    \"\"\"
    Renders standard 4-panel diagnostic montage (Input, GT, Pred, Error Map).
    \"\"\"
    img = (orig_rgb_norm.transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    H, W = bgr.shape[:2]
    
    # Panel 2: GT Overlay
    p2 = bgr.copy()
    for c_id in [1, 2, 3]:
        mask_c = (gt_mask == c_id)
        if mask_c.any():
            col = COLOR_MAP_RGB[c_id][::-1]
            p2[mask_c] = (p2[mask_c] * 0.4 + np.array(col) * 0.6).astype(np.uint8)
    for k in range(4):
        if gt_j_vis[k] > 0.5:
            pt = (gt_j_coords[k] * float(W)).astype(int)
            cv2.circle(p2, tuple(pt), 14, (0, 0, 0), -1)
            cv2.circle(p2, tuple(pt), 10, JUNCTION_COLORS_BGR[k], -1)
            cv2.circle(p2, tuple(pt), 3, (255, 255, 255), -1)
            
    # Panel 3: Pred Overlay
    p3 = bgr.copy()
    for c_id in [1, 2, 3]:
        mask_c = (pred_map == c_id)
        if mask_c.any():
            col = COLOR_MAP_RGB[c_id][::-1]
            p3[mask_c] = (p3[mask_c] * 0.4 + np.array(col) * 0.6).astype(np.uint8)
    for k in range(4):
        pt = (pred_j_coords[k] * float(W)).astype(int)
        cv2.circle(p3, tuple(pt), 14, (0, 0, 0), -1)
        cv2.circle(p3, tuple(pt), 10, JUNCTION_COLORS_BGR[k], -1)
        cv2.circle(p3, tuple(pt), 3, (0, 0, 0), -1)
        
    # Panel 4: Error Map (TP=Green, FP=Cyan, FN=Red)
    p4 = bgr.copy()
    tp = (pred_map > 0) & (gt_mask > 0)
    fp = (pred_map > 0) & (gt_mask == 0)
    fn = (pred_map == 0) & (gt_mask > 0)
    
    p4[tp] = (p4[tp] * 0.3 + np.array([0, 255, 0]) * 0.7).astype(np.uint8)
    p4[fp] = (p4[fp] * 0.3 + np.array([255, 255, 0]) * 0.7).astype(np.uint8)
    p4[fn] = (p4[fn] * 0.3 + np.array([0, 0, 255]) * 0.7).astype(np.uint8)
    
    top_row = np.hstack([bgr, p2])
    bot_row = np.hstack([p3, p4])
    montage = np.vstack([top_row, bot_row])
    
    dice_val = metrics.get('macro_dice', 0.0)
    assd_val = metrics.get('macro_assd', 0.0)
    cv2.putText(montage, '1. Input Laparoscopic Frame', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(montage, '2. Ground Truth + GT Junctions', (W + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(montage, f'3. Prediction (Dice: {dice_val:.1%}, ASSD: {assd_val:.1f}px)', (20, H + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 200), 2)
    cv2.putText(montage, '4. Error Map: Green=TP, Cyan=FP, Red=FN', (W + 20, H + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, montage, [cv2.IMWRITE_JPEG_QUALITY, 88])

def run_evaluation(model, dataloader, device, split_name='Val', out_dir=None, save_diagnostics=False):
    model.eval()
    records = []
    latencies = []
    
    diag_dir = os.path.join(out_dir, f'diagnostics_{split_name.lower()}') if (out_dir and save_diagnostics) else None
    print(f'\\n🔍 Running {split_name} Evaluation ({len(dataloader.dataset)} frames)...')
    
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            pixel_values = batch['pixel_values'].to(device)
            masks = batch['mask'].cpu().numpy()
            gt_j_coords = batch['junction_coords'].cpu().numpy()
            gt_j_vis = batch['junction_vis'].cpu().numpy()
            filenames = batch['filename']
            
            t0 = time.time()
            outputs = model(pixel_values)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            latencies.append((time.time() - t0) * 1000.0 / float(pixel_values.shape[0]))
            
            pred_masks_logits = outputs['masks_queries_logits']
            pred_cls_logits = outputs['class_queries_logits']
            pred_j_coords_t = outputs['pred_junction_coords'].float().cpu().numpy()
            
            for b in range(pixel_values.shape[0]):
                pred_map = rasterize_class_map(pred_masks_logits[b], pred_cls_logits[b])
                
                m = evaluate_frame_metrics(
                    pred_map=pred_map,
                    target_map=masks[b],
                    pred_j_coords=pred_j_coords_t[b],
                    gt_j_coords=gt_j_coords[b],
                    gt_j_vis=gt_j_vis[b]
                )
                m['filename'] = filenames[b]
                parts = filenames[b].split('_')
                pat_id = f'{parts[0]}_{parts[1]}' if len(parts) >= 2 else 'Unknown'
                m['patient'] = pat_id
                records.append(m)
                
                if diag_dir:
                    stem = Path(filenames[b]).stem
                    out_diag_path = os.path.join(diag_dir, f'{stem}_diag.jpg')
                    render_frame_diagnostic(
                        orig_rgb_norm=pixel_values[b].float().cpu().numpy(),
                        gt_mask=masks[b],
                        pred_map=pred_map,
                        pred_j_coords=pred_j_coords_t[b],
                        gt_j_coords=gt_j_coords[b],
                        gt_j_vis=gt_j_vis[b],
                        out_path=out_diag_path,
                        metrics=m
                    )
                    
            if (idx + 1) % 20 == 0 or (idx + 1) == len(dataloader):
                cur_dice = np.mean([r['macro_dice'] for r in records])
                cur_assd = np.mean([r['macro_assd'] for r in records])
                print(f'   [{split_name} {idx+1}/{len(dataloader)}] Macro Dice: {cur_dice:.2%} | Macro ASSD: {cur_assd:.2f}px')

    df = pd.DataFrame(records)
    summary = {
        'split': split_name,
        'total_frames': len(df),
        'macro_dice': float(df['macro_dice'].mean()),
        'macro_iou': float(df['macro_iou'].mean()),
        'macro_assd': float(df['macro_assd'].mean()),
        'ridge_dice': float(df['ridge_dice'].mean()),
        'sil_dice': float(df['sil_dice'].mean()),
        'falc_dice': float(df['falc_dice'].mean()),
        'fg_dice': float(df['fg_dice'].mean()),
        'mean_latency_ms': float(np.mean(latencies)),
        'fps': float(1000.0 / np.mean(latencies)),
        'gpu_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU/MPS'
    }
    
    pat_stats = {}
    for pat, pat_group in df.groupby('patient'):
        pat_stats[pat] = {
            'frames': len(pat_group),
            'macro_dice': float(pat_group['macro_dice'].mean()),
            'macro_iou': float(pat_group['macro_iou'].mean()),
            'macro_assd': float(pat_group['macro_assd'].mean()),
            'ridge_dice': float(pat_group['ridge_dice'].mean()),
            'sil_dice': float(pat_group['sil_dice'].mean()),
            'falc_dice': float(pat_group['falc_dice'].mean())
        }
    summary['per_patient_metrics'] = pat_stats
    
    valid_j = df['mean_j_err_px'].dropna()
    if len(valid_j) > 0:
        summary['mean_junction_err_px'] = float(valid_j.mean())
        
    return summary, df
"""

cell_10_code = """# ==============================================================================
# 8. Main Training Execution
# ==============================================================================
def collate_fn_l3d(batch):
    pixel_values = torch.stack([b['pixel_values'] for b in batch])
    masks = torch.stack([b['mask'] for b in batch])
    junction_coords = torch.stack([b['junction_coords'] for b in batch])
    junction_vis = torch.stack([b['junction_vis'] for b in batch])
    filenames = [b['filename'] for b in batch]
    is_p40s = [b['is_patient_40'] for b in batch]
    
    mask_labels_list = []
    class_labels_list = []
    
    for b in range(len(batch)):
        m = masks[b]
        unique_classes = torch.unique(m)
        unique_classes = unique_classes[unique_classes > 0]
        
        if len(unique_classes) == 0:
            mask_labels_list.append(torch.zeros((1, m.shape[0], m.shape[1]), dtype=torch.float32))
            class_labels_list.append(torch.zeros(1, dtype=torch.int64))
        else:
            binary_masks = torch.stack([(m == c).float() for c in unique_classes])
            mask_labels_list.append(binary_masks)
            class_labels_list.append(unique_classes.long())
            
    return {
        'pixel_values': pixel_values,
        'mask': masks,
        'mask_labels': mask_labels_list,
        'class_labels': class_labels_list,
        'junction_coords': junction_coords,
        'junction_vis': junction_vis,
        'filename': filenames,
        'is_patient_40': is_p40s
    }

def build_optimizer_and_scheduler(model, lr_backbone=1e-5, lr_head=1e-4, weight_decay=1e-4, epochs=60):
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'pixel_level_module.encoder' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
            
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': lr_backbone},
        {'params': head_params, 'lr': lr_head}
    ], weight_decay=weight_decay)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    return optimizer, scheduler

EPOCHS = 60
BATCH_SIZE = 2
ACCUM_STEPS = 2
LR_HEAD = 1e-4
LR_BACKBONE = 1e-5

train_dataset = StratifiedL3DDataset(split='Train', train_dir=TRAIN_DIR, val_dir=VAL_DIR)
val_dataset = StratifiedL3DDataset(split='Val', train_dir=TRAIN_DIR, val_dir=VAL_DIR)
test_dataset = StratifiedL3DDataset(split='Test', test_dir=TEST_DIR)

num_workers = 2 if device.type == 'cuda' else 0
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, collate_fn=collate_fn_l3d)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, collate_fn=collate_fn_l3d)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, collate_fn=collate_fn_l3d)

model = JunctionSteeredMask2Former(num_labels=4).to(device)
criterion = JunctionSteeredLoss(lambda_m2f=1.0, lambda_coord=5.0, lambda_vis=1.0)
optimizer, scheduler = build_optimizer_and_scheduler(model, lr_backbone=LR_BACKBONE, lr_head=LR_HEAD, epochs=EPOCHS)

use_amp = (device.type == 'cuda')
amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

print(f"🚀 Starting {EPOCHS}-epoch training on {device}...")
best_val_dice = -1.0
best_ckpt_path = os.path.join(OUT_DIR, 'best_model.pth')
training_log = []

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    model.train()
    losses = []
    optimizer.zero_grad()
    for step, batch in enumerate(train_loader):
        pixel_values = batch['pixel_values'].to(device)
        mask_labels = [m.to(device) for m in batch['mask_labels']]
        class_labels = [c.to(device) for c in batch['class_labels']]
        gt_j_coords = batch['junction_coords'].to(device)
        gt_j_vis = batch['junction_vis'].to(device)
        
        with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
            outputs = model(pixel_values, mask_labels=mask_labels, class_labels=class_labels)
            total_loss, loss_dict = criterion(outputs, gt_j_coords, gt_j_vis)
            loss_scaled = total_loss / float(ACCUM_STEPS)
            
        scaler.scale(loss_scaled).backward()
        losses.append(loss_dict['loss'])
        
        if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
    scheduler.step()
    
    # Validation evaluation
    val_summary, val_df = run_evaluation(model, val_loader, device, split_name='Val', out_dir=OUT_DIR)
    vdice = val_summary['macro_dice']
    vassd = val_summary['macro_assd']
    
    log_entry = {
        'epoch': epoch,
        'train_loss': float(np.mean(losses)),
        'val_macro_dice': vdice,
        'val_macro_assd': vassd,
        'duration_sec': time.time() - t0
    }
    training_log.append(log_entry)
    pd.DataFrame(training_log).to_csv(os.path.join(OUT_DIR, 'training_log.csv'), index=False)
    
    print(f"Epoch [{epoch}/{EPOCHS}] Train Loss: {np.mean(losses):.4f} | Val Dice: {vdice:.2%} | ASSD: {vassd:.2f}px ({time.time()-t0:.1f}s)")
    
    if vdice > best_val_dice:
        best_val_dice = vdice
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'val_macro_dice': vdice}, best_ckpt_path)
        print(f"⭐ New best model saved: {best_val_dice:.2%}")

# Final Comprehensive Evaluation
print('\\n🏁 Running Final Comprehensive Evaluation on Best Checkpoint...')
ckpt = torch.load(best_ckpt_path, map_location=device)
model.load_state_dict(ckpt['model_state_dict'])

val_summary, val_df = run_evaluation(model, val_loader, device, split_name='Val-Final', out_dir=OUT_DIR, save_diagnostics=True)
val_df.to_csv(os.path.join(OUT_DIR, 'val_predictions.csv'), index=False)

test_summary, test_df = run_evaluation(model, test_loader, device, split_name='Test-Final', out_dir=OUT_DIR, save_diagnostics=True)
test_df.to_csv(os.path.join(OUT_DIR, 'test_predictions.csv'), index=False)

full_metrics = {'val_summary': val_summary, 'test_summary': test_summary, 'best_epoch': ckpt.get('epoch', None)}
with open(os.path.join(OUT_DIR, 'metrics_summary.json'), 'w') as f:
    json.dump(full_metrics, f, indent=4)

# Zip results for download
zip_path = os.path.join(OUT_DIR, 'results.zip')
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, _, files in os.walk(OUT_DIR):
        for f in files:
            if f != 'results.zip':
                p = os.path.join(root, f)
                zipf.write(p, os.path.relpath(p, OUT_DIR))

print(f"🎉 All done! Download results archive from: {zip_path}")
"""

def make_cell(cell_type, source):
    return {
        'cell_type': cell_type,
        'metadata': {},
        'source': source.splitlines(keepends=True)
    }

cells = [
    make_cell('markdown', cell_0_md),
    make_cell('code', cell_1_code),
    make_cell('code', cell_2_code),
    make_cell('code', cell_3_code),
    make_cell('code', cell_4_code),
    make_cell('code', cell_5_code),
    make_cell('code', cell_6_code),
    make_cell('code', cell_7_code),
    make_cell('code', cell_8_code),
    make_cell('code', cell_9_code),
    make_cell('code', cell_10_code),
]

nb = {
    'cells': cells,
    'metadata': {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {
            'name': 'python',
            'version': '3.10.0'
        }
    },
    'nbformat': 4,
    'nbformat_minor': 5
}

with open(OUT_NOTEBOOK, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Generated notebook at:', OUT_NOTEBOOK)
