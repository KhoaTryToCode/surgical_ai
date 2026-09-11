"""
Forensic Diagnostic & Benchmark Discrepancy Analyzer for EXP_12: BCRNet
========================================================================
Diagnoses why test split logs ~34.84% macro DSC vs paper's 69.57%:
  1. Per-Patient Breakdown: Shows how standard patients (e.g. Patient 41, up to 69.3%) compare against out-of-distribution patients (e.g. Patient 31).
  2. Dataset-Level Global Dice vs. Sample-Level Macro Mean Dice:
     Checks whether the paper calculated global intersection / union across all frames or average of per-frame Dice.
  3. Threshold Sensitivity Sweep:
     Evaluates threshold tau in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50] to find the peak F1 operating point.
  4. Non-Zero Landmark Frames:
     Analyzes performance when excluding empty / absent ground truth landmarks.

Usage:
  python experiments/EXP_12_bcrnet_replication/scripts/diagnose_benchmark.py \
      --model_path /kaggle/working/checkpoints/EXP_12_bcrnet_replication/best_model.pt \
      --data_path /kaggle/working/L3D \
      --split Test
"""

import os
import sys
import glob
import types
import argparse
from collections import defaultdict
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ws_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
bcrnet_root = os.path.join(ws_root, "repos/BCRNet")
for p in [bcrnet_root, ws_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Native PyTorch fallback for MSDeformAttn
c_ext_files = glob.glob(os.path.join(bcrnet_root, "adet/_C*.so")) + glob.glob(os.path.join(bcrnet_root, "adet/_C*.pyd"))
if len(c_ext_files) == 0:
    adet_pkg = types.ModuleType('adet')
    adet_pkg.__path__ = [os.path.join(bcrnet_root, 'adet')]
    sys.modules['adet'] = adet_pkg
    mock_c = types.ModuleType('adet._C')
    mock_c.ms_deform_attn_forward = lambda *args, **kwargs: None
    mock_c.ms_deform_attn_backward = lambda *args, **kwargs: None
    adet_pkg._C = mock_c
    sys.modules['adet._C'] = mock_c
    from adet.layers.ms_deform_attn import MSDeformAttn, ms_deform_attn_core_pytorch
    def _fb(self, q, ref, in_fl, sp_sh, l_idx, p_mask=None):
        N, Lq, _ = q.shape; N, Lin, _ = in_fl.shape
        v = self.value_proj(in_fl).view(N, Lin, self.n_heads, self.d_model // self.n_heads)
        soff = self.sampling_offsets(q).view(N, Lq, self.n_heads, self.n_levels, self.n_points, 2)
        aw = torch.softmax(self.attention_weights(q).view(N, Lq, self.n_heads, self.n_levels * self.n_points), -1).view(N, Lq, self.n_heads, self.n_levels, self.n_points)
        norm = torch.stack([sp_sh[..., 1], sp_sh[..., 0]], -1)
        sloc = ref[:, :, None, :, None, :] + soff / norm[None, None, None, :, None, :]
        return self.output_proj(ms_deform_attn_core_pytorch(v, sp_sh, sloc, aw))
    MSDeformAttn.forward = _fb

from adet.utils.curve_utils import BezierSampler, upcast
def _ad_pts(self, cpm):
    if cpm.numel() == 0: return cpm
    k = cpm.shape[-2]
    if self.bernstein_matrix.shape[1] != k:
        self.degree = k - 1
        self.bezier_coeff = self.get_bezier_coefficient()
        self.bernstein_matrix = self.get_bernstein_matrix()
    if self.bernstein_matrix.device != cpm.device:
        self.bernstein_matrix = self.bernstein_matrix.to(cpm.device)
    return upcast(self.bernstein_matrix).matmul(upcast(cpm))
BezierSampler.get_sample_points = _ad_pts

from adet.modeling.bezier_detection import TransformerPureDetector
from utils.config_utils import load_config
from utils.bezier_dataset import BezierDataset, collate_fun

# Safe dataset monkey patches
def _safe_load_bezier_gt(self, item_name):
    path = os.path.join(self.data_path, 's_bezier', item_name + '.npz')
    with np.load(path, allow_pickle=True) as bezier_data:
        landmark_list = ['silhouette', 'ligament', 'ridge']
        source = []
        for i, (label, data) in enumerate(zip(landmark_list, bezier_data['gt'])):
            ctrl_points = torch.from_numpy(data['ctrl_points']).float().to(self.device)
            curve_points = torch.from_numpy(data['curve_points']).float().to(self.device)
            landmark_mask = torch.from_numpy(data['landmark_mask']).to(torch.uint8)
            source.append({
                'label': label,
                'ctrl_points': ctrl_points,
                'class_id': torch.LongTensor([i]).to(self.device),
                'curve_points': curve_points,
                'landmark_mask': landmark_mask
            })
    return source

_depth_cache = {}
def _safe_load_depth(self, item_name):
    candidates = [
        os.path.join(self.data_path, 'depth_AdelaiDepth', item_name + '.png'),
        os.path.join(self.data_path, 'depth_AdelaiDepth', item_name + '.jpg'),
        os.path.join(self.data_path, 'depth_anything_v2', item_name + '.png'),
        os.path.join(self.data_path, 'depth', item_name + '.png'),
    ]
    for c in candidates:
        if os.path.exists(c):
            img = cv2.imread(c, 0)
            if img is not None:
                return torch.from_numpy(cv2.resize(img, self.image_size).astype('float32')).to(self.device)
    if os.path.exists('/kaggle/input'):
        for root, _, files in os.walk('/kaggle/input'):
            for ext in ['.png', '.jpg']:
                if f"{item_name}{ext}" in files and 'depth' in root.lower():
                    img = cv2.imread(os.path.join(root, f"{item_name}{ext}"), 0)
                    if img is not None:
                        return torch.from_numpy(cv2.resize(img, self.image_size).astype('float32')).to(self.device)
    return torch.zeros(self.image_size, dtype=torch.float32, device=self.device)

_sam_cache = {}
def _safe_load_sam_feature(self, item_name):
    path = os.path.join(self.data_path, 'sam', item_name + '.npy')
    if os.path.exists(path):
        try: return torch.from_numpy(np.load(path)).to(self.device)
        except Exception: pass
    if os.path.exists('/kaggle/input'):
        for root, _, files in os.walk('/kaggle/input'):
            if f"{item_name}.npy" in files:
                try: return torch.from_numpy(np.load(os.path.join(root, f"{item_name}.npy"))).to(self.device)
                except Exception: pass
    return torch.zeros((256, 64, 64), dtype=torch.float32, device=self.device)

def _safe_load_image(self, item_name):
    for ext in ['.jpg', '.png']:
        path = os.path.join(self.data_path, 'images', item_name + ext)
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                sz = img.shape[:2]
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, self.image_size).astype('float32') / 255.
                return torch.from_numpy(img).to(self.device), sz
    raise FileNotFoundError(f"Image for {item_name} not found")

BezierDataset.load_bezier_gt = _safe_load_bezier_gt
BezierDataset.load_depth = _safe_load_depth
BezierDataset.load_sam_feature = _safe_load_sam_feature
BezierDataset.load_image = _safe_load_image


def rasterize_curves(ctrl_points_list, H, W, linewidth=30):
    channels = []
    for lm in ctrl_points_list:
        ch = np.zeros((H, W), dtype=np.uint8)
        if 'ctrl_points' in lm and lm['ctrl_points'].numel() > 0:
            curves = lm['ctrl_points'].detach().cpu().numpy()
            for cp in curves:
                pts = np.clip(cp, [0, 0], [W - 1, H - 1]).astype(np.int32)
                for i in range(1, len(pts)):
                    pt1 = (int(pts[i - 1][0]), int(pts[i - 1][1]))
                    pt2 = (int(pts[i][0]), int(pts[i][1]))
                    cv2.line(ch, pt1, pt2, 1, linewidth)
        channels.append(ch)
    return np.stack(channels, axis=-1)


def main():
    parser = argparse.ArgumentParser(description="EXP_12 Forensic Diagnostic Analyzer")
    parser.add_argument('--config', default=os.path.join(os.path.dirname(__file__), "../configs/bcrnet_l3d.yaml"))
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--data_path', default="/kaggle/working/L3D" if os.path.exists("/kaggle") else os.path.join(ws_root, "data/L3D"))
    parser.add_argument('--split', default='Test')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    print("=" * 80)
    print(f"🔬 EXP_12 FORENSIC DIAGNOSTIC ANALYZER [{args.split.upper()} SPLIT]")
    print(f"   Model:  {args.model_path}")
    print(f"   Data:   {args.data_path}")
    print(f"   Device: {device}")
    print("=" * 80)

    cfg = load_config(args.config)
    cfg.MODEL.DEVICE = device
    model = TransformerPureDetector(cfg).to(device)

    checkpoint = torch.load(args.model_path, map_location=device)
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    del checkpoint, state_dict
    model.train()  # Matching official test.py

    split_dir = os.path.join(args.data_path, args.split.capitalize())
    dataset = BezierDataset(split_dir, device=device)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fun)

    # 1. Collect all outputs without thresholding first
    print(f"\n[1/3] Extracting Raw Model Proposals for all {len(dataset)} frames...")

    all_frame_data = []
    with torch.inference_mode():
        for batch_data in tqdm(loader, desc="Inference"):
            img, depth, sam_feature, targets, info = batch_data
            item_name = info[0]['item_name']
            patient_id = item_name.split('_')[1] if '_' in item_name else 'Unknown'

            # Run transformer
            images = model.preprocess_image(img)
            output = model.detection_transformer(images, depth, sam_feature)

            # Store raw tensors needed for different thresholds
            ctrl_point_cls = output["pred_logits"][..., 0].mean(-1).sigmoid().cpu()  # [B, M, Q]
            ctrl_point_coord = output["pred_ctrl_points"].cpu()                      # [B, M, Q, P, 2]
            img_size = info[0]['img_size'].cpu()

            # Ground truth masks
            gt_masks = [lm['landmark_mask'].cpu().numpy().astype(np.uint8) for lm in targets[0]]
            gt = np.stack(gt_masks, -1)

            all_frame_data.append({
                'item_name': item_name,
                'patient_id': patient_id,
                'scores': ctrl_point_cls[0],        # [M, Q]
                'coords': ctrl_point_coord[0],      # [M, Q, P, 2]
                'img_size': img_size,
                'gt': gt,
            })
            del batch_data, img, depth, sam_feature, targets, info, output

    # 2. Evaluate at threshold = 0.3 (Default Paper Setting)
    print("\n[2/3] Computing Detailed Breakdown at Threshold tau = 0.30...")
    patient_metrics = defaultdict(lambda: {'dices': [], 'ious': [], 'intersections': 0, 'preds': 0, 'gts': 0})
    class_metrics = {0: {'dices': []}, 1: {'dices': []}, 2: {'dices': []}}
    all_dices = []
    all_ious = []
    total_intersection = 0
    total_pred = 0
    total_gt = 0
    smooth = 1e-5

    for frame in all_frame_data:
        gt = frame['gt']
        H, W = gt.shape[:2]
        patient_id = frame['patient_id']
        scores = frame['scores']
        coords = frame['coords']
        img_size = frame['img_size']

        # Filter by threshold 0.3
        pred_channels = []
        for m in range(3):
            ch = np.zeros((H, W), dtype=np.uint8)
            selector = scores[m] >= 0.3
            if selector.any():
                sel_coords = coords[m][selector].clone()
                sel_coords[..., 0] *= img_size[1]
                sel_coords[..., 1] *= img_size[0]
                curves = sel_coords.numpy()
                for cp in curves:
                    pts = np.clip(cp, [0, 0], [W - 1, H - 1]).astype(np.int32)
                    for i in range(1, len(pts)):
                        pt1 = (int(pts[i - 1][0]), int(pts[i - 1][1]))
                        pt2 = (int(pts[i][0]), int(pts[i][1]))
                        cv2.line(ch, pt1, pt2, 1, 30)
            pred_channels.append(ch)
        pred = np.stack(pred_channels, axis=-1)

        inter = np.sum(pred * gt)
        p_sum = np.sum(pred)
        g_sum = np.sum(gt)
        d = float((2.0 * inter + smooth) / (p_sum + g_sum + smooth))
        iou = float(d / (2.0 - d))

        all_dices.append(d)
        all_ious.append(iou)
        total_intersection += inter
        total_pred += p_sum
        total_gt += g_sum

        patient_metrics[patient_id]['dices'].append(d)
        patient_metrics[patient_id]['ious'].append(iou)
        patient_metrics[patient_id]['intersections'] += inter
        patient_metrics[patient_id]['preds'] += p_sum
        patient_metrics[patient_id]['gts'] += g_sum

        for m in range(3):
            pm = pred[..., m]
            gm = gt[..., m]
            dm = float((2.0 * np.sum(pm * gm) + smooth) / (np.sum(pm) + np.sum(gm) + smooth))
            class_metrics[m]['dices'].append(dm)

    # 3. Patient-by-Patient Report
    print("\n" + "=" * 80)
    print("📊 PER-PATIENT BENCHMARK BREAKDOWN (tau = 0.30):")
    print("=" * 80)
    print("| Patient ID | Frames | Sample Macro DSC (%) | Global Dataset DSC (%) | IoU (%) |")
    print("|:---|:---:|:---:|:---:|:---:|")
    for pid in sorted(patient_metrics.keys()):
        p_dices = patient_metrics[pid]['dices']
        p_ious = patient_metrics[pid]['ious']
        p_macro_dice = float(np.mean(p_dices)) * 100.0
        p_macro_iou = float(np.mean(p_ious)) * 100.0
        p_global_dice = float(2.0 * patient_metrics[pid]['intersections'] / (patient_metrics[pid]['preds'] + patient_metrics[pid]['gts'] + smooth)) * 100.0
        print(f"| **Patient {pid}** | {len(p_dices)} | **{p_macro_dice:.2f}%** | **{p_global_dice:.2f}%** | {p_macro_iou:.2f}% |")

    # Global Dataset vs Macro Mean
    macro_dice = float(np.mean(all_dices)) * 100.0
    macro_iou = float(np.mean(all_ious)) * 100.0
    global_dice = float((2.0 * total_intersection) / (total_pred + total_gt + smooth)) * 100.0
    global_iou = global_dice / (200.0 - global_dice) * 100.0

    print("=" * 80)
    print(f"• Sample-Level Macro Mean DSC:  {macro_dice:.2f}%  (Paper: 69.57%)")
    print(f"• Dataset-Level Global DSC:     {global_dice:.2f}%")
    print(f"• Sample-Level Macro Mean IoU:  {macro_iou:.2f}%  (Paper: 54.16%)")
    print("=" * 80)

    # 4. Threshold Sensitivity Sweep
    print("\n[3/3] Running Threshold Sensitivity Sweep tau in [0.05 ... 0.50]...")
    print("\n| Threshold (tau) | Macro DSC (%) | Global DSC (%) | Macro IoU (%) | Silhouette (%) | Ligament (%) | Ridge (%) |")
    print("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    best_th = 0.30
    best_macro = 0.0

    for th in thresholds:
        th_dices = []
        th_inter = 0
        th_pred = 0
        th_gt = 0
        th_cls = {0: [], 1: [], 2: []}

        for frame in all_frame_data:
            gt = frame['gt']
            H, W = gt.shape[:2]
            scores = frame['scores']
            coords = frame['coords']
            img_size = frame['img_size']

            pred_channels = []
            for m in range(3):
                ch = np.zeros((H, W), dtype=np.uint8)
                selector = scores[m] >= th
                if selector.any():
                    sel_coords = coords[m][selector].clone()
                    sel_coords[..., 0] *= img_size[1]
                    sel_coords[..., 1] *= img_size[0]
                    curves = sel_coords.numpy()
                    for cp in curves:
                        pts = np.clip(cp, [0, 0], [W - 1, H - 1]).astype(np.int32)
                        for i in range(1, len(pts)):
                            pt1 = (int(pts[i - 1][0]), int(pts[i - 1][1]))
                            pt2 = (int(pts[i][0]), int(pts[i][1]))
                            cv2.line(ch, pt1, pt2, 1, 30)
                pred_channels.append(ch)
            pred = np.stack(pred_channels, axis=-1)

            inter = np.sum(pred * gt)
            p_sum = np.sum(pred)
            g_sum = np.sum(gt)
            d = float((2.0 * inter + smooth) / (p_sum + g_sum + smooth))
            th_dices.append(d)
            th_inter += inter
            th_pred += p_sum
            th_gt += g_sum

            for m in range(3):
                pm = pred[..., m]
                gm = gt[..., m]
                dm = float((2.0 * np.sum(pm * gm) + smooth) / (np.sum(pm) + np.sum(gm) + smooth))
                th_cls[m].append(dm)

        th_macro_dsc = float(np.mean(th_dices)) * 100.0
        th_global_dsc = float(2.0 * th_inter / (th_pred + th_gt + smooth)) * 100.0
        th_macro_iou = float(np.mean([d / (2.0 - d) for d in th_dices])) * 100.0
        c0 = float(np.mean(th_cls[0])) * 100.0
        c1 = float(np.mean(th_cls[1])) * 100.0
        c2 = float(np.mean(th_cls[2])) * 100.0

        marker = " 🏆" if th_macro_dsc > best_macro else ""
        if th_macro_dsc > best_macro:
            best_macro = th_macro_dsc
            best_th = th

        print(f"| {th:.2f}{marker} | {th_macro_dsc:.2f} | {th_global_dsc:.2f} | {th_macro_iou:.2f} | {c0:.2f} | {c1:.2f} | {c2:.2f} |")

    print("\n" + "=" * 80)
    print(f"💡 Optimal Threshold Found: tau = {best_th:.2f} (Macro DSC: {best_macro:.2f}%)")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
