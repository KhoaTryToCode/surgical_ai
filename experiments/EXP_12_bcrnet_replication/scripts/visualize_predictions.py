"""
Visual Diagnostic & Landmark Prediction Overlay Tool for EXP_12: BCRNet
========================================================================
Generates high-resolution multi-panel visual comparisons for surgical landmark detection:
  - Panel 1: Original Laparoscopic Frame (RGB)
  - Panel 2: Ground Truth Landmark Curves Overlaid on Surgical View
  - Panel 3: BCRNet Predicted Bézier Curves Overlaid on Surgical View (with confidence)
  - Panel 4: Pixel Confusion Overlay (Green = True Positive, Red = False Positive, Yellow = False Negative)

Allows visual inspection of why pixel-wise DSC is ~35% on L3D:
  - Checks geometric curve alignment vs. 30px rasterization sensitivity
  - Inspects per-landmark predictions (Silhouette, Falciform Ligament, Anterior Ridge)
  - Detects threshold cutoffs and false positive / false negative patterns

Usage:
  python experiments/EXP_12_bcrnet_replication/scripts/visualize_predictions.py \
      --model_path /kaggle/working/checkpoints/EXP_12_bcrnet_replication/best_model.pt \
      --data_path /kaggle/working/L3D \
      --split Test \
      --num_samples 10 \
      --select diverse \
      --threshold 0.3
"""

import os
import sys
import glob
import types
import argparse
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

# Native PyTorch fallback for MSDeformAttn if CUDA extension not found
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

# Resilient dataset monkey patches
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
        os.path.join(self.data_path, 'depth_anything_v2', item_name + '.jpg'),
        os.path.join(self.data_path, 'depth', item_name + '.png'),
    ]
    img = None
    for c in candidates:
        if os.path.exists(c):
            img = cv2.imread(c, 0)
            if img is not None: break
    if img is None and os.path.exists('/kaggle/input'):
        if item_name in _depth_cache:
            img = cv2.imread(_depth_cache[item_name], 0)
        else:
            for root, _, files in os.walk('/kaggle/input'):
                for ext in ['.png', '.jpg']:
                    if f"{item_name}{ext}" in files and 'depth' in root.lower():
                        p = os.path.join(root, f"{item_name}{ext}")
                        img = cv2.imread(p, 0)
                        if img is not None:
                            _depth_cache[item_name] = p
                            break
                if img is not None: break
    if img is None:
        return torch.zeros(self.image_size, dtype=torch.float32, device=self.device)
    img = cv2.resize(img, self.image_size).astype('float32')
    return torch.from_numpy(img).to(self.device)

_sam_cache = {}
def _safe_load_sam_feature(self, item_name):
    path = os.path.join(self.data_path, 'sam', item_name + '.npy')
    if os.path.exists(path):
        try: return torch.from_numpy(np.load(path)).to(self.device)
        except Exception: pass
    if os.path.exists('/kaggle/input'):
        if item_name in _sam_cache:
            try: return torch.from_numpy(np.load(_sam_cache[item_name])).to(self.device)
            except Exception: pass
        else:
            for root, _, files in os.walk('/kaggle/input'):
                if f"{item_name}.npy" in files:
                    p = os.path.join(root, f"{item_name}.npy")
                    try:
                        feat = np.load(p)
                        _sam_cache[item_name] = p
                        return torch.from_numpy(feat).to(self.device)
                    except Exception: pass
    return torch.zeros((256, 64, 64), dtype=torch.float32, device=self.device)

def _safe_load_image(self, item_name):
    candidates = [
        os.path.join(self.data_path, 'images', item_name + '.jpg'),
        os.path.join(self.data_path, 'images', item_name + '.png'),
    ]
    img = None
    for c in candidates:
        if os.path.exists(c):
            img = cv2.imread(c)
            if img is not None: break
    if img is None and os.path.exists('/kaggle/input'):
        for root, _, files in os.walk('/kaggle/input'):
            for ext in ['.jpg', '.png']:
                if f"{item_name}{ext}" in files:
                    img = cv2.imread(os.path.join(root, f"{item_name}{ext}"))
                    if img is not None: break
            if img is not None: break
    if img is None:
        raise FileNotFoundError(f"Image for '{item_name}' not found")
    image_size = img.shape[:2]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, self.image_size).astype('float32') / 255.
    return torch.from_numpy(img).to(self.device), image_size

BezierDataset.load_bezier_gt = _safe_load_bezier_gt
BezierDataset.load_depth = _safe_load_depth
BezierDataset.load_sam_feature = _safe_load_sam_feature
BezierDataset.load_image = _safe_load_image


# BGR Color definitions for landmark classes
CLASS_COLORS = {
    0: (0, 255, 0),      # Silhouette -> Green
    1: (255, 165, 0),    # Ligament   -> Cyan/SkyBlue in BGR: (255, 165, 0)
    2: (0, 0, 255),      # Ridge      -> Red
}
CLASS_NAMES = ['Silhouette', 'Ligament', 'Ridge']


def create_comparison_panel(raw_bgr, results, targets, item_name, dice_val, iou_val, class_dices):
    """
    Creates a 4-panel visual comparison:
      [Raw Image] | [Ground Truth] | [BCRNet Prediction] | [Pixel Error Map]
    """
    H, W = raw_bgr.shape[:2]

    # 1. Ground Truth Rasterization & Overlay
    gt_canvas = raw_bgr.copy()
    gt_mask_3ch = np.zeros((H, W, 3), dtype=np.uint8)
    for c_idx, lm in enumerate(targets[0]):
        m = lm['landmark_mask']
        m_np = (m.cpu().numpy() if torch.is_tensor(m) else m).astype(np.uint8)
        if m_np.shape[:2] != (H, W):
            m_np = cv2.resize(m_np, (W, H), interpolation=cv2.INTER_NEAREST)
        gt_mask_3ch[:, :, c_idx] = m_np
        color = CLASS_COLORS[c_idx]
        gt_canvas[m_np > 0] = cv2.addWeighted(gt_canvas[m_np > 0], 0.3, np.full_like(gt_canvas[m_np > 0], color), 0.7, 0)

    # 2. Prediction Rasterization & Overlay
    pred_canvas = raw_bgr.copy()
    pred_mask_3ch = np.zeros((H, W, 3), dtype=np.uint8)
    detected_counts = {c: 0 for c in range(3)}

    for c_idx, lm in enumerate(results[0]):
        color = CLASS_COLORS[c_idx]
        if 'ctrl_points' in lm and lm['ctrl_points'].numel() > 0:
            curves = lm['ctrl_points'].detach().cpu().numpy()
            scores = lm.get('scores', torch.tensor([])).detach().cpu().numpy() if 'scores' in lm else None
            detected_counts[c_idx] = len(curves)
            for k, cp in enumerate(curves):
                # Scale if normalized to 1024 or image_size
                pts = np.clip(cp, [0, 0], [W - 1, H - 1]).astype(np.int32)
                for i in range(1, len(pts)):
                    pt1 = (int(pts[i - 1][0]), int(pts[i - 1][1]))
                    pt2 = (int(pts[i][0]), int(pts[i][1]))
                    cv2.line(pred_mask_3ch[:, :, c_idx], pt1, pt2, 1, 30)
                    cv2.line(pred_canvas, pt1, pt2, color, 4)
                # Draw control points markers
                for p in pts:
                    cv2.circle(pred_canvas, (int(p[0]), int(p[1])), 4, (255, 255, 255), -1)

    # 3. Pixel Confusion Overlay
    # Green = True Positive (overlap), Red = False Positive, Yellow = False Negative
    error_canvas = np.zeros((H, W, 3), dtype=np.uint8)
    pred_any = np.sum(pred_mask_3ch, axis=-1) > 0
    gt_any = np.sum(gt_mask_3ch, axis=-1) > 0

    tp = pred_any & gt_any
    fp = pred_any & ~gt_any
    fn = ~pred_any & gt_any

    error_canvas[tp] = [0, 255, 0]     # TP = Green
    error_canvas[fp] = [0, 0, 255]     # FP = Red
    error_canvas[fn] = [0, 255, 255]   # FN = Yellow/Orange
    # Blend with background image slightly for context
    error_canvas = cv2.addWeighted(raw_bgr, 0.4, error_canvas, 0.6, 0)

    # Add Panel Subtitles
    font = cv2.FONT_HERSHEY_SIMPLEX
    def add_title(img, text):
        out = img.copy()
        cv2.rectangle(out, (0, 0), (W, 40), (20, 20, 20), -1)
        cv2.putText(out, text, (15, 28), font, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
        return out

    p1 = add_title(raw_bgr, "1. Laparoscopic View (RGB)")
    p2 = add_title(gt_canvas, "2. Ground Truth Landmarks")
    p3 = add_title(pred_canvas, f"3. BCRNet Predictions (Sil:{detected_counts[0]} Lig:{detected_counts[1]} Rid:{detected_counts[2]})")
    p4 = add_title(error_canvas, "4. Overlap (Green=TP, Red=FP, Yellow=FN)")

    # Legend on Panel 2
    cv2.putText(p2, "Sil (Green)", (15, H - 20), font, 0.65, (0, 255, 0), 2)
    cv2.putText(p2, "Lig (Cyan)", (160, H - 20), font, 0.65, (255, 165, 0), 2)
    cv2.putText(p2, "Rid (Red)", (300, H - 20), font, 0.65, (0, 0, 255), 2)

    # Top & Bottom 2x2 Grid or 1x4 Strip
    # Resize panels for display (e.g. 640x360 each)
    pw, ph = 640, 360
    r1 = np.hstack([cv2.resize(p1, (pw, ph)), cv2.resize(p2, (pw, ph))])
    r2 = np.hstack([cv2.resize(p3, (pw, ph)), cv2.resize(p4, (pw, ph))])
    composite = np.vstack([r1, r2])

    # Add Global Header Banner
    header = np.zeros((50, composite.shape[1], 3), dtype=np.uint8)
    banner_text = (
        f"Frame: {item_name} | Overall DSC: {dice_val*100:.1f}% | IoU: {iou_val*100:.1f}% | "
        f"Sil: {class_dices['Silhouette']*100:.1f}% | Lig: {class_dices['Ligament']*100:.1f}% | Rid: {class_dices['Ridge']*100:.1f}%"
    )
    cv2.putText(header, banner_text, (20, 33), font, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
    final_image = np.vstack([header, composite])

    return final_image


def main():
    parser = argparse.ArgumentParser(description="EXP_12: BCRNet Visual Prediction Inspector")
    parser.add_argument('--config', default=os.path.join(os.path.dirname(__file__), "../configs/bcrnet_l3d.yaml"))
    parser.add_argument('--model_path', required=True, help="Path to checkpoint (.pt)")
    parser.add_argument('--data_path', default="/kaggle/working/L3D" if os.path.exists("/kaggle") else os.path.join(ws_root, "data/L3D"))
    parser.add_argument('--split', default='Test', choices=['Test', 'Val'])
    parser.add_argument('--num_samples', type=int, default=10, help="Number of comparison figures to generate")
    parser.add_argument('--select', default='diverse', choices=['diverse', 'best', 'worst', 'first'], help="Sample selection strategy")
    parser.add_argument('--threshold', type=float, default=0.3, help="Confidence threshold")
    parser.add_argument('--model_mode', default='train', choices=['train', 'eval'])
    parser.add_argument('--save_dir', default="/kaggle/working/visualizations/EXP_12" if os.path.exists("/kaggle") else os.path.join(ws_root, "experiments/EXP_12_bcrnet_replication/visualizations"))
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() else 'cpu'

    print("=" * 80)
    print(f"🎨 BCRNET SURGICAL PREDICTION VISUALIZER [{args.split.upper()} SPLIT]")
    print(f"   Model:     {args.model_path}")
    print(f"   Dataset:   {args.data_path}")
    print(f"   Output:    {args.save_dir}")
    print(f"   Threshold: {args.threshold} | Mode: {args.model_mode}")
    print("=" * 80)

    cfg = load_config(args.config)
    cfg.MODEL.DEVICE = device
    model = TransformerPureDetector(cfg).to(device)
    model.test_score_threshold = args.threshold

    checkpoint = torch.load(args.model_path, map_location=device)
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    del checkpoint, state_dict

    if args.model_mode == 'train':
        model.train()
    else:
        model.eval()

    split_dir = os.path.join(args.data_path, args.split.capitalize())
    dataset = BezierDataset(split_dir, device=device)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fun)

    print(f"🔍 Running inference on all {len(dataset)} frames to compute metrics and rank candidates...", flush=True)

    records = []
    smooth = 1e-5

    with torch.inference_mode():
        for idx, batch_data in enumerate(tqdm(loader, desc="Analyzing Frames")):
            img, depth, sam_feature, targets, info = batch_data
            item_name = info[0]['item_name']
            results = model(batch_data)

            # Rasterize for metrics
            gt_masks = [lm['landmark_mask'].cpu().numpy().astype(np.uint8) for lm in targets[0]]
            gt = np.stack(gt_masks, -1)
            H, W = gt.shape[:2]

            pred_channels = []
            for m, lm in enumerate(results[0]):
                channel = np.zeros((H, W), dtype=np.uint8)
                if 'ctrl_points' in lm and lm['ctrl_points'].numel() > 0:
                    curves = lm['ctrl_points'].detach().cpu().numpy()
                    for cp in curves:
                        pts = np.clip(cp, [0, 0], [W - 1, H - 1]).astype(np.int32)
                        for i in range(1, len(pts)):
                            cv2.line(channel, (int(pts[i - 1][0]), int(pts[i - 1][1])), (int(pts[i][0]), int(pts[i][1])), 1, 30)
                pred_channels.append(channel)
            pred = np.stack(pred_channels, axis=-1)

            intersection = np.sum(pred * gt)
            dice = float((2.0 * intersection + smooth) / (np.sum(pred) + np.sum(gt) + smooth))
            iou = float(dice / (2.0 - dice))

            class_d = {}
            for c_idx, c_name in enumerate(CLASS_NAMES):
                p_c = pred[:, :, c_idx]
                g_c = gt[:, :, c_idx]
                d_c = float((2.0 * np.sum(p_c * g_c) + smooth) / (np.sum(p_c) + np.sum(g_c) + smooth))
                class_d[c_name] = d_c

            records.append({
                'index': idx,
                'item_name': item_name,
                'dice': dice,
                'iou': iou,
                'class_dices': class_d,
                'results': results,
                'targets': targets,
                'info': info
            })

            del batch_data, results, pred, gt, img, depth, sam_feature, targets, info

    # Selection Strategy
    records_sorted = sorted(records, key=lambda r: r['dice'], reverse=True)
    N = min(args.num_samples, len(records))

    if args.select == 'best':
        selected = records_sorted[:N]
    elif args.select == 'worst':
        selected = records_sorted[-N:]
    elif args.select == 'first':
        selected = records[:N]
    else:  # diverse
        # Pick evenly distributed quantiles across the Dice distribution
        indices = np.linspace(0, len(records_sorted) - 1, N, dtype=int)
        selected = [records_sorted[i] for i in indices]

    print(f"\n📸 Rendering {len(selected)} high-resolution diagnostic panels to: {args.save_dir}")

    saved_paths = []
    for rank, rec in enumerate(selected):
        item_name = rec['item_name']
        dice = rec['dice']
        iou = rec['iou']
        class_d = rec['class_dices']

        # Read original image
        img_cand = [
            os.path.join(split_dir, 'images', item_name + '.jpg'),
            os.path.join(split_dir, 'images', item_name + '.png'),
        ]
        raw_bgr = None
        for p in img_cand:
            if os.path.exists(p):
                raw_bgr = cv2.imread(p)
                if raw_bgr is not None: break
        if raw_bgr is None:
            raw_bgr = np.zeros((1080, 1920, 3), dtype=np.uint8)

        panel = create_comparison_panel(
            raw_bgr=raw_bgr,
            results=rec['results'],
            targets=rec['targets'],
            item_name=item_name,
            dice_val=dice,
            iou_val=iou,
            class_dices=class_d
        )

        out_name = f"rank{rank+1:02d}_{item_name}_dice{dice*100:.1f}.png"
        out_path = os.path.join(args.save_dir, out_name)
        cv2.imwrite(out_path, panel)
        saved_paths.append(out_path)
        print(f"   [{rank+1:02d}/{N}] Saved: {out_name} (DSC: {dice*100:.2f}%)")

    print("\n" + "=" * 80)
    print(f"✅ Generated {len(saved_paths)} diagnostic visualization panels in: {args.save_dir}")
    print("=" * 80)
    print("\n💡 To view these images directly inside your Kaggle notebook cell, run:")
    print("```python")
    print("import glob")
    print("from IPython.display import Image, display")
    print(f"for p in sorted(glob.glob('{args.save_dir}/*.png'))[:5]:")
    print("    display(Image(p, width=950))")
    print("```\n")


if __name__ == '__main__':
    main()
