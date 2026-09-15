"""
Minimal, Bulletproof Evaluation Script for EXP_12: BCRNet Replication
====================================================================
Direct extraction of the proven validation loop from `train_bcrnet.py`
that ran 70 epochs on Kaggle without a single memory error.

Usage:
  python experiments/EXP_12_bcrnet_replication/scripts/eval_simple.py \
      --model_path /kaggle/working/checkpoints/EXP_12_bcrnet_replication/best_model.pt \
      --data_path /kaggle/working/L3D \
      --split Test
"""

import os
import sys
import types
import glob
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

try:
    import fvcore
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "fvcore", "iopath"], check=True)

from adet.modeling.bezier_detection import TransformerPureDetector
from utils.config_utils import load_config
from utils.bezier_dataset import BezierDataset, collate_fun

# Safe npz loading
def _safe_load_bezier_gt(self, item_name):
    path = os.path.join(self.data_path, 's_bezier', item_name + '.npz')
    with np.load(path, allow_pickle=True) as bezier_data:
        landmark_list = ['silhouette', 'ligament', 'ridge']
        source = []
        for i, (label, data) in enumerate(zip(landmark_list, bezier_data['gt'])):
            ctrl_points = torch.from_numpy(data['ctrl_points']).float().to(self.device)
            curve_points = torch.from_numpy(data['curve_points']).float().to(self.device)
            landmark_mask = torch.from_numpy(data['landmark_mask']).float().to(self.device)
            source.append({
                'label': label,
                'ctrl_points': ctrl_points,
                'class_id': torch.LongTensor([i]).to(self.device),
                'curve_points': curve_points,
                'landmark_mask': landmark_mask
            })
    return source
_depth_cache = {}
_warned_missing_depth = False

def _safe_load_depth(self, item_name):
    global _warned_missing_depth
    img = None

    candidates = [
        os.path.join(self.data_path, 'depth_AdelaiDepth', item_name + '.png'),
        os.path.join(self.data_path, 'depth_AdelaiDepth', item_name + '.jpg'),
        os.path.join(self.data_path, 'depth_anything_v2', item_name + '.png'),
        os.path.join(self.data_path, 'depth_anything_v2', item_name + '.jpg'),
        os.path.join(self.data_path, 'depth', item_name + '.png'),
        os.path.join(self.data_path, 'depth', item_name + '.jpg'),
    ]
    for c in candidates:
        if os.path.exists(c):
            img = cv2.imread(c, 0)
            if img is not None:
                break

    if img is None and os.path.exists('/kaggle/input'):
        if item_name in _depth_cache:
            img = cv2.imread(_depth_cache[item_name], 0)
        else:
            for root, _, files in os.walk('/kaggle/input'):
                for ext in ['.png', '.jpg']:
                    fname = f"{item_name}{ext}"
                    if fname in files and 'depth' in root.lower():
                        p = os.path.join(root, fname)
                        img = cv2.imread(p, 0)
                        if img is not None:
                            _depth_cache[item_name] = p
                            break
                if img is not None:
                    break

    if img is None:
        if not _warned_missing_depth:
            print(f"⚠️ [Safe Depth] Missing depth map for '{item_name}'. Using zero-filled fallback tensor (image_size: {self.image_size}).", flush=True)
            _warned_missing_depth = True
        return torch.zeros(self.image_size, dtype=torch.float32, device=self.device)

    img = cv2.resize(img, self.image_size).astype('float32')
    return torch.from_numpy(img).to(self.device)

_sam_cache = {}
_warned_missing_sam = False

def _safe_load_sam_feature(self, item_name):
    global _warned_missing_sam
    path = os.path.join(self.data_path, 'sam', item_name + '.npy')
    if os.path.exists(path):
        try:
            feat = np.load(path)
            return torch.from_numpy(feat).to(self.device)
        except Exception:
            pass

    if os.path.exists('/kaggle/input'):
        if item_name in _sam_cache:
            try:
                feat = np.load(_sam_cache[item_name])
                return torch.from_numpy(feat).to(self.device)
            except Exception:
                pass
        else:
            for root, _, files in os.walk('/kaggle/input'):
                if f"{item_name}.npy" in files:
                    p = os.path.join(root, f"{item_name}.npy")
                    try:
                        feat = np.load(p)
                        _sam_cache[item_name] = p
                        return torch.from_numpy(feat).to(self.device)
                    except Exception:
                        pass

    if not _warned_missing_sam:
        print(f"⚠️ [Safe SAM] Missing SAM feature for '{item_name}'. Using zero-filled fallback tensor (256, 64, 64).", flush=True)
        _warned_missing_sam = True
    return torch.zeros((256, 64, 64), dtype=torch.float32, device=self.device)

def _safe_load_image(self, item_name):
    candidates = [
        os.path.join(self.data_path, 'images', item_name + '.jpg'),
        os.path.join(self.data_path, 'images', item_name + '.png'),
        os.path.join(self.data_path, 'images', item_name + '.jpeg'),
    ]
    img = None
    for c in candidates:
        if os.path.exists(c):
            img = cv2.imread(c)
            if img is not None:
                break
    if img is None and os.path.exists('/kaggle/input'):
        for root, _, files in os.walk('/kaggle/input'):
            for ext in ['.jpg', '.png', '.jpeg']:
                if f"{item_name}{ext}" in files:
                    img = cv2.imread(os.path.join(root, f"{item_name}{ext}"))
                    if img is not None:
                        break
            if img is not None:
                break
    if img is None:
        raise FileNotFoundError(f"Image for '{item_name}' not found in {self.data_path} or /kaggle/input")
    image_size = img.shape[:2]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, self.image_size).astype('float32') / 255.
    return torch.from_numpy(img).to(self.device), image_size

BezierDataset.load_bezier_gt = _safe_load_bezier_gt
BezierDataset.load_depth = _safe_load_depth
BezierDataset.load_sam_feature = _safe_load_sam_feature
BezierDataset.load_image = _safe_load_image


def evaluation(pred, gt):
    smooth = 1e-5
    intersection = np.sum(pred * gt)
    dice = (2.0 * intersection + smooth) / (np.sum(pred) + np.sum(gt) + smooth)
    iou = dice / (2.0 - dice)
    return iou, dice


def metrix(results, targets):
    gt = np.stack([lm['landmark_mask'].cpu().numpy().astype(np.uint8) for lm in targets[0]], -1)
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
    iou, dice = evaluation(pred, gt)
    return iou, dice, pred, gt


def evaluate_single_split(model, split_name, data_path, device):
    split_dir = os.path.join(data_path, split_name.capitalize())
    if not os.path.exists(split_dir):
        print(f"⚠️ Split directory not found: {split_dir}. Skipping.")
        return None

    dataset = BezierDataset(split_dir, device=device)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fun)
    print(f"\n🔍 Evaluating split [{split_name.upper()}] ({len(dataset)} frames)...", flush=True)

    class_names = ['silhouette', 'ligament', 'ridge']
    ious = []
    dices = []
    class_dices = {c: [] for c in class_names}

    with torch.inference_mode():
        pbar = tqdm(loader, desc=f"Evaluating {split_name.capitalize()}")
        for step_idx, batch_data in enumerate(pbar):
            img, depth, sam_feature, targets, info = batch_data
            results = model(batch_data)
            iou, dice, pred, gt = metrix(results, targets)

            ious.append(iou)
            dices.append(dice)

            # Per-class Dice
            smooth = 1e-5
            for c_idx, c_name in enumerate(class_names):
                p_c = pred[:, :, c_idx]
                g_c = gt[:, :, c_idx]
                d_c = float((2.0 * np.sum(p_c * g_c) + smooth) / (np.sum(p_c) + np.sum(g_c) + smooth))
                class_dices[c_name].append(d_c)

            pbar.set_postfix({'Dice': f"{dice * 100:.2f}%", 'IoU': f"{iou * 100:.2f}%"})
            del batch_data, results, img, depth, sam_feature, targets, info, pred, gt

            if (step_idx + 1) % 10 == 0:
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    mean_dice = float(np.mean(dices)) * 100.0
    mean_iou = float(np.mean(ious)) * 100.0

    print("\n" + "=" * 70, flush=True)
    print(f"📊 BENCHMARK RESULTS FOR [{split_name.upper()}]:", flush=True)
    print(f"   • Mean DSC: {mean_dice:.2f}%  (Paper Target: 69.57%)", flush=True)
    print(f"   • Mean IoU: {mean_iou:.2f}%  (Paper Target: 54.16%)", flush=True)
    for c_name in class_names:
        c_mean = float(np.mean(class_dices[c_name])) * 100.0
        print(f"     - {c_name.capitalize()}: {c_mean:.2f}% DSC", flush=True)
    print("=" * 70 + "\n", flush=True)

    return {'mean_dice': mean_dice, 'mean_iou': mean_iou, 'class_dices': class_dices}


def main():
    parser = argparse.ArgumentParser(description="EXP_12: Lightweight BCRNet Evaluation")
    parser.add_argument('--config', default=os.path.join(os.path.dirname(__file__), "../configs/bcrnet_l3d.yaml"))
    parser.add_argument('--model_path', required=True, help="Path to checkpoint (.pt)")
    parser.add_argument('--data_path', default="/kaggle/working/L3D" if os.path.exists("/kaggle") else os.path.join(ws_root, "data/L3D"))
    parser.add_argument('--split', default='Test', choices=['Val', 'Test', 'both'])
    parser.add_argument('--model_mode', default='train', choices=['train', 'eval'], help="Model mode ('train' matches BCRNet test.py line 26; 'eval' for eval mode)")
    parser.add_argument('--threshold', type=float, default=0.3)
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Loading BCRNet model: {args.model_path} on {device}", flush=True)

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
        print("   Inference mode: model.train() (matching official repos/BCRNet/test.py line 26)")
    else:
        model.eval()
        print("   Inference mode: model.eval() (standard evaluation)")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    splits_to_eval = ['Val', 'Test'] if args.split == 'both' else [args.split]
    for sp in splits_to_eval:
        evaluate_single_split(model, sp, args.data_path, device)


if __name__ == '__main__':
    main()
