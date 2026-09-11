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
BezierDataset.load_bezier_gt = _safe_load_bezier_gt


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


def main():
    parser = argparse.ArgumentParser(description="EXP_12: Lightweight BCRNet Evaluation")
    parser.add_argument('--config', default=os.path.join(os.path.dirname(__file__), "../configs/bcrnet_l3d.yaml"))
    parser.add_argument('--model_path', required=True, help="Path to checkpoint (.pt)")
    parser.add_argument('--data_path', default="/kaggle/working/L3D" if os.path.exists("/kaggle") else os.path.join(ws_root, "data/L3D"))
    parser.add_argument('--split', default='Test', choices=['Val', 'Test'])
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
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    split_dir = os.path.join(args.data_path, args.split.capitalize())
    if not os.path.exists(split_dir):
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    dataset = BezierDataset(split_dir, device=device)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fun)
    print(f"🔍 Evaluating split [{args.split.upper()}] ({len(dataset)} frames)...", flush=True)

    class_names = ['silhouette', 'ligament', 'ridge']
    ious = []
    dices = []
    class_dices = {c: [] for c in class_names}

    with torch.inference_mode():
        pbar = tqdm(loader, desc=f"Evaluating {args.split}")
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
    print(f"📊 BENCHMARK RESULTS FOR [{args.split.upper()}]:", flush=True)
    print(f"   • Mean DSC: {mean_dice:.2f}%  (Paper Target: 69.57%)", flush=True)
    print(f"   • Mean IoU: {mean_iou:.2f}%  (Paper Target: 54.16%)", flush=True)
    for c_name in class_names:
        c_mean = float(np.mean(class_dices[c_name])) * 100.0
        print(f"     - {c_name.capitalize()}: {c_mean:.2f}% DSC", flush=True)
    print("=" * 70 + "\n", flush=True)


if __name__ == '__main__':
    main()
