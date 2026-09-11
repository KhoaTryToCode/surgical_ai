"""
Comprehensive Evaluation Script for EXP_12: BCRNet Replication
==============================================================
Evaluates trained BCRNet model on both **Val** and **Test** splits.
Computes all benchmark metrics reported in the BCRNet paper:
  - DSC (%)  — Dice Similarity Coefficient
  - IoU (%)  — Intersection over Union
  - ASSD (px) — Average Symmetric Surface Distance

Usage:
  python experiments/EXP_12_bcrnet_replication/scripts/evaluate_bcrnet.py \
      --model_path /kaggle/working/checkpoints/EXP_12_bcrnet_replication/best_model.pt \
      --data_path /kaggle/working/L3D \
      --split Test \
      --save_path /kaggle/working/results/EXP_12_bcrnet_replication
"""

import os
import sys
import types
import glob
import argparse
import json
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ws_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
bcrnet_root = os.path.join(ws_root, "repos/BCRNet")
for p in [bcrnet_root, ws_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Check if compiled _C extension exists on disk
c_ext_files = glob.glob(os.path.join(bcrnet_root, "adet/_C*.so")) + glob.glob(os.path.join(bcrnet_root, "adet/_C*.pyd"))
if len(c_ext_files) == 0:
    print("⚠️ adet._C CUDA extension not found. Enabling native PyTorch autograd MSDeformAttn fallback...")
    adet_pkg = types.ModuleType('adet')
    adet_pkg.__path__ = [os.path.join(bcrnet_root, 'adet')]
    sys.modules['adet'] = adet_pkg

    mock_c = types.ModuleType('adet._C')
    mock_c.ms_deform_attn_forward = lambda *args, **kwargs: None
    mock_c.ms_deform_attn_backward = lambda *args, **kwargs: None
    adet_pkg._C = mock_c
    sys.modules['adet._C'] = mock_c

    from adet.layers.ms_deform_attn import MSDeformAttn, ms_deform_attn_core_pytorch

    def _fallback_forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(f"Last dim of reference_points must be 2 or 4, got {reference_points.shape[-1]}")

        output = ms_deform_attn_core_pytorch(value, input_spatial_shapes, sampling_locations, attention_weights)
        return self.output_proj(output)

    MSDeformAttn.forward = _fallback_forward

try:
    import fvcore
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "fvcore", "iopath"], check=True)

from adet.utils.curve_utils import BezierSampler, upcast

def _adaptive_get_sample_points(self, control_points_matrix):
    if control_points_matrix.numel() == 0:
        return control_points_matrix
    k = control_points_matrix.shape[-2]
    if self.bernstein_matrix.shape[1] != k:
        self.degree = k - 1
        self.bezier_coeff = self.get_bezier_coefficient()
        self.bernstein_matrix = self.get_bernstein_matrix()
    if self.bernstein_matrix.device != control_points_matrix.device:
        self.bernstein_matrix = self.bernstein_matrix.to(control_points_matrix.device)
    return upcast(self.bernstein_matrix).matmul(upcast(control_points_matrix))

BezierSampler.get_sample_points = _adaptive_get_sample_points

_orig_bezier_sampler_init = BezierSampler.__init__
def _patched_bezier_sampler_init(self, num_sample_points, degree=5):
    _orig_bezier_sampler_init(self, num_sample_points, degree=degree)
BezierSampler.__init__ = _patched_bezier_sampler_init

from adet.modeling.bezier_detection import TransformerPureDetector
from utils.bezier_dataset import BezierDataset, collate_fun

# Memory leak prevention: ensure .npz files are properly closed and masks kept on CPU as uint8
def _safe_load_bezier_gt(self, item_name):
    path = os.path.join(self.data_path, 's_bezier', item_name + '.npz')
    with np.load(path, allow_pickle=True) as bezier_data:
        landmark_list = ['silhouette', 'ligament', 'ridge']
        source = []
        for i, (label, data) in enumerate(zip(landmark_list, bezier_data['gt'])):
            ctrl_points = torch.from_numpy(data['ctrl_points']).float().to(self.device)
            curve_points = torch.from_numpy(data['curve_points']).float().to(self.device)
            # Store landmark mask as uint8 on CPU to prevent 25MB GPU allocation per frame
            landmark_mask = torch.from_numpy(data['landmark_mask']).to(torch.uint8)
            source.append({
                'label': label,
                'ctrl_points': ctrl_points,
                'class_id': torch.LongTensor([i]).to(self.device),
                'curve_points': curve_points,
                'landmark_mask': landmark_mask
            })
    return source

BezierDataset.load_bezier_gt = _safe_load_bezier_gt


def compute_assd(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Computes Average Symmetric Surface Distance (ASSD) in pixels.
    Uses medpy if available, otherwise exact distance-transform boundary method.
    """
    if np.sum(pred_mask) == 0 or np.sum(gt_mask) == 0:
        return np.nan

    try:
        from medpy.metric.binary import assd
        return float(assd(pred_mask, gt_mask))
    except Exception:
        from scipy.ndimage import distance_transform_edt
        kernel = np.ones((3, 3), np.uint8)
        pred_boundary = pred_mask & ~cv2.erode(pred_mask.astype(np.uint8), kernel, iterations=1)
        gt_boundary = gt_mask & ~cv2.erode(gt_mask.astype(np.uint8), kernel, iterations=1)

        if np.sum(pred_boundary) == 0 or np.sum(gt_boundary) == 0:
            return np.nan

        d_gt = distance_transform_edt(~gt_boundary)
        d_pred = distance_transform_edt(~pred_boundary)

        assd_val = (np.mean(d_gt[pred_boundary]) + np.mean(d_pred[gt_boundary])) / 2.0
        return float(assd_val)


def render_prediction_and_gt(results, targets):
    """
    Renders 30px thick landmark strokes in-place on C-contiguous 2D channel buffers.
    Ensures full OpenCV cv::Mat memory compatibility with zero extra allocations.
    """
    gt_list = []
    for lm in targets[0]:
        m = lm['landmark_mask']
        if torch.is_tensor(m):
            gt_list.append(m.cpu().numpy().astype(np.uint8))
        else:
            gt_list.append(m.astype(np.uint8))
    gt = np.stack(gt_list, -1)
    H, W = gt.shape[:2]

    pred_channels = []
    for m, lm in enumerate(results[0]):
        channel = np.zeros((H, W), dtype=np.uint8)
        if 'ctrl_points' in lm and lm['ctrl_points'].numel() > 0:
            curves = lm['ctrl_points'].detach().cpu().numpy()
            for curve_points in curves:
                for i in range(1, len(curve_points)):
                    pt1 = (int(curve_points[i - 1][0]), int(curve_points[i - 1][1]))
                    pt2 = (int(curve_points[i][0]), int(curve_points[i][1]))
                    cv2.line(channel, pt1, pt2, 1, 30)
        pred_channels.append(channel)
    pred = np.stack(pred_channels, axis=-1)
    return pred, gt


def get_mem_mb():
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def evaluate_split(model, dataset_dir, split_name, save_dir, device, save_images=False, do_assd=True):
    split_cap = split_name.capitalize()
    data_split_dir = os.path.join(dataset_dir, split_cap)
    if not os.path.exists(data_split_dir):
        print(f"⚠️ Directory for split '{split_name}' does not exist at {data_split_dir}. Skipping.", flush=True)
        return None

    dataset = BezierDataset(data_split_dir, device=device)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fun)

    class_names = ['silhouette', 'ligament', 'ridge']
    sample_metrics = []
    seg_save_dir = os.path.join(save_dir, split_cap, 'seg_results')
    if save_images:
        os.makedirs(seg_save_dir, exist_ok=True)

    print(f"\n" + "=" * 70, flush=True)
    print(f"🔍 EVALUATING BCRNET ON: [{split_cap.upper()}] ({len(dataset)} frames)", flush=True)
    print(f"   Initial Host RAM: {get_mem_mb():.1f} MB", flush=True)
    if torch.cuda.is_available():
        print(f"   Initial GPU VRAM: {torch.cuda.memory_allocated() / (1024 * 1024):.1f} MB", flush=True)
    print(f"=" * 70, flush=True)

    pbar = tqdm(loader, desc=f"Evaluating {split_cap}")
    for step_idx, batch_data in enumerate(pbar):
        img, depth, sam_feature, targets, info = batch_data
        item_name = info[0]['item_name']

        with torch.no_grad():
            results = model(batch_data)

        pred, gt = render_prediction_and_gt(results, targets)

        # Overall Dice & IoU
        smooth = 1e-5
        intersection = np.sum(pred * gt)
        sample_dice = float((2.0 * intersection + smooth) / (np.sum(pred) + np.sum(gt) + smooth))
        sample_iou = float(sample_dice / (2.0 - sample_dice))

        # Overall ASSD
        if do_assd:
            pred_flat = (np.sum(pred, axis=-1) > 0).astype(np.bool_)
            gt_flat = (np.sum(gt, axis=-1) > 0).astype(np.bool_)
            sample_assd = compute_assd(pred_flat, gt_flat)
        else:
            sample_assd = float('nan')

        # Per-class Dice
        class_dices = {}
        for c_idx, c_name in enumerate(class_names):
            p_c = pred[:, :, c_idx]
            g_c = gt[:, :, c_idx]
            inter_c = np.sum(p_c * g_c)
            d_c = (2.0 * inter_c + smooth) / (np.sum(p_c) + np.sum(g_c) + smooth)
            class_dices[f"{c_name}_dice"] = float(d_c)

        sample_record = {
            'item_name': item_name,
            'dice': sample_dice,
            'iou': sample_iou,
            'assd': float(sample_assd) if not np.isnan(sample_assd) else None,
            **class_dices,
        }
        sample_metrics.append(sample_record)

        # Save Visual Overlays only if requested
        if save_images:
            pred_bgr = np.stack([pred[:, :, 1], pred[:, :, 0], pred[:, :, 2]], -1) * 255
            gt_bgr = np.stack([gt[:, :, 1], gt[:, :, 0], gt[:, :, 2]], -1) * 255
            cv2.imwrite(os.path.join(seg_save_dir, f"{item_name}-{sample_dice:.3f}.png"), pred_bgr.astype(np.uint8))
            cv2.imwrite(os.path.join(seg_save_dir, f"{item_name}-gt.png"), gt_bgr.astype(np.uint8))

        pbar.set_postfix({'DSC': f"{sample_dice*100:.2f}%", 'IoU': f"{sample_iou*100:.2f}%"})
        del batch_data, results, pred, gt, img, depth, sam_feature, targets, info

        # Release system memory every 20 frames
        if (step_idx + 1) % 20 == 0:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            try:
                import ctypes
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception:
                pass

    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Aggregate metrics
    mean_dice = float(np.mean([s['dice'] for s in sample_metrics]))
    mean_iou = float(np.mean([s['iou'] for s in sample_metrics]))
    valid_assds = [s['assd'] for s in sample_metrics if s['assd'] is not None]
    mean_assd = float(np.mean(valid_assds)) if valid_assds else float('nan')

    summary = {
        'split': split_cap,
        'num_samples': len(sample_metrics),
        'mean_dice': mean_dice * 100.0,
        'mean_iou': mean_iou * 100.0,
        'mean_assd': mean_assd,
        'paper_target_dice': 69.57,
        'paper_target_iou': 54.16,
        'paper_target_assd': 43.55,
    }

    for c_name in class_names:
        c_mean = float(np.mean([s[f"{c_name}_dice"] for s in sample_metrics]))
        summary[f"{c_name}_dice"] = c_mean * 100.0

    print(f"\n📊 Summary for {split_cap}:", flush=True)
    print(f"   • Mean DSC:  {summary['mean_dice']:.2f}%  (Paper: 69.57%)", flush=True)
    print(f"   • Mean IoU:  {summary['mean_iou']:.2f}%  (Paper: 54.16%)", flush=True)
    if not np.isnan(mean_assd):
        print(f"   • Mean ASSD: {summary['mean_assd']:.2f} px (Paper: 43.55 px)", flush=True)
    for c_name in class_names:
        print(f"     - {c_name.capitalize()}: {summary[f'{c_name}_dice']:.2f}% DSC", flush=True)

    with open(os.path.join(save_dir, f"metrics_{split_cap.lower()}.json"), "w") as f:
        json.dump({'summary': summary, 'samples': sample_metrics}, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="EXP_12: Evaluate BCRNet Model")
    default_cfg = os.path.join(os.path.dirname(__file__), "../configs/bcrnet_l3d.yaml")
    default_data = "/kaggle/working/L3D" if os.path.exists("/kaggle") else os.path.join(ws_root, "data/L3D")
    default_save = "/kaggle/working/results/EXP_12_bcrnet_replication" if os.path.exists("/kaggle") else os.path.join(ws_root, "experiments/EXP_12_bcrnet_replication/results")

    parser.add_argument('--config', default=default_cfg)
    parser.add_argument('--model_path', required=True, help="Path to checkpoint (.pt)")
    parser.add_argument('--data_path', default=default_data)
    parser.add_argument('--split', default='Test', choices=['Val', 'Test', 'both'])
    parser.add_argument('--threshold', type=float, default=0.3, help="Proposal confidence threshold (Paper: 0.3, default config: 0.35)")
    parser.add_argument('--model_mode', default='eval', choices=['eval', 'train'], help="Model mode during inference (default 'eval' for stable stats; 'train' for test.py parity)")
    parser.add_argument('--no_assd', action='store_true', default=False, help="Skip ASSD computation for fastest evaluation")
    parser.add_argument('--save_images', action='store_true', default=False, help="Save PNG prediction overlays")
    parser.add_argument('--save_path', default=default_save)
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Loading BCRNet model from: {args.model_path}", flush=True)
    cfg = load_config(args.config)
    cfg.MODEL.DEVICE = device
    model = TransformerPureDetector(cfg).to(device)

    # Configure inference threshold (Paper: 0.3)
    model.test_score_threshold = args.threshold
    print(f"   Inference threshold: {model.test_score_threshold} (Paper: 0.3)", flush=True)

    checkpoint = torch.load(args.model_path, map_location=device)
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(state_dict)

    if args.model_mode == 'train':
        model.train()
        print("   Inference mode: model.train() (matching official repos/BCRNet/test.py line 26)", flush=True)
    else:
        model.eval()
        print("   Inference mode: model.eval() (standard stable evaluation)", flush=True)
    print("✅ Model loaded successfully.", flush=True)

    splits_to_eval = ['Val', 'Test'] if args.split == 'both' else [args.split]

    all_summaries = {}
    for sp in splits_to_eval:
        summary = evaluate_split(
            model=model,
            dataset_dir=args.data_path,
            split_name=sp,
            save_dir=args.save_path,
            device=device,
            save_images=args.save_images,
            do_assd=not args.no_assd
        )
        if summary:
            all_summaries[sp] = summary

    # Print comparative Markdown table
    print("\n" + "=" * 80, flush=True)
    print("📋 REPLICATION BENCHMARK SUMMARY TABLE:", flush=True)
    print("=" * 80, flush=True)
    print("| Split | DSC (%) | IoU (%) | ASSD (px) | Silhouette (%) | Ligament (%) | Ridge (%) |", flush=True)
    print("|:---|:---:|:---:|:---:|:---:|:---:|:---:|", flush=True)
    print(f"| **Paper Target (Test)** | **69.57** | **54.16** | **43.55** | -- | -- | -- |", flush=True)
    for sp, s in all_summaries.items():
        assd_str = f"{s['mean_assd']:.2f}" if not np.isnan(s['mean_assd']) else "N/A"
        print(f"| {sp} | {s['mean_dice']:.2f} | {s['mean_iou']:.2f} | {assd_str} | {s['silhouette_dice']:.2f} | {s['ligament_dice']:.2f} | {s['ridge_dice']:.2f} |", flush=True)
    print("=" * 80 + "\n", flush=True)


if __name__ == '__main__':
    main()
