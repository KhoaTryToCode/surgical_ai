"""
Training Runner for EXP_12: BCRNet Paper Replication
====================================================
Faithful execution wrapper for BCRNet (MICCAI 2025):
- Dynamically imports TransformerPureDetector & BezierDataset from `repos/BCRNet/`
- Implements exact BCRNet dynamic loss scheduling: lambda(epoch) = 1 - sigmoid((epoch - 10) / 2)
- Supports W&B offline / online logging
- Tracks best validation Dice and saves periodic checkpoints
- Includes native PyTorch autograd fallback if CUDA extension is not compiled

Usage:
  python experiments/EXP_12_bcrnet_replication/scripts/train_bcrnet.py \
      --config experiments/EXP_12_bcrnet_replication/configs/bcrnet_l3d.yaml \
      --data_path /kaggle/working/L3D \
      --epochs 80 \
      --bs 2 \
      --save_path /kaggle/working/checkpoints/EXP_12_bcrnet_replication
"""

import os
import sys
import types
import glob
import argparse
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
from utils.config_utils import load_config
from utils.bezier_dataset import BezierDataset, collate_fun

# Memory leak prevention: ensure .npz files are properly closed
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


def cal_loss(loss_dict, loss_weight):
    loss = 0
    for key, value in loss_dict.items():
        if key in loss_weight:
            loss = loss + loss_weight[key] * value
        else:
            loss = loss + value
    return loss


def evaluation(pred, gt):
    smooth = 1e-5
    intersection = np.sum(pred * gt)
    dice = (2.0 * intersection + smooth) / (np.sum(pred) + np.sum(gt) + smooth)
    iou = dice / (2.0 - dice)
    return iou, dice


def metrix(results, targets):
    gt = np.stack([lm['landmark_mask'].cpu().numpy() for lm in targets[0]], -1).astype(np.uint8)
    pred = np.zeros(gt.shape, dtype=np.uint8)
    for m, lm in enumerate(results[0]):
        curves = lm['ctrl_points'].cpu().numpy()
        for curve_points in curves:
            for i in range(1, len(curve_points)):
                pt1 = (int(curve_points[i - 1][0]), int(curve_points[i - 1][1]))
                pt2 = (int(curve_points[i][0]), int(curve_points[i][1]))
                cv2.line(pred[:, :, m], pt1, pt2, 1, 30)
    iou, dice = evaluation(pred, gt)
    return iou, dice


def main(args):
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"🚀 Initializing BCRNet training on device: {device}")
    print(f"   Config:    {args.config}")
    print(f"   Data path: {args.data_path}")
    print(f"   Save path: {args.save_path}")
    print(f"   Epochs:    {args.epochs} | Batch size: {args.bs}")

    # Configure W&B
    use_wandb = False
    if args.wandb:
        try:
            import wandb
            if args.wandb_key:
                wandb.login(key=args.wandb_key)
            else:
                os.environ.setdefault("WANDB_MODE", "offline")
            wandb.init(project='landmark_bcrnet_replication', name=os.path.basename(args.save_path))
            use_wandb = True
            print("   ✅ W&B tracking initialized.")
        except Exception as e:
            print(f"   ⚠️ Could not initialize W&B ({e}). Proceeding without W&B.")

    # Datasets & Loaders
    train_dir = os.path.join(args.data_path, 'Train')
    val_dir = os.path.join(args.data_path, 'Val')

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Train directory not found: {train_dir}. Please run prepare_data.py first.")

    train_dataset = BezierDataset(train_dir, device=device)
    val_dataset = BezierDataset(val_dir, device=device)
    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, collate_fn=collate_fun)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fun)
    print(f"   📊 Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    # Model & Config
    cfg = load_config(args.config)
    cfg.MODEL.DEVICE = device
    model = TransformerPureDetector(cfg).to(device)

    start_epoch = 0
    best_dice = 0.0

    if args.model_path and os.path.exists(args.model_path):
        print(f"   🔄 Loading checkpoint: {args.model_path}")
        model_checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(model_checkpoint['model'])
        if 'epoch-' in args.model_path:
            start_epoch = int(args.model_path.split('-')[-1].split('.')[0])
        print(f"   Resuming from epoch: {start_epoch}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Sigmoid loss schedule: lambda(epoch) = 1 - sigmoid((epoch - 10) / 2)
    def sigmoid_weight(e):
        t = torch.tensor((float(e) - 10.0) / 2.0, device=device)
        return 1.0 - torch.sigmoid(t)

    for epoch in range(start_epoch + 1, args.epochs + 1):
        lam = sigmoid_weight(epoch)
        loss_weights = {
            'loss_ce_enc': 1.0 - lam,
            'loss_pos_enc': lam,
            'segmentation_loss': lam,
        }

        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        epoch_losses = []

        for batch_data in pbar:
            results = model(batch_data, return_loss=True)
            loss = cal_loss(results, loss_weights)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            epoch_losses.append(loss_val)
            pbar.set_postfix({'epoch': epoch, 'loss': f"{loss_val:.4f}"})
            del batch_data, results, loss

        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        if use_wandb:
            wandb.log({'epoch': epoch, 'train_loss': mean_loss, 'lambda': lam.item()})

        # Periodic Evaluation (default every 10 epochs or final epoch)
        if epoch % args.eval_period == 0 or epoch == args.epochs:
            ckpt_file = os.path.join(args.save_path, f"epoch-{epoch}.pt")
            torch.save({'model': model.state_dict(), 'epoch': epoch}, ckpt_file)
            print(f"\n💾 Saved checkpoint: {ckpt_file}")

            # Validation pass
            model.eval()
            with torch.no_grad():
                ious = []
                dices = []
                val_pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch}")
                for batch_data in val_pbar:
                    img, depth, sam_feature, targets, info = batch_data
                    results = model(batch_data)
                    iou, dice = metrix(results, targets)
                    ious.append(iou)
                    dices.append(dice)
                    val_pbar.set_postfix({'Val_epoch': epoch, 'Dice': f"{dice*100:.2f}%", 'IoU': f"{iou*100:.2f}%"})
                    del batch_data, results, img, depth, sam_feature, targets, info

                mean_val_dice = float(np.mean(dices)) if dices else 0.0
                mean_val_iou = float(np.mean(ious)) if ious else 0.0
                print(f"📊 [Epoch {epoch}] Val Mean Dice: {mean_val_dice*100:.2f}% | Val Mean IoU: {mean_val_iou*100:.2f}%")

                if use_wandb:
                    wandb.log({'epoch': epoch, 'val_dice': mean_val_dice, 'val_iou': mean_val_iou})

                if mean_val_dice > best_dice:
                    best_dice = mean_val_dice
                    best_file = os.path.join(args.save_path, "best_model.pt")
                    torch.save({'model': model.state_dict(), 'epoch': epoch, 'val_dice': best_dice}, best_file)
                    print(f"⭐ New best model saved ({best_dice*100:.2f}% DSC) -> {best_file}")

        # End of epoch garbage collection and CUDA cache release
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print(f"✅ Training completed! Best validation DSC: {best_dice*100:.2f}%")
    print(f"   Checkpoints stored in: {args.save_path}")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="EXP_12: BCRNet Training Runner")
    default_cfg = os.path.join(os.path.dirname(__file__), "../configs/bcrnet_l3d.yaml")
    default_data = "/kaggle/working/L3D" if os.path.exists("/kaggle") else os.path.join(ws_root, "data/L3D")
    default_save = "/kaggle/working/checkpoints/EXP_12_bcrnet_replication" if os.path.exists("/kaggle") else os.path.join(ws_root, "checkpoints/EXP_12_bcrnet_replication")

    parser.add_argument('--config', default=default_cfg)
    parser.add_argument('--data_path', default=default_data)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--bs', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--eval_period', type=int, default=10)
    parser.add_argument('--save_path', default=default_save)
    parser.add_argument('--model_path', default=None)
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--wandb', action='store_true', default=False)
    parser.add_argument('--wandb_key', default="")

    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    main(args=args)
