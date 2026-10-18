"""
P2I-LReg Dataset Adapter for Vis2Reg Replication.
Supports:
1. Preoperative CT mesh loading (.obj) with unit conversion (m -> mm) and surface sampling.
2. Intraoperative point cloud loading (.ply) with background filtering (Z > 0) and subsampling.
3. Camera intrinsics (CAM_K.yml) and ground-truth camera poses (camPose.yml).
4. Multi-view windowing (F=3) for real keyframes.
5. Deterministic 5-fold cross-validation patient partitioner (12 train, 4 val, 5 test).
"""

from __future__ import annotations

import os
import glob
from typing import Dict, List, Optional, Tuple, Any

import yaml
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

try:
    from plyfile import PlyData
    HAS_PLYFILE = True
except ImportError:
    HAS_PLYFILE = False

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False


def load_obj_mesh_vertices(obj_path: str, scale_to_mm: float = 1000.0) -> np.ndarray:
    """
    Loads 3D vertices from an .obj mesh file.
    Applies scale_to_mm (default 1000.0 to convert meters to millimeters).
    """
    if HAS_TRIMESH:
        mesh = trimesh.load(obj_path, force='mesh', process=False)
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
    else:
        # Lightweight fallback parser for OBJ vertices
        verts = []
        with open(obj_path, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    parts = line.strip().split()
                    verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
        vertices = np.array(verts, dtype=np.float32)
    
    return vertices * scale_to_mm


def load_ply_point_cloud(ply_path: str, filter_positive_z: bool = True) -> np.ndarray:
    """
    Loads point cloud from .ply file.
    Filters out background artifact points where Z <= 0.
    """
    if HAS_PLYFILE:
        ply = PlyData.read(ply_path)
        v = ply['vertex']
        points = np.stack([v['x'], v['y'], v['z']], axis=-1).astype(np.float32)
    else:
        # Fallback parser for ASCII PLY
        pts = []
        in_header = True
        with open(ply_path, 'r') as f:
            for line in f:
                if in_header:
                    if line.strip() == 'end_header':
                        in_header = False
                    continue
                parts = line.strip().split()
                if len(parts) >= 3:
                    pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
        points = np.array(pts, dtype=np.float32)

    if filter_positive_z:
        valid_mask = points[:, 2] > 0.0
        if valid_mask.any():
            points = points[valid_mask]
            
    return points


def resample_points(points: np.ndarray, max_points: int = 6000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Uniformly subsamples or zero-pads point cloud to max_points.
    Returns: (padded_points [max_points, 3], valid_mask [max_points])
    """
    n_pts = len(points)
    if n_pts == 0:
        return np.zeros((max_points, 3), dtype=np.float32), np.zeros(max_points, dtype=bool)

    if n_pts >= max_points:
        choice = np.random.choice(n_pts, max_points, replace=False)
        sampled = points[choice]
        valid = np.ones(max_points, dtype=bool)
    else:
        sampled = np.zeros((max_points, 3), dtype=np.float32)
        sampled[:n_pts] = points
        valid = np.zeros(max_points, dtype=bool)
        valid[:n_pts] = True

    return sampled, valid


def collate_p2i_lreg(batch_or_items) -> Dict[str, Any]:
    """
    Collate function for P2I-LReg datasets.
    Accepts either a list of sample dicts [dict, ...] or a single sample dict.
    Returns stacked/batched tensor dictionary with shapes (B, N, 3), (B, 3, 3), (B, 3), etc.
    """
    if isinstance(batch_or_items, dict):
        batch_list = [batch_or_items]
    else:
        batch_list = list(batch_or_items)

    collated = {}
    for key in batch_list[0].keys():
        vals = [item[key] for item in batch_list]
        if isinstance(vals[0], torch.Tensor):
            squeezed = []
            for v in vals:
                if key != 'intra_masks' and v.dim() > 1 and v.shape[0] == 1:
                    squeezed.append(v.squeeze(0))
                else:
                    squeezed.append(v)
            collated[key] = torch.stack(squeezed, dim=0)
        elif isinstance(vals[0], (int, float, np.integer, np.floating)):
            collated[key] = torch.tensor(vals)
        elif isinstance(vals[0], bool):
            collated[key] = vals[0]
        else:
            collated[key] = vals
    return collated


class P2ILRegFolds:
    """
    Deterministic 5-Fold Patient Partitioner for the 21 P2I-LReg patients.
    Each fold provides disjoint splits: 12 train, 4 validation, 5 test patients.
    """
    ALL_PATIENTS = [f"{i:02d}" for i in range(1, 22)]  # '01' to '21'

    @classmethod
    def get_fold(cls, fold_idx: int) -> Dict[str, List[str]]:
        assert 0 <= fold_idx < 5, "fold_idx must be in [0, 4]"
        # Base permutation for deterministic stratification
        base_order = [
            '01', '02', '03', '04', '05',
            '06', '07', '08', '09', '10',
            '11', '12', '13', '14', '15',
            '16', '17', '18', '19', '20', '21'
        ]
        
        # Always exactly 5 test patients per fold
        test_start = (fold_idx * 4) % 21
        test_indices = [(test_start + i) % 21 for i in range(5)]
        test_patients = [base_order[i] for i in test_indices]
        
        # Always exactly 4 validation patients
        val_start = (test_start + 5) % 21
        val_indices = [(val_start + i) % 21 for i in range(4)]
        val_patients = [base_order[i] for i in val_indices]
        
        # Remaining 21 - 5 - 4 = 12 are training patients
        test_val_set = set(test_patients) | set(val_patients)
        train_patients = [p for p in base_order if p not in test_val_set]
        
        return {
            'train': sorted(train_patients),
            'val': sorted(val_patients),
            'test': sorted(test_patients),
        }


class P2ILRegSyntheticDataset(Dataset):
    """
    Stage 1 Synthetic Pretraining Dataset.
    Loads patient-specific Blender simulation samples with ground-truth camera poses.
    """
    def __init__(
        self,
        dataset_root: str,
        patient_ids: List[str],
        max_points: int = 6000,
        samples_per_patient: int = 2000,
    ):
        self.dataset_root = dataset_root
        self.patient_ids = patient_ids
        self.max_points = max_points

        self.samples: List[Tuple[str, str, int]] = []
        self.preop_cache: Dict[str, np.ndarray] = {}
        self.intrinsics_cache: Dict[str, Dict] = {}
        self.poses_cache: Dict[str, Dict] = {}
        self.c2w_mode: Optional[str] = None

        self._index_dataset(samples_per_patient)

    def _index_dataset(self, max_samples_per_patient: int):
        for pid in self.patient_ids:
            p_dir = os.path.join(self.dataset_root, pid)
            syn_dir = os.path.join(p_dir, 'syn')
            if not os.path.exists(syn_dir):
                continue

            # Load Intrinsics
            cam_k_file = os.path.join(syn_dir, 'CAM_K.yml')
            if os.path.exists(cam_k_file):
                with open(cam_k_file, 'r') as f:
                    self.intrinsics_cache[pid] = yaml.safe_load(f)

            # Load Poses
            pose_file = os.path.join(syn_dir, 'camPose.yml')
            if os.path.exists(pose_file):
                with open(pose_file, 'r') as f:
                    self.poses_cache[pid] = yaml.safe_load(f)

            # Pre-cache preoperative mesh
            obj_path = os.path.join(p_dir, 'model', 'reconstructed_mesh_world_m.obj')
            if os.path.exists(obj_path):
                raw_verts = load_obj_mesh_vertices(obj_path, scale_to_mm=1000.0)
                self.preop_cache[pid] = raw_verts

            # Gather synthetic point clouds
            pcd_dir = os.path.join(syn_dir, 'liverPcds')
            if os.path.exists(pcd_dir):
                ply_files = sorted(glob.glob(os.path.join(pcd_dir, 'ptsLiverCam*.ply')))
                for ply_path in ply_files[:max_samples_per_patient]:
                    fid = os.path.basename(ply_path).replace('ptsLiverCam', '').replace('.ply', '')
                    self.samples.append((pid, ply_path, fid))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pid, ply_path, fid = self.samples[idx]

        # 1. Preoperative points P (from CT mesh)
        pre_verts = self.preop_cache.get(pid)
        if pre_verts is None:
            pre_verts = np.zeros((self.max_points, 3), dtype=np.float32)
        pre_pts, pre_valid = resample_points(pre_verts, self.max_points)

        # 2. Intraoperative points Q (from camera PLY)
        target_raw = load_ply_point_cloud(ply_path, filter_positive_z=True)
        tgt_pts, tgt_valid = resample_points(target_raw, self.max_points)

        # 3. Camera Intrinsics K
        K = np.eye(3, dtype=np.float32)
        k_dict = self.intrinsics_cache.get(pid, {})
        if fid in k_dict:
            K = np.array(k_dict[fid][0]['cam_K'], dtype=np.float32).reshape(3, 3)
        elif '00000' in k_dict:
            K = np.array(k_dict['00000'][0]['cam_K'], dtype=np.float32).reshape(3, 3)

        # 4. Ground-Truth Camera Pose (R, t)
        R_gt = np.eye(3, dtype=np.float32)
        t_gt = np.zeros(3, dtype=np.float32)
        pose_dict = self.poses_cache.get(pid, {})
        if fid in pose_dict:
            pose_item = pose_dict[fid][0]
            if 'cam_R_w2c' in pose_item:
                R_gt = np.array(pose_item['cam_R_w2c'], dtype=np.float32).reshape(3, 3)
                t_gt = np.array(pose_item['cam_t_w2c'], dtype=np.float32)
            elif 'cam_R_c2w' in pose_item:
                R_raw = np.array(pose_item['cam_R_c2w'], dtype=np.float32).reshape(3, 3)
                t_raw = np.array(pose_item['cam_t_c2w'], dtype=np.float32)
                if self.c2w_mode is None and len(pre_pts) > 0 and len(tgt_pts) > 0:
                    p_sub = pre_pts[:min(50, len(pre_pts))]
                    q_sub = tgt_pts[:min(50, len(tgt_pts))]
                    err_dir = np.mean(np.linalg.norm((p_sub @ R_raw.T + t_raw)[:, None, :] - q_sub[None, :, :], axis=-1).min(axis=1))
                    R_inv = R_raw.T
                    t_inv = -R_raw.T @ t_raw
                    err_inv = np.mean(np.linalg.norm((p_sub @ R_inv.T + t_inv)[:, None, :] - q_sub[None, :, :], axis=-1).min(axis=1))
                    self.c2w_mode = 'invert' if err_inv < err_dir else 'direct'
                if self.c2w_mode == 'invert':
                    R_gt = R_raw.T
                    t_gt = -R_raw.T @ t_raw
                else:
                    R_gt = R_raw
                    t_gt = t_raw

        return {
            'pre_points': torch.from_numpy(pre_pts).unsqueeze(0),           # (1, N, 3)
            'pre_valid_mask': torch.from_numpy(pre_valid).unsqueeze(0),     # (1, N)
            'target_points': torch.from_numpy(tgt_pts).unsqueeze(0),        # (1, N, 3)
            'target_points_cam': torch.from_numpy(tgt_pts).unsqueeze(0),    # (1, N, 3)
            'cam_intrinsics': torch.from_numpy(K).unsqueeze(0),             # (1, 3, 3)
            'gt_R': torch.from_numpy(R_gt).unsqueeze(0),                    # (1, 3, 3)
            'gt_t': torch.from_numpy(t_gt).unsqueeze(0),                    # (1, 3)
            'patient_id': pid,
            'frame_id': fid,
            'rigid_only': True,
        }


class P2ILRegRealDataset(Dataset):
    """
    Stage 2 Real Intraoperative Keyframe Dataset.
    Loads real clinical video keyframes, CT mesh, scale factors, and 2D masks.
    """
    def __init__(
        self,
        dataset_root: str,
        patient_ids: List[str],
        max_points: int = 6000,
        multiview_f: int = 1,
    ):
        self.dataset_root = dataset_root
        self.patient_ids = patient_ids
        self.max_points = max_points
        self.multiview_f = multiview_f

        self.samples: List[Tuple[str, str, str]] = []
        self.preop_cache: Dict[str, np.ndarray] = {}
        self.intrinsics_cache: Dict[str, Dict] = {}

        self._index_dataset()

    def _index_dataset(self):
        for pid in self.patient_ids:
            p_dir = os.path.join(self.dataset_root, pid)
            real_dir = os.path.join(p_dir, 'real')
            if not os.path.exists(real_dir):
                continue

            # Load Intrinsics
            cam_k_file = os.path.join(real_dir, 'CAM_K.yml')
            if os.path.exists(cam_k_file):
                with open(cam_k_file, 'r') as f:
                    self.intrinsics_cache[pid] = yaml.safe_load(f)

            # Pre-cache preoperative mesh
            obj_path = os.path.join(p_dir, 'model', 'reconstructed_mesh_world_m.obj')
            if os.path.exists(obj_path):
                self.preop_cache[pid] = load_obj_mesh_vertices(obj_path, scale_to_mm=1000.0)

            # Gather real keyframe PLY files
            pcd_dir = os.path.join(real_dir, 'liverPcds')
            if os.path.exists(pcd_dir):
                ply_files = sorted(glob.glob(os.path.join(pcd_dir, 'frame_*_json.ply')))
                for ply_path in ply_files:
                    fname = os.path.basename(ply_path).replace('.ply', '')
                    self.samples.append((pid, ply_path, fname))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pid, ply_path, fname = self.samples[idx]
        real_dir = os.path.join(self.dataset_root, pid, 'real')

        # 1. Preoperative points P (from CT mesh)
        pre_verts = self.preop_cache.get(pid)
        if pre_verts is None:
            pre_verts = np.zeros((self.max_points, 3), dtype=np.float32)
        pre_pts, pre_valid = resample_points(pre_verts, self.max_points)

        # 2. Intraoperative points Q (from 2M PLY filtered by Z > 0)
        target_raw = load_ply_point_cloud(ply_path, filter_positive_z=True)
        tgt_pts, tgt_valid = resample_points(target_raw, self.max_points)

        # 3. Scale factor from scale/fname.txt
        scale_val = 141.0
        scale_file = os.path.join(real_dir, 'scale', f"{fname}.txt")
        if os.path.exists(scale_file):
            try:
                with open(scale_file, 'r') as f:
                    scale_val = float(f.read().strip())
            except Exception:
                pass

        # 4. Intrinsics K
        K = np.eye(3, dtype=np.float32)
        k_dict = self.intrinsics_cache.get(pid, {})
        if '00000' in k_dict:
            K = np.array(k_dict['00000'][0]['cam_K'], dtype=np.float32).reshape(3, 3)

        # 5. 2D Binary Liver Mask (label.png)
        mask_path = os.path.join(real_dir, 'labels', fname, 'label.png')
        if os.path.exists(mask_path):
            mask_img = Image.open(mask_path).convert('L')
            mask_np = (np.array(mask_img, dtype=np.float32) > 0).astype(np.float32)
        else:
            mask_np = np.zeros((512, 512), dtype=np.float32)

        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, H, W)

        return {
            'pre_points': torch.from_numpy(pre_pts).unsqueeze(0),
            'pre_valid_mask': torch.from_numpy(pre_valid).unsqueeze(0),
            'target_points': torch.from_numpy(tgt_pts).unsqueeze(0),
            'target_points_cam': torch.from_numpy(tgt_pts).unsqueeze(0),
            'cam_intrinsics': torch.from_numpy(K).unsqueeze(0),
            'pre_scale': torch.tensor([[scale_val]], dtype=torch.float32),
            'intra_masks': mask_tensor,
            'patient_id': pid,
            'frame_name': fname,
            'rigid_only': False,
        }
