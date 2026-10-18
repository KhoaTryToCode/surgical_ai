from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from .encoders.geometric_encoder import GeometricEncoder
from .encoders.visual_encoder import VisualEncoder
from .fusion.cross_modal_fusion import CrossModalFusion
from .fields.deformation_field import ImplicitDeformationField
from .rigid.rigid_pose_net import RigidPoseNet
from .matching.match_head import MatchHead
from .matching.match_head_ot import MatchHeadOT
from .matching.geotransformer_matcher import GeoTransformerMatcher
from .rigid.weighted_svd_solver import WeightedSVDSolver
from .rigid.ransac_icp import robust_pose_from_soft_matches
from .local_matcher import local_matcher_forward
from .renderer.point_renderer import PointsSilhouetteRenderer
from vis2reg.utils.geometry import apply_transform
from vis2reg.utils.logger import get_logger

class Vis2Reg(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        model_cfg = getattr(cfg, 'model', {})
        geo_cfg = model_cfg.get('geometric_encoder', {})
        vis_cfg = model_cfg.get('visual_encoder', {})
        fusion_cfg = model_cfg.get('fusion', {})
        field_cfg = model_cfg.get('deformation_field', {})
        pose_cfg = model_cfg.get('pose_net', {})
        renderer_cfg = model_cfg.get('renderer', None)
        data_cfg = getattr(cfg, 'data', {})
        self.dataset_name = data_cfg.get('dataset', '')

        geo_feature_dim = geo_cfg.get('feature_dim', 256)
        self.geometric_encoder = GeometricEncoder(
            input_dim=geo_cfg.get('input_dim'),
            feature_dim=geo_feature_dim,
            layers=geo_cfg.get('layers', 4),
            attr_dim=geo_cfg.get('attr_dim', 6),
            hidden_dim=geo_cfg.get('hidden_dim', 256),
            heads=geo_cfg.get('heads', 4),
            coarse_points=geo_cfg.get('coarse_points', 256),
            knn_k=geo_cfg.get('knn_k', 20),
            pool_k=geo_cfg.get('pool_k', 32),
            norm_groups=geo_cfg.get('norm_groups', 8),
        )
        self.visual_enabled = bool(vis_cfg.get('enabled', True))
        self.fusion_enabled = bool(fusion_cfg.get('enabled', True)) and self.visual_enabled
        self.visual_encoder = None
        self.visual_proj = None
        self.fusion = None
        self.render_condition_proj = None
        if self.visual_enabled:
            self.visual_encoder = VisualEncoder(
                backbone=vis_cfg.get('backbone', 'resnet50'),
                pretrained=vis_cfg.get('pretrained', True),
                feature_dim=vis_cfg.get('feature_dim', 256),
                num_frames=vis_cfg.get('num_frames', 1),
                temporal_layers=vis_cfg.get('temporal_layers', 2),
                aux_channels=vis_cfg.get('aux_channels', 0),
                temporal_heads=vis_cfg.get('temporal_heads', 4),
            )
            self.freeze_visual_encoder = bool(vis_cfg.get('freeze', False))
        else:
            self.freeze_visual_encoder = False
        fusion_dim = geo_feature_dim
        if self.fusion_enabled:
            self.fusion = CrossModalFusion(
                dim=fusion_dim,
                heads=fusion_cfg.get('heads', 4),
                layers=fusion_cfg.get('layers', 2),
                dropout=fusion_cfg.get('dropout', 0.1),
                temporal_layers=fusion_cfg.get('temporal_layers', 2),
            )
        if self.visual_enabled:
            self.visual_proj = nn.Linear(vis_cfg.get('feature_dim', 256), fusion_dim)
            self.render_condition_proj = nn.Sequential(
                nn.Linear(4, fusion_dim),
                nn.GELU(),
                nn.LayerNorm(fusion_dim),
            )
        self.deformation_field = ImplicitDeformationField(
            feature_dim=fusion_dim,
            hidden_dim=field_cfg.get('hidden_dim', 256),
            num_layers=field_cfg.get('num_layers', 5),
            pe_frequencies=field_cfg.get('pe_frequencies', 8),
            skip_connections=field_cfg.get('skip_connections', [])
        )
        fused_global_dim = geo_cfg.get('feature_dim', 256) + (fusion_dim if self.visual_enabled else 0)
        self.rigid_pose_net = RigidPoseNet(fused_global_dim, pose_cfg)
        self.point_renderer = None
        self.real_flip_enabled = bool(data_cfg.get('real_ocv2renderer', True))
        self.flip_vector = torch.tensor([1.0, -1.0, -1.0])
        self.use_decoupled_rigid = bool(model_cfg.get('use_decoupled_rigid', False))
        rigid_cfg = model_cfg.get('rigid', {})
        self.use_matching_solver = bool(rigid_cfg.get('use_matching_solver', False))
        self.use_geom_feature_v1 = rigid_cfg.get('use_geom_feature_v1', False)
        self.use_target_geom_v2 = bool(rigid_cfg.get('use_target_geom_v2', False))
        self.use_region_vessel_embedding = bool(rigid_cfg.get('use_region_vessel_embedding', False))
        self.region_emb_dim = int(rigid_cfg.get('region_emb_dim', 8))
        self.vessel_emb_dim = int(rigid_cfg.get('vessel_emb_dim', 8))
        self.num_regions_cfg = int(rigid_cfg.get('num_regions', 16))
        self.num_vessels_cfg = int(rigid_cfg.get('num_vessels', 16))
        self.match_row_topk = int(rigid_cfg.get('match_row_topk', 0))
        self.match_geom_window_mm = float(rigid_cfg.get('match_geom_window_mm', 0.0))
        # Local matcher: geometry filtering + sparse K-way softmax
        self.match_use_local = bool(rigid_cfg.get('match_use_local', False))
        self.match_use_ot = bool(rigid_cfg.get('match_use_ot', False))
        self.match_ot_coarse_only = bool(rigid_cfg.get('match_ot_coarse_only', False))
        self.use_geo_transformer_matcher = bool(rigid_cfg.get('use_geo_transformer_matcher', False))
        self.match_local_k = int(rigid_cfg.get('match_local_k', 64))
        self.match_local_radius_mm = float(rigid_cfg.get('match_local_radius_mm', 15.0))
        self.use_ransac_solver = bool(
            rigid_cfg.get('use_ransac_solver', rigid_cfg.get('pairs_gnn_ransac_enable', False))
        )
        self.use_icp_refine = bool(
            rigid_cfg.get('use_icp_refine', rigid_cfg.get('pairs_gnn_ransac_icp_enable', False))
        )
        self.ransac_topk = int(rigid_cfg.get('ransac_topk', 512))
        self.ransac_hypotheses = int(rigid_cfg.get('ransac_hypotheses', 128))
        self.ransac_inlier_threshold = float(rigid_cfg.get('ransac_inlier_threshold', 0.01))
        self.icp_iters = int(rigid_cfg.get('icp_iters', 10))
        self.icp_threshold = float(rigid_cfg.get('icp_threshold', self.ransac_inlier_threshold))
        self.match_temperature = float(model_cfg.get('match_temperature', rigid_cfg.get('match_temperature', 0.1)))
        self.use_camera_frame = bool(rigid_cfg.get('use_camera_frame', True))
        self.svd_conf_filter_enabled = bool(rigid_cfg.get('svd_conf_filter_enabled', False))
        self.svd_conf_threshold = float(rigid_cfg.get('svd_conf_threshold', 0.0))
        self.svd_conf_min_points = int(rigid_cfg.get('svd_conf_min_points', 3))
        self.svd_conf_fallback_topk = int(rigid_cfg.get('svd_conf_fallback_topk', 32))
        attr_dim_cfg = geo_cfg.get('attr_dim', 6)
        self.local_geom_dim_cfg = geo_cfg.get('local_geom_dim', 16)
        if self.use_geom_feature_v1:
            self.match_attr_dim = 6
            self.local_geom_dim = self.local_geom_dim_cfg
        else:
            self.match_attr_dim = attr_dim_cfg
            self.local_geom_dim = max(0, self.match_attr_dim - 6)
        if self.use_region_vessel_embedding:
            self.match_attr_dim = max(0, self.match_attr_dim - 2)
            self.region_emb = nn.Embedding(max(1, self.num_regions_cfg), self.region_emb_dim)
            self.vessel_emb = nn.Embedding(max(1, self.num_vessels_cfg), self.vessel_emb_dim)
            self.regves_proj = nn.Linear(fusion_dim + self.region_emb_dim + self.vessel_emb_dim, fusion_dim)
        else:
            self.region_emb = None
            self.vessel_emb = None
            self.regves_proj = None
        self.match_pre_proj = nn.Linear(fusion_dim + 3 + self.match_attr_dim, fusion_dim)
        self.match_tgt_proj = nn.Linear(fusion_dim + 3 + self.match_attr_dim, fusion_dim)
        self.match_pre_mlp = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.LayerNorm(fusion_dim),
        )
        self.match_tgt_mlp = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.LayerNorm(fusion_dim),
        )
        self.match_pre_norm = nn.LayerNorm(fusion_dim)
        self.match_tgt_norm = nn.LayerNorm(fusion_dim)
        self.tgt_global_proj = nn.Linear(fusion_dim, fusion_dim) if self.visual_enabled else None
        self.match_geom_encoder = GeometricEncoder(
            input_dim=None if geo_cfg.get('input_dim', None) is None else 3 + self.match_attr_dim,
            feature_dim=fusion_dim,
            layers=geo_cfg.get('layers', 4),
            attr_dim=self.match_attr_dim,
            hidden_dim=geo_cfg.get('hidden_dim', 256),
            heads=geo_cfg.get('heads', 4),
            knn_k=geo_cfg.get('knn_k', 20),
            pool_k=geo_cfg.get('pool_k', 32),
            norm_groups=geo_cfg.get('norm_groups', 8),
        )
        # local geom injection branch (16-dim default) -> fusion_dim, added to geom features
        if self.local_geom_dim > 0:
            self.local_geom_mlp = nn.Sequential(
                nn.Linear(self.local_geom_dim, fusion_dim),
                nn.GELU(),
                nn.LayerNorm(fusion_dim),
            )
        else:
            self.local_geom_mlp = None
        # target 侧密度/曲率正规化 + 小 MLP（可选）
        if self.use_target_geom_v2:
            self.target_geom_mlp = nn.Sequential(
                nn.Linear(2, fusion_dim),
                nn.GELU(),
                nn.LayerNorm(fusion_dim),
            )
        else:
            self.target_geom_mlp = None
        # coarse-to-fine matching scaffolding
        self.match_head_coarse = MatchHead(
            fusion_dim,
            heads=fusion_cfg.get('heads', 4),
            dropout=fusion_cfg.get('dropout', 0.0),
            temperature=self.match_temperature,
        )
        self.match_head = MatchHead(
            fusion_dim,
            heads=fusion_cfg.get('heads', 4),
            dropout=fusion_cfg.get('dropout', 0.0),
            temperature=self.match_temperature,
        )
        self.match_head_ot = MatchHeadOT(
            fusion_dim,
            heads=fusion_cfg.get('heads', 4),
            dropout=fusion_cfg.get('dropout', 0.0),
            temperature=self.match_temperature,
            sinkhorn_iters=rigid_cfg.get('sinkhorn_iters', 20),
            dustbin_score=rigid_cfg.get('dustbin_score', 1.0),
        )
        self.geo_matcher = GeoTransformerMatcher(
            fusion_dim,
            heads=fusion_cfg.get('heads', 4),
            dropout=fusion_cfg.get('dropout', 0.0),
            temperature=self.match_temperature,
            coarse_points=rigid_cfg.get('geo_coarse_points', rigid_cfg.get('match_coarse_points', 512)),
            sinkhorn_iters=rigid_cfg.get('sinkhorn_iters', 20),
            coarse_bias_weight=rigid_cfg.get('geo_coarse_bias_weight', 1.0),
        )
        logit_scale_init = float(rigid_cfg.get('match_logit_scale_init', 1.0))
        self.match_logit_scale = nn.Parameter(torch.tensor(logit_scale_init))
        local_logit_scale_init = float(rigid_cfg.get('local_logit_scale_init', 2.0))
        self.local_logit_scale_log = nn.Parameter(torch.log(torch.tensor(local_logit_scale_init)))
        self.match_coarse_points = int(rigid_cfg.get('match_coarse_points', 1024))
        self.rigid_solver = WeightedSVDSolver()
        self.logger = get_logger('Vis2Reg')
        self.logger.info('Decoupled rigid mode: %s', 'ON' if self.use_decoupled_rigid else 'OFF')
        if self.svd_conf_filter_enabled:
            self.logger.info(
                'SVD confidence filter: enabled=True | thr=%.3f | min_points=%d | fallback_topk=%d',
                self.svd_conf_threshold,
                self.svd_conf_min_points,
                self.svd_conf_fallback_topk,
            )
        if self.match_use_local:
            self.logger.info(
                "Local logit scale | init_scale=%.4f (log=%.4f)",
                float(local_logit_scale_init),
                float(torch.log(torch.tensor(local_logit_scale_init)).item()),
            )
        if self.use_matching_solver:
            self.logger.info('Rigid matching solver enabled')
        if renderer_cfg:
            renderer_type = renderer_cfg.get('type', 'points')
            if renderer_type == 'points':
                self.point_renderer = PointsSilhouetteRenderer(
                    image_size=renderer_cfg.get('image_size', (512, 512)),
                    radius=renderer_cfg.get('radius', 0.01),
                    points_per_pixel=renderer_cfg.get('points_per_pixel', 16),
                )

    @staticmethod
    def _denormalize_points(points: torch.Tensor, centroid: Optional[torch.Tensor], scale: Optional[torch.Tensor]) -> torch.Tensor:
        if centroid is None or scale is None:
            return points
        centroid = centroid.to(points.device, dtype=points.dtype)
        scale = scale.to(points.device, dtype=points.dtype)
        if centroid.dim() == 1:
            centroid = centroid.view(1, 1, -1)
        elif centroid.dim() == 2:
            centroid = centroid.unsqueeze(1)
        if scale.dim() == 0:
            scale = scale.view(1)
        while scale.dim() < points.dim():
            scale = scale.unsqueeze(-1)
        return points * scale + centroid

    def forward(self, batch: Dict[str, torch.Tensor], render: bool = True) -> Dict[str, torch.Tensor]:
        pre_points = batch['pre_points']  # (B, N, 3)
        pre_attrs = batch.get('pre_point_attrs')
        geo = self.geometric_encoder(pre_points, pre_attrs, valid_mask=batch.get('pre_valid_mask', None))
        frame_feats = None
        temporal_feats = None
        render_contrast_feat = None
        render_condition = None
        structure_mats = []
        frame_weights = None
        fused_points = geo['point_features']
        fused_global = geo['global_feature']
        render_depths = None
        render_masks = None
        if self.visual_enabled and self.visual_encoder is not None and self.fusion_enabled and self.fusion is not None:
            render_depths = batch.get('render_depths')
            render_masks = batch.get('render_masks')
            disable_render = bool(batch.get('disable_render_condition', False))
            if self.dataset_name == 'synthetic':
                disable_render = True
            if disable_render:
                render_depths = None
                render_masks = None
            vis = self.visual_encoder(batch['intra_images'], render_depths, render_masks)
            frame_feats = vis['frame_features']
            temporal_feats = vis.get('temporal_features')
            render_contrast_feat = vis.get('render_contrast_feat')
            render_condition = self._build_render_condition(
                render_depths,
                render_masks,
                pre_points.shape[1],
                pre_points.device,
                disable=disable_render,
            )
            fused_points, structure_mats, frame_weights = self.fusion(
                geo['point_features'],
                frame_feats,
                render_condition=render_condition,
                temporal_feats=temporal_feats,
                render_contrast_feat=render_contrast_feat,
            )
            fused_global = torch.cat([geo['global_feature'], self.visual_proj(vis['global_features'])], dim=-1)
        pose = self._run_rigid_pose(fused_global)
        match_logits = None
        soft_match = None
        match_stats = {}
        pose_out = (pose, match_logits, soft_match, match_stats)
        if self.use_matching_solver and self._is_rigid_only(batch):
            pose_out = self._run_matching_rigid(
                batch,
                pre_points,
                batch.get('target_points'),
                geo['point_features'],
                frame_feats,
                geo_tgt_feats=geo['point_features'],  # pass tgt geometry feats
            )
        pose, match_logits, soft_match, match_stats = pose_out
        rigid_only = bool(batch.get('rigid_only', False))
        if rigid_only:
            delta = torch.zeros_like(pre_points)
            field_out = {
                'warped_points': apply_transform(pre_points, pose[0], pose[1], pose[2]),
                'delta': delta,
                'latent_features': fused_points,
                'regularization': delta.mean() * 0,
            }
        else:
            field_out = self._run_deformation(pre_points, fused_points, pose)
        centroid = batch.get('pre_centroid')
        scale = batch.get('pre_scale')
        warped_world = self._denormalize_points(field_out['warped_points'], centroid, scale)
        if self._apply_real_flip(batch):
            warped_world = self._flip_tensor(warped_world)

        render_masks_all, render_depths_all = self._render_point_views(warped_world, batch)

        outputs = {
            'warped_pcd': warped_world,
            'warped_pcd_normalized': field_out['warped_points'],
            'deformation': field_out['delta'],
            'latent_field': field_out['latent_features'],
            'structure_mats': structure_mats,
            'rigid_pose': pose,
            'regularization': field_out['regularization'],
            'frame_attention': frame_weights,
            'match_stats': match_stats,
        }
        extra_match_keys = [
            'match_eff_K_mean', 'match_eff_K_median', 'match_eff_K_min', 'match_eff_K_max',
            'match_logK_mean', 'match_prob_ot', 'geo_pre_idx', 'geo_tgt_idx',
            'ransac_inliers', 'ransac_residual', 'icp_inliers',
        ]
        extra_match_dict = {k: match_stats[k] for k in extra_match_keys if k in match_stats}
        if extra_match_dict:
            outputs['extra_match'] = extra_match_dict
        if render_masks_all is not None:
            render_masks = render_masks_all

        if render_masks is not None:
            outputs['render_mask'] = render_masks[:, 0]
        elif render and self.point_renderer is not None:
            image_hw = batch['intra_images'].shape[-2:]
            renders, _ = self.point_renderer.render_with_depth(
                warped_world,
                batch['cam_intrinsics'][:, 0],
                image_size=image_hw,
                return_depth=True,
            )
            outputs['render_mask'] = renders

        # Cycle consistency over remaining frames if available and not rigid-only
        if frame_feats is not None and frame_feats.size(1) > 1 and not rigid_only:
            cycle_masks = []
            for idx in range(1, frame_feats.size(1)):
                single_frame = frame_feats[:, idx : idx + 1]
                fused_i, _, _ = self.fusion(
                    geo['point_features'],
                    single_frame,
                    render_condition=render_condition,
                    temporal_feats=None,
                    render_contrast_feat=render_contrast_feat[:, idx:idx+1] if render_contrast_feat is not None else None,
                )
                field_i = self.deformation_field(pre_points, fused_i, pose)
                cycle_world = self._denormalize_points(field_i['warped_points'], centroid, scale)
                cycle_masks.append(cycle_world)
            outputs['cycle_predictions'] = cycle_masks

        return outputs

    def _apply_real_flip(self, batch: Dict[str, torch.Tensor]) -> bool:
        if not self.real_flip_enabled:
            return False
        flag = batch.get('real_ocv2renderer')
        if isinstance(flag, torch.Tensor):
            return bool(flag[0].item() > 0)
        if isinstance(flag, list) and flag:
            val = flag[0]
            if isinstance(val, torch.Tensor):
                return bool(val.item() > 0)
            return bool(val)
        return bool(flag)

    def _is_rigid_only(self, batch: Dict[str, torch.Tensor]) -> bool:
        return bool(batch.get('rigid_only', False))

    def _run_matching_rigid(
        self,
        batch: Dict[str, torch.Tensor],
        pre_points: torch.Tensor,
        target_points: Optional[torch.Tensor],
        pre_point_feats: torch.Tensor,
        tgt_frame_feats: torch.Tensor,
        geo_tgt_feats: Optional[torch.Tensor] = None,
    ):
        match_stats = {}
        tgt = None
        tgt_cam = None
        if target_points is not None and isinstance(target_points, torch.Tensor):
            if target_points.ndim == 3:
                tgt = target_points  # (B, N_tgt, 3) world
            elif target_points.ndim == 4:
                tgt = target_points[:, 0]  # first frame world
        # camera-frame target if available
        tgt_cam_batch = batch.get('target_points_cam')
        if tgt_cam_batch is not None and isinstance(tgt_cam_batch, torch.Tensor):
            if tgt_cam_batch.ndim == 3:
                tgt_cam = tgt_cam_batch
            elif tgt_cam_batch.ndim == 4:
                tgt_cam = tgt_cam_batch[:, 0]
        if tgt is None:
            pose_fallback = self._run_rigid_pose(torch.cat([pre_point_feats.mean(dim=1), tgt_frame_feats.mean(dim=1)], dim=-1))
            return pose_fallback, None, None, {}

        pre_cam_batch = batch.get('pre_points_cam')
        use_cam = self.use_camera_frame and pre_cam_batch is not None and tgt_cam is not None
        if use_cam:
            pre_coords = pre_cam_batch
            tgt_coords = tgt_cam
        else:
            pre_coords = pre_points
            tgt_coords = tgt

        # gather optional point attributes
        pre_attrs_full = batch.get('pre_point_attrs')
        if isinstance(pre_attrs_full, torch.Tensor):
            pre_attrs_full = pre_attrs_full.to(pre_points.device, dtype=pre_points.dtype)
        else:
            pre_attrs_full = torch.zeros(pre_points.shape[0], pre_points.shape[1], 0, device=pre_points.device, dtype=pre_points.dtype)
        tgt_attrs_full = batch.get('target_point_features')
        if isinstance(tgt_attrs_full, torch.Tensor):
            tgt_attrs_full = tgt_attrs_full[:, 0] if tgt_attrs_full.dim() == 4 else tgt_attrs_full
            tgt_attrs_full = tgt_attrs_full.to(pre_points.device, dtype=pre_points.dtype)
        else:
            tgt_attrs_full = torch.zeros(tgt_coords.shape[0], tgt_coords.shape[1], pre_attrs_full.shape[-1], device=pre_points.device, dtype=pre_points.dtype)

        # 基础 attrs（前 6 维）与 local_geom 分离，便于开关控制
        base_dim = min(pre_attrs_full.shape[-1], 6)
        pre_base_attr = pre_attrs_full[..., :base_dim]
        tgt_base_attr = tgt_attrs_full[..., :base_dim]
        if self.match_attr_dim > base_dim:
            pad = torch.zeros(pre_base_attr.shape[0], pre_base_attr.shape[1], self.match_attr_dim - base_dim, device=pre_base_attr.device, dtype=pre_base_attr.dtype)
            pre_attr_slice = torch.cat([pre_base_attr, pad], dim=-1)
            pad_t = torch.zeros(tgt_base_attr.shape[0], tgt_base_attr.shape[1], self.match_attr_dim - base_dim, device=tgt_base_attr.device, dtype=tgt_base_attr.dtype)
            tgt_attr_slice = torch.cat([tgt_base_attr, pad_t], dim=-1)
        else:
            pre_attr_slice = pre_base_attr[..., : self.match_attr_dim]
            tgt_attr_slice = tgt_base_attr[..., : self.match_attr_dim]

        # local_geom 单独抽取（不再平铺到 attrs，当 use_geom_feature_v1 开启时）
        pre_local_geom = None
        tgt_local_geom = None
        if self.local_geom_dim > 0:
            if pre_attrs_full.shape[-1] >= 6 + self.local_geom_dim:
                pre_local_geom = pre_attrs_full[..., 6 : 6 + self.local_geom_dim]
            if tgt_attrs_full.shape[-1] >= 6 + self.local_geom_dim:
                tgt_local_geom = tgt_attrs_full[..., 6 : 6 + self.local_geom_dim]
            # 兜底
            if pre_local_geom is None:
                pre_local_geom = torch.zeros(pre_points.shape[0], pre_points.shape[1], self.local_geom_dim, device=pre_points.device, dtype=pre_points.dtype)
            if tgt_local_geom is None:
                tgt_local_geom = torch.zeros(tgt_coords.shape[0], tgt_coords.shape[1], self.local_geom_dim, device=pre_points.device, dtype=pre_points.dtype)
        # 确保 attr 维度完全一致（防止 target 特征缺失导致长度偏差）
        if tgt_attr_slice.shape[-1] != pre_attr_slice.shape[-1]:
            if tgt_attr_slice.shape[-1] < pre_attr_slice.shape[-1]:
                pad = torch.zeros(tgt_attr_slice.shape[0], tgt_attr_slice.shape[1], pre_attr_slice.shape[-1] - tgt_attr_slice.shape[-1], device=tgt_attr_slice.device, dtype=tgt_attr_slice.dtype)
                tgt_attr_slice = torch.cat([tgt_attr_slice, pad], dim=-1)
            else:
                tgt_attr_slice = tgt_attr_slice[..., : pre_attr_slice.shape[-1]]

        # geometric match features (camera frame)
        pre_match_geo = self.match_geom_encoder(pre_coords, pre_attr_slice)['point_features']
        tgt_match_geo = self.match_geom_encoder(tgt_coords, tgt_attr_slice)['point_features']
        extra_match = {}
        # Optional local geometry branch.
        if self.local_geom_mlp is not None and self.local_geom_dim > 0:
            pre_match_geo = pre_match_geo + self.local_geom_mlp(pre_local_geom)
            tgt_match_geo = tgt_match_geo + self.local_geom_mlp(tgt_local_geom)
        # Optional target density/curvature conditioning.
        if self.target_geom_mlp is not None and tgt_attrs_full.shape[-1] >= 5:
            tgt_dc = tgt_attrs_full[..., 3:5]
            mean_dc = tgt_dc.mean(dim=1, keepdim=True)
            std_dc = tgt_dc.std(dim=1, keepdim=True).clamp_min(1e-6)
            tgt_dc_norm = (tgt_dc - mean_dc) / std_dc
            tgt_match_geo = tgt_match_geo + self.target_geom_mlp(tgt_dc_norm)
        # Optional source-side region/vessel embeddings.
        if (
            self.use_region_vessel_embedding
            and self.region_emb is not None
            and self.vessel_emb is not None
            and base_dim >= 6
        ):
            reg_idx = 4
            ves_idx = 5
            reg_ids = pre_base_attr[..., reg_idx].long().clamp(min=0, max=self.region_emb.num_embeddings - 1)
            ves_ids = pre_base_attr[..., ves_idx].long().clamp(min=0, max=self.vessel_emb.num_embeddings - 1)
            reg_emb = self.region_emb(reg_ids)
            ves_emb = self.vessel_emb(ves_ids)
            pre_match_geo = torch.cat([pre_match_geo, reg_emb, ves_emb], dim=-1)
            if self.regves_proj is not None:
                pre_match_geo = self.regves_proj(pre_match_geo)
        # Add visual global context to target geometry features.
        if self.tgt_global_proj is not None and tgt_frame_feats is not None:
            tgt_global = tgt_frame_feats[:, 0]
            if tgt_global.dim() == 3:
                tgt_global = tgt_global.mean(dim=1)
            tgt_global_proj = self.tgt_global_proj(tgt_global).unsqueeze(1).expand_as(tgt_match_geo)
            tgt_match_geo = tgt_match_geo + tgt_global_proj
        # Concatenate geometry, coordinates, and attributes for matching.
        pre_feats = torch.cat([pre_match_geo, pre_coords, pre_attr_slice], dim=-1)
        tgt_feats = torch.cat([tgt_match_geo, tgt_coords, tgt_attr_slice], dim=-1)
        pre_feats = self.match_pre_proj(pre_feats)
        tgt_feats = self.match_tgt_proj(tgt_feats)

        def _nan_check(tag: str, tensor: torch.Tensor) -> None:
            if not torch.is_tensor(tensor):
                return
            finite = torch.isfinite(tensor)
            if finite.all():
                return
            nan_count = int(torch.isnan(tensor).sum().item())
            inf_count = int(torch.isinf(tensor).sum().item())
            if finite.any():
                finite_vals = tensor[finite]
                t_min = float(finite_vals.min().item())
                t_max = float(finite_vals.max().item())
                t_mean = float(finite_vals.mean().item())
            else:
                t_min = t_max = t_mean = float("nan")
            msg = (
                f"{tag} contains NaN/Inf | nan={nan_count} inf={inf_count} | "
                f"min={t_min:.6g} max={t_max:.6g} mean={t_mean:.6g} | "
                f"shape={tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device}"
            )
            self.logger.error(msg)
            raise ValueError(msg)

        _nan_check("pre_feats_before_mlp", pre_feats)
        _nan_check("tgt_feats_before_mlp", tgt_feats)
        # residual refinement
        pre_feats = self.match_pre_mlp(pre_feats) + pre_feats
        tgt_feats = self.match_tgt_mlp(tgt_feats) + tgt_feats
        _nan_check("pre_feats_after_mlp", pre_feats)
        _nan_check("tgt_feats_after_mlp", tgt_feats)
        # Keep source and target match features on comparable scales.
        pre_feats = self.match_pre_norm(pre_feats)
        tgt_feats = self.match_tgt_norm(tgt_feats)

        # Local matcher: geometry filtering + sparse K-way softmax.
        if self.match_use_local:
            local_logit_scale = torch.exp(torch.clamp(self.local_logit_scale_log, min=-4.6, max=4.6))

            sparse_soft_match, _local_logits, _local_probs, _candidate_indices, _candidate_mask = local_matcher_forward(
                features_pre=pre_feats,
                features_tgt=tgt_feats,
                pre_coords=pre_coords,
                tgt_coords=tgt_coords,
                k_local=self.match_local_k,
                radius_mm=self.match_local_radius_mm,
                temperature=self.match_temperature,
                logit_scale=local_logit_scale,
            )

            sim_logits = None
            soft_match = sparse_soft_match
            soft_match_mixed = soft_match
        elif self.use_geo_transformer_matcher:
            (
                sim_logits_raw,
                soft_match_init,
                match_prob_full,
                pre_idx_coarse,
                tgt_idx_coarse,
                pre_cluster_mid,
                tgt_cluster_mid,
                coarse_match,
                coarse_topk,
                _mask_applied_geo,
                _coarse_logits,
                _coarse_cosine,
                _overlap_logits_pre,
                _overlap_logits_tgt,
                _coarse_pre_coords_used,
                _coarse_tgt_coords_used,
            ) = self.geo_matcher(pre_coords, tgt_coords, pre_feats, tgt_feats)
            sim_logits = sim_logits_raw.view(sim_logits_raw.shape[0], sim_logits_raw.shape[-2], sim_logits_raw.shape[-1])
            soft_match = soft_match_init.view(soft_match_init.shape[0], soft_match_init.shape[-2], soft_match_init.shape[-1])
            soft_match_mixed = soft_match
            extra_match['match_prob_ot'] = match_prob_full
            extra_match['geo_pre_idx'] = pre_idx_coarse
            extra_match['geo_tgt_idx'] = tgt_idx_coarse
        else:
            # coarse-to-fine softmax matcher
            coarse_bias = None
            match_prob_full = None
            logit_scale = torch.clamp(self.match_logit_scale, min=0.01, max=100.0)
            B, N_pre, _ = pre_feats.shape
            N_tgt = tgt_feats.shape[1]
            Nc = min(self.match_coarse_points, N_pre, N_tgt)
            if Nc > 8:
                idx_pre = torch.linspace(0, N_pre - 1, Nc, device=pre_feats.device).long()
                idx_tgt = torch.linspace(0, N_tgt - 1, Nc, device=pre_feats.device).long()
                pre_coarse = pre_feats[:, idx_pre]
                tgt_coarse = tgt_feats[:, idx_tgt]
                if self.match_use_ot:
                    coarse_logits, coarse_prob = self.match_head_ot(pre_coarse, tgt_coarse)
                else:
                    coarse_logits, _ = self.match_head_coarse(pre_coarse, tgt_coarse)
                coarse_logits = torch.nan_to_num(coarse_logits, nan=0.0, posinf=0.0, neginf=0.0)
                pre_map = torch.clamp((torch.arange(N_pre, device=pre_feats.device, dtype=torch.float32) * (Nc - 1) / max(N_pre - 1, 1)).round().long(), 0, Nc - 1)
                tgt_map = torch.clamp((torch.arange(N_tgt, device=pre_feats.device, dtype=torch.float32) * (Nc - 1) / max(N_tgt - 1, 1)).round().long(), 0, Nc - 1)
                coarse_bias = coarse_logits[:, pre_map][:, :, tgt_map] * logit_scale

            if self.match_use_ot and not self.match_ot_coarse_only:
                sim_logits, prob_full = self.match_head_ot(pre_feats, tgt_feats, bias=coarse_bias)
                sim_logits = torch.nan_to_num(sim_logits, nan=0.0, posinf=0.0, neginf=0.0)
                soft_match = prob_full[:, : pre_feats.shape[1], : tgt_feats.shape[1]]
                soft_match_mixed = soft_match
                match_prob_full = prob_full
                extra_match['match_prob_ot'] = prob_full
            else:
                sim_logits, _ = self.match_head(pre_feats, tgt_feats)
                sim_logits = torch.nan_to_num(sim_logits, nan=0.0, posinf=0.0, neginf=0.0)
                sim_logits = sim_logits * logit_scale
                if coarse_bias is not None:
                    sim_logits = sim_logits + coarse_bias
            # Row-wise candidate pruning by top-k logits and/or geometric window.
            if sim_logits is not None and (
                (self.match_row_topk and self.match_row_topk > 0)
                or (self.match_geom_window_mm and self.match_geom_window_mm > 0)
            ):
                if extra_match is None:
                    extra_match = {}
                B, N_pre, N_tgt = sim_logits.shape
                cand_mask = torch.ones((B, N_pre, N_tgt), device=sim_logits.device, dtype=torch.bool)
                # 几何窗口
                if self.match_geom_window_mm and self.match_geom_window_mm > 0:
                    radius_m = self.match_geom_window_mm / 1000.0
                    dist = torch.cdist(pre_coords, tgt_coords)  # (B,N_pre,N_tgt)
                    cand_mask &= dist <= radius_m
                # top-k 按 logits
                if self.match_row_topk and self.match_row_topk > 0:
                    k = min(self.match_row_topk, N_tgt)
                    topk_idx = sim_logits.topk(k=k, dim=2).indices  # (B,N_pre,k)
                    topk_mask = torch.zeros_like(cand_mask)
                    topk_mask.scatter_(2, topk_idx, True)
                    cand_mask &= topk_mask
                # 若某行全 False，则保底选第一列
                row_has = cand_mask.any(dim=2, keepdim=True)
                cand_mask = torch.where(row_has, cand_mask, torch.zeros_like(cand_mask))
                cand_mask = torch.where(row_has, cand_mask, cand_mask.scatter(2, torch.zeros_like(row_has, dtype=torch.long), True))
                eff_K = cand_mask.sum(dim=2).float()
                extra_match['match_eff_K_mean'] = eff_K.mean()
                extra_match['match_eff_K_median'] = eff_K.median()
                extra_match['match_eff_K_min'] = eff_K.min()
                extra_match['match_eff_K_max'] = eff_K.max()
                if eff_K.mean() > 0:
                    log_k = torch.log(eff_K.mean())
                    extra_match['match_logK_mean'] = log_k
                sim_logits = sim_logits.masked_fill(~cand_mask, -1e9)
            if not (self.match_use_ot and not self.match_ot_coarse_only):
                soft_match = F.softmax(sim_logits, dim=-1)
                soft_match_mixed = soft_match
        pre_for_svd = pre_coords
        tgt_for_svd = tgt_coords
        svd_weights = soft_match_mixed if soft_match_mixed is not None else soft_match
        if svd_weights is None:
            svd_weights = torch.zeros(
                pre_points.shape[0],
                pre_points.shape[1],
                tgt_for_svd.shape[1],
                device=pre_points.device,
                dtype=pre_points.dtype,
            )

        if self.svd_conf_filter_enabled:
            confidence = svd_weights.max(dim=2).values
            pass_mask = confidence > float(self.svd_conf_threshold)
            used_mask = pass_mask.clone()
            batch_size, n_pre = confidence.shape
            min_pts = max(1, int(self.svd_conf_min_points))
            fallback_k = max(int(self.svd_conf_fallback_topk), min_pts)
            for b in range(batch_size):
                if int(used_mask[b].sum().item()) < min_pts:
                    k = min(fallback_k, n_pre)
                    top_idx = torch.topk(confidence[b], k=k, largest=True, sorted=False).indices
                    used_mask[b, top_idx] = True
            svd_weights = svd_weights * used_mask.to(dtype=svd_weights.dtype).unsqueeze(-1)
            match_stats['svd_conf_pass_mean'] = pass_mask.sum(dim=1).float().mean().detach()
            match_stats['svd_conf_used_mean'] = used_mask.sum(dim=1).float().mean().detach()

        if self.use_ransac_solver:
            R_pred, t_pred, robust_stats = robust_pose_from_soft_matches(
                pre_for_svd,
                tgt_for_svd,
                svd_weights,
                topk=self.ransac_topk,
                hypotheses=self.ransac_hypotheses,
                inlier_threshold=self.ransac_inlier_threshold,
                use_icp=self.use_icp_refine,
                icp_iters=self.icp_iters,
                icp_threshold=self.icp_threshold,
            )
            match_stats['ransac_inliers'] = robust_stats['ransac_inliers'].mean().detach()
            match_stats['ransac_residual'] = robust_stats['ransac_residual'].mean().detach()
            if self.use_icp_refine:
                match_stats['icp_inliers'] = robust_stats['icp_inliers'].mean().detach()
        else:
            R_pred, t_pred = self.rigid_solver(pre_for_svd, tgt_for_svd, svd_weights)

        scale = torch.ones((pre_points.shape[0], 1), device=pre_points.device, dtype=pre_points.dtype)
        if sim_logits is not None:
            extra_match['match_logK_mean'] = sim_logits.new_tensor(float(sim_logits.shape[-1])).log()
        soft_match_to_return = soft_match_mixed if soft_match_mixed is not None else soft_match
        return (R_pred, t_pred, scale), sim_logits, soft_match_to_return, {**extra_match, **match_stats}

    def _flip_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        flip_vec = self.flip_vector.to(tensor.device).view(1, 1, 3)
        return tensor * flip_vec

    def _run_rigid_pose(self, fused_global: torch.Tensor):
        return self.rigid_pose_net(fused_global)

    def _run_deformation(self, pre_points: torch.Tensor, fused_points: torch.Tensor, pose):
        if self.use_decoupled_rigid:
            return self.deformation_field(pre_points, fused_points, pose)
        return self.deformation_field(pre_points, fused_points, pose)

    def _render_point_views(self, warped_points: torch.Tensor, batch: Dict[str, torch.Tensor]):
        if self.point_renderer is None:
            return None, None
        intrinsics = batch.get('cam_intrinsics')
        if intrinsics is None:
            return None, None
        masks = batch.get('intra_masks')
        if masks is None:
            return None, None
        B, F, _, H, W = masks.shape
        points = warped_points.unsqueeze(1).expand(-1, F, -1, -1).reshape(B * F, -1, 3)
        intr_flat = intrinsics.reshape(B * F, 3, 3)
        image_hw = batch['intra_images'].shape[-2:]
        mask, depth = self.point_renderer.render_with_depth(points, intr_flat, image_size=image_hw, return_depth=True)
        mask = mask.view(B, F, image_hw[0], image_hw[1])
        depth = depth.view(B, F, image_hw[0], image_hw[1])
        return mask, depth

    def _build_render_condition(
        self,
        depths: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
        num_points: int,
        device: torch.device,
        disable: bool = False,
    ) -> Optional[torch.Tensor]:
        if disable:
            return None
        stats = []
        def _summary(tensor: torch.Tensor):
            flat = tensor.view(tensor.shape[0], -1)
            return flat.mean(dim=-1, keepdim=True), flat.std(dim=-1, keepdim=True)

        if isinstance(depths, torch.Tensor) and depths.numel() > 0:
            mean, std = _summary(depths)
            stats.extend([mean, std])
        if isinstance(masks, torch.Tensor) and masks.numel() > 0:
            mean, std = _summary(masks)
            stats.extend([mean, std])
        if not stats:
            return None
        summary = torch.cat(stats, dim=-1)
        if summary.shape[-1] < 4:
            pad = torch.zeros(summary.shape[0], 4 - summary.shape[-1], device=summary.device, dtype=summary.dtype)
            summary = torch.cat([summary, pad], dim=-1)
        elif summary.shape[-1] > 4:
            summary = summary[:, :4]
        cond = self.render_condition_proj(summary)
        return cond.unsqueeze(1).expand(-1, num_points, -1)
