from __future__ import annotations

import torch
from torch import nn


def _pairwise_sq_dists(x: torch.Tensor) -> torch.Tensor:
    """Compute squared pairwise distances for x: (B, N, C) -> (B, N, N)."""
    xx = (x ** 2).sum(dim=-1, keepdim=True)  # (B,N,1)
    dist2 = xx + xx.transpose(1, 2) - 2.0 * torch.matmul(x, x.transpose(1, 2))
    return torch.clamp(dist2, min=0.0)


def _knn_indices(coords: torch.Tensor, k: int) -> torch.Tensor:
    """kNN indices based on coordinate space. coords: (B,N,3) -> (B,N,k)."""
    B, N, _ = coords.shape
    k = max(1, min(int(k), N - 1))
    dist2 = _pairwise_sq_dists(coords)
    # take k+1 to skip self at rank 0
    idx = dist2.topk(k=k + 1, dim=-1, largest=False).indices[:, :, 1:]
    return idx


def _gather_neighbors(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Gather neighbor features.
    x:   (B, N, C)
    idx: (B, N, k)
    returns: (B, N, k, C)
    """
    B, N, C = x.shape
    k = idx.shape[-1]
    idx_base = (torch.arange(B, device=x.device).view(B, 1, 1) * N).long()
    idx_flat = (idx.long() + idx_base).reshape(-1)
    x_flat = x.reshape(B * N, C)
    neigh = x_flat[idx_flat].reshape(B, N, k, C)
    return neigh


def _edge_features(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Build EdgeConv features: [x_i, x_j - x_i].
    x:   (B, N, C)
    idx: (B, N, k)
    returns: (B, 2C, N, k)
    """
    neigh = _gather_neighbors(x, idx)  # (B,N,k,C)
    x_i = x.unsqueeze(2).expand_as(neigh)
    edge = torch.cat([x_i, neigh - x_i], dim=-1)  # (B,N,k,2C)
    return edge.permute(0, 3, 1, 2).contiguous()


class EdgeConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int = 8):
        super().__init__()
        def _valid_groups(num_channels: int, max_groups: int) -> int:
            g = max(1, min(int(max_groups), int(num_channels)))
            while g > 1 and (num_channels % g) != 0:
                g -= 1
            return g

        g1 = _valid_groups(out_channels, groups)
        g2 = _valid_groups(out_channels, groups)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(g1, out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(g2, out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor, knn_idx: torch.Tensor) -> torch.Tensor:
        """
        x:       (B, N, C)
        knn_idx: (B, N, k)
        returns: (B, N, out_channels)
        """
        edge = _edge_features(x, knn_idx)  # (B,2C,N,k)
        out = self.mlp(edge)  # (B,out,N,k)
        out = out.max(dim=-1).values  # (B,out,N)
        return out.transpose(1, 2).contiguous()


class GeometricEncoder(nn.Module):
    """
    DGCNN-style geometric encoder (EdgeConv) with meaningful coarse pooling.

    - Local aggregation via EdgeConv (kNN in coordinate space).
    - Multi-scale features via concatenation of 3 EdgeConv stages.
    - Coarse features computed by FPS + neighborhood max-pooling from fine features (not indexing).

    Output dict keys are kept compatible with the previous GeometricEncoder.
    """

    def __init__(
        self,
        input_dim: int | None = None,
        feature_dim: int = 256,
        layers: int = 4,  # kept for config compatibility (EdgeConv stages fixed to 3)
        attr_dim: int = 6,
        hidden_dim: int = 256,  # unused (compat)
        heads: int = 4,  # unused (compat)
        coarse_points: int = 256,
        knn_k: int = 20,
        pool_k: int = 32,
        norm_groups: int = 8,
    ):
        super().__init__()
        self.attr_dim = max(0, int(attr_dim))
        self.coarse_points = max(1, int(coarse_points))
        self.knn_k = int(knn_k)
        self.pool_k = int(pool_k)
        self.norm_groups = int(norm_groups)
        self.feature_dim = int(feature_dim)

        base_in = 3 + self.attr_dim
        target_in = int(input_dim) if input_dim is not None else base_in
        self.input_proj = None
        if target_in != base_in:
            self.input_proj = nn.Linear(base_in, target_in, bias=False)
        in_channels = target_in
        out1 = max(16, self.feature_dim // 4)
        out2 = max(16, self.feature_dim // 4)
        out3 = max(16, self.feature_dim - out1 - out2)

        self.edge1 = EdgeConv(in_channels, out1, groups=self.norm_groups)
        self.edge2 = EdgeConv(out1, out2, groups=self.norm_groups)
        self.edge3 = EdgeConv(out2, out3, groups=self.norm_groups)

        # normalize concatenated fine features to keep stable scale
        self.out_norm = nn.LayerNorm(out1 + out2 + out3)

    @staticmethod
    def _fps_single(coords: torch.Tensor, k: int) -> torch.Tensor:
        """Deterministic FPS on a single point cloud coords: (N,3) -> (k,)."""
        k = min(int(k), coords.shape[0])
        if k <= 0:
            return torch.zeros(0, dtype=torch.long, device=coords.device)
        idx = torch.zeros(k, dtype=torch.long, device=coords.device)
        centroid = coords.mean(dim=0, keepdim=True)
        dist0 = torch.cdist(centroid, coords).squeeze(0)
        idx[0] = torch.argmax(dist0)
        dist = torch.full((coords.shape[0],), float("inf"), device=coords.device)
        for i in range(1, k):
            last = coords[idx[i - 1]].unsqueeze(0)
            dist = torch.minimum(dist, torch.cdist(last, coords).squeeze(0))
            idx[i] = torch.argmax(dist)
        return idx

    def _prepare_attrs(self, points: torch.Tensor, point_attrs: torch.Tensor | None) -> torch.Tensor:
        """Return attrs padded/truncated to attr_dim. Output shape: (B,N,attr_dim)."""
        B, N, _ = points.shape
        if self.attr_dim == 0:
            return points.new_zeros(B, N, 0)
        if point_attrs is None:
            return points.new_zeros(B, N, self.attr_dim)
        if point_attrs.dim() == 2:
            point_attrs = point_attrs.unsqueeze(0)
        if point_attrs.shape[0] != B:
            # best-effort broadcast for B=1
            if point_attrs.shape[0] == 1 and B > 1:
                point_attrs = point_attrs.expand(B, -1, -1)
        if point_attrs.shape[-1] > self.attr_dim:
            point_attrs = point_attrs[..., : self.attr_dim]
        if point_attrs.shape[-1] < self.attr_dim:
            pad = point_attrs.new_zeros(B, N, self.attr_dim - point_attrs.shape[-1])
            point_attrs = torch.cat([point_attrs, pad], dim=-1)
        return torch.nan_to_num(point_attrs, nan=0.0, posinf=0.0, neginf=0.0)

    def _pool_to_coarse(self, fine_coords: torch.Tensor, fine_feats: torch.Tensor, coarse_coords: torch.Tensor) -> torch.Tensor:
        """Neighborhood max-pool from fine_feats to coarse_coords using coordinate kNN."""
        B, N, _ = fine_coords.shape
        Nc = coarse_coords.shape[1]
        k = max(1, min(int(self.pool_k), N))
        # (B, Nc, N)
        dist = torch.cdist(coarse_coords, fine_coords)
        idx = dist.topk(k=k, dim=-1, largest=False).indices  # (B,Nc,k)
        C = fine_feats.shape[-1]
        idx_base = (torch.arange(B, device=fine_coords.device).view(B, 1, 1) * N).long()
        idx_flat = (idx.long() + idx_base).reshape(-1)
        fine_flat = fine_feats.reshape(B * N, C)
        neigh = fine_flat[idx_flat].reshape(B, Nc, k, C)
        return neigh.max(dim=2).values

    def forward(
        self,
        points: torch.Tensor,
        point_attrs: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> dict:
        """
        points: (B, N, 3)
        point_attrs: (B, N, D) or (N, D)
        valid_mask: (B, N) bool, True for real points (no padding)
        """
        if points.dim() == 2:
            points = points.unsqueeze(0)
        B, N, _ = points.shape
        if valid_mask is not None:
            if valid_mask.dim() == 1:
                valid_mask = valid_mask.unsqueeze(0)
            valid_mask = valid_mask.to(device=points.device, dtype=torch.bool)

        attrs_full = self._prepare_attrs(points, point_attrs)
        fine_feats = points.new_zeros(B, N, self.out_norm.normalized_shape[0])
        global_feat = points.new_zeros(B, self.out_norm.normalized_shape[0])
        coarse_coords = points.new_zeros(B, self.coarse_points, 3)
        coarse_feats = points.new_zeros(B, self.coarse_points, self.out_norm.normalized_shape[0])
        coarse_valid_mask = points.new_zeros(B, self.coarse_points, dtype=torch.bool)
        map_full2coarse = points.new_full((B, N), -1, dtype=torch.long)

        for b in range(B):
            mask_b = valid_mask[b] if valid_mask is not None else None
            if mask_b is None:
                idx_valid = torch.arange(N, device=points.device)
            else:
                idx_valid = mask_b.nonzero(as_tuple=False).squeeze(-1)
            if idx_valid.numel() < 2:
                continue
            coords_b = points[b, idx_valid]  # (Nv,3)
            attrs_b = attrs_full[b, idx_valid]  # (Nv,D)
            x0 = torch.cat([coords_b, attrs_b], dim=-1).unsqueeze(0)  # (1,Nv,3+D)
            if self.input_proj is not None:
                x0 = self.input_proj(x0)
            knn_idx = _knn_indices(coords_b.unsqueeze(0), self.knn_k)  # (1,Nv,k)
            x1 = self.edge1(x0, knn_idx)
            x2 = self.edge2(x1, knn_idx)
            x3 = self.edge3(x2, knn_idx)
            fv = torch.cat([x1, x2, x3], dim=-1)  # (1,Nv,C)
            fv = self.out_norm(fv)
            fine_feats[b, idx_valid] = fv[0]
            global_feat[b] = fv[0].mean(dim=0)

            # coarse FPS on valid points, then neighborhood pool features
            Nv = coords_b.shape[0]
            k = min(self.coarse_points, Nv)
            if Nv <= self.coarse_points:
                idx_local = torch.arange(Nv, device=points.device)
            else:
                idx_local = self._fps_single(coords_b, k)
            c_coords_valid = coords_b[idx_local]
            c_feats_valid = self._pool_to_coarse(coords_b.unsqueeze(0), fv, c_coords_valid.unsqueeze(0))[0]
            coarse_coords[b, :k] = c_coords_valid
            coarse_feats[b, :k] = c_feats_valid
            coarse_valid_mask[b, :k] = True
            # full->coarse nearest mapping for valid points only
            dist = torch.cdist(coords_b.unsqueeze(0), c_coords_valid.unsqueeze(0)).squeeze(0)  # (Nv,k)
            nn_idx = dist.argmin(dim=-1).clamp(max=k - 1)
            map_full2coarse[b, idx_valid] = nn_idx

        return {
            "point_features": fine_feats,
            "global_feature": global_feat,
            "fine_points": points,
            "fine_features": fine_feats,
            "coarse_points": coarse_coords,
            "coarse_features": coarse_feats,
            "map_full2coarse": map_full2coarse,
            "fine_valid_mask": valid_mask if valid_mask is not None else points.new_ones(B, N, dtype=torch.bool),
            "coarse_valid_mask": coarse_valid_mask,
        }
