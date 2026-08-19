"""
EXP_07 — Iterative Bézier Spline Decoder (Visualization-Only, No Loss).

Each query holds a cubic Bézier curve [P0, C1, C2, P3] and a content embedding.
At every decoder layer the query:
  1. Evaluates B(t) to get sample points along the curve.
  2. Probes the encoder feature map at those coordinates via grid_sample.
  3. Runs self-attention across the sampled features.
  4. Predicts small (ΔP0, ΔC1, ΔC2, ΔP3) corrections and a class logit.
  5. Updates control points and passes them to the next layer.

Returns intermediate control points at every layer for visualization.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BezierDecoderLayer(nn.Module):
    """
    Single decoder layer: probe → self-attention → correction MLP.
    """

    def __init__(self, d_model: int = 256, nhead: int = 8, num_sample_t: int = 20):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_sample_t = num_sample_t

        # Register fixed t-values for Bézier evaluation
        t = torch.linspace(0.0, 1.0, num_sample_t)  # (T,)
        self.register_buffer("t_vals", t)

        # Project probed features into d_model
        # Encoder output: geometric_features(128) + saliency(1) + tangent(2) = 131 channels
        self.probe_proj = nn.Sequential(
            nn.Linear(131, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        # Self-attention across sampled curve points
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.self_attn_norm = nn.LayerNorm(d_model)

        # Cross-attention: query content ↔ probed sample features
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.cross_attn_norm = nn.LayerNorm(d_model)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.ffn_norm = nn.LayerNorm(d_model)

        # Correction head: predicts (ΔP0, ΔC1, ΔC2, ΔP3) = 8 values
        self.correction_head = nn.Linear(d_model, 8)

    def _evaluate_bezier(self, control_points: torch.Tensor) -> torch.Tensor:
        """
        Evaluate cubic Bézier curve at self.t_vals.

        Args:
            control_points: (B, Q, 4, 2) — P0, C1, C2, P3 for each query.
        Returns:
            sample_xy: (B, Q, T, 2) — sampled (x, y) in [0, 1].
        """
        t = self.t_vals.view(1, 1, -1, 1)  # (1, 1, T, 1)
        p0 = control_points[:, :, 0:1, :]  # (B, Q, 1, 2)
        c1 = control_points[:, :, 1:2, :]
        c2 = control_points[:, :, 2:3, :]
        p3 = control_points[:, :, 3:4, :]

        # B(t) = (1-t)^3 P0 + 3(1-t)^2 t C1 + 3(1-t) t^2 C2 + t^3 P3
        one_minus_t = 1.0 - t
        pts = (
            (one_minus_t ** 3) * p0
            + 3.0 * (one_minus_t ** 2) * t * c1
            + 3.0 * one_minus_t * (t ** 2) * c2
            + (t ** 3) * p3
        )  # (B, Q, T, 2)
        return pts.clamp(0.0, 1.0)

    def _probe_features(self, sample_xy: torch.Tensor, feature_map: torch.Tensor) -> torch.Tensor:
        """
        Bilinear-sample the feature map at the curve's sample points.

        Args:
            sample_xy: (B, Q, T, 2) in [0, 1], where [.., 0]=x, [.., 1]=y.
            feature_map: (B, C, H, W).
        Returns:
            probed: (B, Q, T, C).
        """
        B, Q, T, _ = sample_xy.shape
        C = feature_map.shape[1]

        # grid_sample expects grid in [-1, 1] with (x, y) layout
        grid = sample_xy * 2.0 - 1.0  # map [0,1] → [-1,1]
        grid = grid.view(B, Q * T, 1, 2)  # (B, Q*T, 1, 2)

        # Sample: (B, C, Q*T, 1)
        sampled = F.grid_sample(feature_map, grid, mode="bilinear", padding_mode="border", align_corners=True)
        sampled = sampled.squeeze(-1).permute(0, 2, 1)  # (B, Q*T, C)
        return sampled.view(B, Q, T, C)

    def forward(
        self,
        query_content: torch.Tensor,
        control_points: torch.Tensor,
        feature_map: torch.Tensor,
    ) -> tuple:
        """
        Args:
            query_content: (B, Q, d_model) — content embeddings per query.
            control_points: (B, Q, 4, 2) — current [P0, C1, C2, P3].
            feature_map: (B, 131, H, W) — concatenated encoder output.
        Returns:
            query_content: (B, Q, d_model) — updated content embeddings.
            control_points: (B, Q, 4, 2) — refined control points.
            sample_xy: (B, Q, T, 2) — the sample points used for probing.
        """
        B, Q, _, _ = control_points.shape

        # 1. Evaluate Bézier curve → sample points
        sample_xy = self._evaluate_bezier(control_points)  # (B, Q, T, 2)

        # 2. Probe the feature map at sample points
        probed = self._probe_features(sample_xy, feature_map)  # (B, Q, T, 131)
        probed = self.probe_proj(probed)  # (B, Q, T, d_model)

        # 3. Self-attention along each curve's sample points
        # Reshape to (B*Q, T, d_model) for per-curve self-attention
        probed_flat = probed.view(B * Q, self.num_sample_t, self.d_model)
        sa_out, _ = self.self_attn(probed_flat, probed_flat, probed_flat)
        sa_out = self.self_attn_norm(probed_flat + sa_out)  # (B*Q, T, d_model)

        # Pool curve features into single vector per query
        curve_feat = sa_out.mean(dim=1)  # (B*Q, d_model)
        curve_feat = curve_feat.view(B, Q, self.d_model)  # (B, Q, d_model)

        # 4. Cross-attention: query_content attends to pooled curve features
        ca_out, _ = self.cross_attn(
            query_content,   # (B, Q, d_model)
            curve_feat,      # (B, Q, d_model)
            curve_feat,      # (B, Q, d_model)
        )
        query_content = self.cross_attn_norm(query_content + ca_out)

        # 5. FFN
        ffn_out = self.ffn(query_content)
        query_content = self.ffn_norm(query_content + ffn_out)

        # 6. Predict control point corrections
        delta = self.correction_head(query_content)  # (B, Q, 8)
        delta = delta.view(B, Q, 4, 2)

        # Scale corrections (small refinement steps)
        delta = delta * 0.1

        # 7. Update control points
        control_points = (control_points + delta).clamp(0.0, 1.0)

        return query_content, control_points, sample_xy


class BezierSplineDecoder(nn.Module):
    """
    Full iterative Bézier spline decoder with N layers.
    Returns intermediate control points at every layer for visualization.
    """

    def __init__(
        self,
        num_queries: int = 10,
        num_layers: int = 6,
        d_model: int = 256,
        nhead: int = 8,
        num_sample_t: int = 20,
        num_classes: int = 5,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_sample_t = num_sample_t

        # Learnable query content embeddings
        self.query_embed = nn.Embedding(num_queries, d_model)

        # Learnable initial control points (P0, C1, C2, P3) per query
        # Initialized spread across the image
        self.init_control_points = nn.Parameter(
            self._create_initial_control_points(num_queries)
        )

        # Decoder layers
        self.layers = nn.ModuleList([
            BezierDecoderLayer(d_model, nhead, num_sample_t)
            for _ in range(num_layers)
        ])

        # Classification head
        self.class_head = nn.Linear(d_model, num_classes + 1)  # +1 for "no object"

    @staticmethod
    def _create_initial_control_points(num_queries: int) -> torch.Tensor:
        """
        Create initial control points spread across the image canvas.
        Each query gets a short horizontal curve segment at a different vertical position.
        """
        cps = torch.zeros(num_queries, 4, 2)
        for i in range(num_queries):
            y_center = (i + 0.5) / num_queries
            # P0 (start), C1 (handle 1), C2 (handle 2), P3 (end)
            cps[i, 0] = torch.tensor([0.2, y_center])          # P0
            cps[i, 1] = torch.tensor([0.35, y_center + 0.02])  # C1
            cps[i, 2] = torch.tensor([0.65, y_center - 0.02])  # C2
            cps[i, 3] = torch.tensor([0.8, y_center])          # P3
        return cps

    def forward(self, encoder_output: dict) -> dict:
        """
        Args:
            encoder_output: dict from SVGVectorAwareFeatureEncoder.forward().
        Returns:
            Dictionary with:
              - "final_control_points": (B, Q, 4, 2)
              - "class_logits": (B, Q, num_classes+1)
              - "layer_states": list of dicts, one per layer, each containing:
                  - "control_points": (B, Q, 4, 2) BEFORE this layer's correction
                  - "sample_xy": (B, Q, T, 2)
                  - "control_points_after": (B, Q, 4, 2) AFTER correction
        """
        geo_feat = encoder_output["geometric_features"]  # (B, 128, H, W)
        saliency = encoder_output["saliency_field"]       # (B, 1, H, W)
        tangent = encoder_output["tangent_field"]          # (B, 2, H, W)

        B = geo_feat.shape[0]
        H, W = geo_feat.shape[2], geo_feat.shape[3]

        # Resize saliency & tangent to match geometric features spatial size
        if saliency.shape[2:] != geo_feat.shape[2:]:
            saliency = F.interpolate(saliency, size=(H, W), mode="bilinear", align_corners=False)
        if tangent.shape[2:] != geo_feat.shape[2:]:
            tangent = F.interpolate(tangent, size=(H, W), mode="bilinear", align_corners=False)

        # Concatenate all encoder outputs into a single feature map
        feature_map = torch.cat([geo_feat, saliency, tangent], dim=1)  # (B, 131, H, W)

        # Initialize queries
        query_content = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # (B, Q, d_model)
        control_points = self.init_control_points.unsqueeze(0).expand(B, -1, -1, -1).clone()  # (B, Q, 4, 2)

        # Run decoder layers, recording intermediate states
        layer_states = []
        for layer in self.layers:
            cp_before = control_points.clone()
            query_content, control_points, sample_xy = layer(
                query_content, control_points, feature_map
            )
            layer_states.append({
                "control_points": cp_before,
                "sample_xy": sample_xy,
                "control_points_after": control_points.clone(),
            })

        # Classification
        class_logits = self.class_head(query_content)  # (B, Q, num_classes+1)

        return {
            "final_control_points": control_points,
            "class_logits": class_logits,
            "layer_states": layer_states,
            "feature_map": feature_map,  # for visualization
        }
