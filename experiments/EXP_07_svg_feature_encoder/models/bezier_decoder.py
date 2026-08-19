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

        # 1. Global Cross-Attention: query content ↔ full image memory tokens (Radar/GPS)
        self.global_cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.global_cross_norm = nn.LayerNorm(d_model)

        # 2. Local Probing: Project probed 131-channel features into d_model
        # Encoder output: geometric_features(128) + saliency(1) + tangent(2) = 131 channels
        self.probe_proj = nn.Sequential(
            nn.Linear(131, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        # 3. Intra-Curve Self-Attention across sampled curve points (Microscope context)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.self_attn_norm = nn.LayerNorm(d_model)

        # 4. Local Cross-Attention: query content ↔ sampled curve features
        self.local_cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.local_cross_norm = nn.LayerNorm(d_model)

        # 5. FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.ffn_norm = nn.LayerNorm(d_model)

        # 6. Correction head: predicts (ΔP0, ΔC1, ΔC2, ΔP3) = 8 values
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
        global_memory: torch.Tensor = None,
    ) -> tuple:
        """
        Args:
            query_content: (B, Q, d_model) — content embeddings per query.
            control_points: (B, Q, 4, 2) — current [P0, C1, C2, P3].
            feature_map: (B, 131, H, W) — concatenated encoder output (high-res canvas).
            global_memory: (B, HW, d_model) — global image tokens (Radar/GPS view).
        Returns:
            query_content: (B, Q, d_model) — updated content embeddings.
            control_points: (B, Q, 4, 2) — refined control points.
            sample_xy: (B, Q, T, 2) — the sample points used for probing.
        """
        B, Q, _, _ = control_points.shape

        # ── PHASE 1: GLOBAL PROBING (The Radar / GPS) ──
        # Query attends across the ENTIRE image canvas to discover distant landmarks
        if global_memory is not None:
            g_out, _ = self.global_cross_attn(
                query_content,  # (B, Q, d_model)
                global_memory,  # (B, HW, d_model)
                global_memory,  # (B, HW, d_model)
            )
            query_content = self.global_cross_norm(query_content + g_out)

        # ── PHASE 2: LOCAL PROBING (The Microscope) ──
        # 1. Evaluate Bézier curve → sample 20 physical coordinates along B(t)
        sample_xy = self._evaluate_bezier(control_points)  # (B, Q, T, 2)

        # 2. Probe the high-res feature map at those 20 sample coordinates
        probed = self._probe_features(sample_xy, feature_map)  # (B, Q, T, 131)
        probed = self.probe_proj(probed)  # (B, Q, T, d_model)

        # 3. Intra-curve self-attention along the 20 sample points (shares context across glares/smoke)
        probed_flat = probed.view(B * Q, self.num_sample_t, self.d_model)
        sa_out, _ = self.self_attn(probed_flat, probed_flat, probed_flat)
        sa_out = self.self_attn_norm(probed_flat + sa_out)  # (B*Q, T, d_model)

        # Pool curve features into single vector per query
        curve_feat = sa_out.mean(dim=1).view(B, Q, self.d_model)  # (B, Q, d_model)

        # 4. Local Cross-Attention: query_content attends to its own pooled curve features
        ca_out, _ = self.local_cross_attn(
            query_content,   # (B, Q, d_model)
            curve_feat,      # (B, Q, d_model)
            curve_feat,      # (B, Q, d_model)
        )
        query_content = self.local_cross_norm(query_content + ca_out)

        # 5. FFN
        ffn_out = self.ffn(query_content)
        query_content = self.ffn_norm(query_content + ffn_out)

        # 6. Predict control point corrections [ΔP0, ΔC1, ΔC2, ΔP3]
        delta = self.correction_head(query_content).view(B, Q, 4, 2)
        delta = delta * 0.1  # Stable refinement step

        # 7. Update control points
        control_points = (control_points + delta).clamp(0.0, 1.0)

        return query_content, control_points, sample_xy

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
        Create structured Spatial Anchor Priors covering canonical laparoscopic anatomical positions:
          - Queries 0-2: Horizontal curved spans (Upper/Mid/Lower Anterior Ridges & Silhouettes)
          - Queries 3-5: Vertical spans (Center & Left/Right Falciform Ligament / Ligamentum Teres)
          - Queries 6-7: Lateral curved boundaries (Left & Right Liver Silhouettes)
          - Queries 8-9: Diagonal / Oblique spans (Gallbladder Boundary & Transverse Ridges)
        """
        anchors = [
            # 0: Upper Horizontal Ridge (Anterior Ridge high)
            [[0.15, 0.32], [0.35, 0.22], [0.65, 0.22], [0.85, 0.32]],
            # 1: Mid Horizontal Ridge (Anterior Ridge mid)
            [[0.15, 0.50], [0.35, 0.45], [0.65, 0.45], [0.85, 0.50]],
            # 2: Lower Horizontal Silhouette (Liver bottom contour)
            [[0.20, 0.72], [0.40, 0.78], [0.60, 0.78], [0.80, 0.72]],
            # 3: Center Vertical Midline (Falciform Ligament)
            [[0.50, 0.20], [0.50, 0.40], [0.50, 0.65], [0.50, 0.85]],
            # 4: Left-Midline Vertical (Ligamentum Teres / Vessel)
            [[0.42, 0.28], [0.45, 0.48], [0.48, 0.68], [0.50, 0.88]],
            # 5: Right-Midline Vertical
            [[0.58, 0.28], [0.55, 0.48], [0.52, 0.68], [0.50, 0.88]],
            # 6: Left Lateral Curved Boundary (Left Lobe Silhouette)
            [[0.15, 0.30], [0.12, 0.50], [0.15, 0.70], [0.25, 0.85]],
            # 7: Right Lateral Curved Boundary (Right Lobe Silhouette)
            [[0.85, 0.30], [0.88, 0.50], [0.85, 0.70], [0.75, 0.85]],
            # 8: Right-Lower Oblique (Gallbladder Boundary / Fossa)
            [[0.65, 0.45], [0.70, 0.55], [0.68, 0.65], [0.60, 0.75]],
            # 9: Diagonal Transverse Span (Transverse liver contour)
            [[0.25, 0.30], [0.45, 0.45], [0.65, 0.55], [0.80, 0.65]],
        ]

        cps = torch.zeros(num_queries, 4, 2)
        for i in range(num_queries):
            if i < len(anchors):
                cps[i] = torch.tensor(anchors[i], dtype=torch.float32)
            else:
                y_c = (i + 0.5) / num_queries
                cps[i, 0] = torch.tensor([0.2, y_c])
                cps[i, 1] = torch.tensor([0.35, y_c + 0.02])
                cps[i, 2] = torch.tensor([0.65, y_c - 0.02])
                cps[i, 3] = torch.tensor([0.8, y_c])
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

        # Global Memory Tokens for Phase 1 Global Cross-Attention (Stride 32 / Stride 16 tokens)
        swin_out = encoder_output.get("swin_outputs", {})
        if "stride32_features" in swin_out:
            # (B, 256, 32, 32) -> (B, 1024, 256)
            s32 = swin_out["stride32_features"]
            global_memory = s32.flatten(2).permute(0, 2, 1)
        else:
            # Fallback to downsampled geometric features
            s32 = F.adaptive_avg_pool2d(geo_feat, (32, 32))
            if s32.shape[1] != self.d_model:
                s32 = F.interpolate(s32, size=(32, 32), mode="bilinear")
            global_memory = s32.flatten(2).permute(0, 2, 1)

        # Initialize queries
        query_content = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # (B, Q, d_model)
        control_points = self.init_control_points.unsqueeze(0).expand(B, -1, -1, -1).clone()  # (B, Q, 4, 2)

        # Run decoder layers with Global Radar + Local Microscope probing
        layer_states = []
        for layer in self.layers:
            cp_before = control_points.clone()
            query_content, control_points, sample_xy = layer(
                query_content=query_content,
                control_points=control_points,
                feature_map=feature_map,
                global_memory=global_memory,
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
