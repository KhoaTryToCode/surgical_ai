"""
EXP_13: Query-to-Curve Bridge (Bypassing Blind ACPI)
===================================================
Bridges Mask2Former's refined semantic query embeddings into
parametric 5th-order Bézier curve control points.
Eliminates BCRNet's blind 16x16 grid search.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class QueryCurveBridge(nn.Module):
    """
    Directly converts Mask2Former query embeddings (B, M*K, 256)
    into initial 5th-order Bézier control points (B, M, K, 6, 2)
    and proposal confidence scores (B, M, K).
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_classes: int = 3,
        num_queries_per_class: int = 5,
        num_ctrl_pts: int = 6,
        scale: float = 0.35,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries_per_class = num_queries_per_class
        self.num_ctrl_pts = num_ctrl_pts
        self.scale = scale

        # 1. Anatomical Prior Anchors (Ridge, Silhouette, Ligament) in [0, 1]^2
        # Ridge: transverse arc across liver surface
        ridge_anchor = torch.tensor([
            [0.20, 0.40], [0.35, 0.45], [0.50, 0.50],
            [0.65, 0.52], [0.78, 0.55], [0.88, 0.58]
        ], dtype=torch.float32)

        # Silhouette: outer dome contour
        silh_anchor = torch.tensor([
            [0.15, 0.60], [0.25, 0.32], [0.45, 0.20],
            [0.65, 0.22], [0.80, 0.35], [0.85, 0.55]
        ], dtype=torch.float32)

        # Ligament: near-vertical notch
        lig_anchor = torch.tensor([
            [0.48, 0.22], [0.50, 0.35], [0.52, 0.50],
            [0.53, 0.65], [0.54, 0.78], [0.55, 0.88]
        ], dtype=torch.float32)

        # Stack to (M, 6, 2)
        anchors = torch.stack([ridge_anchor, silh_anchor, lig_anchor], dim=0)
        self.register_buffer("anchors", anchors)

        # 2. Control Point Offset MLP: maps query embedding (256) -> 6 * 2 = 12 offsets
        self.ctrl_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, num_ctrl_pts * 2),
        )

        # 3. Proposal Confidence Score MLP
        self.conf_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(
        self,
        query_embeddings: torch.Tensor,  # (B, M*K, 256)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            initial_ctrl_pts: (B, M, K, 6, 2) in [0, 1]^2
            proposal_scores:  (B, M, K) in [0, 1]
        """
        B, total_q, D = query_embeddings.shape
        M = self.num_classes
        K = self.num_queries_per_class

        # Reshape queries to (B, M, K, D)
        q_reshaped = query_embeddings.view(B, M, K, D)

        # Predict control point offsets: (B, M, K, 6*2) -> (B, M, K, 6, 2)
        offsets = self.ctrl_mlp(q_reshaped).view(B, M, K, self.num_ctrl_pts, 2)

        # Expand anchor to (B, M, K, 6, 2)
        anchor_expanded = self.anchors.unsqueeze(0).unsqueeze(2).expand(B, M, K, -1, -1)

        # Bounded Tanh Residual Update: strictly kept within [0, 1]
        init_ctrl_pts = torch.clamp(
            anchor_expanded + self.scale * torch.tanh(offsets),
            0.0, 1.0
        )

        # Confidence scores
        conf_logits = self.conf_mlp(q_reshaped).squeeze(-1)  # (B, M, K)
        conf_scores = torch.sigmoid(conf_logits)

        return init_ctrl_pts, conf_scores
