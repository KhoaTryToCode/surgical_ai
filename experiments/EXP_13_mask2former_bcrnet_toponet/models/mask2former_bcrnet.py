"""
EXP_13: Mask2Former-BCRNet Master Architecture
==============================================
The unified master model combining:
1. Mask2Former 4-Channel RGB-D Pixel Decoder & Masked Attention Engine
2. TopoNet Centerline Topological Regularization
3. Query-to-Curve Geometric Bridge
4. Mask-Gated Hierarchical Curve Refinement (M-HCR)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple

from .mask2former_engine import Mask2FormerEngine
from .query_curve_bridge import QueryCurveBridge
from .masked_hcr import MaskedHCR
try:
    from ..configs.exp13_config import EXP13Config
except (ImportError, ValueError):
    from configs.exp13_config import EXP13Config


class Mask2FormerBCRNet(nn.Module):
    """
    Master Architecture: Mask2Former + TopoNet Loss + BCRNet.
    Outputs:
    - 4-class pixel segmentation map (Background, Ridge, Silhouette, Ligament)
    - 5th-order continuous parametric Bézier curves (6 control points per landmark)
    """

    def __init__(self, cfg: EXP13Config):
        super().__init__()
        self.cfg = cfg
        self.num_classes = cfg.num_classes
        self.num_queries_per_class = cfg.num_queries_per_class
        self.bezier_degree = cfg.bezier_degree
        self.num_ctrl_pts = cfg.num_ctrl_pts
        self.num_sample_pts = cfg.num_sample_pts

        # 1. Mask2Former Engine
        self.m2f_engine = Mask2FormerEngine(
            in_channels=cfg.in_channels,
            num_classes=cfg.num_classes,
            num_queries_per_class=cfg.num_queries_per_class,
            fpn_dim=cfg.fpn_dim,
            num_decoder_layers=cfg.num_m2f_decoder_layers,
            num_heads=cfg.m2f_num_heads,
        )

        # 2. Query-to-Curve Bridge
        self.bridge = QueryCurveBridge(
            embed_dim=cfg.fpn_dim,
            num_classes=cfg.num_classes,
            num_queries_per_class=cfg.num_queries_per_class,
            num_ctrl_pts=cfg.num_ctrl_pts,
            scale=cfg.bounded_tanh_scale,
        )

        # 3. Mask-Gated Hierarchical Curve Refinement
        self.hcr = MaskedHCR(
            in_channels=cfg.fpn_dim,
            embed_dim=128,
            num_stages=cfg.hcr_stages,
            num_ctrl_pts=cfg.num_ctrl_pts,
            num_sample_pts=cfg.num_sample_pts,
        )

    def forward(
        self,
        rgbd: torch.Tensor,
        target_size: Tuple[int, int] = (512, 512),
    ) -> Dict[str, Any]:
        """
        Forward pass through the unified architecture.
        Args:
            rgbd: (B, 4, H, W) 4-channel RGB-D input tensor
        Returns:
            Dictionary with both pixel-level and curve-level predictions.
        """
        # 1. Mask2Former forward pass
        m2f_out = self.m2f_engine(rgbd, target_size=target_size)
        fpn_features = m2f_out["fpn_features"]         # [f1, f2, f3, f4]
        query_embeddings = m2f_out["query_embeddings"] # (B, M*K, 256)
        semantic_logits = m2f_out["semantic_logits"]   # (B, 4, H, W)
        query_masks = m2f_out["query_masks"]           # (B, M*K, H, W)

        # 2. Bridge: Mask2Former queries -> Initial Bézier control points
        init_ctrl_pts, init_scores = self.bridge(query_embeddings)
        # init_ctrl_pts: (B, M, K, 6, 2)
        # init_scores:   (B, M, K)

        # 3. Mask-Gated HCR: Iterative curve refinement
        final_ctrl_pts, final_curve_pts, final_scores = self.hcr(
            fpn_features=fpn_features,
            mask_logits=semantic_logits,
            init_ctrl_pts=init_ctrl_pts,
            init_scores=init_scores,
        )

        return {
            # Pixel predictions (Mask2Former + TopoNet)
            "semantic_logits": semantic_logits,       # (B, 4, H, W)
            "query_masks": query_masks,               # (B, M*K, H, W)
            "query_classes": m2f_out["query_classes"], # (B, M*K, 4)
            # Parametric curve predictions (BCRNet)
            "init_ctrl_pts": init_ctrl_pts,           # (B, M, K, 6, 2)
            "final_ctrl_pts": final_ctrl_pts,         # (B, M, K, 6, 2)
            "final_curve_pts": final_curve_pts,       # (B, M, K, 26, 2)
            "final_scores": final_scores,             # (B, M, K)
        }

    @torch.no_grad()
    def predict_best_curves(
        self,
        rgbd: torch.Tensor,
        target_size: Tuple[int, int] = (512, 512),
    ) -> Dict[str, Any]:
        """
        Inference helper: selects best proposal per landmark class.
        """
        self.eval()
        outputs = self.forward(rgbd, target_size=target_size)
        B, M, K = outputs["final_scores"].shape

        best_ctrl = []
        best_curves = []
        best_scores = []

        for b in range(B):
            b_ctrl = []
            b_curv = []
            b_score = []
            for m in range(M):
                scores_m = outputs["final_scores"][b, m]  # (K,)
                best_k = torch.argmax(scores_m).item()
                b_ctrl.append(outputs["final_ctrl_pts"][b, m, best_k])
                b_curv.append(outputs["final_curve_pts"][b, m, best_k])
                b_score.append(scores_m[best_k])
            best_ctrl.append(torch.stack(b_ctrl, dim=0))
            best_curves.append(torch.stack(b_curv, dim=0))
            best_scores.append(torch.stack(b_score, dim=0))

        # Semantic map prediction via argmax over softmax probabilities
        sem_probs = F.softmax(outputs["semantic_logits"], dim=1)
        sem_pred = torch.argmax(sem_probs, dim=1)  # (B, H, W) in {0, 1, 2, 3}

        return {
            "best_ctrl_pts": torch.stack(best_ctrl, dim=0),      # (B, M, 6, 2)
            "best_curve_pts": torch.stack(best_curves, dim=0),   # (B, M, 26, 2)
            "best_scores": torch.stack(best_scores, dim=0),      # (B, M)
            "semantic_pred": sem_pred,                           # (B, H, W)
            "semantic_probs": sem_probs,                         # (B, 4, H, W)
        }
