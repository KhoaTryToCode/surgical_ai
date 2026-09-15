import os
import sys
import torch
import torch.nn as nn

EXP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if EXP_DIR not in sys.path:
    sys.path.append(EXP_DIR)

from models.backbone_2d import SurgicalBackbone2D
from models.transformer_decoder_2d import HierarchicalMaskedDecoder2D
from models.vector_losses_2d import Vector2DLossSuite

class Surgical2DVectorTransformer(nn.Module):
    """
    EXP_06: Direct 2D Vector Space Transformer for Laparoscopic Liver Landmark Segmentation.
    Operates natively in normalized 2D image coordinates (u, v) in [0.0, 1.0]^2 with Learned Query Embeddings.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 1. Swin-Tiny 2D Feature Backbone
        self.backbone = SurgicalBackbone2D(config)
        
        # 2. Hierarchical 2D Masked Decoder with Learned Query Embeddings
        self.decoder = HierarchicalMaskedDecoder2D(config)
        
        # 3. Deep Supervision 2D Loss Suite (3 Core Objectives: L_cls, L_pos, L_mask)
        self.loss_suite = Vector2DLossSuite(
            lambda_cls=config.lambda_cls,
            lambda_pos=config.lambda_pos,
            lambda_mask=config.lambda_mask
        )

    def forward(self, images: torch.Tensor, targets: dict = None) -> dict:
        """
        images: (B, 3, 1024, 1024) RGB surgical video frames
        targets: Dict containing GT annotations (target_classes, target_polylines, target_masks, valid_mask)
        """
        # 1. Multi-scale 2D visual feature extraction
        fused_features = self.backbone(images)
        
        # 2. 6-Layer Hierarchical Masked Decoding
        decoder_outputs = self.decoder(fused_features)
        
        # 3. Loss computation if targets provided
        loss = None
        loss_dict = {}
        if targets is not None:
            loss, loss_dict = self.loss_suite(
                outputs_cls=decoder_outputs["aux_cls"],
                outputs_polylines=decoder_outputs["aux_polylines"],
                outputs_masks=decoder_outputs["aux_masks"],
                targets=targets
            )

        return {
            "loss": loss,
            "loss_dict": loss_dict,
            "pred_cls": decoder_outputs["pred_cls"],
            "pred_polylines": decoder_outputs["pred_polylines"],
            "pred_masks": decoder_outputs["pred_masks"],
            "aux_cls": decoder_outputs["aux_cls"],
            "aux_polylines": decoder_outputs["aux_polylines"],
            "aux_masks": decoder_outputs["aux_masks"]
        }
