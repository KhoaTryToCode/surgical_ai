import torch
import torch.nn as nn
from .backbone import SurgicalBackbone3DLifting
from .proposal_head import ProposalHead3D
from .transformer_decoder import HierarchicalMaskedDecoder3D
from .vector_losses_3d import Vector3DLossSuite

class Surgical3DVectorTransformer(nn.Module):
    """
    Unified Architecture for Monocular 3D Vector Space Landmark Detection (EXP_05).
    Pipeline:
      1. 2D Visual Backbone (Swin/ResNet FPN) + Canonical 3D Pinhole Unprojection + PE_3D Injection
      2. 2D/3D Proposal Head (Predicts initial 3D anchor points p_anchor^{(0)})
      3. Hierarchical Masked Attention Transformer Decoder (Iterative Dual 2D Mask & 3D Vertices Refinement)
      4. Deep Supervision Vector Loss Computation
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 1. Backbone & 3D Geometry Encoder
        self.backbone = SurgicalBackbone3DLifting(
            embed_dim=config.embed_dim,
            fov_degrees=config.fov_degrees,
            mask2former_model_name=config.mask2former_model_name
        )

        # 2. Option B 3D Proposal Head
        self.proposal_head = ProposalHead3D(
            embed_dim=config.embed_dim,
            num_instances=config.num_instances,
            num_points=config.num_points
        )

        # 3. Transformer Decoder with Dual Heads
        self.decoder = HierarchicalMaskedDecoder3D(
            embed_dim=config.embed_dim,
            num_instances=config.num_instances,
            num_points=config.num_points,
            num_classes=config.num_classes,
            num_layers=config.num_decoder_layers
        )

        # 4. Loss Engine
        self.criterion = Vector3DLossSuite(
            lambda_cls=config.lambda_cls,
            lambda_pos=config.lambda_pos,
            lambda_tan=config.lambda_tan,
            lambda_curv=config.lambda_curv,
            lambda_mask=config.lambda_mask
        )

    def forward(self, images: torch.Tensor, depth: torch.Tensor, targets: dict = None):
        """
        images: (B, 3, 1024, 1024)
        depth: (B, 1, 1024, 1024)
        targets: Optional dict containing target_classes, target_polylines, target_masks, valid_mask
        """
        # Step 1: Feature Extraction & 3D Unprojection
        fused_features = self.backbone(images, depth) # List of 4 feature maps

        # Step 2: Generate Initial 3D Anchors from Stride-8 Feature Map
        initial_anchors = self.proposal_head(fused_features[1]) # (B, N, K, 3)

        # Step 3: Decoder Iterative Refinement (Defaulting to Stride-8 High Precision for A100)
        stride_idx = getattr(self.config, "decoder_stride_idx", 1)
        outputs_cls, outputs_polylines, outputs_masks = self.decoder(fused_features, initial_anchors, stride_idx=stride_idx)

        output_dict = {
            "outputs_cls": outputs_cls,               # List of L tensors (B, N, num_classes+1)
            "outputs_polylines": outputs_polylines,   # List of L tensors (B, N, K, 3)
            "outputs_masks": outputs_masks,           # List of L tensors (B, N, 1024, 1024)
            "pred_cls": outputs_cls[-1],              # Final layer outputs
            "pred_polylines": outputs_polylines[-1],
            "pred_masks": outputs_masks[-1]
        }

        if targets is not None:
            loss, loss_dict = self.criterion(outputs_cls, outputs_polylines, outputs_masks, targets)
            output_dict["loss"] = loss
            output_dict["loss_dict"] = loss_dict

        return output_dict
