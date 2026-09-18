import os
import sys
import torch
import torch.nn as nn

# Ensure workspace root is on path for cross-environment portability
_WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)

from transformers import Mask2FormerForUniversalSegmentation
from experiments.EXPERIMENT_3.models.bezier_decoder import BezierPatchDecoder

class BezierPatchModel(nn.Module):
    """
    Combined BezierPatchModel leveraging HF Mask2Former backbone.
    """
    def __init__(self, grid_size=8, num_classes=4, embed_dim=256, num_decoder_layers=6):
        super().__init__()
        hf_model = Mask2FormerForUniversalSegmentation.from_pretrained(
            'facebook/mask2former-swin-tiny-ade-semantic',
            ignore_mismatched_sizes=True
        )
        self.backbone = hf_model.model.pixel_level_module.encoder
        self.pixel_decoder = hf_model.model.pixel_level_module.decoder
        del hf_model
        
        self.bezier_decoder = BezierPatchDecoder(
            embed_dim=embed_dim,
            grid_size=grid_size,
            num_classes=num_classes,
            num_decoder_layers=num_decoder_layers,
        )
        
    def forward(self, pixel_values):
        backbone_output = self.backbone(pixel_values)
        backbone_features = backbone_output.feature_maps
        
        pixel_decoder_output = self.pixel_decoder(backbone_features, output_hidden_states=False)
        
        if hasattr(pixel_decoder_output, 'multi_scale_features') and pixel_decoder_output.multi_scale_features is not None:
            multi_scale_features = list(pixel_decoder_output.multi_scale_features)
        elif hasattr(pixel_decoder_output, 'hidden_states') and pixel_decoder_output.hidden_states:
            multi_scale_features = list(pixel_decoder_output.hidden_states)[-3:]
        else:
            multi_scale_features = list(backbone_features)[-3:]
            
        pred_class, pred_bezier = self.bezier_decoder(multi_scale_features)
        return pred_class, pred_bezier

def load_bezier_patch_model(checkpoint_path=None, grid_size=8, num_classes=4, device='cpu'):
    """
    Loads BezierPatchModel and optionally its checkpoint weights.
    """
    model = BezierPatchModel(grid_size=grid_size, num_classes=num_classes)
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f'Loaded checkpoint from {checkpoint_path}')
    return model.to(device)
