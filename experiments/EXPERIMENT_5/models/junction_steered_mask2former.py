"""
Junction-Steered Mask2Former Model for EXPERIMENT_5.
Combines:
  1. Pretrained Swin-Tiny Backbone + MSDeformAttn Pixel Decoder (from HF Mask2Former)
  2. 4-Query Junction Anchor Head (predicting Top, Bottom, Lat_R, Lat_L)
  3. Dynamic Junction Cross-Attention Query Steering Block
  4. 9-Layer Mask2Former Transformer Decoder
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure workspace root is on sys.path
_WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)

from experiments.EXPERIMENT_5.models.junction_head import JunctionAnchorHead

try:
    from transformers import Mask2FormerForUniversalSegmentation, AutoImageProcessor
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

class JunctionQuerySteering(nn.Module):
    """
    Dynamic Query Steering Block:
    Allows 100 Mask2Former queries to cross-attend to the 4 junction anchor vectors.
    """
    def __init__(self, embed_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.proj_q = nn.Linear(embed_dim, embed_dim)
        self.proj_k = nn.Linear(embed_dim, embed_dim)
        self.proj_v = nn.Linear(embed_dim, embed_dim)
        
        # Learnable gating scalar initialized to 0.1 for smooth gradient highway
        self.gate = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, junction_features, junction_coords=None):
        """
        Args:
            queries (Tensor): (B, 100, C) base query features
            junction_features (Tensor): (B, 4, C) junction anchor tokens
            junction_coords (Tensor, optional): (B, 4, 2) normalized coordinates
        Returns:
            steered_queries (Tensor): (B, 100, C)
            attn_weights (Tensor): (B, 100, 4)
        """
        # Linear projections
        q = self.proj_q(queries)
        k = self.proj_k(junction_features)
        v = self.proj_v(junction_features)
        
        # Cross-attention: queries attend to 4 junctions
        delta_q, attn_weights = self.cross_attn(query=q, key=k, value=v)
        
        # Gated residual update: Q_steered = Norm(Q + gate * delta_Q)
        steered_queries = self.norm(queries + self.gate * self.dropout(delta_q))
        
        return steered_queries, attn_weights

class JunctionSteeredMask2Former(nn.Module):
    """
    Integrated Junction-Steered Mask2Former.
    """
    def __init__(self, model_name="facebook/mask2former-swin-tiny-ade-semantic", num_labels=4):
        super().__init__()
        if not HAS_TRANSFORMERS:
            raise ImportError("HuggingFace 'transformers' library is required. Please install it or activate your PyTorch CUDA env.")
            
        print(f"📦 Building JunctionSteeredMask2Former with base: '{model_name}'...")
        self.m2f = Mask2FormerForUniversalSegmentation.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )
        
        embed_dim = 256
        
        # 1. Junction Anchor Head (4 biological queries)
        self.junction_head = JunctionAnchorHead(embed_dim=embed_dim, num_layers=2, num_heads=8)
        
        # 2. Query Steering Block
        self.query_steering = JunctionQuerySteering(embed_dim=embed_dim, num_heads=8)
        
    def forward(self, pixel_values, mask_labels=None, class_labels=None):
        """
        Forward pass with dynamic query steering.
        Args:
            pixel_values (Tensor): (B, 3, H, W)
            mask_labels (list of Tensors, optional): GT binary masks per image for training
            class_labels (list of Tensors, optional): GT class IDs per image for training
        Returns:
            dict containing:
                - 'loss': total loss (if mask_labels is provided)
                - 'masks_queries_logits': (B, 100, H/4, W/4)
                - 'class_queries_logits': (B, 100, num_classes + 1)
                - 'pred_junction_coords': (B, 4, 2) in [0, 1]^2
                - 'pred_junction_vis': (B, 4) visibility logits
                - 'junction_attn_weights': (B, 100, 4)
                - 'm2f_loss': standard Hungarian loss (if training)
        """
        B = pixel_values.shape[0]
        
        # 1. Swin-Tiny Backbone + MSDeformAttn Pixel Decoder
        pixel_level_outputs = self.m2f.model.pixel_level_module(pixel_values, output_hidden_states=True)
        # multi_scale_features has 3 tensors: [stride 32, stride 16, stride 8]
        multi_scale_features = list(pixel_level_outputs.decoder_hidden_states)
        mask_features = pixel_level_outputs.decoder_last_hidden_state # (B, 256, H/4, W/4)
        
        # 2. Extract 4 Anatomical Junctions from stride-16 features
        stride_16_feat = multi_scale_features[1] # (B, 256, H_16, W_16)
        pred_j_coords, pred_j_vis, j_features = self.junction_head(stride_16_feat)
        
        # 3. Dynamic Query Steering:
        # Base query features from Mask2Former
        tm = self.m2f.model.transformer_module
        base_query_feat = tm.queries_features.weight.unsqueeze(0).repeat(B, 1, 1) # (B, 100, 256)
        
        # Steer the queries with the 4 junction vectors
        steered_query_feat, j_attn_weights = self.query_steering(base_query_feat, j_features, pred_j_coords)
        
        # 4. Forward through Mask2Former Transformer Decoder with steered queries
        # Temporarily patch queries_features during this forward pass
        orig_queries_features = tm.queries_features
        class DynamicQueryFeatures(nn.Module):
            def __init__(self, steered_tensor):
                super().__init__()
                self.steered = steered_tensor
            def forward(self, *args, **kwargs):
                return self.steered
                
        # Run standard transformer module forward
        # In HuggingFace, transformer_module forward expects (multi_scale_features, mask_features)
        # We temporarily hook the query features
        old_weight = tm.queries_features.weight
        # Steered mean across batch as weight, or pass per-sample into decoder:
        # HuggingFace transformer_module:
        # query_features = self.queries_features.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        # We hook tm.queries_features:
        tm.queries_features = DynamicQueryFeatures(steered_query_feat)
        
        try:
            decoder_outputs = tm(
                multi_scale_features,
                mask_features,
                output_hidden_states=False,
                output_attentions=False
            )
        finally:
            tm.queries_features = orig_queries_features # Restore
            
        # Extract class & mask logits
        class_queries_logits = self.m2f.class_predictor(decoder_outputs[0]) # (B, 100, num_classes + 1)
        mask_embeddings = self.m2f.mask_embedder(decoder_outputs[0])       # (B, 100, 256)
        # Dot product with mask_features: (B, 100, 256) x (B, 256, H/4, W/4) -> (B, 100, H/4, W/4)
        masks_queries_logits = torch.einsum("bqc,bchw->bqhw", mask_embeddings, mask_features)
        
        output = {
            'masks_queries_logits': masks_queries_logits,
            'class_queries_logits': class_queries_logits,
            'pred_junction_coords': pred_j_coords,
            'pred_junction_vis': pred_j_vis,
            'junction_attn_weights': j_attn_weights
        }
        
        # 5. Compute Hungarian Matching Loss if GT labels are provided
        if mask_labels is not None and class_labels is not None:
            # Build loss using Mask2Former's built-in criterion
            loss_dict = self.m2f.criterion(
                masks_queries_logits=masks_queries_logits,
                class_queries_logits=class_queries_logits,
                mask_labels=mask_labels,
                class_labels=class_labels
            )
            # Weighted sum according to HuggingFace weight_dict
            weight_dict = self.m2f.criterion.weight_dict
            m2f_loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            output['m2f_loss'] = m2f_loss
            output['m2f_loss_dict'] = loss_dict
            
        return output

def load_junction_steered_model(checkpoint_path=None, model_name="facebook/mask2former-swin-tiny-ade-semantic", device="cpu"):
    """
    Helper to instantiate and load checkpoint weights.
    """
    model = JunctionSteeredMask2Former(model_name=model_name, num_labels=4)
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(state_dict)
        print(f"✅ Loaded checkpoint from: {checkpoint_path}")
    return model.to(device)
