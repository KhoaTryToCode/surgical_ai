import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation

from .prototype_pooler import LandmarkPrototypePooler
from .temporal_decoder_layer import TemporalQueryInterleavedBlock


class InterleavedDecoderLayerWrapper(nn.Module):
    """
    Wraps a native Mask2Former transformer decoder layer to inject
    interleaved temporal cross-attention before standard masked cross-attention.
    """
    def __init__(self, base_layer, temporal_block):
        super().__init__()
        self.base_layer = base_layer
        self.temporal_block = temporal_block
        self.past_prototypes = None

    def forward(self, hidden_states, *args, **kwargs):
        # hidden_states: (num_queries, B, C) e.g., (100, B, 256)
        if self.past_prototypes is not None:
            # Batch-first cross-attention: (B, num_queries, C)
            q_b = hidden_states.permute(1, 0, 2)
            q_steered = self.temporal_block(q_b, self.past_prototypes)
            hidden_states = q_steered.permute(1, 0, 2)

        return self.base_layer(hidden_states, *args, **kwargs)


class TemporalMask2Former(nn.Module):
    """
    Temporal Mask2Former with Interleaved Landmark Prototype Conditioning.
    
    Architecture Overview:
      1. Context frames (t-2, t-1) pass through pixel module under torch.no_grad().
      2. Prototypes are extracted via mask-pooling (100% GT in train for Teacher Forcing,
         model prediction in eval).
      3. The resulting 6 prototype tokens are injected into each of the 9 decoder layers
         via learnable gated cross-attention.
      4. Target frame (t) is supervised using standard Hungarian matching loss.
    """
    def __init__(
        self,
        base_checkpoint="facebook/mask2former-swin-tiny-ade-semantic",
        config=None,
        num_classes=3,
        hidden_dim=256,
        temporal_gate_init=0.1
    ):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        # Load base Mask2Former model
        if config is not None:
            self.base_model = Mask2FormerForUniversalSegmentation(config)
        else:
            try:
                self.base_model = Mask2FormerForUniversalSegmentation.from_pretrained(
                    base_checkpoint,
                    num_labels=num_classes,
                    ignore_mismatched_sizes=True
                )
            except Exception:
                cfg = Mask2FormerConfig(
                    num_labels=num_classes,
                    hidden_dim=hidden_dim,
                    num_queries=100
                )
                self.base_model = Mask2FormerForUniversalSegmentation(cfg)

        # Prototype Pooler for context frames (t-2, t-1)
        self.prototype_pooler = LandmarkPrototypePooler(
            num_classes=num_classes,
            hidden_dim=hidden_dim
        )

        # Interleave temporal cross-attention inside each of the 9 decoder layers
        self.decoder_wrappers = nn.ModuleList()
        decoder_layers = self.base_model.model.transformer_module.decoder.layers
        for i in range(len(decoder_layers)):
            base_l = decoder_layers[i]
            t_blk = TemporalQueryInterleavedBlock(
                hidden_dim=hidden_dim,
                gate_init=temporal_gate_init
            )
            wrapper = InterleavedDecoderLayerWrapper(base_l, t_blk)
            decoder_layers[i] = wrapper
            self.decoder_wrappers.append(wrapper)

    def _predict_mask_from_pix_decoder(self, pixel_decoder_output):
        """
        Fast prediction of masks for context frames using the base model without autograd.
        """
        # Run transformer decoder without temporal prototypes
        transformer_out = self.base_model.model.transformer_module(
            multi_scale_features=pixel_decoder_output.decoder_hidden_states,
            mask_features=pixel_decoder_output.decoder_last_hidden_state
        )
        class_logits = self.base_model.class_predictor(transformer_out.last_hidden_state)
        mask_logits = transformer_out.masks_queries_logits[-1]

        mask_probs = mask_logits.sigmoid()
        class_probs = class_logits.softmax(dim=-1)[..., :self.num_classes]
        sem_seg = torch.einsum("bqc, bqhw -> bchw", class_probs, mask_probs)
        return (sem_seg > 0.4).float()

    def forward(self, pixel_values, mask_labels=None, return_dict=None):
        """
        Args:
            pixel_values: (B, T, 3, H, W) where T=3
            mask_labels: (B, T, num_classes, H, W) ground truth masks (optional)
        """
        B, T, C, H, W = pixel_values.shape
        c0 = pixel_values[:, 0]      # Frame t-2
        c1 = pixel_values[:, 1]      # Frame t-1
        target = pixel_values[:, 2]  # Target frame t

        # --- Context Extraction (strictly causal, torch.no_grad) ---
        with torch.no_grad():
            pix_0 = self.base_model.model.pixel_level_module(c0)
            feat_0 = pix_0.decoder_last_hidden_state  # (B, 256, H/4, W/4)

            pix_1 = self.base_model.model.pixel_level_module(c1)
            feat_1 = pix_1.decoder_last_hidden_state  # (B, 256, H/4, W/4)

            if self.training and mask_labels is not None:
                # 100% Teacher Forcing with GT past masks
                m0 = mask_labels[:, 0]
                m1 = mask_labels[:, 1]
            else:
                # Inference / Evaluation: use model's own predicted masks
                m0 = self._predict_mask_from_pix_decoder(pix_0)
                m1 = self._predict_mask_from_pix_decoder(pix_1)

            # Extract 6 landmark prototypes: (B, 6, 256)
            K_past = self.prototype_pooler([feat_0, feat_1], [m0, m1])

        # --- Target Frame Forward Pass ---
        # Set past prototypes on all 9 interleaved layers
        for wrapper in self.decoder_wrappers:
            wrapper.past_prototypes = K_past

        # Prepare Hungarian matching targets if mask_labels is supplied
        batch_mask_labels = None
        batch_class_labels = None
        if mask_labels is not None:
            target_masks = mask_labels[:, -1]  # (B, num_classes, H, W)
            batch_mask_labels = []
            batch_class_labels = []
            for b in range(B):
                masks_b = []
                classes_b = []
                for c in range(self.num_classes):
                    if target_masks[b, c].sum() > 0:
                        masks_b.append(target_masks[b, c])
                        classes_b.append(c)
                if len(masks_b) == 0:
                    # Guard for empty image
                    masks_b.append(torch.zeros_like(target_masks[b, 0]))
                    classes_b.append(0)
                batch_mask_labels.append(torch.stack(masks_b))
                batch_class_labels.append(torch.tensor(classes_b, dtype=torch.long, device=target_masks.device))

        try:
            outputs = self.base_model(
                pixel_values=target,
                mask_labels=batch_mask_labels,
                class_labels=batch_class_labels,
                return_dict=return_dict
            )
        finally:
            # Always reset prototypes to maintain clean state
            for wrapper in self.decoder_wrappers:
                wrapper.past_prototypes = None

        return outputs
