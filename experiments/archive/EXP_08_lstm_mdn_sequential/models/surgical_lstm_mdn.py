import torch
import torch.nn as nn

from models.backbone_resnet import ResNetBackbone
from models.lstm_mdn_decoder import LSTMMDNDecoder


class SurgicalLSTM_MDN(nn.Module):
    """
    Full Model: ResNet-18 Backbone → LSTM-MDN Sequential Decoder
    
    EXP_08: CNN-LSTM-MDN Sequential Surgical Landmark Detection
    
    Training flow (teacher forced):
        1. Extract spatial features: F = backbone(image) → (B, D, H_f, W_f)
        2. For each valid GT instance in the batch:
           - Initialize LSTM with [GAP(F); class_embedding]
           - Run K steps, feeding GT coordinates as input (teacher forcing)
           - Collect MDN outputs for loss computation
    
    Inference flow (autoregressive):
        1. Extract features once
        2. For each landmark class, run LSTM decoder autoregressively
        3. Collect predicted polylines
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = ResNetBackbone(config)
        self.decoder = LSTMMDNDecoder(config)
    
    def forward(
        self,
        images: torch.Tensor,
        gt_polylines: torch.Tensor = None,
        gt_classes: torch.Tensor = None,
        valid_mask: torch.Tensor = None
    ) -> dict:
        """
        Args:
            images: (B, 3, H, W) normalized RGB images
            gt_polylines: (B, N, K, 2) GT polylines in [0, 1]^2 (training only)
            gt_classes: (B, N) class IDs per instance (training only)
            valid_mask: (B, N) boolean mask for valid instances (training only)
            
        Returns:
            Training: dict with 'instance_outputs' list for loss computation
            Inference: dict with 'predicted_polylines' per class
        """
        B = images.shape[0]
        
        # 1. Extract backbone features
        feature_maps = self.backbone(images)  # (B, D, H_f, W_f)
        
        if gt_polylines is not None and gt_classes is not None and valid_mask is not None:
            # ──── TRAINING MODE (Teacher Forcing) ────
            return self._forward_train(feature_maps, gt_polylines, gt_classes, valid_mask)
        else:
            # ──── INFERENCE MODE (Autoregressive) ────
            return self._forward_inference(feature_maps)
    
    def _forward_train(
        self,
        feature_maps: torch.Tensor,
        gt_polylines: torch.Tensor,
        gt_classes: torch.Tensor,
        valid_mask: torch.Tensor
    ) -> dict:
        """
        Teacher-forced training: one LSTM pass per valid GT instance.
        
        Returns:
            dict with:
                instance_outputs: list of decoder outputs (one per valid instance)
                gt_polylines_matched: list of (K, 2) GT polylines (aligned with outputs)
                gt_masks: will be populated by the training script (passed through)
        """
        B, N, K, _ = gt_polylines.shape
        
        all_instance_outputs = []
        all_gt_polylines = []
        
        for b in range(B):
            feat_b = feature_maps[b]  # (D, H_f, W_f)
            
            for i in range(N):
                if not valid_mask[b, i]:
                    continue
                
                cls_id = gt_classes[b, i].item()
                gt_poly = gt_polylines[b, i]  # (K, 2)
                
                # Run teacher-forced decoder for this instance
                out = self.decoder.forward_teacher_forced(feat_b, cls_id, gt_poly)
                
                all_instance_outputs.append(out)
                all_gt_polylines.append(gt_poly)
        
        return {
            "instance_outputs": all_instance_outputs,
            "gt_polylines_matched": all_gt_polylines
        }
    
    @torch.no_grad()
    def _forward_inference(self, feature_maps: torch.Tensor) -> dict:
        """
        Autoregressive inference: one LSTM pass per landmark class per image.
        
        Returns:
            dict with:
                predicted_polylines: (B, num_classes, K, 2) predicted polylines
                eos_probs: (B, num_classes, K) EOS probabilities
        """
        B = feature_maps.shape[0]
        num_classes = self.config.num_classes
        K = self.config.num_points
        device = feature_maps.device
        
        all_polylines = torch.zeros(B, num_classes, K, 2, device=device)
        all_eos = torch.zeros(B, num_classes, K, device=device)
        
        for b in range(B):
            feat_b = feature_maps[b]  # (D, H_f, W_f)
            
            for cls_id in range(1, num_classes + 1):  # Classes 1..4
                out = self.decoder.forward_autoregressive(feat_b, cls_id, max_steps=K)
                
                pred_pts = out["predicted_points"]  # (T, 2), T may differ from K
                eos = out["eos_probs"]              # (T,)
                
                # Pad or truncate to exactly K points
                T = min(pred_pts.shape[0], K)
                all_polylines[b, cls_id - 1, :T] = pred_pts[:T].clamp(0.0, 1.0)
                all_eos[b, cls_id - 1, :T] = eos[:T]
                
                # If T < K, repeat the last predicted point to fill
                if T < K:
                    all_polylines[b, cls_id - 1, T:] = pred_pts[-1].clamp(0.0, 1.0)
        
        return {
            "predicted_polylines": all_polylines,  # (B, num_classes, K, 2)
            "eos_probs": all_eos                    # (B, num_classes, K)
        }
    
    def get_param_groups(self, base_lr: float) -> list:
        """
        Returns optimizer parameter groups with differential learning rates.
        
        - Backbone: base_lr × backbone_lr_mult (0.1)
        - Decoder + MDN Head: base_lr × 1.0
        """
        backbone_groups = self.backbone.get_param_groups(self.config.backbone_lr_mult)
        decoder_params = list(self.decoder.parameters())
        
        param_groups = []
        for group in backbone_groups:
            param_groups.append({
                "params": group["params"],
                "lr": base_lr * group["lr_mult"]
            })
        
        param_groups.append({
            "params": decoder_params,
            "lr": base_lr
        })
        
        return param_groups
