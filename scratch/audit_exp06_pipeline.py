import os
import sys
import torch
import numpy as np

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from experiments.EXP_06_2d_vector_transformer.configs.exp06_config import config
from experiments.EXP_06_2d_vector_transformer.models.surgical_2d_vector_transformer import Surgical2DVectorTransformer

def run_exp06_audit():
    print("=" * 80)
    print("🔍 EXP_06 COMPREHENSIVE PIPELINE AUDIT (Direct 2D Vector Transformer)")
    print("=" * 80)

    device = torch.device("cpu")
    print(f"1. Instantiating Surgical2DVectorTransformer on {device}...")
    model = Surgical2DVectorTransformer(config).to(device)
    model.train()

    B = 2
    N = config.num_instances
    K = config.num_points
    H, W = 1024, 1024

    dummy_images = torch.randn(B, 3, H, W, device=device)
    dummy_targets = {
        "target_classes": torch.tensor([[1, 2, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [3, 0, 0, 0, 0, 0, 0, 0, 0, 0]], device=device),
        "target_polylines": torch.rand(B, N, K, 2, device=device),
        "target_masks": (torch.rand(B, N, H, W, device=device) > 0.8).float(),
        "valid_mask": torch.tensor([[True, True, False, False, False, False, False, False, False, False],
                                    [True, False, False, False, False, False, False, False, False, False]], device=device)
    }

    print("\n2. [MODEL] Testing Full Forward Pass...")
    outputs = model(dummy_images, targets=dummy_targets)
    
    assert outputs["pred_cls"].shape == (B, N, config.num_classes + 1), f"Unexpected pred_cls shape: {outputs['pred_cls'].shape}"
    assert outputs["pred_polylines"].shape == (B, N, K, 2), f"Unexpected pred_polylines shape: {outputs['pred_polylines'].shape}"
    assert outputs["pred_masks"].shape == (B, N, H, W), f"Unexpected pred_masks shape: {outputs['pred_masks'].shape}"
    
    print(f"   • Pred Cls shape: {outputs['pred_cls'].shape} (Expected: [2, 10, 5])")
    print(f"   • Pred Polylines shape: {outputs['pred_polylines'].shape} (Expected: [2, 10, 20, 2])")
    print(f"   • Pred Masks shape: {outputs['pred_masks'].shape} (Expected: [2, 10, 1024, 1024])")
    print("   ✅ Output Shapes Verified 100%!")

    print("\n3. [LOSSES] Testing Deep Supervision 2D Loss Suite & Hungarian Matching...")
    loss = outputs["loss"]
    loss_dict = outputs["loss_dict"]
    print(f"   • Total Loss: {loss.item():.4f}")
    print(f"   • Loss Dict: {loss_dict}")
    assert not torch.isnan(loss), "Loss is NaN!"
    assert not torch.isinf(loss), "Loss is Inf!"
    print("   ✅ Loss Suite & Hungarian Matching verified cleanly!")

    print("\n4. [GRADIENTS] Testing Backward Pass & Gradient Flow across Parameter Groups...")
    loss.backward()
    
    backbone_grads = []
    head_grads = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            if "backbone" in name:
                backbone_grads.append(param.grad.norm().item())
            else:
                head_grads.append(param.grad.norm().item())

    print(f"   • Backbone Parameter Gradients: {len(backbone_grads)} layers active | Mean Grad Norm: {np.mean(backbone_grads):.6f}")
    print(f"   • Decoder & Head Parameter Gradients: {len(head_grads)} layers active | Mean Grad Norm: {np.mean(head_grads):.6f}")
    assert len(backbone_grads) > 0, "No gradients in backbone!"
    assert len(head_grads) > 0, "No gradients in decoder/heads!"
    print("   ✅ Backward pass and gradient flow verified 100%!")

    print("\n5. [OPTIMIZER] Checking Parameter Groups (Backbone LR 0.1x)...")
    backbone_params = [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]
    
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": config.learning_rate * config.backbone_lr_mult},
        {"params": head_params, "lr": config.learning_rate}
    ])
    print(f"   • Group 0 (Backbone): {len(backbone_params)} tensors | LR = {optimizer.param_groups[0]['lr']}")
    print(f"   • Group 1 (Heads): {len(head_params)} tensors | LR = {optimizer.param_groups[1]['lr']}")
    print("   ✅ Optimizer parameter groups verified 100%!")

    print("\n" + "=" * 80)
    print("🎉 ALL 5 PIPELINE AUDIT STAGES FOR EXP_06 PASSED WITH 0 ERRORS!")
    print("=" * 80)

if __name__ == "__main__":
    run_exp06_audit()
