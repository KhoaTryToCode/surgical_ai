import os
import sys
import torch

# Ensure experiment directory is in sys.path
exp_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if exp_root not in sys.path:
    sys.path.insert(0, exp_root)

from configs.exp10_config import EXP10Config
from models.super_token_vit import SuperTokenGeometricViT
from models.dual_domain_loss import DualDomainGeometricLoss
from utils.dataset_super_token import SuperTokenLandmarkDataset


def run_smoke_test():
    print("=" * 75)
    print("🔬 [EXP_10 SMOKE TEST] Super-Token Geometric ViT")
    print("=" * 75)
    
    config = EXP10Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Execution Device: {device}")
    
    # -------------------------------------------------------------
    # 1. Dataset Sanity Check
    # -------------------------------------------------------------
    print("\n📦 [1/4] Testing SuperTokenLandmarkDataset...")
    dataset = SuperTokenLandmarkDataset(
        dataset_dir="data/laparoscopic_liver",
        mode="train",
        image_size=config.image_size,
        patch_size=config.patch_size,
        num_ctrl_points=config.num_ctrl_points,
        render_size=128,
        use_depth=config.use_depth
    )
    sample = dataset[0]
    
    img = sample["image"].unsqueeze(0).to(device)                      # (1, 4, 512, 512)
    target_exists = sample["target_exists"].unsqueeze(0).to(device)    # (1, 4)
    target_ctrl = sample["target_ctrl_points"].unsqueeze(0).to(device) # (1, 4, 6, 2)
    target_attn = sample["target_attn_masks"].unsqueeze(0).to(device)  # (1, 4, 32, 32)
    target_render = sample["target_render_masks"].unsqueeze(0).to(device) # (1, 4, 128, 128)
    
    print(f"  • Image Tensor Shape:        {img.shape} (RGB-D: {img.shape[1]} channels)")
    print(f"  • Target Exists Shape:       {target_exists.shape}")
    print(f"  • Target Ctrl Points Shape:  {target_ctrl.shape} (K={config.num_ctrl_points})")
    print(f"  • Target Attn Mask Shape:    {target_attn.shape} (Grid: 32x32)")
    print(f"  • Target Render Mask Shape:  {target_render.shape} (Render: 128x128)")
    
    assert img.shape == (1, 4, 512, 512), f"Unexpected img shape: {img.shape}"
    assert target_ctrl.shape == (1, 4, config.num_ctrl_points, 2), f"Unexpected ctrl shape: {target_ctrl.shape}"
    print("  ✅ Dataset loading passed.")

    # -------------------------------------------------------------
    # 2. Model Forward Pass
    # -------------------------------------------------------------
    print("\n🧠 [2/4] Testing SuperTokenGeometricViT Forward Pass...")
    # Using modular fallback or lightweight config for instant smoke test execution
    model = SuperTokenGeometricViT(
        backbone_name="vit_tiny_patch16_224",  # Tiny for instantaneous CPU testing
        in_chans=config.in_chans,
        pretrained=False,
        image_size=config.image_size,
        patch_size=config.patch_size,
        num_classes=config.num_classes,
        num_ctrl_points=config.num_ctrl_points,
        embed_dim=192,                         # Tiny embed_dim
        hidden_dim=256,
        render_size=128,
        cross_attn_heads=4
    ).to(device)
    
    pred_dict = model(img)
    exist_logits = pred_dict["exist_logits"]
    exist_probs = pred_dict["exist_probs"]
    ctrl_points = pred_dict["ctrl_points"]
    attn_heatmaps = pred_dict["attn_heatmaps"]
    soft_masks = pred_dict["soft_masks"]
    cls_token = pred_dict["cls_token"]
    super_tokens = pred_dict["super_tokens"]
    
    print(f"  • CLS Organ Pose Token:      {cls_token.shape}")
    print(f"  • Super-Tokens:              {super_tokens.shape} (C={config.num_classes} landmarks)")
    print(f"  • Patch Attn Heatmaps:       {attn_heatmaps.shape}")
    print(f"  • Exist Probabilities:       {exist_probs.shape} | Values: {exist_probs.detach().cpu().numpy().round(3)}")
    print(f"  • Predicted Ctrl Points:     {ctrl_points.shape} (Min: {ctrl_points.min().item():.3f}, Max: {ctrl_points.max().item():.3f})")
    print(f"  • Soft Rendered Masks:       {soft_masks.shape}")
    
    assert exist_probs.shape == (1, config.num_classes)
    assert ctrl_points.shape == (1, config.num_classes, config.num_ctrl_points, 2)
    assert attn_heatmaps.shape == (1, config.num_classes, 32, 32)
    assert soft_masks.shape == (1, config.num_classes, 128, 128)
    assert 0.0 <= ctrl_points.min() and ctrl_points.max() <= 1.0, "Control points outside [0, 1] range!"
    print("  ✅ Model forward pass and output shapes verified.")

    # -------------------------------------------------------------
    # 3. Dual-Domain Loss Computation
    # -------------------------------------------------------------
    print("\n⚖️ [3/4] Testing DualDomainGeometricLoss...")
    criterion = DualDomainGeometricLoss(
        lambda_attn=config.lambda_attn,
        lambda_vector=config.lambda_vector,
        lambda_dice=config.lambda_dice,
        lambda_exist=config.lambda_exist
    )
    
    target_dict = {
        "target_exists": target_exists,
        "target_ctrl_points": target_ctrl,
        "target_attn_masks": target_attn,
        "target_render_masks": target_render
    }
    
    loss_dict = criterion(pred_dict, target_dict)
    total_loss = loss_dict["loss"]
    print(f"  • Total Loss:       {total_loss.item():.4f}")
    print(f"    - L_exist:        {loss_dict['loss_exist'].item():.4f}")
    print(f"    - L_attn (BCE):   {loss_dict['loss_attn'].item():.4f}")
    print(f"    - L_vector (L1):  {loss_dict['loss_vector'].item():.4f}")
    print(f"    - L_dice (Soft):  {loss_dict['loss_dice'].item():.4f}")
    
    assert not torch.isnan(total_loss), "Loss contains NaN!"
    print("  ✅ Loss function calculation passed.")

    # -------------------------------------------------------------
    # 4. End-to-End Backward Pass & Gradient Flow
    # -------------------------------------------------------------
    print("\n🔄 [4/4] Testing Backpropagation & Gradient Flow...")
    total_loss.backward()
    
    # Verify gradients reach the key architectural modules
    assert model.base_queries.grad is not None, "Gradients failed to reach base_queries!"
    assert model.pose_proj[1].weight.grad is not None, "Gradients failed to reach pose_proj!"
    assert model.cross_attention.q_proj.weight.grad is not None, "Gradients failed to reach cross_attention!"
    assert model.decoder.curve_head[-2].weight.grad is not None, "Gradients failed to reach curve_decoder!"
    
    print(f"  • base_queries Grad Norm:    {model.base_queries.grad.norm().item():.5f}")
    print(f"  • pose_proj Grad Norm:       {model.pose_proj[1].weight.grad.norm().item():.5f}")
    print(f"  • cross_attention Grad Norm: {model.cross_attention.q_proj.weight.grad.norm().item():.5f}")
    print(f"  • curve_decoder Grad Norm:   {model.decoder.curve_head[-2].weight.grad.norm().item():.5f}")
    print("  ✅ Full end-to-end gradient backpropagation verified.")

    print("\n" + "=" * 75)
    print("🎉 ALL SMOKE TESTS PASSED FOR EXP_10 Super-Token Geometric ViT!")
    print("=" * 75)


if __name__ == "__main__":
    run_smoke_test()
