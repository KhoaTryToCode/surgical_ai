import os
import sys
import torch

exp_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if exp_root not in sys.path:
    sys.path.insert(0, exp_root)

from configs.exp10_config import EXP10Config
from models.macro_patch_vit import MacroPatchViT
from models.macro_losses import MacroPatchLoss
from models.macro_merger import merge_macro_beziers_to_image
from utils.dataset_macro_vit import MacroPatchLandmarkDataset


def run_smoke_test():
    print("=" * 75)
    print("🔬 [EXP_10 SMOKE TEST] Macro-Patch Geometric ViT (Way A: 64px Grid)")
    print("=" * 75)
    
    config = EXP10Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Execution Device: {device}")
    
    # -------------------------------------------------------------
    # 1. Dataset Sanity Check
    # -------------------------------------------------------------
    print("\n📦 [1/4] Testing MacroPatchLandmarkDataset...")
    dataset = MacroPatchLandmarkDataset(
        dataset_dir="data/laparoscopic_liver",
        mode="train",
        image_size=config.image_size,
        macro_patch_size=config.macro_patch_size,
        use_depth=config.use_depth
    )
    sample = dataset[0]
    
    img = sample["image"].unsqueeze(0).to(device)                      # (1, 4, 512, 512)
    target_classes = sample["target_classes"].unsqueeze(0).to(device)  # (1, 8, 8)
    target_beziers = sample["target_beziers"].unsqueeze(0).to(device)  # (1, 8, 8, 4, 2)
    active_mask = sample["active_mask"].unsqueeze(0).to(device)        # (1, 8, 8)
    
    print(f"  • Image Tensor Shape:        {img.shape} (RGB-D: {img.shape[1]} channels)")
    print(f"  • Target Classes Shape:      {target_classes.shape} (Grid: 8x8)")
    print(f"  • Target Béziers Shape:      {target_beziers.shape} (P0..P3 in [0, 1]^2)")
    print(f"  • Active Mask Shape:         {active_mask.shape} | Active Count: {active_mask.sum().item()}")
    
    assert img.shape == (1, 4, 512, 512)
    assert target_classes.shape == (1, 8, 8)
    assert target_beziers.shape == (1, 8, 8, 4, 2)
    print("  ✅ Macro-patch dataset loading passed.")

    # -------------------------------------------------------------
    # 2. Model Forward Pass
    # -------------------------------------------------------------
    print("\n🧠 [2/4] Testing MacroPatchViT Forward Pass...")
    model = MacroPatchViT(
        backbone_name="vit_tiny_patch16_224",  # Tiny for instant CPU smoke test
        in_chans=config.in_chans,
        pretrained=False,
        image_size=config.image_size,
        micro_patch_size=config.micro_patch_size,
        macro_patch_size=config.macro_patch_size,
        num_classes=config.num_classes,
        embed_dim=192,
        hidden_dim=256,
        macro_depth=2,
        macro_heads=4
    ).to(device)
    
    pred_dict = model(img)
    macro_logits = pred_dict["macro_logits"]   # (1, 8, 8, 5)
    macro_beziers = pred_dict["macro_beziers"] # (1, 8, 8, 4, 2)
    macro_tokens = pred_dict["macro_tokens"]   # (1, 64, 192)
    
    print(f"  • Macro-Tokens:              {macro_tokens.shape} (8x8 = 64 tokens)")
    print(f"  • Macro Logits Shape:        {macro_logits.shape} (5 classes)")
    print(f"  • Macro Béziers Shape:       {macro_beziers.shape} (P0..P3 in [0, 1]^2)")
    print(f"  • Control Point Bounds:      Min: {macro_beziers.min().item():.3f}, Max: {macro_beziers.max().item():.3f}")
    
    assert macro_logits.shape == (1, 8, 8, 5)
    assert macro_beziers.shape == (1, 8, 8, 4, 2)
    assert 0.0 <= macro_beziers.min() and macro_beziers.max() <= 1.0, "Control points outside [0, 1] range!"
    print("  ✅ Macro-patch forward pass passed.")

    # -------------------------------------------------------------
    # 3. Multi-Task Loss Computation
    # -------------------------------------------------------------
    print("\n⚖️ [3/4] Testing MacroPatchLoss...")
    criterion = MacroPatchLoss(
        lambda_cls=config.lambda_cls,
        lambda_ctrl=config.lambda_ctrl,
        lambda_sample=config.lambda_sample,
        lambda_tan=config.lambda_tan,
        lambda_cont=config.lambda_cont,
        macro_grid_size=8,
        macro_patch_size=64
    )
    
    target_dict = {
        "target_classes": target_classes,
        "target_beziers": target_beziers,
        "active_mask": active_mask
    }
    
    loss_dict = criterion(pred_dict, target_dict)
    total_loss = loss_dict["loss"]
    print(f"  • Total Loss:                {total_loss.item():.4f}")
    print(f"    - L_cls (Focal):           {loss_dict['loss_cls'].item():.4f}")
    print(f"    - L_ctrl (Smooth L1):      {loss_dict['loss_ctrl'].item():.4f}")
    print(f"    - L_sample (Curve L1):     {loss_dict['loss_sample'].item():.4f}")
    print(f"    - L_tan (Tangent):         {loss_dict['loss_tan'].item():.4f}")
    print(f"    - L_cont (Continuity):     {loss_dict['loss_cont'].item():.4f}")
    
    assert not torch.isnan(total_loss), "Loss contains NaN!"
    print("  ✅ Macro loss computation passed.")

    # -------------------------------------------------------------
    # 4. Backward Pass & Global Spatial Reconstruction
    # -------------------------------------------------------------
    print("\n🔄 [4/4] Testing Backpropagation & Global Macro-Merge...")
    total_loss.backward()
    
    assert model.macro_merge[0].weight.grad is not None, "Gradients failed to reach macro_merge conv!"
    assert model.relational_transformer.layers[0].linear1.weight.grad is not None, "Gradients failed to reach relational transformer!"
    assert model.bezier_head[1].weight.grad is not None, "Gradients failed to reach bezier_head!"
    
    print(f"  • macro_merge Grad Norm:     {model.macro_merge[0].weight.grad.norm().item():.5f}")
    print(f"  • relational_trans Grad Norm:{model.relational_transformer.layers[0].linear1.weight.grad.norm().item():.5f}")
    print(f"  • bezier_head Grad Norm:     {model.bezier_head[1].weight.grad.norm().item():.5f}")
    
    # Test global coordinate reconstruction and rendering
    render_res = merge_macro_beziers_to_image(
        macro_logits[0].detach().cpu().numpy(),
        macro_beziers[0].detach().cpu().numpy(),
        macro_patch_size=64,
        image_size=512,
        confidence_thresh=0.0  # Force rendering for testing
    )
    print(f"  • Rendered Pixel Masks Shape:{render_res['pixel_masks'].shape} (4, 512, 512)")
    print(f"  • Rendered Combined Mask:    {render_res['combined_mask'].shape}")
    assert render_res["pixel_masks"].shape == (4, 512, 512)
    print("  ✅ Global coordinate shifting and rendering passed.")

    print("\n" + "=" * 75)
    print("🎉 ALL SMOKE TESTS PASSED FOR EXP_10 Macro-Patch Geometric ViT (Way A)!")
    print("=" * 75)


if __name__ == "__main__":
    run_smoke_test()
