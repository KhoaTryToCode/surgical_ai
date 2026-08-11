import os
import glob
import cv2
import gc
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path

# ==================== ABLATION CONFIGURATION ====================
# Experiment Modes:
#   1. "Swin_MaskedAttn"   : Swin-Tiny Backbone  + Masked Attention (Baseline, NO Topo Loss)
#   2. "ResNet_MaskedAttn" : ResNet-50 Backbone  + Masked Attention (Backbone Ablation)
#   3. "Swin_FullAttn"     : Swin-Tiny Backbone  + Full Global Attention (Attention Ablation)
#   4. "ResNet_FullAttn"   : ResNet-50 Backbone  + Full Global Attention (Combined Ablation)

DEFAULT_MODE = "Swin_MaskedAttn"

parser = argparse.ArgumentParser(description="Mask2Former 2x2 Factorial Ablation Study")
parser.add_argument("--mode", type=str, default=DEFAULT_MODE, 
                    choices=["Swin_MaskedAttn", "ResNet_MaskedAttn", "Swin_FullAttn", "ResNet_FullAttn"],
                    help="Ablation mode to run")
parser.add_argument("--data_path", type=str, default="/kaggle/working/L3D", help="Dataset path")
parser.add_argument("--save_dir", type=str, default="/kaggle/working/results_ablation", help="Save directory")
parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs (default: 60)")
parser.add_argument("--lr", type=float, default=8e-5, help="Learning rate")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size per step")
parser.add_argument("--accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
args, unknown = parser.parse_known_args()

ABLATION_MODE = args.mode
EPOCHS = args.epochs

print(f"==================================================")
print(f"🚀 RUNNING MASK2FORMER ABLATION EXPERIMENT: {ABLATION_MODE}")
print(f"   Epochs: {EPOCHS}")
print(f"   Topological Loss: DISABLED (Standard BCE + Dice Loss only)")
print(f"==================================================")

# 0. FREE PREVIOUS GPU MEMORY CACHE
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# 1. Install Dependencies if missing
try:
    # pyrefly: ignore [missing-import]
    import transformers
except ImportError:
    os.system("pip install -q transformers")
    # pyrefly: ignore [missing-import]
    import transformers

try:
    import wandb
except ImportError:
    os.system("pip install -q wandb")
    import wandb

# pyrefly: ignore [missing-import]
from transformers import (
    AutoImageProcessor,
    Mask2FormerForUniversalSegmentation,
    MaskFormerForInstanceSegmentation
)

# ── Setup paths ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(os.path.dirname(EXP_DIR))

for p in [SCRIPT_DIR, os.path.join(EXP_DIR, 'models'), os.path.join(REPO_ROOT, 'shared'), REPO_ROOT]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

try:
    from utils.prepare_dataset import get_split
    from utils.dataset import load_image, load_mask
except ImportError:
    from shared.utils.prepare_dataset import get_split
    from shared.utils.dataset import load_image, load_mask

try:
    # pyrefly: ignore [missing-import]
    import surface_distance
    # pyrefly: ignore [missing-import]
    from surface_distance import metrics as sd_metrics
    HAS_SURFACE_DIST = True
except Exception:
    HAS_SURFACE_DIST = False

# Official TopoNet metric evaluation function
def evaluation(pred, gt):
    smooth = 1e-5
    intersection = np.sum(pred * gt)
    dice = (2 * intersection + smooth) / (np.sum(pred) + np.sum(gt) + smooth)
    iou = dice / (2 - dice)
    return iou, dice

def compute_toponet_metrics(pred_map, gt_2d):
    pred_channels = np.array([pred_map == i for i in range(4)]).astype(np.uint8)
    gt_channels = np.array([gt_2d == i for i in range(4)]).astype(np.uint8)
    iou, dice = evaluation(pred_channels[1:].flatten(), gt_channels[1:].flatten())
    
    assd = None
    if HAS_SURFACE_DIST:
        if 0 == np.count_nonzero(pred_channels[1:]):
            assd = 80.0
        else:
            temp_assd = []
            for i in range(3):
                sd = sd_metrics.compute_surface_distances(
                    np.array(gt_channels[i + 1], dtype=bool),
                    np.array(pred_channels[i + 1], dtype=bool),
                    (1.0, 1.0)
                )
                avg_sd = surface_distance.compute_average_surface_distance(sd)
                temp_assd.append(avg_sd[1])
            if np.mean(temp_assd) < 500:
                assd = np.mean(temp_assd)
            else:
                assd = 80.0
                
    return dice, iou, assd

# Helper function to disable Masked Attention in Transformer Decoder
def disable_masked_attention(model):
    """
    Patches the Transformer Decoder layers in Hugging Face Mask2Former
    to ignore the predicted attention mask (attn_mask=None), converting Masked Attention
    into Standard Full/Global Cross-Attention across all spatial pixels.
    """
    disabled_count = 0
    for module in model.modules():
        # Target Mask2Former Transformer Decoder Layers
        if module.__class__.__name__ == "Mask2FormerTransformerDecoderLayer":
            original_forward = module.forward
            def make_unmasked_forward(orig_fn):
                def unmasked_forward(*args, **kwargs):
                    kwargs['attn_mask'] = None
                    return orig_fn(*args, **kwargs)
                return unmasked_forward
            module.forward = make_unmasked_forward(original_forward)
            disabled_count += 1
        elif hasattr(module, "cross_attn"):
            original_cross_attn = module.cross_attn.forward
            def make_unmasked_cross_attn(orig_fn):
                def unmasked_cross_attn(*args, **kwargs):
                    kwargs['attn_mask'] = None
                    if 'attention_mask' in kwargs:
                        kwargs['attention_mask'] = None
                    return orig_fn(*args, **kwargs)
                return unmasked_cross_attn
            module.cross_attn.forward = make_unmasked_cross_attn(original_cross_attn)

    print(f"⚠️ Masked Attention DISABLED ({disabled_count} decoder layers modified -> FULL GLOBAL ATTENTION mode active).")
    return model

# ==================== WANDB LOGIN & SETUP ====================
student_id = "10423057"
api_key = os.environ.get("WANDB_API_KEY", "83f4544a22543e319c6009abceaac90b634c68a3")

if api_key == "":
    print("⚠️ Warning: WANDB_API_KEY not set. Running WandB in offline mode.")
    wandb.init(mode="offline")
else:
    print("WandB API key detected. Logging into WandB...")
    wandb.login(key=api_key)

# ==================== MODEL & BACKBONE SELECTION ====================
if "ResNet" in ABLATION_MODE:
    MODEL_NAME = "facebook/maskformer-resnet50-ade"
    BACKBONE_NAME = "ResNet-50"
else:
    MODEL_NAME = "facebook/mask2former-swin-tiny-ade-semantic"
    BACKBONE_NAME = "Swin-Tiny"

USE_MASKED_ATTENTION = ("MaskedAttn" in ABLATION_MODE)

SAVE_DIR = os.path.join(args.save_dir, ABLATION_MODE.lower())
os.makedirs(SAVE_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} | Backbone: {BACKBONE_NAME} | Masked Attention: {USE_MASKED_ATTENTION}")

BEST_CKPT_PATH = os.path.join(SAVE_DIR, f"best_{ABLATION_MODE.lower()}.pth")
LATEST_CKPT_PATH = os.path.join(SAVE_DIR, f"latest_{ABLATION_MODE.lower()}.pth")

# Initialize WandB Run
wandb.init(
    project="liver-landmark-segmentation-ablation",
    name=f"Ablation-{ABLATION_MODE}",
    id=f"ablation_{ABLATION_MODE.lower()}_{student_id}",
    resume="allow",
    config={
        "student_id": student_id,
        "ablation_mode": ABLATION_MODE,
        "backbone": BACKBONE_NAME,
        "masked_attention": USE_MASKED_ATTENTION,
        "topological_loss": False,
        "epochs": EPOCHS,
        "effective_batch_size": args.batch_size * args.accumulation_steps,
        "lr": args.lr,
        "weight_decay": 3e-5,
        "resolution": "1024x1024"
    }
)

# ==================== DATASET DEFINITION ====================
class Mask2FormerDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        rgb_img = load_image(img_path)
        gt_masks = load_mask(img_path)
        gt_2d = np.argmax(gt_masks, axis=0).astype(np.int32)
        return rgb_img, gt_2d, str(img_path)

if not os.path.exists(args.data_path):
    print(f"⚠️ Warning: Dataset path '{args.data_path}' does not exist on local machine. Ensure this runs on Kaggle where dataset is present.")

train_files, test_files, val_files = get_split(args.data_path)
print(f"Train samples: {len(train_files)} | Val samples: {len(val_files)}")

train_dataset = Mask2FormerDataset(train_files)
val_dataset = Mask2FormerDataset(val_files)

# ==================== MODEL & PROCESSOR INITIALIZATION ====================
processor = AutoImageProcessor.from_pretrained(
    MODEL_NAME,
    reduce_labels=False,
    ignore_index=255
)

if "ResNet" in ABLATION_MODE:
    print(f"✅ Initializing ResNet-50 backbone model from '{MODEL_NAME}'...")
    model = MaskFormerForInstanceSegmentation.from_pretrained(
        MODEL_NAME,
        num_labels=4,
        ignore_mismatched_sizes=True
    ).to(device)
else:
    print(f"✅ Initializing Swin-Tiny backbone model from '{MODEL_NAME}'...")
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        MODEL_NAME,
        num_labels=4,
        ignore_mismatched_sizes=True
    ).to(device)

# Apply Attention Ablation if Full Attention selected
if not USE_MASKED_ATTENTION:
    model = disable_masked_attention(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=3e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

# ==================== RESUME FROM CHECKPOINT ====================
start_epoch = 1
best_val_dice = 0.0

if os.path.exists(LATEST_CKPT_PATH):
    print(f"\n🔄 Found checkpoint at '{LATEST_CKPT_PATH}', resuming training...")
    checkpoint = torch.load(LATEST_CKPT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_dice = checkpoint['best_val_dice']
    print(f"   Resuming from epoch {start_epoch} | Best Val Dice so far: {best_val_dice:.4f}")
else:
    print(f"\n🆕 Starting training from scratch for {EPOCHS} epochs.")

# ==================== TRAINING LOOP ====================
for epoch in range(start_epoch, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    train_dices = []
    train_ious = []
    optimizer.zero_grad()

    pbar = tqdm(enumerate(train_dataset), total=len(train_dataset), desc=f"Epoch {epoch}/{EPOCHS} [{ABLATION_MODE}]")
    for step, (rgb_img, gt_2d, _) in pbar:

        inputs = processor(
            images=[rgb_img],
            segmentation_maps=[gt_2d],
            return_tensors="pt"
        )

        pixel_values = inputs["pixel_values"].to(device)
        mask_labels = [m.to(device) for m in inputs["mask_labels"]]
        class_labels = [c.to(device) for c in inputs["class_labels"]]

        outputs = model(
            pixel_values=pixel_values,
            mask_labels=mask_labels,
            class_labels=class_labels
        )

        # Standard Mask2Former Loss (BCE + Dice per Query) - NO Topological Loss
        step_loss = outputs.loss

        loss = step_loss / args.accumulation_steps
        loss.backward()
        
        total_loss += step_loss.item()

        if (step + 1) % args.accumulation_steps == 0 or (step + 1) == len(train_dataset):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        with torch.no_grad():
            pred_map = processor.post_process_semantic_segmentation(
                outputs, target_sizes=[(1024, 1024)]
            )[0].cpu().numpy()
            t_dice, t_iou, t_assd = compute_toponet_metrics(pred_map, gt_2d)
            train_dices.append(t_dice)
            train_ious.append(t_iou)

        pbar.set_postfix({
            "loss": f"{total_loss / (step + 1):.4f}", 
            "tr_dice": f"{np.mean(train_dices):.4f}"
        })

    current_lr = scheduler.get_last_lr()[0]
    scheduler.step()

    # ==================== EVALUATION ON VAL SPLIT ====================
    model.eval()
    val_losses = []
    val_dices = []
    val_ious = []
    val_assds = []

    with torch.no_grad():
        for rgb_img, gt_2d, _ in tqdm(val_dataset, desc=f"Epoch {epoch}/{EPOCHS} [Val]"):
            inputs = processor(images=[rgb_img], segmentation_maps=[gt_2d], return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            mask_labels = [m.to(device) for m in inputs["mask_labels"]]
            class_labels = [c.to(device) for c in inputs["class_labels"]]

            outputs = model(
                pixel_values=pixel_values,
                mask_labels=mask_labels,
                class_labels=class_labels
            )

            val_losses.append(outputs.loss.item())

            pred_map = processor.post_process_semantic_segmentation(
                outputs, target_sizes=[(1024, 1024)]
            )[0].cpu().numpy()

            v_dice, v_iou, v_assd = compute_toponet_metrics(pred_map, gt_2d)
            val_dices.append(v_dice)
            val_ious.append(v_iou)
            if v_assd is not None:
                val_assds.append(v_assd)

    epoch_train_loss = total_loss / len(train_dataset)
    mean_val_loss = np.mean(val_losses)
    mean_train_dice = np.mean(train_dices)
    mean_val_iou = np.mean(val_ious)
    mean_val_dice = np.mean(val_dices)
    mean_val_assd = np.mean(val_assds) if len(val_assds) > 0 else 0.0

    print_msg = f"👉 Epoch {epoch:03d} | Tr Loss: {epoch_train_loss:.4f} | Val Loss: {mean_val_loss:.4f} | Tr Dice: {mean_train_dice:.4f} | Val Dice: {mean_val_dice:.4f} | Val IoU: {mean_val_iou:.4f}"
    if len(val_assds) > 0:
        print_msg += f" | Val ASSD: {mean_val_assd:.4f}"
    print(print_msg)

    log_dict = {
        "epoch": epoch,
        "train_loss": epoch_train_loss,
        "train_dice": mean_train_dice,
        "val_loss": mean_val_loss,
        "val_dice": mean_val_dice,
        "val_iou": mean_val_iou,
        "learning_rate": current_lr
    }
    if len(val_assds) > 0:
        log_dict["val_assd"] = mean_val_assd
    wandb.log(log_dict)

    if mean_val_dice > best_val_dice:
        best_val_dice = mean_val_dice
        torch.save(model.state_dict(), BEST_CKPT_PATH)
        wandb.run.summary["best_val_dice"] = best_val_dice
        print(f"  🏆 New Best Model! Val Dice: {best_val_dice:.4f} -> '{BEST_CKPT_PATH}'")

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_dice': best_val_dice
    }, LATEST_CKPT_PATH)

wandb.finish()
print(f"\n✅ Ablation [{ABLATION_MODE}] Complete! Best Validation Dice: {best_val_dice:.4f}")
