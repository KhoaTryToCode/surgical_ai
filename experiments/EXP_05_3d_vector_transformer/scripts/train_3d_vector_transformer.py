import os
import sys
import time
import argparse
import torch
from torch.utils.data import DataLoader

# Add experiment root to path
EXP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if EXP_DIR not in sys.path:
    sys.path.append(EXP_DIR)

from configs.exp05_config import config
from utils.dataset_3d import Surgical3DVectorDataset
from models.surgical_3d_vector_transformer import Surgical3DVectorTransformer

def parse_args():
    parser = argparse.ArgumentParser(description="Train EXP_05 3D Vector Space Transformer")
    parser.add_argument("--dataset_dir", type=str, default=config.dataset_dir, help="Path to surgical dataset")
    parser.add_argument("--epochs", type=int, default=config.num_epochs, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=config.batch_size, help="Batch size")
    parser.add_argument("--lr", type=float, default=config.learning_rate, help="Learning rate")
    parser.add_argument("--save_dir", type=str, default="checkpoints/EXP_05", help="Checkpoint save directory")
    return parser.parse_args()

def main():
    args = parse_args()
    print("=" * 70)
    print("🚀 Starting Training: EXP_05 Monocular 3D Vector Space Transformer")
    print("=" * 70)
    print(f"📁 Dataset Directory: {args.dataset_dir}")
    print(f"⚙️ Hyperparameters: Batch Size={args.batch_size}, LR={args.lr}, Epochs={args.epochs}")

    # Device selection (CUDA -> MPS -> CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"💻 Execution Device: CUDA ({torch.cuda.get_device_name(0)})")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("💻 Execution Device: Apple Silicon MPS")
    else:
        device = torch.device("cpu")
        print("💻 Execution Device: CPU")

    os.makedirs(args.save_dir, exist_ok=True)

    # 1. Dataset & DataLoader
    if not os.path.exists(args.dataset_dir):
        print(f"⚠️ Warning: Dataset path '{args.dataset_dir}' does not exist yet.")
        print("💡 Make sure to run dataset preparation script first: python shared/utils/prepare_dataset.py --target_dir /kaggle/working/L3D")
        return

    dataset = Surgical3DVectorDataset(
        dataset_dir=args.dataset_dir,
        num_instances=config.num_instances,
        num_points=config.num_points,
        mode="train"
    )
    print(f"📊 Training Dataset Size: {len(dataset)} images")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=True if len(dataset) > args.batch_size else False
    )

    # 2. Instantiate Model & Optimizer
    model = Surgical3DVectorTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    use_cuda_amp = (device.type == "cuda")
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_cuda_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_cuda_amp)

    # 3. Training Loop
    best_loss = float("inf")
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_loss_dict = {"l_cls": 0.0, "l_pos": 0.0, "l_tan": 0.0, "l_curv": 0.0, "l_mask": 0.0}
        start_time = time.time()

        pbar = tqdm(loader, desc=f"Epoch {epoch:2d}/{args.epochs:2d}", leave=False)
        for batch_idx, batch in enumerate(pbar):
            images = batch["image"].to(device)
            depth = batch["depth"].to(device)
            targets = {
                "target_classes": batch["target_classes"].to(device),
                "target_polylines": batch["target_polylines"].to(device),
                "target_masks": batch["target_masks"].to(device),
                "valid_mask": batch["valid_mask"].to(device)
            }

            optimizer.zero_grad()
            if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                autocast_ctx = torch.amp.autocast("cuda", enabled=use_cuda_amp)
            else:
                autocast_ctx = torch.cuda.amp.autocast(enabled=use_cuda_amp)

            with autocast_ctx:
                outputs = model(images, depth, targets=targets)
                loss = outputs["loss"]
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            for k, v in outputs["loss_dict"].items():
                epoch_loss_dict[k] += v

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        elapsed = time.time() - start_time
        avg_loss = epoch_loss / max(len(loader), 1)
        for k in epoch_loss_dict:
            epoch_loss_dict[k] /= max(len(loader), 1)

        print(f"Epoch [{epoch:2d}/{args.epochs:2d}] ({elapsed:.1f}s) | Total Loss: {avg_loss:.4f} | "
              f"Cls: {epoch_loss_dict['l_cls']:.3f} | Pos3D: {epoch_loss_dict['l_pos']:.3f} | "
              f"Tan: {epoch_loss_dict['l_tan']:.3f} | Mask: {epoch_loss_dict['l_mask']:.3f}", flush=True)

        # Save Best Checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = os.path.join(args.save_dir, "best_surgical_3d_vector.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_loss": best_loss,
                "config": config
            }, ckpt_path)
            print(f"  💾 Best model checkpoint saved to '{ckpt_path}'")

    print("=" * 70)
    print(f"✅ EXP_05 Training Finished cleanly! Best Loss = {best_loss:.4f}")
    print("=" * 70)

if __name__ == "__main__":
    main()
