# Execution Manual: EXP_12 — BCRNet Paper Replication (L3D Dataset)

Step-by-step, copy-pasteable execution cells for Kaggle GPU sessions (Single T4 / P100 / A100).
Replicates BCRNet on both **Val** and **Test** splits to verify the paper's reported benchmark (**69.57% DSC / 54.16% IoU / 43.55px ASSD**).

---

## 1. Kaggle Session Setup (Run Once per Notebook Session)

### Cell 1: Clone / Pull Repository
```bash
cd /kaggle/working
if [ ! -d "surgical_ai" ]; then
    git clone https://github.com/KhoaTryToCode/surgical_ai.git
fi
cd surgical_ai
git pull origin main
```

### Cell 2: Install Dependencies & Build CUDA Extension
```bash
cd /kaggle/working/surgical_ai
bash experiments/EXP_12_bcrnet_replication/scripts/setup_kaggle.sh
```

---

## 2. Dataset Preparation & Asset Generation

Prepares the target `/kaggle/working/L3D` directory by symlinking images, authentic `depth_AdelaiDepth` maps, labels, and precomputing any missing 5th-order Bézier GT (`s_bezier`) or SAM ViT-B embeddings (`sam`).

```bash
cd /kaggle/working/surgical_ai
export PYTHONPATH="/kaggle/working/surgical_ai/repos/BCRNet:/kaggle/working/surgical_ai:$PYTHONPATH"

python experiments/EXP_12_bcrnet_replication/scripts/prepare_data.py \
    --target_dir /kaggle/working/L3D \
    --sam_checkpoint /kaggle/working/surgical_ai/checkpoints/sam_vit_b_01ec64.pth \
    --device cuda
```

---

## 3. Training Execution (80 Epochs)

Executes the official BCRNet training schedule:
- Architecture: `TransformerPureDetector` (ResNet-50 + AdelaiDepth + SAM ViT-B + ACPI + 3-Stage HCR)
- Optimizer: Adam (`lr = 1e-5`, `weight_decay = 1e-4`, `batch_size = 2`)
- Loss Annealing: $\lambda(epoch) = 1 - \sigma((epoch - 10) / 2)$

```bash
cd /kaggle/working/surgical_ai
export PYTHONPATH="/kaggle/working/surgical_ai/repos/BCRNet:/kaggle/working/surgical_ai:$PYTHONPATH"

python experiments/EXP_12_bcrnet_replication/scripts/train_bcrnet.py \
    --config experiments/EXP_12_bcrnet_replication/configs/bcrnet_l3d.yaml \
    --data_path /kaggle/working/L3D \
    --epochs 80 \
    --bs 2 \
    --eval_period 10 \
    --save_path /kaggle/working/checkpoints/EXP_12_bcrnet_replication \
    --device cuda:0
```

> **Optional: Enable W&B Online Logging**
> Add `--wandb --wandb_key <YOUR_WANDB_KEY>` to the command above. Without keys, logging defaults gracefully to local progress tracking.

---

## 4. Evaluation & Metric Verification (Val & Test)

Evaluates the best checkpoint (`best_model.pt`) on the **Test** split (exact paper benchmark: 109 frames) or **Val** split.

### Option 1 (Recommended — Lightweight & Bulletproof):
Uses `eval_simple.py`, directly mirroring the exact validation loop from `train_bcrnet.py` that ran 70 epochs on Kaggle without a single memory issue:

```bash
cd /kaggle/working/surgical_ai
git pull origin main
export PYTHONPATH="/kaggle/working/surgical_ai/repos/BCRNet:/kaggle/working/surgical_ai:$PYTHONPATH"

python experiments/EXP_12_bcrnet_replication/scripts/eval_simple.py \
    --model_path /kaggle/working/checkpoints/EXP_12_bcrnet_replication/best_model.pt \
    --data_path /kaggle/working/L3D \
    --split Test \
    --threshold 0.3
```

### Option 2 (Comprehensive with Logging & ASSD):
Uses `evaluate_bcrnet.py` with real-time RAM/VRAM telemetry written to `/kaggle/working/eval_debug.log`:

```bash
python experiments/EXP_12_bcrnet_replication/scripts/evaluate_bcrnet.py \
    --model_path /kaggle/working/checkpoints/EXP_12_bcrnet_replication/best_model.pt \
    --data_path /kaggle/working/L3D \
    --split Test \
    --threshold 0.3
```
*(Add `--compute_assd` if you wish to compute surface distance in pixels).*

---

## 5. Paper Target Benchmark Reference

Compare your evaluation output against the published results from the BCRNet paper (MICCAI 2025):

| Split / Benchmark | DSC (%) | IoU (%) | ASSD (px) | Silhouette (%) | Ligament (%) | Ridge (%) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Paper Benchmark (Test - 109 frames)** | **69.57** | **54.16** | **43.55** | -- | -- | -- |
| **EXP_12 Measured (Val - 122 frames)** | **35.49** | **23.07** | -- | 41.37 | 20.53 | 33.83 |
| **EXP_12 Measured (Test - 109 frames)** | *(Run Option 1 or 2)* | *(Run Option 1 or 2)* | *(Optional)* | -- | -- | -- |
