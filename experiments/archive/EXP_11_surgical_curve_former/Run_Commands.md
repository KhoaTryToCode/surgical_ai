# EXP_11 Run Commands — SurgicalCurveFormer (BCRNet ACPI + HCR + EXP_10)

Exact copy-pasteable commands to run EXP_11 on **Kaggle GPU** (T4 / P100 / A100).
Architecture: SAM-ViT-B (frozen) + ResNet-50 FPN (RGB-D) + ACPI + HCR + Existence Gate + Soft Rasterizer.
Target: beat BCRNet SOTA (69.57% DSC / 54.16% IoU / 43.55px ASSD).

---

## 1. Kaggle Notebook Setup (run once per session)

```python
# ── Cell 1: Clone / pull repository ───────────────────────────────────────
import subprocess, os

REPO_URL = "https://github.com/KhoaTryToCode/surgical_ai.git"  # update if needed
WORK_DIR = "/kaggle/working/surgical_ai"

if not os.path.exists(WORK_DIR):
    subprocess.run(["git", "clone", REPO_URL, WORK_DIR], check=True)
else:
    subprocess.run(["git", "-C", WORK_DIR, "pull", "origin", "main"], check=True)

print("✅ Repo ready at", WORK_DIR)
```

```python
# ── Cell 2: Install dependencies ──────────────────────────────────────────
import subprocess
subprocess.run([
    "pip", "install", "-q",
    "timm",           # ViT-B backbone
    "wandb",          # Experiment tracking
    "scipy",          # Bernstein basis (comb)
    "opencv-python-headless",
], check=True)
print("✅ Dependencies installed")
```

```python
# ── Cell 3: Set PYTHONPATH ────────────────────────────────────────────────
import sys, os

EXP_ROOT = "/kaggle/working/surgical_ai/experiments/EXP_11_surgical_curve_former"
WS_ROOT  = "/kaggle/working/surgical_ai"

for p in [EXP_ROOT, WS_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ["PYTHONPATH"] = f"{EXP_ROOT}:{WS_ROOT}:" + os.environ.get("PYTHONPATH", "")
print("✅ PYTHONPATH set")
```

```python
# ── Cell 4: Verify GPU ────────────────────────────────────────────────────
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

---

## 2. Full Training Run — EXP_11 Run 2 (Fixed Hyperparameters)

> **Run 1 result:** 13.81% Dice. Root causes diagnosed and fixed. See below.

```bash
cd /kaggle/working/surgical_ai && git pull origin main
export PYTHONPATH="/kaggle/working/surgical_ai/experiments/EXP_11_surgical_curve_former:/kaggle/working/surgical_ai:$PYTHONPATH"

python experiments/EXP_11_surgical_curve_former/scripts/train_exp11.py \
    --dataset_dir       /kaggle/working/L3D \
    --epochs            80 \
    --batch_size        4 \
    --lr                5e-5 \
    --backbone_lr_mult  0.1 \
    --acpi_top_k        10 \
    --hcr_stages        3 \
    --anneal_center     20.0 \
    --anneal_slope      4.0 \
    --lambda_d_min      0.05 \
    --sigma_start_px    30.0 \
    --sigma_end_px      2.0 \
    --sigma_anneal_epochs 30 \
    --amp \
    --use_depth \
    --save_dir          /kaggle/working/checkpoints/EXP_11_run2 \
    --wandb \
    --wandb_key         83f4544a22543e319c6009abceaac90b634c68a3
```

### What each fix does

| Fix | Parameter | Run 1 (broken) | Run 2 (fixed) | Why |
|:---|:---|:---:|:---:|:---|
| λ_d floor | `lambda_d_min` | 0.0 | **0.05** | CNN decoder never loses gradient → no forgetting |
| Slower anneal | `anneal_center` | 10 | **20** | Dense head gets 20 full epochs before transition |
| Slower anneal | `anneal_slope` | 2 | **4** | Ramp spans 20 epochs not 8 |
| Sigma warm-start | `sigma_start_px` | 2px | **30px** | Wide Gaussian → useful gradients even when curves are far |
| Sigma decay | `sigma_end_px` | 2px | **2px** | Final precision unchanged |
| L_dice gating | (in losses.py) | constant | **scales with (1-λ_d)** | Rasterizer only dominates when curve head is active |
| Higher LR | `--lr` | 1e-5 | **5e-5** | Faster convergence for random-initialized curve head |



---

## 3. Fast Prototyping Run (20 Epochs, Fewer Proposals — Debug Mode)

Use this to confirm the full pipeline runs end-to-end before committing to 60 epochs.

```bash
python experiments/EXP_11_surgical_curve_former/scripts/train_exp11.py \
    --dataset_dir      /kaggle/working/L3D \
    --epochs           20 \
    --batch_size       4 \
    --lr               1e-5 \
    --backbone_lr_mult 0.1 \
    --acpi_top_k       5 \
    --hcr_stages       2 \
    --amp \
    --use_depth \
    --save_dir         /kaggle/working/checkpoints/EXP_11_proto
```

---

## 4. Expected Training Behavior

| Epoch | lambda_d | Focus | Watch for |
|:---:|:---:|:---|:---|
| 1–5 | ~0.99 | `L_s` + `L_ind` (dense pixel) | `L_s` should drop from ~0.9 → ~0.4 |
| 6–10 | 0.99→0.5 | Transition | `L_crv` should start decreasing |
| 11–25 | 0.5→0.05 | `L_cs` + `L_crv` (curve refinement) | Proposals converge to GT curves |
| 26–60 | ~0.01 | Curve + Existence + Dice | Val Dice should climb past 65% |
| ≥50 | ~0.00 | Fully curve-mode | Target: 72%+ Dice |

**BCRNet baseline for comparison:** 69.57% DSC / 54.16% IoU / 43.55px ASSD

---

## 5. Dataset Path Fallback Logic

The `resolve_dataset_dir()` in `configs/exp11_config.py` searches these paths in order:

```
/kaggle/working/L3D                                      ← preferred
/kaggle/input/datasets/khoatrytopublish/l3d-train/Train
/kaggle/input/l3d-train/Train
data/laparoscopic_liver                                  ← local fallback
```

If your Kaggle dataset is mounted at a different path, override with `--dataset_dir <your_path>`.

---

## 6. Architecture Summary

```
RGB + AdelaiDepth (4ch, 512×512)
  → SAM-ViT-B (frozen, 768d)           ← global semantic prior
  → ResNet-50 FPN (4ch, trainable)     ← multi-scale local features
  → FPN {f1(128²), f2(64²), f3(32²), f4(16²)}, C=256 + SAM fusion at f4
  → CNN Seg Decoder (4-level deep supervision L_s, BCRNet annealing)
  → ACPI (per-class, f4 level):
      Δb → sigma(Δb + logit(c)) → K=6 ctrl pts in (0,1)²
      Top-10 per class → Induction Loss L_ind
  → HCR (3 stages, coarse→fine):
      Stage 0: {f3, f4}  Stage 1: {f2, f3}  Stage 2: {f1, f2}
      3-way self-attn: intra-curve × inter-curve × inter-category
      → Δref_pts → updated Bézier
  → Existence Gate (CLS-pose super-token, EXP_10) → L_exist
  → Soft Gaussian Rasterizer (σ=2px) → Differentiable Dice L_dice
```

---

## 7. Saving / Restoring Checkpoints

```python
# Save manually (already handled in training loop)
import torch
torch.save(model.state_dict(), "/kaggle/working/checkpoints/EXP_11/manual_save.pth")

# Copy best checkpoint to Kaggle output for persistence
import shutil
shutil.copy(
    "/kaggle/working/checkpoints/EXP_11/best_model.pth",
    "/kaggle/working/EXP_11_best_model.pth"
)
```
