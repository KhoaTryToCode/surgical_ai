# Run Commands: EXPERIMENT_2 — Mask2Former Component Ablation Study (Set A Suite)

---

## 1. Remote HPC Server Execution via Slurm (`gpu-a240`)

The primary training environment is the `gpu-a240` cluster (`100.82.42.48`). All jobs are submitted via Slurm (`sbatch`) and operate entirely within `/data/khoalq/`.

### Storage & Environment Reference:
- **Repository:** `/data/khoalq/surgical_ai`
- **Environment:** `/data/khoalq/miniconda3/envs/surgical_ai`
- **Dataset:** `/data/khoalq/data/L3D/` (`Train`, `Val`, `Test`)
- **Checkpoints:** `/data/khoalq/checkpoints/`
- **Logs:** `/data/khoalq/logs/`

---

### Option A: Dedicated Full A100 GPU (40GB VRAM — Fast ~3 min/epoch)
Targets GPU 0 (`#SBATCH --gres=gpu:a100:1`) with micro-batch size 4 and gradient accumulation 1 (effective batch size = 4):

```bash
# Submit all 4 Set A ablation runs sequentially:
sbatch /data/khoalq/surgical_ai/experiments/EXPERIMENT_2/scripts/run_mask2former_suite.sbatch

# Or submit a specific ablation mode:
# [Run 0] Baseline Control (Pretrained Mask2Former Swin-Tiny -> Replicates 0.68 DSC Benchmark)
sbatch /data/khoalq/surgical_ai/experiments/EXPERIMENT_2/scripts/run_mask2former_suite.sbatch baseline

# [Run 1] Spatial Gating Ablation (w/o Masked Cross-Attention -> Full Global Attention)
sbatch /data/khoalq/surgical_ai/experiments/EXPERIMENT_2/scripts/run_mask2former_suite.sbatch wo_masked_attn

# [Run 2] Scale Engine Ablation (w/o Multi-Scale Feature Cycling -> Stride 16 Only)
sbatch /data/khoalq/surgical_ai/experiments/EXPERIMENT_2/scripts/run_mask2former_suite.sbatch wo_multiscale

# [Run 3] Query Interaction Ablation (w/o Query Self-Attention -> Independent Parallel Queries)
sbatch /data/khoalq/surgical_ai/experiments/EXPERIMENT_2/scripts/run_mask2former_suite.sbatch wo_self_attn
```

---

### Option B: Partitioned 10GB GPU (MIG `gpu:a100_2g.10gb:1`)
Targets a 10GB MIG slice on GPU 1 with micro-batch size 1 and gradient accumulation 4 (effective batch size = 4):

```bash
# Submit all 4 ablation runs sequentially:
sbatch /data/khoalq/surgical_ai/experiments/EXPERIMENT_2/scripts/run_mask2former_10gb.sbatch

# Or submit a specific ablation mode:
sbatch /data/khoalq/surgical_ai/experiments/EXPERIMENT_2/scripts/run_mask2former_10gb.sbatch baseline
```

---

### Option C: Direct Interactive GPU Execution
If executing within an interactive session (`srun --pty --gres=gpu:a100:1 bash`):

```bash
source "/data/khoalq/miniconda3/etc/profile.d/conda.sh"
conda activate surgical_ai
export PYTHONPATH=/data/khoalq/surgical_ai:$PYTHONPATH
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u /data/khoalq/surgical_ai/experiments/EXPERIMENT_2/scripts/train_server.py \
    --ablation baseline \
    --epochs 60 \
    --batch_size 4 \
    --accumulation_steps 1 \
    --train_dir /data/khoalq/data/L3D/Train \
    --val_dir /data/khoalq/data/L3D/Val \
    --test_dir /data/khoalq/data/L3D/Test \
    --save_dir /data/khoalq/checkpoints/mask2former_baseline_60ep \
    --eval_splits both
```

---

### Cluster Monitoring Commands:
```bash
# View active and queued jobs:
squeue -u khoalq

# Check real-time GPU slice allocation:
scontrol show node gpu-a240 | grep -E 'CfgTRES|AllocTRES'

# Live stream training log:
tail -f /data/khoalq/logs/mask2former_baseline.log
```

---

## 2. Dedicated Kaggle Execution (1-Click Run & Download)

Each ablation mode has a self-contained Jupyter notebook located in `experiments/EXPERIMENT_2/`:

| Target Mode | Notebook File | Output Zip File (1-Click Download) | Description |
| :--- | :--- | :--- | :--- |
| **Run 0: Control Baseline (0.68 Benchmark)** | `Mask2Former_Run_0_Baseline.ipynb` | `EXPERIMENT_2_RESULTS_RUN_0_BASELINE.zip` | Pretrained Swin-Tiny with native Masked Attention + Multi-Scale + Self-Attention |
| **Run 1: Spatial Gating Ablation** | `Mask2Former_Run_1_wo_MaskedAttn.ipynb` | `EXPERIMENT_2_RESULTS_RUN_1_WO_MASKED_ATTN.zip` | Pretrained Swin-Tiny with Full Global Cross-Attention (`attn_mask = None`) |
| **Run 2: Scale Engine Ablation** | `Mask2Former_Run_2_wo_MultiScale.ipynb` | `EXPERIMENT_2_RESULTS_RUN_2_WO_MULTISCALE.zip` | Pretrained Swin-Tiny with Single-Scale Feature Map (stride 16 only) |
| **Run 3: Query Interaction Ablation** | `Mask2Former_Run_3_wo_SelfAttn.ipynb` | `EXPERIMENT_2_RESULTS_RUN_3_WO_SELF_ATTN.zip` | Pretrained Swin-Tiny with Independent Parallel Queries (w/o Query Self-Attn) |

### How to Run on Kaggle:
1. Open [Kaggle](https://www.kaggle.com/) -> **New Notebook**.
2. Go to **File -> Upload Notebook** and select one of the 4 `.ipynb` files above.
3. In the right-hand settings panel:
   - **Accelerator:** Set to **GPU T4 x2** (or GPU P100).
   - **Persistence:** Set to **Variables and Files**.
   - **Internet:** Set to **On** (to download Hugging Face pretrained weights).
4. Click **+ Add Input** and attach your L3D dataset.
5. Click **Run All**.
6. When execution completes, find the `.zip` file in `/kaggle/working/` on the right sidebar and click **Download**.

---

## 3. Local macOS Verification (MPS / CPU)

To verify dataset loading and a 1-iteration dry run locally:
```bash
/opt/anaconda3/envs/surgical_ai/bin/python experiments/EXPERIMENT_2/scripts/train_server.py \
    --ablation baseline \
    --smoke_test \
    --train_dir data/L3D/Train \
    --val_dir data/L3D/Val \
    --test_dir data/L3D/Test \
    --save_dir experiments/EXPERIMENT_2/results/smoke_test
```

---

## 4. Visual Diagnostics & Multi-Model Comparative Viewer

To visually inspect and compare predictions across all 9 experimental configurations (Mask2Former EXP_2 and TopoNet EXP_1) on the challenging Patient 40 validation sequence (`08460` to `09390`):

```bash
# Open directly in your default browser on macOS:
open "experiments/patient_40_diagnostics_comparison.html"

# Or serve via local HTTP server:
python -m http.server 8000
# Then navigate to: http://localhost:8000/experiments/patient_40_diagnostics_comparison.html
```

### Visualizer Features:
- **Sequential Timeline Scrubber:** Quick buttons (`<`, `>`) and 18 clickable thumbnail pills.
- **Keyboard Controls:** `Left Arrow` (Prev), `Right Arrow` (Next), `Space` (Play/Pause auto-advance).
- **View Focus Switcher:** Full 4-Panel Montage (`RGB | GT | Pred | Error`), Prediction Only, GT vs Pred, and Error Map Only.
- **Filter Chips:** View All (9 models), Mask2Former Suite (4 models), TopoNet Suite (5 models), or Baseline Head-to-Head.
- **Per-Frame Scorecard:** Macro Dice, FG Dice, ASSD, and per-class breakdown (Ridge, Silhouette, Falciform) updated dynamically.
