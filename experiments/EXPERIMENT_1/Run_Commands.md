# Execution Manual: EXPERIMENT_1 — TopoNet Paper Replication & Ablations

---

## 1. Local Verification (macOS)

Run the full 4-stage pipeline verification test locally before launching any cloud jobs:

```bash
conda activate surgical_ai
python experiments/EXPERIMENT_1/scripts/smoke_test_local.py
```

Expected output:
```
================================================================================
🎉 ALL 4 LOCAL VERIFICATION TESTS PASSED WITH ZERO ERRORS!
================================================================================
```

---

## 2. Dedicated Kaggle Notebook Execution (Parallel 6x-10x Faster Workflow)

The ablation suite is split into **6 dedicated, independent Jupyter notebooks** that run in parallel without timing out or hitting Kaggle's 12-hour limit:

| Notebook | Run ID | Ablation Mode | Evaluation Splits | Output ZIP |
| :--- | :--- | :--- | :---: | :--- |
| [`TopoNet_Run_1_Full.ipynb`](TopoNet_Run_1_Full.ipynb) | `Run_1.0` | `full` (Full TopoNet) | Val (122) + Test (109) | `EXPERIMENT_1_RESULTS_FULL.zip` |
| [`TopoNet_Run_2_Baseline.ipynb`](TopoNet_Run_2_Baseline.ipynb) | `Run_1.1` | `baseline` (Standard Conv) | Val (122) | `EXPERIMENT_1_RESULTS_BASELINE.zip` |
| [`TopoNet_Run_3_wo_Lper.ipynb`](TopoNet_Run_3_wo_Lper.ipynb) | `Run_1.2` | `wo_lper` (No Betti Loss) | Val (122) | `EXPERIMENT_1_RESULTS_WO_LPER.zip` |
| [`TopoNet_Run_4_wo_Lcl.ipynb`](TopoNet_Run_4_wo_Lcl.ipynb) | `Run_1.3` | `wo_lcl` (No clDice Loss) | Val (122) | `EXPERIMENT_1_RESULTS_WO_LCL.zip` |
| [`TopoNet_Run_5_wo_Lper_Lcl.ipynb`](TopoNet_Run_5_wo_Lper_Lcl.ipynb) | `Run_1.4` | `wo_lper_lcl` (Soft Dice only) | Val (122) | `EXPERIMENT_1_RESULTS_WO_LPER_LCL.zip` |
| [`TopoNet_Run_6_wo_BTF.ipynb`](TopoNet_Run_6_wo_BTF.ipynb) | `Run_1.5` | `wo_btf` (Simple Concat) | Val (122) | `EXPERIMENT_1_RESULTS_WO_BTF.zip` |

### Step-by-Step Kaggle Instructions:
1. **Upload Desired Notebook:**
   - Go to [kaggle.com/code](https://www.kaggle.com/code) -> **New Notebook** -> **File** -> **Import Notebook**.
   - Drag and drop any of the `TopoNet_Run_*.ipynb` notebooks.
2. **Attach Datasets in Kaggle Right-Hand Sidebar:**
   - Attach the 3 image datasets: `l3d-train`, `l3d-val`, `l3d-test`.
   - Attach the precomputed depth dataset: `l3d-depth` (`khoale05/l3d-depth`).
3. **Configure GPU Accelerator:**
   - **Accelerator:** Set to **GPU T4** or **GPU P100**.
   - **Internet:** Turn **ON** (required for `pip` packages and Betti git clone).
4. **Run Notebook:**
   - Click **Save Version** -> **Save & Run All (Commit)**, or run cells interactively.
   - Precomputed depth map loading cuts per-epoch time to **~3 minutes**!
5. **Download Results:**
   - When finished, download the dedicated zip archive directly from the notebook output (e.g. `EXPERIMENT_1_RESULTS_FULL.zip`).

---

## 3. Kaggle Terminal / Script CLI Execution (Alternative)

If you prefer running via Kaggle terminal or script mode:

### Full TopoNet (Run 1.0 on Val & Test splits):
```bash
python experiments/EXPERIMENT_1/scripts/train_toponet.py \
  --train_dir /kaggle/input/datasets/khoatrytopublish/l3d-train/Train \
  --val_dir /kaggle/input/datasets/khoatrytopublish/l3d-val/Val \
  --test_dir /kaggle/input/datasets/khoatrytopublish/l3d-test/Test \
  --depth_weights /kaggle/working/checkpoints/depth_anything_v2_vitb.pth \
  --save_dir /kaggle/working/results/run_full \
  --ablation full \
  --epochs 100 \
  --batch_size 1 \
  --accumulation_steps 4 \
  --lr 8e-5 \
  --weight_decay 3e-5 \
  --eval_splits both
```

### Baseline (Run 1.1 on Val split):
```bash
python experiments/EXPERIMENT_1/scripts/train_toponet.py \
  --train_dir /kaggle/input/datasets/khoatrytopublish/l3d-train/Train \
  --val_dir /kaggle/input/datasets/khoatrytopublish/l3d-val/Val \
  --depth_weights /kaggle/working/checkpoints/depth_anything_v2_vitb.pth \
  --save_dir /kaggle/working/results/run_baseline \
  --ablation baseline \
  --epochs 100 \
  --batch_size 1 \
  --accumulation_steps 4 \
  --eval_splits val
```

### Without L_per (Betti) (Run 1.2 on Val split):
```bash
python experiments/EXPERIMENT_1/scripts/train_toponet.py \
  --train_dir /kaggle/input/datasets/khoatrytopublish/l3d-train/Train \
  --val_dir /kaggle/input/datasets/khoatrytopublish/l3d-val/Val \
  --depth_weights /kaggle/working/checkpoints/depth_anything_v2_vitb.pth \
  --save_dir /kaggle/working/results/run_wo_lper \
  --ablation wo_lper \
  --epochs 100 \
  --batch_size 1 \
  --accumulation_steps 4 \
  --eval_splits val
```

### Without L_cl (clDice) (Run 1.3 on Val split):
```bash
python experiments/EXPERIMENT_1/scripts/train_toponet.py \
  --train_dir /kaggle/input/datasets/khoatrytopublish/l3d-train/Train \
  --val_dir /kaggle/input/datasets/khoatrytopublish/l3d-val/Val \
  --depth_weights /kaggle/working/checkpoints/depth_anything_v2_vitb.pth \
  --save_dir /kaggle/working/results/run_wo_lcl \
  --ablation wo_lcl \
  --epochs 100 \
  --batch_size 1 \
  --accumulation_steps 4 \
  --eval_splits val
```

### Without L_per & L_cl (Run 1.4 on Val split):
```bash
python experiments/EXPERIMENT_1/scripts/train_toponet.py \
  --train_dir /kaggle/input/datasets/khoatrytopublish/l3d-train/Train \
  --val_dir /kaggle/input/datasets/khoatrytopublish/l3d-val/Val \
  --depth_weights /kaggle/working/checkpoints/depth_anything_v2_vitb.pth \
  --save_dir /kaggle/working/results/run_wo_lper_lcl \
  --ablation wo_lper_lcl \
  --epochs 100 \
  --batch_size 1 \
  --accumulation_steps 4 \
  --eval_splits val
```

### Without BTF (Run 1.5 on Val split):
```bash
python experiments/EXPERIMENT_1/scripts/train_toponet.py \
  --train_dir /kaggle/input/datasets/khoatrytopublish/l3d-train/Train \
  --val_dir /kaggle/input/datasets/khoatrytopublish/l3d-val/Val \
  --depth_weights /kaggle/working/checkpoints/depth_anything_v2_vitb.pth \
  --save_dir /kaggle/working/results/run_wo_btf \
  --ablation wo_btf \
  --epochs 100 \
  --batch_size 1 \
  --accumulation_steps 4 \
  --eval_splits val
```

---

## 4. Local Results Unpacking

When you download `EXPERIMENT_1_RESULTS.zip` from Kaggle to your Mac, extract it directly into `experiments/EXPERIMENT_1/results/`:

```bash
unzip -o ~/Downloads/EXPERIMENT_1_RESULTS.zip -d experiments/EXPERIMENT_1/
```

---

## 5. HPC Remote Cluster Execution (Slurm on `gpu-a240` / A100 40GB)

For running the computationally intensive topological clDice ablations (`full`, `wo_lper`, `wo_btf`) on the remote server:

### Pull Latest Optimized Code on Server:
```bash
cd /data/khoalq/surgical_ai
git pull origin main
```

### Submit Slurm Suite (10GB GPU Partition — Peak VRAM 7.51 GB):
```bash
sbatch /data/khoalq/surgical_ai/experiments/EXPERIMENT_1/scripts/run_toponet_10gb.sbatch
```

### Submit Slurm Suite (Dedicated Full A100 GPU / 40GB):
```bash
sbatch /data/khoalq/surgical_ai/experiments/EXPERIMENT_1/scripts/run_toponet_suite.sbatch
```

### Submit Individual Ablation (Optional):
```bash
# Run only full TopoNet
sbatch /data/khoalq/surgical_ai/experiments/EXPERIMENT_1/scripts/run_toponet_suite.sbatch full

# Run only without L_per
sbatch /data/khoalq/surgical_ai/experiments/EXPERIMENT_1/scripts/run_toponet_suite.sbatch wo_lper

# Run only without BTF
sbatch /data/khoalq/surgical_ai/experiments/EXPERIMENT_1/scripts/run_toponet_suite.sbatch wo_btf
```

### Monitor Live Progress:
```bash
# Check queue
squeue -u khoalq

# Watch live training output
tail -f /data/khoalq/logs/toponet_full.log

# Monitor GPU VRAM and power
watch -n 2 nvidia-smi
```
