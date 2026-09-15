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

## 2. Kaggle Notebook Execution (Recommended / Easiest Workflow)

The entire pipeline has been packaged into a self-contained, single-click Jupyter notebook:
**[`experiments/EXPERIMENT_1/TopoNet_Ablation_Kaggle.ipynb`](TopoNet_Ablation_Kaggle.ipynb)**

### Step-by-Step Kaggle Instructions:
1. **Upload Notebook:**
   - Go to [kaggle.com/code](https://www.kaggle.com/code) -> **New Notebook** -> **File** -> **Import Notebook**.
   - Drag and drop or select `TopoNet_Ablation_Kaggle.ipynb`.
2. **Configure Settings in Kaggle Right-Hand Sidebar:**
   - **Accelerator:** Set to **GPU T4 x2** or **GPU P100**.
   - **Internet:** Turn **ON** (required for `wget` of Depth Anything V2 weights and `pip` packages).
   - **Input Datasets:** Ensure your 3 L3D datasets are attached (`l3d-train`, `l3d-val`, `l3d-test`).
3. **Choose Execution Mode in Step 7:**
   - Default: `RUN_ALL_ABLATIONS = False` -> Runs **Run 1.0 (Full TopoNet)** on both Validation (122 frames) and Test (109 frames).
   - Full Suite: Set `RUN_ALL_ABLATIONS = True` -> Runs all 6 paper ablation configurations sequentially (`full`, `baseline`, `wo_lper`, `wo_lcl`, `wo_lper_lcl`, `wo_btf`).
4. **Run Notebook:**
   - Click **Save Version** -> **Save & Run All (Commit)**, or run cells interactively.
5. **Download Results:**
   - When finished, download the automatically generated archive:
     `/kaggle/working/EXPERIMENT_1_RESULTS.zip`
   - Contains:
     - `results/run_*/summary_metrics.json`
     - `results/run_*/validation_per_frame_results.csv`
     - `results/run_*/test_per_frame_results.csv`
     - `results/run_*/visualizations_patient40/*.png` (4-panel visual diagnostic plots)
     - `results/run_*/best_model.pth`

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
