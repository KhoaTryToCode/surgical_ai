# Run Commands — EXPERIMENT_15: Temporal Mask2Former

## 1. Local Environment Execution (macOS / Local GPU)

### Train
```bash
# From workspace root
python3 experiments/EXPERIMENT_15_temporal_augmented_mask2former/scripts/train.py \
  --config experiments/EXPERIMENT_15_temporal_augmented_mask2former/configs/temporal_m2f_t3.yaml \
  --save-dir checkpoints/exp15_temporal_m2f
```

### Evaluate
```bash
# Run official evaluation on test split (109 frames)
python3 experiments/EXPERIMENT_15_temporal_augmented_mask2former/scripts/evaluate.py \
  --checkpoint checkpoints/exp15_temporal_m2f/best_temporal_m2f.pth \
  --split test \
  --output-dir experiments/EXPERIMENT_15_temporal_augmented_mask2former/eval_results
```

---

## 2. Kaggle CUDA Environment Execution

### Option A: Interactive Notebook Run
1. Open Kaggle and create a new notebook with GPU accelerator (Tesla T4 or P100).
2. Attach datasets: `l3d-train`, `l3d-val`, `l3d-test` (from `khoatrytopublish`) or any unified L3D dataset.
3. Upload `experiments/EXPERIMENT_15_temporal_augmented_mask2former/notebooks/EXP15_Temporal_Mask2Former_Kaggle.ipynb`.
4. Click **Run All**.
5. All 30 epochs will execute with Teacher Forcing on training, evaluate validation at each epoch, and output the final test benchmark results to `/kaggle/working/test_summary_metrics.json`.

### Option B: Kaggle CLI Execution
```bash
kaggle kernels push -p experiments/EXPERIMENT_15_temporal_augmented_mask2former/notebooks
```

---

## 3. Rebuilding the Standalone Kaggle Notebook
If modifications are made to model or dataset logic, recompile the notebook with:
```bash
python3 experiments/EXPERIMENT_15_temporal_augmented_mask2former/scripts/build_kaggle_notebook.py
python3 experiments/EXPERIMENT_15_temporal_augmented_mask2former/scripts/build_visual_inspection_notebook.py
```

---

## 4. Visual Inspection Execution (Validation & Test Sets)
To run full inference on both Val (122 frames) and Test (109 frames) and generate 4-panel visual comparison strips with instant zip download:
1. Upload `experiments/EXPERIMENT_15_temporal_augmented_mask2former/notebooks/EXP15_Visual_Inspection_Kaggle.ipynb` to Kaggle.
2. Set `CHECKPOINT_PATH` to your trained `.pth` weight path.
3. Run the notebook to generate `/kaggle/working/exp15_visual_inspections.zip` containing all 231 side-by-side comparative images.
