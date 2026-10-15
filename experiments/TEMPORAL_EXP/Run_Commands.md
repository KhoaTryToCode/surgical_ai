# Run Commands — TEMPORAL_EXP (Universal Video Benchmark)

Master CLI execution guide for running 5-Fold cross validation and compiling leaderboards across video architectures (`minvis`, `cutie`, `sam2`) on macOS and Kaggle.

---

## 1. Local Development (macOS)

### A. Train Any Model (e.g. MinVIS Fold 0)
```bash
python experiments/TEMPORAL_EXP/train.py \
  --model minvis \
  --fold_idx 0 \
  --epochs 40 \
  --batch-size 4 \
  --data-root data/L3D
```

*(For a quick 1-epoch dry-run test, simply pass `--epochs 1 --batch-size 2`)*

### B. Evaluate Checkpoint on Validation or Frozen Test Set
```bash
# Evaluate on Validation Fold 0
python experiments/TEMPORAL_EXP/evaluate.py \
  --model minvis \
  --checkpoint experiments/TEMPORAL_EXP/checkpoints/minvis/best_minvis_fold0.pth \
  --fold_idx 0 \
  --split val \
  --save-vis

# Evaluate on Frozen Test Set (Patients 21, 31, 41, 51)
python experiments/TEMPORAL_EXP/evaluate.py \
  --model minvis \
  --checkpoint experiments/TEMPORAL_EXP/checkpoints/minvis/best_minvis_fold0.pth \
  --split test \
  --save-vis
```

### C. Compile Master Leaderboard (All Models Combined)
Once JSON files are placed into `experiments/TEMPORAL_EXP/results/`:
```bash
python experiments/TEMPORAL_EXP/aggregate.py
```
This automatically scans all model outputs (`minvis`, `cutie`, `sam2`), calculates 5-fold Mean $\pm$ Std and Frozen Test metrics, and exports:
* `experiments/TEMPORAL_EXP/results/LEADERBOARD.csv`
* `experiments/TEMPORAL_EXP/results/LEADERBOARD.md`

---

## 2. Kaggle GPU Execution (Zero Code-Drift Git Runner)

### A. Runner Notebook
Notebook files:
* [`experiments/TEMPORAL_EXP/notebooks/Kaggle_Temporal_Runner.ipynb`](file:///Users/khoale/Downloads/Surgical%20AI/experiments/TEMPORAL_EXP/notebooks/Kaggle_Temporal_Runner.ipynb)
* [`experiments/TEMPORAL_EXP/notebooks/EXP17_MinVIS_Kaggle.ipynb`](file:///Users/khoale/Downloads/Surgical%20AI/experiments/TEMPORAL_EXP/notebooks/EXP17_MinVIS_Kaggle.ipynb)

* **Architecture**: The notebook does not hardcode model classes or training loops. It runs `git clone` / `git pull` from `https://github.com/KhoaTryToCode/surgical_ai.git`, enters `experiments/TEMPORAL_EXP/`, and directly invokes the modular CLIs (`train.py`, `evaluate.py`, `aggregate.py`).
* **Dataset Inputs**: Attach `l3d-train`, `l3d-val`, and `l3d-test` (or `laparoscopic-liver-landmarks`). The scripts automatically resolve all three sibling mounts via `resolve_all_split_dirs()`.

### B. Configurable Parameters Exposed in the Runner Cell
```python
MODEL = "minvis"          # Architecture: 'minvis', 'cutie', 'sam2'
FOLD_IDX = 0              # 5-fold cross-validation index: 0, 1, 2, 3, 4
EPOCHS = 40               # Total training epochs (e.g. 40 full, 1 for dry run)
BATCH_SIZE = 4            # Batch size (4 per GPU on Kaggle T4)
LR_DECODER = 1.0e-4       # Learning rate for Mask2Former decoder and temporal heads
LR_BACKBONE = 1.0e-5      # Learning rate for vision backbone (Swin-Tiny)
WEIGHT_DECAY = 0.05       # AdamW weight decay
DATA_ROOT = "data/L3D"    # Dataset root (auto-resolves on Kaggle /kaggle/input)
DEVICE = "cuda"           # Compute device: 'cuda', 'cpu'
```

### C. Running 5 Folds in Parallel (Under 1 Hour)
1. Open 5 Kaggle notebook sessions with GPU T4 x 2.
2. In the execution cell, set:
   * Tab 0: `FOLD_IDX = 0`
   * Tab 1: `FOLD_IDX = 1`
   * Tab 2: `FOLD_IDX = 2`
   * Tab 3: `FOLD_IDX = 3`
   * Tab 4: `FOLD_IDX = 4`
3. Click **"Save Version" -> "Save & Run All (Commit)"**.
4. When finished, download the 5 pairs of JSON outputs (`results_fold_{0..4}.json` and `test_results_fold_{0..4}.json`) into `experiments/TEMPORAL_EXP/results/minvis/`.
5. Run the master aggregator locally:
```bash
python experiments/TEMPORAL_EXP/aggregate.py
```
