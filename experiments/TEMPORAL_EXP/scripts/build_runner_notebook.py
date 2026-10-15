"""
Generates the clean, lightweight Kaggle Git Runner Notebook for TEMPORAL_EXP.
Eliminates code duplication and implementation drift by cloning and running
directly from the git repository on Kaggle.
"""

import json
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def build_runner_notebook():
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            },
            "accelerator": "GPU",
            "gpuClass": "standard"
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    def add_cell(cell_type, source):
        nb["cells"].append({
            "cell_type": cell_type,
            "metadata": {},
            "source": source if isinstance(source, list) else source.splitlines(keepends=True)
        })

    # Cell 0: Header Markdown
    add_cell("markdown", """# 🚀 Surgical AI: Universal Video Landmark Benchmark Runner
### (MinVIS [EXP 17] • Cutie [EXP 18] • SAM 2 [EXP 19])

This lightweight runner executes temporal experiments directly from the git repository on Kaggle GPUs.
* **Zero Implementation Drift**: Clones repository directly from GitHub so training logic and metrics are always 100% in sync with the codebase.
* **Multi-Dataset Auto-Mount**: Natively resolves Kaggle sibling datasets (`l3d-train`, `l3d-val`, `l3d-test`).
* **Canonical 5-Fold Cross-Validation**: Evaluates on locked patient splits (`_config/5fold_patient_splits.json`) and the frozen test set.
""")

    # Cell 1: Git Clone, Pull & Dependencies
    add_cell("code", """# --- Step 1: Clone or Update Repository & Install Dependencies ---
import os

REPO_URL = "https://github.com/KhoaTryToCode/surgical_ai.git"
WORKDIR = "/kaggle/working" if os.path.exists("/kaggle") else os.path.abspath(".")
REPO_DIR = os.path.join(WORKDIR, "surgical_ai")

# 1. Clone repository if not present, otherwise pull latest updates
if not os.path.exists(REPO_DIR):
    !git clone {REPO_URL} {REPO_DIR}
else:
    !cd {REPO_DIR} && git pull

# 2. Change working directory into the temporal experiment suite
EXP_DIR = os.path.join(REPO_DIR, "experiments/TEMPORAL_EXP")
%cd {EXP_DIR}

# 3. Verify git status and active commit
!git status

# 4. Ensure transformers is available (pre-installed in Kaggle)
try:
    import transformers
    print(f"✅ Transformers available: {transformers.__version__}")
except ImportError:
    !pip install -q transformers
""")

    # Cell 2: Configurable Parameters & Master Execution
    add_cell("code", """# ==============================================================================
# 🎮 CONFIGURABLE EXPERIMENT RUNNER PARAMETERS
# Modify any of these parameters to run your desired experiment and fold to your liking!
# ==============================================================================
MODEL = "minvis"          # Architecture to run: 'minvis', 'cutie', 'sam2'
FOLD_IDX = 0              # 5-fold cross-validation index: 0, 1, 2, 3, 4
EPOCHS = 40               # Total training epochs (e.g. 40 full, 1 for dry run)
BATCH_SIZE = 4            # Batch size (4 per GPU on Kaggle T4)
LR_DECODER = 1.0e-4       # Learning rate for Mask2Former decoder and temporal heads
LR_BACKBONE = 1.0e-5      # Learning rate for vision backbone (Swin-Tiny)
WEIGHT_DECAY = 0.05       # AdamW weight decay
DATA_ROOT = "data/L3D"    # Dataset root (auto-resolves on Kaggle /kaggle/input)
DEVICE = "cuda"           # Compute device: 'cuda', 'cpu'
# ==============================================================================

print(f"🚀 Launching Benchmark: Model={MODEL.upper()} | Fold={FOLD_IDX} | Epochs={EPOCHS}")

# Execute 5-Fold Training, Evaluation & Leaderboard Aggregation
!python train.py \\
    --model {MODEL} \\
    --fold_idx {FOLD_IDX} \\
    --epochs {EPOCHS} \\
    --batch-size {BATCH_SIZE} \\
    --lr-decoder {LR_DECODER} \\
    --lr-backbone {LR_BACKBONE} \\
    --weight-decay {WEIGHT_DECAY} \\
    --data-root {DATA_ROOT} \\
    --device {DEVICE} && \\
python evaluate.py \\
    --model {MODEL} \\
    --checkpoint checkpoints/{MODEL}/best_{MODEL}_fold{FOLD_IDX}.pth \\
    --fold_idx {FOLD_IDX} \\
    --split val \\
    --data-root {DATA_ROOT} \\
    --save-vis \\
    --num-vis 20 && \\
python evaluate.py \\
    --model {MODEL} \\
    --checkpoint checkpoints/{MODEL}/best_{MODEL}_fold{FOLD_IDX}.pth \\
    --fold_idx {FOLD_IDX} \\
    --split test \\
    --data-root {DATA_ROOT} \\
    --save-vis \\
    --num-vis 20 && \\
python aggregate.py
""")

    # Cell 3: Interactive Visual Inspection Display
    add_cell("code", """# --- Optional: Display Saved 4-Panel Qualitative Overlays ---
import glob
from IPython.display import Image, display

val_vis = sorted(glob.glob(f"visualizations/{MODEL}/fold{FOLD_IDX}_val/*.png"))
test_vis = sorted(glob.glob(f"visualizations/{MODEL}/fold{FOLD_IDX}_test/*.png"))

print(f"📸 Validation Visualizations ({len(val_vis)} saved):")
for p in val_vis[:3]:
    display(Image(filename=p))

print(f"📸 Frozen Test Visualizations ({len(test_vis)} saved):")
for p in test_vis[:3]:
    display(Image(filename=p))
""")

    # Write out universal runner notebook
    target_path = NOTEBOOKS_DIR / "Kaggle_Temporal_Runner.ipynb"
    with open(target_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print(f"✅ Generated Runner Notebook: '{target_path}'")


if __name__ == '__main__':
    build_runner_notebook()
