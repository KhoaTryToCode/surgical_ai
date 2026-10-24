# EXP_07 Execution & Run Commands

## Part 1: Encoder Comparison Visualization

### Local macOS Run Command

```bash
# Run the side-by-side comparative visualization (synthetic surgical frame)
python3 experiments/EXP_07_svg_feature_encoder/scripts/visualize_encoder_comparison.py

# Run on a custom surgical image
python3 experiments/EXP_07_svg_feature_encoder/scripts/visualize_encoder_comparison.py \
    --image_path /path/to/surgical_frame.png \
    --output_path experiments/EXP_07_svg_feature_encoder/outputs/custom_comparison.png
```

### Kaggle Environment Run Command

```bash
%cd /kaggle/working/surgical_ai
!git pull

# Run comparative visualization
!python experiments/EXP_07_svg_feature_encoder/scripts/visualize_encoder_comparison.py \
    --output_path /kaggle/working/encoder_feature_comparison.png

# Display in Jupyter Notebook
from IPython.display import Image, display
display(Image('/kaggle/working/encoder_feature_comparison.png', width=1200))
```

---

## Part 2: Decoder Layer-by-Layer Forward Pass Visualization

### Local macOS Run Command

```bash
# Run with default 5 queries and 6 layers (synthetic image if no dataset)
python3 experiments/EXP_07_svg_feature_encoder/scripts/visualize_decoder_forward.py

# Custom settings
python3 experiments/EXP_07_svg_feature_encoder/scripts/visualize_decoder_forward.py \
    --num_queries 5 --num_layers 6 \
    --output_path experiments/EXP_07_svg_feature_encoder/outputs/decoder_layer_progression.png
```

### Kaggle Environment Run Command

```bash
%cd /kaggle/working/surgical_ai
!git pull

# Run decoder forward pass visualization on random L3D surgical image
!python experiments/EXP_07_svg_feature_encoder/scripts/visualize_decoder_forward.py \
    --dataset_dir /kaggle/working/L3D \
    --num_queries 5 --num_layers 6 \
    --output_path /kaggle/working/decoder_layer_progression.png

# Display in Jupyter Notebook
from IPython.display import Image, display
display(Image('/kaggle/working/decoder_layer_progression.png', width=1400))
```

---

## Part 3: Training the SVG Bézier Spline Transformer

### Kaggle Environment Run Command

```bash
%cd /kaggle/working/surgical_ai
!git pull

# Train with default 50 epochs, batch_size=4, WandB enabled
!python experiments/EXP_07_svg_feature_encoder/scripts/train_bezier_decoder.py \
    --dataset_dir /kaggle/working/L3D \
    --epochs 50 \
    --batch_size 4 \
    --lr 1e-4 \
    --save_dir /kaggle/working/checkpoints/EXP_07 \
    --wandb \
    --wandb_project Surgical_AI_Bezier \
    --wandb_run_name EXP_07_SVG_Bezier_v1
```

### Display Anchor Progression

```python
from IPython.display import Image, display
import glob

anchors = sorted(glob.glob('/kaggle/working/checkpoints/EXP_07/anchor_progression/epoch_*_train_anchor.png'))
for a in anchors[-5:]:
    display(Image(a, width=900))
```
