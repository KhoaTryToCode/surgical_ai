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
