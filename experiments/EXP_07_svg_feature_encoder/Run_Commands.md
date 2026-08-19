# EXP_07 Execution & Run Commands

## Local macOS Run Command

```bash
# Run the side-by-side comparative visualization (synthetic surgical frame)
python3 experiments/EXP_07_svg_feature_encoder/scripts/visualize_encoder_comparison.py

# Run on a custom surgical image
python3 experiments/EXP_07_svg_feature_encoder/scripts/visualize_encoder_comparison.py \
    --image_path /path/to/surgical_frame.png \
    --output_path experiments/EXP_07_svg_feature_encoder/outputs/custom_comparison.png
```

---

## Kaggle Environment Run Command

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
