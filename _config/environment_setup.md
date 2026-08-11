# Environment Setup & Surgical Dataset Registries (Layer 3)

Documents PyTorch device targeting, dynamic path resolution, and laparoscopic liver dataset mapping.

---

## 1. Hardware & Environment Specifications

### Local Development Environment
- **OS:** macOS (Apple Silicon)
- **Primary Accelerator:** Metal Performance Shaders (`mps`) / CPU
- **PyTorch Device Resolution:**
  ```python
  import torch

  def get_device() -> torch.device:
      if torch.cuda.is_available():
          return torch.device("cuda")
      elif torch.backends.mps.is_available():
          return torch.device("mps")
      return torch.device("cpu")
  ```

### Kaggle Execution Environment
- **OS:** Linux (Ubuntu base)
- **Primary Accelerator:** NVIDIA GPU (`cuda:0`, `cuda:1`)
- **Repository Root:** `/kaggle/working/Surgical_AI`

---

## 2. Dynamic Path Resolution Standard

```python
import os
from pathlib import Path

def get_data_dir() -> Path:
    """Resolves dataset directory across Kaggle and local environments."""
    kaggle_dir = Path("/kaggle/input/laparoscopic-liver-landmarks")
    local_dir = Path("./data/laparoscopic_liver")
    if kaggle_dir.exists():
        return kaggle_dir
    return local_dir
```

---

## 3. Surgical Dataset & Resource Registry

| Dataset / Resource | Local Relative Path | Kaggle Input Path | Used By Experiments | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Laparoscopic Liver Video Frames** | `data/laparoscopic_liver/images/` | `/kaggle/input/laparoscopic-liver/images` | `EXP_01`, `EXP_02` | Surgical video frame images |
| **Landmark Point Annotations** | `data/laparoscopic_liver/annotations/` | `/kaggle/input/laparoscopic-liver/annotations` | `EXP_01`, `EXP_02` | Liver anatomical landmark coordinates |
| **Depth Anything v2 Depth Maps** | `data/depth_maps/` | `/kaggle/input/depth-maps/` | `EXP_02` | Precomputed geometric depth prompt maps |
