# PyTorch & Modular Coding Conventions (Layer 3)

Guidelines for shallow-cloning external surgical repos, importing from `/repos/`, writing modular PyTorch code, and utilizing shared utilities.

---

## 1. External Repository Intake & Import Protocol (`/repos/`)

### Shallow Cloning Protocol
When cloning external repositories (`GeMap`, `TopoNet`), shallow-clone and strip the `.git` directory:

```bash
git clone --depth 1 https://github.com/cuiruize/TopoNet.git repos/TopoNet
rm -rf repos/TopoNet/.git

git clone --depth 1 https://github.com/cnzzx/GeMap.git repos/GeMap
rm -rf repos/GeMap/.git
```

### Strict Immutability Rule
Never edit or modify files inside `/repos/`. They serve purely as static baseline reference codebases.

### Dynamic Import Protocol
When custom models in `experiments/EXP_02_surgical_gemap/models/` import modules from `/repos/GeMap/` or `/repos/TopoNet/`:

```python
import sys
from pathlib import Path

# Add external repo root dynamically to sys.path
REPO_ROOT = Path(__file__).resolve().parents[3] / "repos" / "GeMap"
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
```

---

## 2. Shared Utilities (`/shared/`)

Import common evaluation metrics and image utilities from `/shared/`:

```python
from shared.utils.metrics import compute_landmark_error
```

---

## 3. PyTorch Model Architecture Standards

1. Model classes inherit from `torch.nn.Module`.
2. Decouple geometric prompt encoders (`depth_anything_v2`) from vector loss heads (`vector_losses.py`).
3. Ensure device-agnostic tensor allocation (`device=device`).
