#!/usr/bin/env python3
"""
Mask2Former Kaggle Notebook Generator (EXPERIMENT_2 — Set A Component Suite)
---------------------------------------------------------------------------
Generates 4 dedicated, self-contained Kaggle runner notebooks for the Set A internal component ablations
built directly on top of the authentic pretrained Hugging Face model (`facebook/mask2former-swin-tiny-ade-semantic`):
  - Run 0: Full Standard Mask2Former (Baseline Control -> Replicates ~0.68 Dice Benchmark)
  - Run 1: Spatial Gating Ablation (w/o Masked Attention -> Full Global Cross-Attention)
  - Run 2: Scale Engine Ablation (w/o Multi-Scale Feature Cycling -> Stride-16 Single-Scale)
  - Run 3: Query Interaction Ablation (w/o Query Self-Attention -> Independent Parallel Queries)

All notebooks are completely self-contained with:
  - Scored dynamic Kaggle dataset discovery across /kaggle/input
  - Patient 32 4K canvas dynamic resolution preservation
  - Line thickness 35 standardized annotations
  - Official TopoNet Dice, IoU, and ASSD metrics
  - Deep-supervised Hungarian loss across all decoder layers
  - Multi-query ensemble post-processing
  - Evaluation on both Validation (122 frames) and Test (109 frames)
  - Automated packaging into a one-click .zip archive in /kaggle/working/
"""

import os
import json
import ast
from pathlib import Path

MODES = [
    {
        "id": "Run_0_Baseline",
        "name": "baseline",
        "ablation_type": "baseline",
        "title": "Full Standard Mask2Former (Baseline Control — 0.68 Benchmark)",
        "desc": "Pretrained Swin-Tiny Backbone + MSDeformAttn Multi-Scale Pixel Decoder + Masked Cross-Attention + Query Self-Attention.",
        "zip_name": "EXPERIMENT_2_RESULTS_RUN_0_BASELINE.zip",
        "nb_filename": "Mask2Former_Run_0_Baseline.ipynb"
    },
    {
        "id": "Run_1_wo_MaskedAttn",
        "name": "wo_masked_attn",
        "ablation_type": "wo_masked_attn",
        "title": "Ablation 1: Spatial Gating (w/o Masked Attention -> Global Cross-Attention)",
        "desc": "Pretrained Swin-Tiny with Full Global Cross-Attention across all 9 decoder layers (attn_mask = None). Isolates spatial noise filtering along thin curvilinear landmarks.",
        "zip_name": "EXPERIMENT_2_RESULTS_RUN_1_WO_MASKED_ATTN.zip",
        "nb_filename": "Mask2Former_Run_1_wo_MaskedAttn.ipynb"
    },
    {
        "id": "Run_2_wo_MultiScale",
        "name": "wo_multiscale",
        "ablation_type": "wo_multiscale",
        "title": "Ablation 2: Scale Engine (w/o Multi-Scale Feature Cycling)",
        "desc": "Pretrained Swin-Tiny with Single-Scale Feature Decoding (fixed level_index = 1, stride 16). Isolates the role of multi-scale feature pyramids in resolving thin 35px boundaries.",
        "zip_name": "EXPERIMENT_2_RESULTS_RUN_2_WO_MULTISCALE.zip",
        "nb_filename": "Mask2Former_Run_2_wo_MultiScale.ipynb"
    },
    {
        "id": "Run_3_wo_SelfAttn",
        "name": "wo_self_attn",
        "ablation_type": "wo_self_attn",
        "title": "Ablation 3: Query Interaction (w/o Query Self-Attention)",
        "desc": "Pretrained Swin-Tiny with Independent Parallel Queries bypassing the Query Self-Attention (Q * Q^T) sublayer. Isolates query coordination vs independent landmark detection.",
        "zip_name": "EXPERIMENT_2_RESULTS_RUN_3_WO_SELF_ATTN.zip",
        "nb_filename": "Mask2Former_Run_3_wo_SelfAttn.ipynb"
    }
]


def make_notebook(mode_cfg):
    mode = mode_cfg["name"]
    ablation_type = mode_cfg["ablation_type"]
    title = mode_cfg["title"]
    desc = mode_cfg["desc"]
    zip_name = mode_cfg["zip_name"]
    model_name = "facebook/mask2former-swin-tiny-ade-semantic"
    save_dir = f"results_mask2former_{mode.lower()}"

    cells = [
        # Cell 1: Header
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"# 🏥 Mask2Former Component Ablation: {title}\n",
                f"### Dedicated Kaggle GPU Runner — EXPERIMENT_2 (Set A Suite)\n",
                f"**Ablation Mode:** `{mode}` | **Output Archive:** `{zip_name}`\n",
                "\n",
                "---\n",
                "\n",
                "### Configuration Overview\n",
                f"- **Ablation Mode:** `{mode}` ({desc})\n",
                f"- **Model Base:** `{model_name}` (Authentic Pretrained Swin-Tiny Mask2Former)\n",
                "- **RGB-Only Standard Pipeline:** Pure 3-channel input `(3, 1024, 1024)`. Zero depth dependency.\n",
                "- **Patient 32 4K Canvas Bug Fix:** Dynamically extracts `imageHeight`/`imageWidth` from JSON labels to prevent coordinate truncation.\n",
                "- **Standardized Line Thickness:** Uses thickness `35` on raw canvas matching EXPERIMENT_1 bit-for-bit.\n",
                "- **Dynamic Dataset Discovery:** Automatically scans `/kaggle/input` using scored directory matching.\n",
                "- **Deep Supervision Hungarian Loss:** Evaluated natively at every decoder layer.\n",
                "- **Multi-Query Semantic Post-Processing:** Cooperative multi-query ensemble projection over all 100 queries.\n",
                "- **Automated Packaging:** Packages all outputs into `{zip_name}` directly in `/kaggle/working/` for one-click download.\n"
            ]
        },
        # Cell 2: System Check & Dependencies
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 1: Environment Diagnostics & Package Installation\n",
                "Verify CUDA acceleration, memory properties, and ensure required packages (`transformers`, `surface-distance`) are available.\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import sys\n",
                "os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'\n",
                "\n",
                "# Install required dependencies if missing\n",
                "!pip install -q transformers surface-distance medpy > /dev/null 2>&1 || true\n",
                "\n",
                "import numpy as np\n",
                "# Monkeypatch legacy NumPy aliases removed in NumPy 2.0 (for surface_distance/medpy)\n",
                "for attr, val in [('Inf', np.inf), ('Infinity', np.inf), ('NAN', np.nan), ('NaN', np.nan)]:\n",
                "    if not hasattr(np, attr):\n",
                "        setattr(np, attr, val)\n",
                "if not hasattr(np, 'bool8'):\n",
                "    np.bool8 = np.bool_\n",
                "if not hasattr(np, 'float_'):\n",
                "    np.float_ = np.float64\n",
                "\n",
                "import torch\n",
                "print('=' * 70)\n",
                "print('🚀 GPU & ENVIRONMENT DIAGNOSTICS')\n",
                "print('=' * 70)\n",
                "print(f'Python Version : {sys.version.split()[0]}')\n",
                "print(f'PyTorch Version: {torch.__version__}')\n",
                "print(f'CUDA Available : {torch.cuda.is_available()}')\n",
                "if torch.cuda.is_available():\n",
                "    print(f'Active GPU     : {torch.cuda.get_device_name(0)}')\n",
                "    print(f'Total VRAM     : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB')\n",
                "else:\n",
                "    print('⚠️ WARNING: GPU is not detected! Enable GPU Accelerator in Kaggle sidebar (Settings -> Accelerator -> GPU).')\n"
            ]
        },
        # Cell 3: Automatic Scored Dataset Discovery
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 2: Automatic Scored Dataset Discovery\n",
                "Scans `/kaggle/input` using scored heuristics to locate `Train`, `Val`, and `Test` image splits.\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import glob\n",
                "from pathlib import Path\n",
                "\n",
                "def find_dataset_split(split_keyword):\n",
                "    candidates = []\n",
                "    search_roots = ['/kaggle/input', '/kaggle/working', './data', '../data']\n",
                "    for search_root in search_roots:\n",
                "        if not os.path.exists(search_root):\n",
                "            continue\n",
                "        for root, dirs, _ in os.walk(search_root, followlinks=True):\n",
                "            parts_lower = [p.lower() for p in Path(root).parts]\n",
                "            if split_keyword.lower() in parts_lower and 'images' in parts_lower:\n",
                "                score = 0\n",
                "                if 'khoatrytopublish' in parts_lower:\n",
                "                    score += 50\n",
                "                if 'l3d' in parts_lower or any('l3d' in p for p in parts_lower):\n",
                "                    score += 30\n",
                "                if 'laparoscopic' in parts_lower:\n",
                "                    score += 20\n",
                "                candidates.append((score, root))\n",
                "            elif os.path.basename(root).lower() == split_keyword.lower():\n",
                "                if 'images' in [d.lower() for d in dirs]:\n",
                "                    img_dir = os.path.join(root, 'images')\n",
                "                    candidates.append((10, img_dir))\n",
                "    if not candidates:\n",
                "        return None\n",
                "    candidates.sort(key=lambda x: x[0], reverse=True)\n",
                "    return candidates[0][1]\n",
                "\n",
                "train_img_dir = find_dataset_split('train')\n",
                "val_img_dir = find_dataset_split('val')\n",
                "test_img_dir = find_dataset_split('test')\n",
                "\n",
                "print('=' * 70)\n",
                "print('📂 DATASET DISCOVERY RESULTS')\n",
                "print('=' * 70)\n",
                "print(f'Train Images: {train_img_dir}')\n",
                "print(f'Val Images  : {val_img_dir}')\n",
                "print(f'Test Images : {test_img_dir}')\n",
                "print('=' * 70)\n"
            ]
        },
        # Cell 4: Standardized Dataset Loader with Patient 32 4K Canvas Fix & Thickness 35
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 3: Standardized L3D Dataset Loader\n",
                "- Preserves dynamic canvas resolution from JSON labels to prevent Patient 32 4K truncation.\n",
                "- Standardizes line thickness to 35.\n",
                "- Employs RGB-only standard inputs `(1024, 1024, 3)` with discrete 2D ground truth maps `(1024, 1024)`.\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import cv2\n",
                "import json\n",
                "import numpy as np\n",
                "from torch.utils.data import Dataset, DataLoader\n",
                "\n",
                "class L3DSurgicalDataset(Dataset):\n",
                "    def __init__(self, img_dir):\n",
                "        self.img_dir = img_dir\n",
                "        valid_exts = {'.png', '.jpg', '.jpeg', '.bmp'}\n",
                "        self.img_paths = sorted([\n",
                "            os.path.join(img_dir, f) for f in os.listdir(img_dir)\n",
                "            if Path(f).suffix.lower() in valid_exts\n",
                "        ])\n",
                "        print(f'Loaded {len(self.img_paths)} frames from {img_dir}')\n",
                "\n",
                "    def __len__(self):\n",
                "        return len(self.img_paths)\n",
                "\n",
                "    def __getitem__(self, idx):\n",
                "        img_path = self.img_paths[idx]\n",
                "        rgb_img = self.load_image(img_path)\n",
                "        gt_2d = self.load_mask_2d(img_path)\n",
                "        return rgb_img, gt_2d, str(img_path)\n",
                "\n",
                "    @staticmethod\n",
                "    def load_image(path):\n",
                "        img = cv2.imread(str(path))\n",
                "        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\n",
                "        if img.shape[0] != 1024 or img.shape[1] != 1024:\n",
                "            img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_LINEAR)\n",
                "        return img\n",
                "\n",
                "    @staticmethod\n",
                "    def load_mask_2d(path):\n",
                "        path_str = str(path)\n",
                "        json_path = os.path.splitext(path_str)[0] + '.json'\n",
                "        if not os.path.exists(json_path):\n",
                "            parent = os.path.dirname(path_str)\n",
                "            base_stem = os.path.splitext(os.path.basename(path_str))[0]\n",
                "            candidates = [\n",
                "                os.path.join(parent, '..', 'labels', base_stem + '.json'),\n",
                "                os.path.join(parent, 'labels', base_stem + '.json'),\n",
                "                os.path.join(parent.replace('images', 'labels'), base_stem + '.json')\n",
                "            ]\n",
                "            for c in candidates:\n",
                "                if os.path.exists(c):\n",
                "                    json_path = c\n",
                "                    break\n",
                "\n",
                "        if not os.path.exists(json_path):\n",
                "            return np.zeros((1024, 1024), dtype=np.int32)\n",
                "\n",
                "        with open(json_path, 'r') as f:\n",
                "            data = json.load(f)\n",
                "\n",
                "        # Dynamic canvas size extraction (Patient 32 4K canvas bug fix)\n",
                "        img_h = data.get('imageHeight', 1080)\n",
                "        img_w = data.get('imageWidth', 1920)\n",
                "        canvas = np.zeros((img_h, img_w), dtype=np.uint8)\n",
                "\n",
                "        for shape in data.get('shapes', []):\n",
                "            label = str(shape.get('label', '')).lower()\n",
                "            if label.startswith('r') or 'ridge' in label or 'rigde' in label:\n",
                "                color = 1\n",
                "            elif label.startswith('s') or 'sil' in label:\n",
                "                color = 2\n",
                "            elif label.startswith('l') or 'lig' in label or 'falc' in label:\n",
                "                color = 3\n",
                "            else:\n",
                "                color = 0\n",
                "\n",
                "            if color > 0:\n",
                "                points = shape.get('points', [])\n",
                "                for i in range(1, len(points)):\n",
                "                    pt1 = tuple(map(int, points[i - 1]))\n",
                "                    pt2 = tuple(map(int, points[i]))\n",
                "                    cv2.line(canvas, pt1, pt2, color, 35)\n",
                "\n",
                "        if canvas.shape[0] != 1024 or canvas.shape[1] != 1024:\n",
                "            canvas = cv2.resize(canvas, (1024, 1024), interpolation=cv2.INTER_NEAREST)\n",
                "\n",
                "        return canvas.astype(np.int32)\n",
                "\n",
                "def collate_fn_l3d(batch):\n",
                "    images = [item[0] for item in batch]\n",
                "    masks = [item[1] for item in batch]\n",
                "    paths = [item[2] for item in batch]\n",
                "    return images, masks, paths\n"
            ]
        },
        # Cell 5: Metrics and Evaluation Functions
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 4: Metric Evaluation (Macro Dice, IoU, ASSD, Patient 40)\n",
                "Standardized TopoNet evaluation functions matching EXPERIMENT_1.\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import numpy as np\n",
                "# Ensure NumPy 2.0 compatibility for legacy surface_distance\n",
                "for attr, val in [('Inf', np.inf), ('Infinity', np.inf), ('NAN', np.nan), ('NaN', np.nan)]:\n",
                "    if not hasattr(np, attr):\n",
                "        setattr(np, attr, val)\n",
                "\n",
                "try:\n",
                "    import surface_distance\n",
                "    from surface_distance import metrics as sd_metrics\n",
                "    HAS_SURFACE_DIST = True\n",
                "except Exception:\n",
                "    HAS_SURFACE_DIST = False\n",
                "\n",
                "def evaluation(pred, gt):\n",
                "    smooth = 1e-5\n",
                "    intersection = np.sum(pred * gt)\n",
                "    dice = (2.0 * intersection + smooth) / (np.sum(pred) + np.sum(gt) + smooth)\n",
                "    iou = dice / (2.0 - dice)\n",
                "    return float(iou), float(dice)\n",
                "\n",
                "def compute_toponet_metrics(pred_map, gt_2d):\n",
                "    pred_channels = np.array([pred_map == i for i in range(4)]).astype(np.uint8)\n",
                "    gt_channels = np.array([gt_2d == i for i in range(4)]).astype(np.uint8)\n",
                "\n",
                "    # Macro foreground metric (Classes 1, 2, 3)\n",
                "    iou, dice = evaluation(pred_channels[1:].flatten(), gt_channels[1:].flatten())\n",
                "\n",
                "    # Per-class Dice\n",
                "    class_dices = {}\n",
                "    class_names = {1: 'ridge', 2: 'silhouette', 3: 'falciform'}\n",
                "    for c, name in class_names.items():\n",
                "        _, c_dice = evaluation(pred_channels[c].flatten(), gt_channels[c].flatten())\n",
                "        class_dices[name] = float(c_dice)\n",
                "\n",
                "    assd = None\n",
                "    if HAS_SURFACE_DIST:\n",
                "        try:\n",
                "            if np.count_nonzero(pred_channels[1:]) == 0:\n",
                "                assd = 80.0\n",
                "            else:\n",
                "                temp_assd = []\n",
                "                for i in range(3):\n",
                "                    gt_c = np.array(gt_channels[i + 1], dtype=bool)\n",
                "                    pred_c = np.array(pred_channels[i + 1], dtype=bool)\n",
                "                    if not gt_c.any() or not pred_c.any():\n",
                "                        temp_assd.append(80.0)\n",
                "                        continue\n",
                "                    sd = sd_metrics.compute_surface_distances(gt_c, pred_c, (1.0, 1.0))\n",
                "                    avg_sd = surface_distance.compute_average_surface_distance(sd)\n",
                "                    val = avg_sd[1]\n",
                "                    temp_assd.append(val if not np.isnan(val) and val < 500.0 else 80.0)\n",
                "                mean_dist = float(np.mean(temp_assd)) if temp_assd else 80.0\n",
                "                assd = mean_dist if mean_dist < 500.0 else 80.0\n",
                "        except Exception:\n",
                "            assd = 80.0\n",
                "\n",
                "    return dice, iou, class_dices, assd\n"
            ]
        },
        # Cell 6: Pretrained Model Setup & Architectural Ablation
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"## Step 5: Model Initialization (`{model_name}`) & Architectural Ablation\n",
                f"Configured for: **{title}**\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from transformers import (\n",
                "    AutoImageProcessor,\n",
                "    Mask2FormerForUniversalSegmentation,\n",
                ")\n",
                "\n",
                "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
                f"MODEL_NAME = '{model_name}'\n",
                f"ABLATION_TYPE = '{ablation_type}'\n",
                "\n",
                "print(f'Loading AutoImageProcessor from {MODEL_NAME}...')\n",
                "processor = AutoImageProcessor.from_pretrained(MODEL_NAME, reduce_labels=False, ignore_index=255)\n",
                "\n",
                "print(f'Loading Mask2FormerForUniversalSegmentation from {MODEL_NAME}...')\n",
                "model = Mask2FormerForUniversalSegmentation.from_pretrained(\n",
                "    MODEL_NAME,\n",
                "    num_labels=4,\n",
                "    ignore_mismatched_sizes=True\n",
                ").to(device)\n",
                "\n",
                "# ==================== ARCHITECTURAL ABLATION PATCHING ====================\n",
                "if ABLATION_TYPE == 'wo_masked_attn':\n",
                "    # Ablation 1: Disable Masked Cross-Attention (Set attn_mask = None -> Full Global Attention)\n",
                "    for l in model.model.transformer_module.decoder.layers:\n",
                "        orig = l.forward_pre\n",
                "        def patch_fn(fn):\n",
                "            def patched(*args, **kwargs):\n",
                "                kwargs['encoder_attention_mask'] = None\n",
                "                return fn(*args, **kwargs)\n",
                "            return patched\n",
                "        l.forward_pre = patch_fn(orig)\n",
                "        l.forward = patch_fn(l.forward)\n",
                "    print('⚠️ Masked Attention DISABLED: All 9 decoder layers operating in FULL GLOBAL CROSS-ATTENTION mode.')\n",
                "\n",
                "elif ABLATION_TYPE == 'wo_multiscale':\n",
                "    # Ablation 2: Disable Multi-Scale Feature Cycling (Lock all layers to single stride-16 feature level)\n",
                "    orig_tm_forward = model.model.transformer_module.forward\n",
                "    def single_scale_tm_forward(multi_scale_features, mask_features, output_hidden_states=False, output_attentions=False):\n",
                "        single_feat = multi_scale_features[1]  # Lock to stride 16 (24x24)\n",
                "        single_multi_scale = [single_feat, single_feat, single_feat]\n",
                "        return orig_tm_forward(single_multi_scale, mask_features, output_hidden_states=output_hidden_states, output_attentions=output_attentions)\n",
                "    model.model.transformer_module.forward = single_scale_tm_forward\n",
                "    print('⚠️ Multi-Scale Feature Cycling DISABLED: All 9 decoder layers locked to single stride-16 level.')\n",
                "\n",
                "elif ABLATION_TYPE == 'wo_self_attn':\n",
                "    # Ablation 3: Disable Query Self-Attention (Bypass self_attn -> Independent Parallel Queries)\n",
                "    for l in model.model.transformer_module.decoder.layers:\n",
                "        def patch_self_attn():\n",
                "            def patched(*args, **kwargs):\n",
                "                hs = kwargs.get('hidden_states', args[0] if len(args) > 0 else None)\n",
                "                return torch.zeros_like(hs), None\n",
                "            return patched\n",
                "        l.self_attn.forward = patch_self_attn()\n",
                "    print('⚠️ Query Self-Attention DISABLED: Queries operating as independent parallel detectors (no inter-query communication).')\n",
                "\n",
                "else:\n",
                "    print('✅ Full Standard Mask2Former Active: Masked Cross-Attention + Multi-Scale Feature Cycling + Query Self-Attention.')\n"
            ]
        },
        # Cell 7: Training Configuration & Loop
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 6: Model Training with Native Deep Supervision Hungarian Loss\n",
                "- Uses native multi-layer Hungarian loss (`outputs.loss`).\n",
                "- Uses multi-query semantic ensemble projection (`processor.post_process_semantic_segmentation`).\n",
                "- Runs 60 epochs with Cosine Annealing scheduler and AMP FP16 mixed precision.\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import time\n",
                "from tqdm.auto import tqdm\n",
                "\n",
                "EPOCHS = 60\n",
                "BATCH_SIZE = 1\n",
                "ACCUMULATION_STEPS = 4\n",
                "LEARNING_RATE = 8e-5\n",
                "WEIGHT_DECAY = 3e-5\n",
                f"SAVE_DIR = '/kaggle/working/{save_dir}'\n",
                "os.makedirs(SAVE_DIR, exist_ok=True)\n",
                "\n",
                "train_dataset = L3DSurgicalDataset(train_img_dir)\n",
                "val_dataset = L3DSurgicalDataset(val_img_dir)\n",
                "test_dataset = L3DSurgicalDataset(test_img_dir) if test_img_dir else None\n",
                "\n",
                "train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_l3d)\n",
                "val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_l3d)\n",
                "\n",
                "optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)\n",
                "scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)\n",
                "scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None\n",
                "\n",
                "best_val_dice = 0.0\n",
                f"best_ckpt_path = os.path.join(SAVE_DIR, 'best_{mode.lower()}.pth')\n",
                f"latest_ckpt_path = os.path.join(SAVE_DIR, 'latest_{mode.lower()}.pth')\n",
                "log_file = os.path.join(SAVE_DIR, 'training_metrics.json')\n",
                "metrics_history = []\n",
                "\n",
                "print('=' * 75)\n",
                f"print('🚀 STARTING TRAINING: {title}')\n",
                f"print(f'   Train Samples: {{len(train_dataset)}} | Val Samples: {{len(val_dataset)}}')\n",
                f"print(f'   Effective Batch Size: {{BATCH_SIZE * ACCUMULATION_STEPS}} | Total Epochs: {{EPOCHS}}')\n",
                "print('=' * 75)\n",
                "\n",
                "for epoch in range(1, EPOCHS + 1):\n",
                "    epoch_start = time.time()\n",
                "    model.train()\n",
                "    total_train_loss = 0.0\n",
                "    optimizer.zero_grad()\n",
                "\n",
                "    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch:02d}/{EPOCHS} [Train]')\n",
                "    for step, (batch_rgb, batch_gt, _) in pbar:\n",
                "        inputs = processor(images=batch_rgb, segmentation_maps=batch_gt, return_tensors='pt')\n",
                "        pixel_values = inputs['pixel_values'].to(device)\n",
                "        mask_labels = [m.to(device) for m in inputs['mask_labels']]\n",
                "        class_labels = [c.to(device) for c in inputs['class_labels']]\n",
                "\n",
                "        if scaler is not None:\n",
                "            with torch.amp.autocast('cuda', dtype=torch.float16):\n",
                "                outputs = model(pixel_values=pixel_values, mask_labels=mask_labels, class_labels=class_labels)\n",
                "                loss = outputs.loss / ACCUMULATION_STEPS\n",
                "            scaler.scale(loss).backward()\n",
                "        else:\n",
                "            outputs = model(pixel_values=pixel_values, mask_labels=mask_labels, class_labels=class_labels)\n",
                "            loss = outputs.loss / ACCUMULATION_STEPS\n",
                "            loss.backward()\n",
                "\n",
                "        total_train_loss += outputs.loss.item() * len(batch_rgb)\n",
                "\n",
                "        if (step + 1) % ACCUMULATION_STEPS == 0 or (step + 1) == len(train_loader):\n",
                "            if scaler is not None:\n",
                "                scaler.unscale_(optimizer)\n",
                "                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)\n",
                "                scaler.step(optimizer)\n",
                "                scaler.update()\n",
                "            else:\n",
                "                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)\n",
                "                optimizer.step()\n",
                "            optimizer.zero_grad()\n",
                "\n",
                "        pbar.set_postfix({'loss': f'{outputs.loss.item():.4f}'})\n",
                "\n",
                "    scheduler.step()\n",
                "    avg_train_loss = total_train_loss / len(train_dataset)\n",
                "    epoch_time = time.time() - epoch_start\n",
                "\n",
                "    # ==================== VALIDATION ====================\n",
                "    model.eval()\n",
                "    val_losses = []\n",
                "    val_dices, val_ious, val_assds = [], [], []\n",
                "    class_dices_accum = {'ridge': [], 'silhouette': [], 'falciform': []}\n",
                "    patient_40_dices = []\n",
                "\n",
                "    with torch.no_grad():\n",
                "        for batch_rgb, batch_gt, batch_paths in tqdm(val_loader, desc=f'Epoch {epoch:02d}/{EPOCHS} [Val]'):\n",
                "            inputs = processor(images=batch_rgb, segmentation_maps=batch_gt, return_tensors='pt')\n",
                "            pixel_values = inputs['pixel_values'].to(device)\n",
                "            mask_labels = [m.to(device) for m in inputs['mask_labels']]\n",
                "            class_labels = [c.to(device) for c in inputs['class_labels']]\n",
                "\n",
                "            if scaler is not None:\n",
                "                with torch.amp.autocast('cuda', dtype=torch.float16):\n",
                "                    outputs = model(pixel_values=pixel_values, mask_labels=mask_labels, class_labels=class_labels)\n",
                "            else:\n",
                "                outputs = model(pixel_values=pixel_values, mask_labels=mask_labels, class_labels=class_labels)\n",
                "\n",
                "            val_losses.append(outputs.loss.item() * len(batch_rgb))\n",
                "\n",
                "            target_sizes = [(1024, 1024)] * len(batch_rgb)\n",
                "            pred_maps = processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)\n",
                "\n",
                "            for p_tensor, gt_arr, p_path in zip(pred_maps, batch_gt, batch_paths):\n",
                "                p_arr = p_tensor.cpu().numpy()\n",
                "                d, iou, c_dices, assd = compute_toponet_metrics(p_arr, gt_arr)\n",
                "                val_dices.append(d)\n",
                "                val_ious.append(iou)\n",
                "                if assd is not None:\n",
                "                    val_assds.append(assd)\n",
                "                for k in class_dices_accum:\n",
                "                    class_dices_accum[k].append(c_dices[k])\n",
                "                if 'patient40' in p_path.lower() or 'p40' in p_path.lower():\n",
                "                    patient_40_dices.append(d)\n",
                "\n",
                "    mean_val_loss = np.sum(val_losses) / len(val_dataset)\n",
                "    mean_val_dice = float(np.mean(val_dices))\n",
                "    mean_val_iou = float(np.mean(val_ious))\n",
                "    mean_val_assd = float(np.mean(val_assds)) if len(val_assds) > 0 else 0.0\n",
                "    mean_ridge = float(np.mean(class_dices_accum['ridge']))\n",
                "    mean_sil = float(np.mean(class_dices_accum['silhouette']))\n",
                "    mean_falc = float(np.mean(class_dices_accum['falciform']))\n",
                "    mean_p40 = float(np.mean(patient_40_dices)) if len(patient_40_dices) > 0 else 0.0\n",
                "\n",
                "    print(\n",
                "        f'👉 Epoch {epoch:02d} ({epoch_time:.1f}s) | '\n",
                "        f'Tr Loss: {avg_train_loss:.4f} | Val Loss: {mean_val_loss:.4f} | '\n",
                "        f'Val Dice: {mean_val_dice:.4f} (Ridge: {mean_ridge:.4f}, Sil: {mean_sil:.4f}, Falc: {mean_falc:.4f}) | '\n",
                "        f'P40 Dice: {mean_p40:.4f}'\n",
                "    )\n",
                "\n",
                "    record = {\n",
                "        'epoch': epoch,\n",
                "        'train_loss': avg_train_loss,\n",
                "        'val_loss': mean_val_loss,\n",
                "        'val_macro_dice': mean_val_dice,\n",
                "        'val_macro_iou': mean_val_iou,\n",
                "        'val_assd': mean_val_assd,\n",
                "        'val_ridge_dice': mean_ridge,\n",
                "        'val_silhouette_dice': mean_sil,\n",
                "        'val_falciform_dice': mean_falc,\n",
                "        'val_patient40_dice': mean_p40,\n",
                "        'epoch_duration_sec': epoch_time\n",
                "    }\n",
                "    metrics_history.append(record)\n",
                "\n",
                "    with open(log_file, 'w') as f:\n",
                "        json.dump(metrics_history, f, indent=2)\n",
                "\n",
                "    if mean_val_dice > best_val_dice:\n",
                "        best_val_dice = mean_val_dice\n",
                "        torch.save({\n",
                "            'epoch': epoch,\n",
                "            'model_state_dict': model.state_dict(),\n",
                "            'best_val_dice': best_val_dice,\n",
                f"            'mode': '{mode}'\n",
                "        }, best_ckpt_path)\n",
                "        print(f'  🏆 New Best Model! Val Dice: {best_val_dice:.4f} -> {best_ckpt_path}')\n",
                "\n",
                "    torch.save({\n",
                "        'epoch': epoch,\n",
                "        'model_state_dict': model.state_dict(),\n",
                "        'best_val_dice': best_val_dice,\n",
                f"        'mode': '{mode}'\n",
                "    }, latest_ckpt_path)\n"
            ]
        },
        # Cell 8: Final Test Evaluation & Automated Packaging
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 7: Final Test Evaluation & One-Click Zip Packaging\n",
                "Evaluates the best checkpoint on the unseen Test split (109 frames) and bundles all outputs into a single downloadable zip file.\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import shutil\n",
                "\n",
                "if test_dataset is not None and os.path.exists(best_ckpt_path):\n",
                "    print('=' * 75)\n",
                "    print(f'🔬 RUNNING FINAL TEST EVALUATION USING BEST CHECKPOINT (Val Dice: {best_val_dice:.4f})')\n",
                "    print('=' * 75)\n",
                "\n",
                "    best_ckpt = torch.load(best_ckpt_path, map_location=device)\n",
                "    model.load_state_dict(best_ckpt['model_state_dict'])\n",
                "    model.eval()\n",
                "\n",
                "    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_l3d)\n",
                "    test_dices, test_ious, test_assds = [], [], []\n",
                "    test_class_dices = {'ridge': [], 'silhouette': [], 'falciform': []}\n",
                "\n",
                "    with torch.no_grad():\n",
                "        for batch_rgb, batch_gt, _ in tqdm(test_loader, desc='Testing'):\n",
                "            inputs = processor(images=batch_rgb, segmentation_maps=batch_gt, return_tensors='pt')\n",
                "            pixel_values = inputs['pixel_values'].to(device)\n",
                "            if scaler is not None:\n",
                "                with torch.amp.autocast('cuda', dtype=torch.float16):\n",
                "                    outputs = model(pixel_values=pixel_values)\n",
                "            else:\n",
                "                outputs = model(pixel_values=pixel_values)\n",
                "\n",
                "            target_sizes = [(1024, 1024)] * len(batch_rgb)\n",
                "            pred_maps = processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)\n",
                "\n",
                "            for p_tensor, gt_arr in zip(pred_maps, batch_gt):\n",
                "                p_arr = p_tensor.cpu().numpy()\n",
                "                d, iou, c_dices, assd = compute_toponet_metrics(p_arr, gt_arr)\n",
                "                test_dices.append(d)\n",
                "                test_ious.append(iou)\n",
                "                if assd is not None:\n",
                "                    test_assds.append(assd)\n",
                "                for k in test_class_dices:\n",
                "                    test_class_dices[k].append(c_dices[k])\n",
                "\n",
                "    final_summary = {\n",
                f"        'mode': '{mode}',\n",
                "        'best_val_dice': best_val_dice,\n",
                "        'test_macro_dice': float(np.mean(test_dices)),\n",
                "        'test_macro_iou': float(np.mean(test_ious)),\n",
                "        'test_assd': float(np.mean(test_assds)) if len(test_assds) > 0 else 0.0,\n",
                "        'test_ridge_dice': float(np.mean(test_class_dices['ridge'])),\n",
                "        'test_silhouette_dice': float(np.mean(test_class_dices['silhouette'])),\n",
                "        'test_falciform_dice': float(np.mean(test_class_dices['falciform']))\n",
                "    }\n",
                "\n",
                "    summary_file = os.path.join(SAVE_DIR, 'final_summary.json')\n",
                "    with open(summary_file, 'w') as f:\n",
                "        json.dump(final_summary, f, indent=2)\n",
                "\n",
                "    print('=' * 75)\n",
                f"    print(f'🏁 FINAL TEST RESULTS [{mode}]:')\n",
                "    print(f'   Best Val Dice      : {final_summary[\"best_val_dice\"]:.4f}')\n",
                "    print(f'   Test Macro Dice    : {final_summary[\"test_macro_dice\"]:.4f}')\n",
                "    print(f'   Test Macro IoU     : {final_summary[\"test_macro_iou\"]:.4f}')\n",
                "    print(f'   Test ASSD (pixels) : {final_summary[\"test_assd\"]:.4f}')\n",
                "    print(f'   Ridge Dice         : {final_summary[\"test_ridge_dice\"]:.4f}')\n",
                "    print(f'   Silhouette Dice    : {final_summary[\"test_silhouette_dice\"]:.4f}')\n",
                "    print(f'   Falciform Dice     : {final_summary[\"test_falciform_dice\"]:.4f}')\n",
                "    print('=' * 75)\n",
                "\n",
                "# ==================== ZIP PACKAGING ====================\n",
                f"zip_base = '/kaggle/working/{Path(zip_name).stem}'\n",
                "print(f'📦 Packaging results into {zip_base}.zip...')\n",
                "shutil.make_archive(zip_base, 'zip', SAVE_DIR)\n",
                f"print(f'🎉 Packaging Complete! Download: /kaggle/working/{zip_name}')\n"
            ]
        }
    ]

    nb = {
        "cells": cells,
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
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    return nb


def main():
    script_dir = Path(__file__).parent.resolve()
    exp_dir = script_dir.parent

    nb_dir = exp_dir / "notebooks"
    nb_dir.mkdir(exist_ok=True)
    print(f"Generating 4 self-contained Mask2Former Kaggle notebooks in: {exp_dir} and {nb_dir}")

    for mode_cfg in MODES:
        nb_filename = mode_cfg["nb_filename"]
        out_path = exp_dir / nb_filename
        nb_json = make_notebook(mode_cfg)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(nb_json, f, indent=1)

        # Also mirror to notebooks/
        mirror_path = nb_dir / nb_filename
        with open(mirror_path, "w", encoding="utf-8") as f:
            json.dump(nb_json, f, indent=1)

        print(f"✅ Generated: {out_path.name} ({mode_cfg['name']})")

    print("\nVerifying generated notebooks syntax...")
    for mode_cfg in MODES:
        out_path = exp_dir / mode_cfg["nb_filename"]
        with open(out_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        code_cells = [c for c in data["cells"] if c["cell_type"] == "code"]
        for idx, cell in enumerate(code_cells):
            code_str = "".join(cell["source"])
            clean_lines = [l for l in code_str.split("\n") if not l.strip().startswith("!")]
            clean_code = "\n".join(clean_lines)
            try:
                ast.parse(clean_code)
            except SyntaxError as e:
                print(f"❌ Syntax Error in {mode_cfg['nb_filename']} cell {idx}: {e}")
                raise e

    print("🎉 All 4 Kaggle notebooks generated and verified with 0 syntax errors!")


if __name__ == "__main__":
    main()
