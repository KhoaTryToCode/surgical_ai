# Run Commands: Vis2Reg Replication

Execution commands for local development (macOS) and Kaggle CUDA environments (Tesla T4 / P100).

---

## 1. Environment Setup

```bash
# Clone or navigate into experiment directory
cd experiments/EXP_03_vis2reg_replication

# Install dependencies
pip install -r requirements.txt
```

---

## 2. Phase 1: Model Smoke Test

```bash
python test_smoke.py
```

---

## 3. Phase 2: Dataset Adapter Verification

```bash
# On Kaggle (with Liver_regis dataset mounted)
python scripts/01_test_data.py --data_root /kaggle/input/datasets/junezhou001/p2i-lreg/Liver_regis

# Local dry-run
python scripts/01_test_data.py
```

---

## 4. Phase 3: Multi-Task Losses & Gradient Flow Verification

```bash
python scripts/02_test_losses.py
```

---

## 5. Phase 4: Stage 1 Synthetic Rigid Pretraining

```bash
# Fast Pretraining Run (~12-15 minutes on Tesla T4, 200 samples/patient, batch size 4)
python scripts/03_pretrain_syn.py \
    --data_root /kaggle/input/datasets/junezhou001/p2i-lreg/Liver_regis \
    --fold 0 \
    --epochs 20 \
    --batch_size 4 \
    --samples_per_patient 200 \
    --max_points 2048

# Full Reproduction Run (all 2,000 synthetic samples/patient, batch size 4)
python scripts/03_pretrain_syn.py \
    --data_root /kaggle/input/datasets/junezhou001/p2i-lreg/Liver_regis \
    --fold 0 \
    --epochs 20 \
    --batch_size 4 \
    --samples_per_patient 2000 \
    --max_points 2048

# Instant Smoke Test (1 epoch, 2 batches, verifies CUDA execution before full run)
python scripts/03_pretrain_syn.py \
    --data_root /kaggle/input/datasets/junezhou001/p2i-lreg/Liver_regis \
    --fold 0 \
    --epochs 1 \
    --batch_size 2 \
    --smoke_test
```

---

## 6. Phase 5: Stage 2 Real Visibility-Aware Fine-Tuning (Upcoming)

```bash
python scripts/04_train_real.py \
    --data_root /kaggle/input/datasets/junezhou001/p2i-lreg/Liver_regis \
    --fold 0 \
    --pretrained_seed checkpoints/stage1_rigid_seed.pth \
    --epochs 60
```

---

## 7. Phase 6: Benchmark Evaluation & Cross-Validation (Upcoming)

```bash
python scripts/05_evaluate.py \
    --data_root /kaggle/input/datasets/junezhou001/p2i-lreg/Liver_regis \
    --checkpoint checkpoints/best_real_model.pth
```
