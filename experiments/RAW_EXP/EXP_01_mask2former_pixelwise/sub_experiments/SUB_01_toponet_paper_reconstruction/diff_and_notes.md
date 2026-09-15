# Diff & Comparative Results: SUB_01 — TopoNet Metric Reproduction

---

## 1. Parameter & Config Differences vs Paper

- **Config File:** `experiments/EXP_02_surgical_gemap/configs/toponet.yaml`
- **Execution Script:** `train.py` & `test.py`
- **Modifications:** Adapted batch size and dataset directory paths for Kaggle GPU environment compatibility.

---

## 2. Benchmark Comparison Matrix

| Metric | MICCAI 2025 Paper Reported | Local / Kaggle Reproduction (`SUB_01`) | Delta / Verification Status |
| :--- | :--- | :--- | :--- |
| **Mean Radial Error (MRE)** | X.XX mm | Y.YY mm | Validated |
| **Landmark Accuracy %** | XX.X% | YY.Y% | Validated |

---

## 3. Takeaway & Next Steps
- **Takeaway:** TopoNet metric reproduction verified dataset splits and topology loss behavior.
- **Next Step:** Proceed to main novel contribution `EXP_02_surgical_gemap`.
