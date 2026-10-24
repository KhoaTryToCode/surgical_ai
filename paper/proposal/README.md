# 2-Page Research Proposal: Geometric Coverage Disparity & Topological Junction Steering

This directory contains the concise, 2-page research proposal and technical report formatted in standard 2-column LaTeX (`main.tex`).

## Document Structure & Layout (Strict 2-Page Budget)

1. **Title & Abstract (Page 1):**
   - High-level overview of laparoscopic landmark detection, empirical diagnosis of coverage disparity, automated junction derivation, state-of-the-art results, and future roadmap.
2. **Section 1: Literature Review & Benchmark State (Page 1):**
   - Summary of prior approaches (U-Net, D2GPLand, TopoNet, A2ONet, BCRNet).
   - **Table 1:** SOTA benchmark comparison highlighting boundary surface distance error (**19.54 px ASSD** for Junction-Steered Mask2Former vs. 43.55 px BCRNet and 49.89 px A2ONet).
3. **Section 2: Problem: Geometric Distribution & Inversion Shift (Page 1):**
   - **The Patient Monopoly Skew:** Val (82.79% Patient 40) and Test (77.06% Patient 41).
   - **Anatomical Pose Inversion:** Canonical dome views (92.0%) vs. retracted states (8.0%) in Train.
4. **Figure 1 (Top of Page 2):**
   - High-resolution side-by-side empirical montage comparing audited Train Flipped frames (Patients 12, 20, 38) vs. Val Flipped frames (Patient 40), proving identical physical traction mechanics.
5. **Section 3: Proposed Methodology: Junction Steering (Page 2):**
   - **Automated Topological Anchor Derivation:** Zero manual overhead algorithm deriving $J_{\text{top}}, J_{\text{bottom}}, J_{\text{lat\_r}}, J_{\text{lat\_l}}$ from raw polylines.
   - **Dynamic Query Steering Block:** Auxiliary junction head and cross-attention injection into Mask2Former object queries.
   - **Table 2:** Patient 40 Inversion Robustness (28.10% baseline $\to$ 70.91% steered, +42.81% DSC gain).
6. **Section 4: Discussion & Strategic Roadmap (Page 2):**
   - **Direction A:** Immediate publication on empirical diagnosis and benchmark-leading ASSD.
   - **Direction B:** Self-Supervised Learning (SSL via DINOv2 / MAE) pretraining on unlabelled surgical video archives (Cholec80).

---

## Assets
- `main.tex`: The primary LaTeX source file.
- `figures/train_vs_val_flipped_montage.png`: High-resolution comparison figure referenced in `main.tex`.

---

## How to Compile to PDF

### Option 1: Overleaf
1. Create a new blank project on [Overleaf](https://www.overleaf.com/).
2. Upload `main.tex` and the `figures/` folder.
3. Select `pdfLaTeX` and hit **Recompile**.

### Option 2: Local TeX Live / MacTeX
Run from this directory:
```bash
pdflatex main.tex
pdflatex main.tex
```
