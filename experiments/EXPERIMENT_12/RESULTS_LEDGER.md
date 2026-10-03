# EXPERIMENT 12: RESULTS LEDGER

## Benchmark Tracking Table

| Model / Experiment | Val Macro Dice | Val Ridge Dice | Val Sil Dice | Val Falc Dice | Val ASSD (px) | Test Macro Dice | P40 Macro Dice | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EXP_01 (Baseline M2F)** | 0.542 | 0.612 | 0.531 | 0.483 | 14.82 | 0.518 | 0.281 | Complete |
| **EXP_05 (Junction-Steered M2F)** | 0.589 | 0.648 | 0.572 | 0.547 | 11.35 | 0.562 | 0.344 | Complete |
| **EXP_11 (Deformed Weighted M2F)** | 0.592 | 0.651 | 0.578 | 0.548 | 11.10 | 0.565 | 0.352 | Complete |
| **EXP_12 (Dual-Decoder L3D + Cholec)** | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | Ready to Launch |

---

## Detailed Metric Targets for EXP_12
1. **L3D Validation Macro Dice:** Target > 0.620 (+3.0% over EXP_11).
2. **Hard-Case (Patient 40) Dice:** Target > 0.400 (+5.0% over EXP_11).
3. **Macro ASSD:** Target < 10.0 px.
4. **Generalization:** Demonstration of reduced false positives on surgical tools and improved boundary continuity along liver margins.
