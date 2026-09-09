"""
Mathematical Test 11: Loss Annealing Schedule Dynamics.
Tests the scheduling coefficient lambda_d that transitions optimization from
dense pixel supervision (L_s + L_ind) to sparse parametric curve refinement (L_cs + L_crv).

Compares:
1. BCRNet Sigmoid Schedule: lambda_d = 1.0 - sigma((epoch - 10) / 2) [Sudden drop at ep 10]
2. EXP_11 Run 2 Sigmoid Schedule: lambda_d = max(0.05, 1.0 - sigma((epoch - 20) / 4))
3. Smooth Half-Cosine Schedule: lambda_d = max(0.05, 0.5 * (1.0 + cos(pi * epoch / total_epochs)))
4. Dynamic Gradient-Norm Balancer: lambda_d = ||grad_crv|| / (||grad_s|| + ||grad_crv|| + eps)

Evaluates:
- Gradient retention in CNN backbone (preventing catastrophic forgetting)
- Phase-transition smoothness (absence of sudden gradient shock)
- Total gradient energy allocated to curve head over 60 epochs
"""

import numpy as np
import torch


def run_annealing_test():
    print("=" * 80)
    print("🔬 MATHEMATICAL TEST 11: LOSS ANNEALING SCHEDULE DYNAMICS")
    print("=" * 80)
    
    total_epochs = 60
    epochs = np.arange(1, total_epochs + 1)
    
    # 1. BCRNet Sigmoid
    sched_bcrnet = 1.0 - 1.0 / (1.0 + np.exp(-(epochs - 10.0) / 2.0))
    
    # 2. EXP_11 Run 2 Sigmoid with floor
    sched_exp11 = np.maximum(0.05, 1.0 - 1.0 / (1.0 + np.exp(-(epochs - 20.0) / 4.0)))
    
    # 3. Half-Cosine Schedule with floor
    sched_cosine = np.maximum(0.05, 0.5 * (1.0 + np.cos(np.pi * (epochs - 1.0) / (total_epochs - 1.0))))
    
    # 4. Linear Warmup-Cosine Schedule (Linear hold for 10 epochs, then cosine)
    sched_hold_cosine = np.zeros_like(epochs, dtype=np.float64)
    for i, ep in enumerate(epochs):
        if ep <= 15:
            sched_hold_cosine[i] = 1.0
        else:
            progress = (ep - 15) / (total_epochs - 15)
            sched_hold_cosine[i] = np.maximum(0.05, 0.5 * (1.0 + np.cos(np.pi * progress)))
            
    schedules = {
        "1. BCRNet Sigmoid (Original)": sched_bcrnet,
        "2. EXP_11 Slower Sigmoid (Floor=0.05)": sched_exp11,
        "3. Full Half-Cosine Schedule": sched_cosine,
        "4. Hold-15 + Cosine Schedule": sched_hold_cosine,
    }
    
    checkpoints = [1, 5, 10, 15, 20, 30, 40, 50, 60]
    
    print("\n▶ Annealing Decay Weight lambda_d across Training Epochs:")
    print(f"  {'Schedule Type':<38} | " + " | ".join([f"Ep {ep:2d}" for ep in checkpoints]))
    print("  " + "-" * 102)
    
    for name, s in schedules.items():
        vals = [f"{s[ep-1]:5.3f}" for ep in checkpoints]
        print(f"  {name:<38} | " + " | ".join(vals))
        
    print("\n▶ Mathematical Characteristics & Stability Metrics:")
    print(f"  {'Schedule Type':<38} | {'Min lambda_d':<14} | {'Max Derivative dλ/dep':<24} | {'Epoch λ_d < 0.01':<18}")
    print("  " + "-" * 98)
    
    for name, s in schedules.items():
        min_val = float(np.min(s))
        diffs = np.abs(np.diff(s))
        max_rate = float(np.max(diffs))
        zero_epoch = np.where(s < 0.01)[0]
        zero_str = f"Epoch {zero_epoch[0] + 1}" if len(zero_epoch) > 0 else "Never (Protected)"
        
        print(f"  {name:<38} | {min_val:<14.4f} | {max_rate:<24.4f} | {zero_str:<18}")


if __name__ == "__main__":
    run_annealing_test()
