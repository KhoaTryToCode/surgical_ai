"""
Mathematical Test 12: Proposal Induction Loss Formulations.
Analyzes the extreme class imbalance in ACPI proposal induction on 16x16 feature map f_4:
Total pixels = 256. True positive midpoint = 1 pixel. Background = 255 pixels (1 : 255 imbalance!).

Compares:
1. BCRNet Plain BCE: BCE(s_hat, s*) with s* in {0, 1}
2. Weighted BCE: BCE(s_hat, s*, pos_weight=20.0)
3. CenterNet-style Gaussian Heatmap BCE: s*(p) = exp(-d^2 / 2*sigma^2)
4. Modified Focal Loss (CornerNet / CenterNet formulation):
   L = - (1 - p)^gamma * log(p) if y=1 else - (1 - y)^beta * p^gamma * log(1 - p)

Evaluates:
- Positive gradient ratio: ||grad_pos|| / ||grad_neg_total||
- Suppression risk (chance of model predicting s_hat = 0 everywhere)
- Convergence steps to recall > 95%
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def run_induction_loss_test():
    print("=" * 80)
    print("🔬 MATHEMATICAL TEST 12: PROPOSAL INDUCTION LOSS FORMULATIONS")
    print("=" * 80)
    
    device = torch.device("cpu")
    H, W = 16, 16  # f_4 level
    midpoint = (8, 8)
    
    # 1. Binary target: 1 at (8, 8), 0 elsewhere
    y_binary = torch.zeros(1, 1, H, W, dtype=torch.float32, device=device)
    y_binary[0, 0, midpoint[0], midpoint[1]] = 1.0
    
    # 2. Gaussian heatmap target: sigma = 1.2 px
    ys = torch.arange(H, dtype=torch.float32, device=device)
    xs = torch.arange(W, dtype=torch.float32, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    dist_sq = (gx - midpoint[1]) ** 2 + (gy - midpoint[0]) ** 2
    y_gaussian = torch.exp(-dist_sq / (2.0 * (1.2 ** 2))).unsqueeze(0).unsqueeze(0)
    
    print(f"\nSpatial Statistics on {H}x{W} Grid:")
    print(f"  • Binary Target:   1 positive pixel, 255 zero pixels (Imbalance 1 : 255)")
    print(f"  • Gaussian Target: Peak=1.00, Effective Area (val > 0.1) = {int(torch.sum(y_gaussian > 0.1).item())} pixels")
    
    loss_formulations = {
        "1. BCRNet Plain BCE": lambda logit: nn.functional.binary_cross_entropy_with_logits(logit, y_binary),
        "2. Weighted BCE (pos=15.0)": lambda logit: nn.functional.binary_cross_entropy_with_logits(logit, y_binary, pos_weight=torch.tensor([15.0])),
        "3. Gaussian Heatmap BCE": lambda logit: nn.functional.binary_cross_entropy_with_logits(logit, y_gaussian),
        "4. CenterNet Focal Loss": lambda logit: focal_heatmap_loss(logit, y_gaussian),
    }
    
    def focal_heatmap_loss(logits, targets, alpha=2.0, beta=4.0):
        probs = torch.sigmoid(logits)
        pos_mask = (targets == 1.0)
        neg_mask = (targets < 1.0)
        
        pos_loss = torch.log(probs + 1e-6) * ((1.0 - probs) ** alpha) * pos_mask
        neg_loss = torch.log(1.0 - probs + 1e-6) * (probs ** alpha) * ((1.0 - targets) ** beta) * neg_mask
        
        num_pos = torch.sum(pos_mask)
        return - (torch.sum(pos_loss) + torch.sum(neg_loss)) / (num_pos + 1e-6)
        
    print("\n▶ Gradient Ratio at Initialization (Logits = 0.0, p = 0.50):")
    print(f"  {'Formulation':<28} | {'Pos Grad':<12} | {'Sum Neg Grads':<16} | {'Signal-to-Noise Ratio':<22}")
    print("  " + "-" * 84)
    
    for name, loss_fn in loss_formulations.items():
        logits = torch.zeros(1, 1, H, W, dtype=torch.float32, device=device, requires_grad=True)
        loss = loss_fn(logits)
        loss.backward()
        
        g = logits.grad[0, 0]
        pos_g = float(torch.abs(g[midpoint[0], midpoint[1]]).item())
        neg_g_sum = float(torch.sum(torch.abs(g[dist_sq > 4.0])).item())
        snr = pos_g / (neg_g_sum + 1e-6)
        
        print(f"  {name:<28} | {pos_g:<12.4f} | {neg_g_sum:<16.4f} | {snr:<22.4f}")
        
    print("\n▶ Optimization Dynamics: Steps to Recall Target at Midpoint (p > 0.80):")
    print(f"  {'Formulation':<28} | {'Steps to p>0.80':<18} | {'Peak Background Prob':<22}")
    print("  " + "-" * 72)
    
    for name, loss_fn in loss_formulations.items():
        logits = torch.zeros(1, 1, H, W, dtype=torch.float32, device=device, requires_grad=True)
        opt = optim.Adam([logits], lr=0.1)
        steps_needed = 100
        
        for s in range(100):
            opt.zero_grad()
            loss = loss_fn(logits)
            loss.backward()
            opt.step()
            
            p = torch.sigmoid(logits)[0, 0]
            if p[midpoint[0], midpoint[1]] > 0.80 and steps_needed == 100:
                steps_needed = s
                
        p_final = torch.sigmoid(logits)[0, 0]
        bg_max = float(torch.max(p_final[dist_sq > 9.0]).item())
        print(f"  {name:<28} | {steps_needed:<18d} | {bg_max:<22.4e}")


if __name__ == "__main__":
    run_induction_loss_test()
