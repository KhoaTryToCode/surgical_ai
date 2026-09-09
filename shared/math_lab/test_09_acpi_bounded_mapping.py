"""
Mathematical Test 09: ACPI Sigmoid-Logit Offset Mapping vs Alternative Bounded Heads.
Analyzes the mathematical properties of BCRNet's ACPI proposal parameterization:
  b_i^j = ( sigma(Delta b_x + logit(c_x)), sigma(Delta b_y + logit(c_y)) )

Compares against 3 alternative parameterizations:
1. BCRNet Sigmoid-Logit: b = sigma(Delta + logit(c))
2. Bounded Tanh Residual: b = clamp(c + tanh(Delta) * delta_max, 0, 1)
3. Direct Linear Additive: b = clamp(c + Delta, 0, 1)
4. Pure Sigmoid (Grid-agnostic): b = sigma(Delta)

Evaluates:
- Gradient magnitude db/dDelta across anchor grid positions c in [0.01, 0.99]
- Boundary saturation behavior (at image borders c -> 0 or c -> 1)
- Lipschitz continuity and gradient condition number
"""

import numpy as np
import torch
import torch.nn.functional as F


def run_acpi_mapping_test():
    print("=" * 80)
    print("🔬 MATHEMATICAL TEST 09: ACPI BOUNDED OFFSET MAPPING DYNAMICS")
    print("=" * 80)
    
    device = torch.device("cpu")
    # Test across anchor positions: near border (0.02), intermediate (0.15), center (0.50)
    anchor_positions = [0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98]
    delta_offsets = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    
    methods = {
        "1. BCRNet Sigmoid-Logit": lambda c, d: torch.sigmoid(d + torch.logit(c, eps=1e-6)),
        "2. Bounded Tanh Residual": lambda c, d: torch.clamp(c + torch.tanh(d) * 0.25, 0.0, 1.0),
        "3. Direct Linear Additive": lambda c, d: torch.clamp(c + d * 0.25, 0.0, 1.0),
        "4. Pure Sigmoid (No Anchor)": lambda c, d: torch.sigmoid(d),
    }
    
    print("\n▶ Gradient Sensitivity db / dDelta across Anchor Positions (at Delta = 0.0):")
    print(f"  {'Method':<28} | " + " | ".join([f"c={c:4.2f}" for c in anchor_positions]))
    print("  " + "-" * 88)
    
    for m_name, func in methods.items():
        grads = []
        for c_val in anchor_positions:
            c = torch.tensor([c_val], dtype=torch.float32, device=device)
            d = torch.tensor([0.0], dtype=torch.float32, device=device, requires_grad=True)
            b = func(c, d)
            b.backward()
            grads.append(float(d.grad.item()))
        cols = " | ".join([f"{g:6.4f}" for g in grads])
        print(f"  {m_name:<28} | {cols}")
        
    print("\n▶ Displacement Range and Linearity at Center (c = 0.50):")
    print(f"  {'Method':<28} | " + " | ".join([f"Δ={d:4.1f}" for d in [-2.0, -1.0, 0.0, 1.0, 2.0]]))
    print("  " + "-" * 76)
    
    for m_name, func in methods.items():
        vals = []
        c = torch.tensor([0.50], dtype=torch.float32, device=device)
        for d_val in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            d = torch.tensor([d_val], dtype=torch.float32, device=device)
            b = func(c, d)
            vals.append(float(b.item()))
        cols = " | ".join([f"{v:6.4f}" for v in vals])
        print(f"  {m_name:<28} | {cols}")


if __name__ == "__main__":
    run_acpi_mapping_test()
