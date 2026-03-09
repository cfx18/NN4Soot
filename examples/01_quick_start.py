"""
Quick start example for NN4Soot.

This example demonstrates the basic workflow:
1. Load a pretrained model if available
2. Make PSD predictions
3. Compute Active Subspace sensitivity
"""

from pathlib import Path

import numpy as np
import torch

from nn4soot import ActiveSubspaceAnalyzer, SootMLP


project_root = Path(__file__).resolve().parents[1]
checkpoint = project_root / "pretrained" / "UF_set3" / "best_MLP_Hp=0.70.pth"

print("Loading model...")
if checkpoint.exists():
    model = SootMLP.from_pretrained(str(checkpoint))
    print(f"Using pretrained checkpoint: {checkpoint}")
else:
    model = SootMLP(input_dim=10, output_dim=20)
    print("Pretrained checkpoint not found, using a randomly initialized model.")

np.random.seed(42)
inputs = np.random.rand(100, 10).astype(np.float32)

print("\nMaking predictions...")
model.eval()
with torch.no_grad():
    x = torch.tensor(inputs, dtype=torch.float32)
    predictions = model(x).cpu().numpy()

print(f"Input shape: {inputs.shape}")
print(f"Output shape: {predictions.shape}")
print(f"Output range: [{predictions.min():.2f}, {predictions.max():.2f}] (log10 PSD)")

print("\nComputing sensitivity analysis for BIN 10...")
analyzer = ActiveSubspaceAnalyzer(model)
result = analyzer.analyze_with_bootstrap(inputs, bin_index=10, n_bootstrap=100)

param_names = [f"P{i+1}" for i in range(inputs.shape[1])]
print("\nTop 3 most sensitive parameters:")
top_idx = np.argsort(result.sensitivity)[::-1][:3]
for rank, idx in enumerate(top_idx, start=1):
    print(f"  {rank}. {param_names[idx]}: {result.sensitivity[idx]:.4f}")

print("\nDone.")