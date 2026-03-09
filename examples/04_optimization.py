"""
Optimization example for NN4Soot.

This example uses synthetic PSD targets defined on the same diameter grid as
the model output so it can run end-to-end without external files.
"""

from pathlib import Path

import numpy as np

from nn4soot import OptimizerConfig, SootMLP, SootOptimizer
from nn4soot.utils.data_loader import DataLoader


model = SootMLP(input_dim=10, output_dim=20)
model.eval()

d_bins = DataLoader.get_default_diameters()
np.random.seed(42)

target_data = {}
for hp, scale in [(0.50, 0.90), (0.70, 1.00), (1.00, 1.10)]:
    profile = 8.0 + scale * 0.25 * np.sin(np.log10(d_bins) * 3.0)
    profile += 0.08 * np.cos(np.log10(d_bins) * 5.0)
    target_data[hp] = {"dia": d_bins.copy(), "psd": 10 ** profile}

print("=" * 60)
print("Parameter Optimization Example")
print("=" * 60)

output_dir = Path(__file__).resolve().parents[1] / "results" / "example_optimization"
output_dir.mkdir(parents=True, exist_ok=True)

config = OptimizerConfig(
    num_epochs=200,
    initial_lr=0.01,
    early_stop_patience=30,
    save_dir=str(output_dir),
)

optimizer = SootOptimizer(model, config)
initial_params = np.full(10, 0.5, dtype=np.float32)
optimize_indices = [0, 1]

print("\nOptimizing parameters: P1, P2")
print(f"Initial values: {initial_params[optimize_indices]}")

optimized_params, history = optimizer.optimize(
    target_data,
    initial_params,
    optimize_indices,
    verbose=True,
)

print("\n" + "=" * 60)
print("Optimization Results")
print("=" * 60)
for i, (init, opt) in enumerate(zip(initial_params, optimized_params)):
    marker = " *" if i in optimize_indices else ""
    print(f"  P{i+1}: {init:.4f} -> {opt:.4f}{marker}")

print(f"\nFinal loss: {history['loss'][-1]:.6f}")
print(f"Number of epochs: {len(history['loss'])}")

results_path = output_dir / "optimization_results.csv"
optimizer.save_results(optimized_params, history["loss"], history["params"], str(results_path))
plot_path = output_dir / "optimization_result.png"
optimizer.plot_optimization_result(
    exp_data_dict=target_data,
    param_history=history["params"],
    model_dia=d_bins,
    save_path=str(plot_path),
    hp_values=[0.50, 0.70, 1.00],
)

print(f"\nSaved results to: {results_path}")
print(f"Saved plot to: {plot_path}")
print("\nDone.")