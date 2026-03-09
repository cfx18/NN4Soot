"""
Sensitivity-analysis example for NN4Soot.

This example demonstrates:
- Active Subspace analysis on several BINs
- Valley sensitivity analysis using synthetic bimodal reference data
"""

import numpy as np

from nn4soot import ActiveSubspaceAnalyzer, SootMLP, ValleyAnalyzer
from nn4soot.utils.data_loader import DataLoader


model = SootMLP(input_dim=10, output_dim=20)
model.eval()

np.random.seed(42)
inputs = np.random.rand(200, 10).astype(np.float32)
d_bins = DataLoader.get_default_diameters()

print("=" * 60)
print("Sensitivity Analysis Example")
print("=" * 60)

print("\n1. Active Subspace Analysis")
print("-" * 40)
as_analyzer = ActiveSubspaceAnalyzer(model)

for bin_idx in [7, 12, 17]:
    result = as_analyzer.analyze_with_bootstrap(inputs, bin_idx, n_bootstrap=100)
    top_params = np.argsort(result.sensitivity)[::-1][:3]
    print(f"\nBIN {bin_idx} top 3 sensitive parameters:")
    for rank, idx in enumerate(top_params, start=1):
        print(
            f"  {rank}. P{idx+1}: {result.sensitivity[idx]:.4f} "
            f"[{result.sensitivity_ci_low[idx]:.4f}, {result.sensitivity_ci_high[idx]:.4f}]"
        )

print("\n2. Valley Sensitivity Analysis")
print("-" * 40)

# A synthetic bimodal PSD reference curve used only to locate peaks and valley.
exp_dia = np.geomspace(2.0, 100.0, 30)
log_d = np.log10(exp_dia)
exp_psd = 10 ** (
    8.2
    + 0.65 * np.exp(-((log_d - np.log10(5.0)) ** 2) / 0.02)
    + 0.55 * np.exp(-((log_d - np.log10(45.0)) ** 2) / 0.03)
)

valley_analyzer = ValleyAnalyzer(model)
peak_valley = valley_analyzer.find_double_peaks_and_valley(exp_dia, np.log10(exp_psd))
baseline = np.full(10, 0.5, dtype=np.float32)

result = valley_analyzer.analyze(
    baseline,
    d_bins,
    peak_valley,
    beta_min=20.0,
    beta_max=20.0,
    compute_fd=False,
)

print("\nDetected reference geometry:")
print(f"  Left peak: {peak_valley['left_peak_d_nm']:.2f} nm")
print(f"  Right peak: {peak_valley['right_peak_d_nm']:.2f} nm")
print(f"  Valley: {peak_valley['valley_d_nm']:.2f} nm")

print("\nValley metrics at baseline:")
print(f"  Valley depth (log10 PSD): {result.y_valley:.4f}")
print(f"  Valley position (nm): {result.d_valley_nm:.2f}")
print(f"  Relative prominence: {result.y_prominence:.4f}")

print("\nValley position sensitivity (d(d_valley)/dP):")
for i, grad in enumerate(result.grad_d_valley_nm):
    sign = "+" if grad > 0 else ""
    print(f"  P{i+1}: {sign}{grad:.4f} nm per unit P")

print("\nDone.")