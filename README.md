# NN4Soot

**Integrated Neural Network Approach for Autonomous Sensitivity Analysis and Optimization of Sectional Soot Kinetic Modeling**

## Overview

NN4Soot provides an integrated analysis framework that couples neural networks with automatic differentiation for sensitivity analysis and gradient-based optimization in strongly nonlinear, high-dimensional sectional soot models. Unlike prior approaches that suffer from severe nonlinearity and parameter couplings, our method provides stable, end-to-end gradients and interpretable sensitivities.

### Key Features

- **Neural Network Surrogate Models**: Efficient prediction of soot particle size distributions (PSD)
- **Active Subspace Sensitivity Analysis**: Global sensitivity ranking with bootstrap confidence intervals
- **Valley Sensitivity Analysis**: Directional sensitivity of bimodal valley characteristics
- **Soft-Valley Global Sensitivity**: Active-subspace sensitivity of the differentiable valley index
- **Mechanism Invariance Analysis**: Compare sensitivity patterns across different parameter sets
- **Gradient-Based Optimization**: Parameter fitting to experimental data with selective optimization

## Installation

```bash
# Clone the repository
git clone https://github.com/cfx18/NN4Soot.git
cd NN4Soot

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Quick Start

```python
from nn4soot import SootMLP, ActiveSubspaceAnalyzer
import numpy as np

# Load pretrained model
model = SootMLP.from_pretrained("pretrained/UF_set3/best_MLP_Hp=0.70.pth")

# Generate sample inputs (normalized kinetic parameters)
inputs = np.random.rand(100, 10)  # 100 samples, 10 parameters

# Perform sensitivity analysis
analyzer = ActiveSubspaceAnalyzer(model)
result = analyzer.analyze_with_bootstrap(inputs, bin_index=10, n_bootstrap=1000)

# View results
print("Sensitivity values:", result.sensitivity)
print("95% CI:", result.sensitivity_ci_low, result.sensitivity_ci_high)
```

## Project Structure

```
NN4Soot/
├── nn4soot/                 # Core package
│   ├── models/              # Neural network models
│   ├── core/                # Training, evaluation, optimization
│   ├── sensitivity/         # Sensitivity analysis methods
│   ├── analysis/            # Comparison and visualization
│   └── utils/               # Data loading and utilities
├── scripts/                 # Numbered pipeline steps (01-05)
├── reproduce_all.py         # Root-level full pipeline entrypoint
├── reproduce_all_skip_pretraining.py
│                            # Full pipeline without step 01
├── examples/                # Usage examples
├── data/                    # Data files
├── pretrained/              # Pretrained model weights
├── docs/                    # Documentation
└── tests/                   # Unit tests
```

## Methodology

### Neural Network Architecture

The surrogate model uses a multi-layer perceptron (MLP) with the following architecture:
- Input: 10 normalized kinetic parameters (P1-P10)
- Hidden layers: 64 → 128 → 256 → 256 → 128 → 64
- Output: 20-bin log10-transformed PSD values
- Activation: ReLU

### Active Subspace Sensitivity

The Active Subspace method computes global sensitivity through gradient covariance eigendecomposition:

```
C = E[∇f(x) · ∇f(x)^T]  (gradient covariance)
C = W Λ W^T             (eigendecomposition)
s_j = Σ_i (w_{j,i}² × λ_i / Σ_k λ_k)  (weighted sensitivity)
```

### Valley Sensitivity

For bimodal PSDs, we analyze sensitivity of valley characteristics using differentiable softmin/softmax:

- **Valley Position**: k_soft = Σ_m w_m × m, where w = softmax(-β × y)
- **Valley Depth**: y_valley = Σ_m w_m × y_m
- **Prominence**: y_prom = y_valley - (y_peak_L + y_peak_R) / 2

## Reproducing Results

```bash
# Step 01: train surrogate models
python scripts/01_train_from_config.py --config configs/train_config.yaml

# Step 02: compare MLP against baseline regressors
python scripts/02_run_model_comparison.py --data_dir data/ --model_dir pretrained/ --output_dir results/

# Step 03: run sensitivity analysis
python scripts/03_run_sensitivity_analysis.py --config configs/sensitivity_analysis_config.yaml

# Step 04: run mechanism similarity analysis
python scripts/04_run_mechanism_similarity.py --config configs/mechanism_similarity_config.yaml

# Step 05: run optimization
python scripts/05_run_optimization.py --config configs/optimization_softmin_config.yaml

# Run the full pipeline from the project root
python reproduce_all.py

# Faster variant: skip step 01 and reuse existing pretrained weights
python reproduce_all_skip_pretraining.py
```

### Example demos

```bash
python examples/01_quick_start.py
python examples/02_train_your_model.py
python examples/03_sensitivity_analysis.py
python examples/04_optimization.py
```


## Citation

If you use NN4Soot in your research, please cite:

```bibtex
@unpublished{cai2026nn4soot,
  author = {Feixue Cai and Patrizia Crepaldi and Andrea Nobili and Matthew J. Cleary and Zhuyin Ren and Assaad R. Masri and Tiziano Faravelli},
  title = {NN4Soot: Integrated Neural Network Approach for Autonomous Sensitivity Analysis and Optimization of Sectional Soot Kinetic Modeling},
  note = {Under review, submitted to Proceedings of the Combustion Institute},
  year = {2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
