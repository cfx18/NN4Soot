"""
NN4Soot: Integrated Neural Network Approach for Autonomous Sensitivity Analysis 
and Optimization of Sectional Soot Kinetic Modeling

Author: Feixue Cai

This package provides an integrated framework that couples neural networks with 
automatic differentiation for sensitivity analysis and gradient-based optimization 
in strongly nonlinear, high-dimensional sectional soot models.

Key Features:
- Neural network surrogate models for soot PSD prediction
- Active Subspace sensitivity analysis with bootstrap confidence intervals
- Gradient-based parameter optimization with experimental data fitting
- Mechanism invariance analysis across different parameter sets
"""

__version__ = "1.0.0"
__author__ = "Feixue Cai"

from nn4soot.models import SootMLP
from nn4soot.core import (
    SootTrainer,
    TrainingConfig,
    SootEvaluator,
    SootOptimizer,
    OptimizerConfig,
    ParameterRecovery,
    ParamRecoveryConfig,
    KineticRunner,
    KineticRunnerConfig,
    GenKineticsConfig,
)
from nn4soot.sensitivity import (
    ActiveSubspaceAnalyzer,
    ValleyAnalyzer,
)
from nn4soot.analysis import (
    MechanismSimilarityAnalyzer,
    ModelComparator,
    CombinedPlotter,
)

__all__ = [
    "SootMLP",
    "SootTrainer",
    "TrainingConfig",
    "SootEvaluator",
    "SootOptimizer",
    "OptimizerConfig",
    "ParameterRecovery",
    "ParamRecoveryConfig",
    "KineticRunner",
    "KineticRunnerConfig",
    "GenKineticsConfig",
    "ActiveSubspaceAnalyzer",
    "ValleyAnalyzer",
    "MechanismSimilarityAnalyzer",
    "ModelComparator",
    "CombinedPlotter",
]
