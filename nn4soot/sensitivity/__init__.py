"""
Sensitivity analysis modules for soot kinetic modeling.

This package provides various sensitivity analysis methods:
- Active Subspace: Global sensitivity via gradient covariance eigendecomposition
- Valley Analysis: Sensitivity of bimodal valley position and depth
"""

from nn4soot.sensitivity.active_subspace import ActiveSubspaceAnalyzer
from nn4soot.sensitivity.valley_analysis import ValleyAnalyzer
from nn4soot.sensitivity.bootstrap import BootstrapCI

__all__ = [
    "ActiveSubspaceAnalyzer",
    "ValleyAnalyzer",
    "BootstrapCI",
]
