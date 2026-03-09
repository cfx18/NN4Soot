"""
Analysis modules for soot kinetic modeling.

This package provides analysis tools:
- Mechanism Similarity: Compare sensitivity patterns across UF sets
- Model Comparison: Compare MLP with baseline models
- Combined Plots: Integrated visualization
"""

from nn4soot.analysis.mechanism_similarity import MechanismSimilarityAnalyzer
from nn4soot.analysis.model_comparison import ModelComparator
from nn4soot.analysis.combined_plots import CombinedPlotter

__all__ = [
    "MechanismSimilarityAnalyzer",
    "ModelComparator",
    "CombinedPlotter",
]
