"""
Bootstrap Utilities for Uncertainty Quantification

This module provides bootstrap methods for computing confidence intervals
in sensitivity analysis.

Author: Feixue Cai
"""

import numpy as np
from typing import Optional, Tuple, List, Callable
from dataclasses import dataclass


@dataclass
class BootstrapCI:
    """Bootstrap confidence interval utilities."""
    
    n_bootstrap: int = 1000
    ci_level: float = 0.95
    seed: int = 42
    
    def compute_ci(
        self,
        samples: np.ndarray,
        axis: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute bootstrap mean and confidence interval.
        
        Parameters
        ----------
        samples : np.ndarray
            Sample array
        axis : int
            Axis along which to compute
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Mean, CI low, CI high
        """
        rng = np.random.default_rng(self.seed)
        
        n = samples.shape[axis]
        boot_samples = []
        
        for _ in range(self.n_bootstrap):
            idx = rng.integers(0, n, size=n)
            if axis == 0:
                boot = samples[idx].mean(axis=axis)
            else:
                boot = np.take(samples, idx, axis=axis).mean(axis=axis)
            boot_samples.append(boot)
        
        boot_samples = np.array(boot_samples)
        
        mean = boot_samples.mean(axis=0)
        
        alpha = (1.0 - self.ci_level) / 2.0
        ci_low = np.quantile(boot_samples, alpha, axis=0)
        ci_high = np.quantile(boot_samples, 1.0 - alpha, axis=0)
        
        return mean, ci_low, ci_high
    
    def compute_ci_from_values(
        self,
        values: np.ndarray,
    ) -> Tuple[float, float, float]:
        """
        Compute CI for a single value from bootstrap samples.
        
        Parameters
        ----------
        values : np.ndarray
            Bootstrap sample values (n_bootstrap,)
        
        Returns
        -------
        Tuple[float, float, float]
            Mean, CI low, CI high
        """
        mean = float(np.mean(values))
        alpha = (1.0 - self.ci_level) / 2.0
        ci_low = float(np.quantile(values, alpha))
        ci_high = float(np.quantile(values, 1.0 - alpha))
        
        return mean, ci_low, ci_high


def bootstrap_percentile(
    data: np.ndarray,
    statistic: Callable,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bootstrap percentile method for confidence intervals.
    
    Parameters
    ----------
    data : np.ndarray
        Data array
    statistic : Callable
        Function to compute statistic
    n_bootstrap : int
        Number of bootstrap samples
    ci_level : float
        Confidence level
    seed : int
        Random seed
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Statistic estimate, CI low, CI high
    """
    rng = np.random.default_rng(seed)
    n = len(data)
    
    boot_stats = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_stat = statistic(data[idx])
        boot_stats.append(boot_stat)
    
    boot_stats = np.array(boot_stats)
    
    estimate = statistic(data)
    
    alpha = (1.0 - ci_level) / 2.0
    ci_low = np.quantile(boot_stats, alpha, axis=0)
    ci_high = np.quantile(boot_stats, 1.0 - alpha, axis=0)
    
    return estimate, ci_low, ci_high
