"""
Visualization Utilities for NN4Soot

This module provides standardized visualization tools for sensitivity
analysis, optimization results, and model comparison.

Author: Feixue Cai
"""

import os
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class SensitivityPlotter:
    """
    Standardized plotter for sensitivity analysis results.
    
    This class provides consistent visualization styles for:
    - Sensitivity bar charts with confidence intervals
    - Eigenvalue spectra
    - Comparison plots across conditions
    
    Parameters
    ----------
    style : str
        Matplotlib style (default: "default")
    dpi : int
        Default DPI for saved figures
    
    Examples
    --------
    >>> from nn4soot import SensitivityPlotter
    >>> plotter = SensitivityPlotter()
    >>> plotter.plot_sensitivity_bars(sensitivity, ci_low, ci_high, "output.png")
    """
    
    def __init__(self, style: str = "default", dpi: int = 300):
        self.style = style
        self.dpi = dpi
        plt.style.use(style)
    
    def plot_sensitivity_bars(
        self,
        sensitivity: np.ndarray,
        ci_low: Optional[np.ndarray] = None,
        ci_high: Optional[np.ndarray] = None,
        param_names: Optional[List[str]] = None,
        title: str = "Sensitivity Analysis",
        ylabel: str = "Sensitivity",
        save_path: Optional[str] = None,
        figsize: Tuple[float, float] = (8, 3),
        color: str = "#34495e",
    ) -> str:
        """
        Plot sensitivity as bar chart with optional confidence intervals.
        
        Parameters
        ----------
        sensitivity : np.ndarray
            Sensitivity values
        ci_low, ci_high : np.ndarray, optional
            Confidence interval bounds
        param_names : List[str], optional
            Parameter names (default: P1, P2, ...)
        title : str
            Plot title
        ylabel : str
            Y-axis label
        save_path : str, optional
            Path to save figure
        figsize : Tuple[float, float]
            Figure size
        color : str
            Bar color
        
        Returns
        -------
        str
            Path to saved figure
        """
        D = len(sensitivity)
        if param_names is None:
            param_names = [f"P{i+1}" for i in range(D)]
        
        x = np.arange(D)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if ci_low is not None and ci_high is not None:
            yerr = np.vstack([sensitivity - ci_low, ci_high - sensitivity])
            yerr = np.clip(np.nan_to_num(yerr, nan=0.0), 0.0, None)
        else:
            yerr = None
        
        ax.bar(x, sensitivity, yerr=yerr, capsize=3, color=color, alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(param_names)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        plt.close()
        return save_path or ""
    
    def plot_eigenvalue_spectrum(
        self,
        eigenvalues: np.ndarray,
        ci_low: Optional[np.ndarray] = None,
        ci_high: Optional[np.ndarray] = None,
        title: str = "Eigenvalue Spectrum",
        save_path: Optional[str] = None,
        figsize: Tuple[float, float] = (5, 3),
    ) -> str:
        """
        Plot eigenvalue spectrum with optional confidence intervals.
        
        Parameters
        ----------
        eigenvalues : np.ndarray
            Eigenvalue values
        ci_low, ci_high : np.ndarray, optional
            Confidence interval bounds
        title : str
            Plot title
        save_path : str, optional
            Path to save figure
        figsize : Tuple[float, float]
            Figure size
        
        Returns
        -------
        str
            Path to saved figure
        """
        D = len(eigenvalues)
        idx = np.arange(1, D + 1)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(idx, eigenvalues, "ko-", ms=4, label="Eigenvalues")
        
        if ci_low is not None and ci_high is not None:
            ax.fill_between(idx, ci_low, ci_high, color="grey", alpha=0.4, label="95% CI")
        
        ax.set_yscale("log")
        ax.set_xlabel("Eigenvalue Index")
        ax.set_ylabel("Eigenvalue")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        plt.close()
        return save_path or ""
    
    def plot_signed_sensitivity(
        self,
        sensitivity: np.ndarray,
        param_names: Optional[List[str]] = None,
        title: str = "Sensitivity",
        ylabel: str = "Sensitivity",
        save_path: Optional[str] = None,
        figsize: Tuple[float, float] = (8, 3),
    ) -> str:
        """
        Plot sensitivity with color indicating sign (red=positive, blue=negative).
        
        Parameters
        ----------
        sensitivity : np.ndarray
            Sensitivity values (can be negative)
        param_names : List[str], optional
            Parameter names
        title : str
            Plot title
        ylabel : str
            Y-axis label
        save_path : str, optional
            Path to save figure
        figsize : Tuple[float, float]
            Figure size
        
        Returns
        -------
        str
            Path to saved figure
        """
        D = len(sensitivity)
        if param_names is None:
            param_names = [f"P{i+1}" for i in range(D)]
        
        colors = ["#e74c3c" if v > 0 else "#3498db" for v in sensitivity]
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(param_names, sensitivity, color=colors, alpha=0.85)
        ax.axhline(0.0, color="k", linewidth=1)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        plt.close()
        return save_path or ""
    
    def plot_comparison_heatmap(
        self,
        data: np.ndarray,
        row_labels: List[str],
        col_labels: List[str],
        title: str = "Comparison",
        save_path: Optional[str] = None,
        cmap: str = "RdYlGn",
        figsize: Optional[Tuple[float, float]] = None,
    ) -> str:
        """
        Plot comparison as heatmap.
        
        Parameters
        ----------
        data : np.ndarray
            2D data array
        row_labels, col_labels : List[str]
            Labels for rows and columns
        title : str
            Plot title
        save_path : str, optional
            Path to save figure
        cmap : str
            Colormap name
        figsize : Tuple[float, float], optional
            Figure size
        
        Returns
        -------
        str
            Path to saved figure
        """
        if figsize is None:
            figsize = (len(col_labels) * 0.8 + 2, len(row_labels) * 0.6 + 1)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.imshow(data, cmap=cmap, aspect="auto")
        
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_xticklabels(col_labels)
        ax.set_yticklabels(row_labels)
        
        plt.colorbar(im, ax=ax)
        ax.set_title(title)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        plt.close()
        return save_path or ""
