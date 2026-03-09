"""
Combined Visualization for Sensitivity Analysis

This module provides integrated visualization combining global sensitivity
with valley-position local sensitivity into unified figures.

Author: Feixue Cai
"""

import os
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class CombinedPlotter:
    """
    Plotter for combined sensitivity visualizations.
    
    This class creates integrated plots combining global sensitivity
    with valley position sensitivity, showing both magnitude and direction.
    
    Examples
    --------
    >>> from nn4soot import CombinedPlotter
    >>> plotter = CombinedPlotter()
    >>> plotter.plot_global_and_dvalley(global_df, valley_df, "output.png")
    """
    
    def __init__(self):
        pass
    
    def plot_global_and_dvalley(
        self,
        global_df: pd.DataFrame,
        valley_df: pd.DataFrame,
        save_path: str,
        param_col: str = "param",
        global_col: str = "boot_median",
        global_low_col: str = "ci95_low",
        global_high_col: str = "ci95_high",
        valley_col: str = "grad_d_valley_nm",
    ) -> str:
        """
        Plot global sensitivity with valley position sensitivity overlay.
        
        Parameters
        ----------
        global_df : pd.DataFrame
            Global sensitivity results
        valley_df : pd.DataFrame
            Valley sensitivity results
        save_path : str
            Path to save figure
        param_col, global_col, global_low_col, global_high_col, valley_col : str
            Column names in the DataFrames
        
        Returns
        -------
        str
            Path to saved figure
        """
        # Merge data
        df = global_df[[param_col, global_col, global_low_col, global_high_col]].copy()
        df = df.merge(valley_df[[param_col, valley_col]], on=param_col, how="inner")
        
        # Sort by parameter index
        df["param_idx"] = df[param_col].str.replace("P", "").astype(int)
        df = df.sort_values("param_idx").reset_index(drop=True)
        
        x = np.arange(len(df))
        
        global_values = df[global_col].to_numpy(float)
        ci_low = df[global_low_col].to_numpy(float)
        ci_high = df[global_high_col].to_numpy(float)
        d_grad = df[valley_col].to_numpy(float)
        
        yerr = np.vstack([global_values - ci_low, ci_high - global_values])
        yerr = np.nan_to_num(yerr, nan=0.0, posinf=0.0, neginf=0.0)
        yerr = np.clip(yerr, 0.0, None)
        
        fig, ax1 = plt.subplots(1, 1, figsize=(5, 2.5))
        
        # Left axis: global-sensitivity bars
        ax1.bar(x, global_values, yerr=yerr, capsize=3, color="#34495e", alpha=0.9,
               label="Global sensitivity")
        ax1.set_xticks(x)
        ax1.set_xticklabels(df[param_col].tolist())
        
        # Right axis: Valley gradient
        ax2 = ax1.twinx()
        ax2.plot(x, d_grad, "-o", color="#e67e22", linewidth=1.2, markersize=4,
                label="Local sensitivity")
        ax2.axhline(0.0, color="#e67e22", linestyle="--", linewidth=1, alpha=0.6)
        ax2.set_ylim([-0.2, 0.2])
        
        # Combined legend
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2, frameon=False, loc="upper right")
        
        plt.tight_layout()
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_sensitivity_by_bin(
        self,
        results_dict: Dict[int, Dict[float, np.ndarray]],
        param_names: List[str],
        save_path: str,
        hp_colors: Optional[Dict[float, str]] = None,
        title: str = "Sensitivity by BIN",
    ) -> str:
        """
        Plot sensitivity comparison across BINs and heights.
        
        Parameters
        ----------
        results_dict : Dict[int, Dict[float, np.ndarray]]
            {bin_index: {hp: sensitivity_array}}
        param_names : List[str]
            Parameter names
        save_path : str
            Path to save figure
        hp_colors : Dict[float, str], optional
            Colors for each height position
        title : str
            Plot title
        
        Returns
        -------
        str
            Path to saved figure
        """
        bin_indices = sorted(results_dict.keys())
        hp_values = sorted(list(results_dict[bin_indices[0]].keys()))
        
        if hp_colors is None:
            hp_colors = {0.50: "#3498db", 0.70: "#e74c3c", 1.00: "#2ecc71"}
        
        n_params = len(param_names)
        
        fig, axes = plt.subplots(len(bin_indices), 1, figsize=(10, 3 * len(bin_indices)),
                                  sharex=True)
        
        if len(bin_indices) == 1:
            axes = [axes]
        
        for ax, bin_idx in zip(axes, bin_indices):
            n_hps = len(hp_values)
            bar_width = 0.25
            x_positions = np.arange(n_params)
            
            for i, hp in enumerate(hp_values):
                sens = results_dict[bin_idx][hp]
                offset = (i - n_hps / 2 + 0.5) * bar_width
                
                ax.bar(x_positions + offset, sens, bar_width,
                      color=hp_colors.get(hp, f"C{i}"), alpha=0.8,
                      label=f"Hp={hp} cm")
            
            ax.set_ylabel("Sensitivity")
            ax.set_title(f"BIN {bin_idx}")
            ax.set_xticks(x_positions)
            ax.set_xticklabels(param_names)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3, axis="y")
        
        axes[-1].set_xlabel("Parameter")
        
        fig.suptitle(title, fontsize=12, fontweight="bold")
        plt.tight_layout()
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_similarity_summary(
        self,
        similarity_df: pd.DataFrame,
        save_path: str,
        metric_col: str = "L2Sim01",
    ) -> str:
        """
        Plot mechanism similarity summary.
        
        Parameters
        ----------
        similarity_df : pd.DataFrame
            Similarity results DataFrame
        save_path : str
            Path to save figure
        metric_col : str
            Similarity metric column name
        
        Returns
        -------
        str
            Path to saved figure
        """
        pairs = similarity_df["Pair"].unique()
        hp_values = sorted(similarity_df["Hp"].unique())
        
        y_base = [0.1, 0.5, 0.9]
        hp_colors = {0.50: "C0", 0.70: "C1", 1.00: "C2"}
        hp_labels = {0.50: "Hp=0.5 cm", 0.70: "Hp=0.7 cm", 1.00: "Hp=1.0 cm"}
        offsets = {0.50: -0.11, 0.70: 0.0, 1.00: 0.11}
        bar_h = 0.1
        
        fig, ax = plt.subplots(1, 1, figsize=(3, 5.5))
        
        for hp in hp_values:
            means = []
            xerr_low = []
            xerr_high = []
            ypos = []
            
            for i, pair in enumerate(pairs):
                vals = similarity_df[(similarity_df["Hp"] == hp) & 
                                     (similarity_df["Pair"] == pair)][metric_col].values
                mu = float(np.nanmean(vals))
                lo = float(np.nanpercentile(vals, 2.5))
                hi = float(np.nanpercentile(vals, 97.5))
                
                means.append(mu)
                xerr_low.append(mu - lo)
                xerr_high.append(hi - mu)
                ypos.append(y_base[i] + offsets[hp])
            
            ax.barh(ypos, means, height=bar_h, color=hp_colors[hp], alpha=0.85,
                   label=hp_labels[hp])
            ax.errorbar(means, ypos, xerr=[xerr_low, xerr_high], fmt="none",
                       ecolor="black", elinewidth=1.2, capsize=3, capthick=1.2, zorder=5)
        
        ax.set_yticks([0.2, 0.6, 1.0])
        ax.set_yticklabels(pairs, fontsize=11, rotation=90)
        ax.set_xlabel("Scaled L2 similarity", fontsize=12)
        ax.set_xlim([0, 1])
        ax.legend(frameon=True, ncol=1, loc="upper left", fontsize=10)
        
        plt.tight_layout()
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close()
        
        return save_path
