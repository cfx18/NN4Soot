"""
Valley Sensitivity Analysis for Bimodal Soot PSD

This module analyzes sensitivity of valley characteristics (position, depth, 
prominence) in bimodal particle size distributions using softmin/softmax 
for differentiable computations.

Key Features:
- Soft-argmin for continuous valley index computation
- Valley depth and prominence sensitivity with directional gradients
- Automatic bimodal peak detection from experimental data
- Bootstrap confidence intervals

Author: Feixue Cai
"""

import os
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class ValleyMetrics:
    """Container for valley metrics."""
    y_valley: torch.Tensor       # Valley depth (log10 PSD)
    d_valley: torch.Tensor       # Valley position (nm)
    y_peak_left: torch.Tensor    # Left peak height
    y_peak_right: torch.Tensor   # Right peak height
    y_prominence: torch.Tensor   # Relative valley depth


@dataclass
class ValleySensitivityResult:
    """Container for valley sensitivity results."""
    
    # Gradients (input_dim,)
    grad_y_valley: np.ndarray
    grad_d_valley_nm: np.ndarray
    grad_y_prominence: np.ndarray
    
    # Valley metrics at baseline
    y_valley: float
    d_valley_nm: float
    y_prominence: float
    x0: Optional[np.ndarray] = None
    left_peak_d_nm: Optional[float] = None
    right_peak_d_nm: Optional[float] = None
    valley_d_nm_exp: Optional[float] = None
    window_inds: Optional[np.ndarray] = None
    split_index_global: Optional[int] = None
    
    # Linear effects (for +/- delta perturbation)
    effects_y_valley: Optional[Dict[str, np.ndarray]] = None
    effects_d_valley: Optional[Dict[str, np.ndarray]] = None
    effects_y_prominence: Optional[Dict[str, np.ndarray]] = None
    
    # Finite difference comparison (optional)
    fd_grad_y_valley: Optional[np.ndarray] = None
    fd_grad_d_valley_nm: Optional[np.ndarray] = None
    fd_grad_y_prominence: Optional[np.ndarray] = None


def softmin_weights(y: torch.Tensor, beta: float) -> torch.Tensor:
    """Compute softmin weights."""
    z = -beta * y
    z = z - z.max()
    return torch.softmax(z, dim=0)


def softmax_weights(y: torch.Tensor, beta: float) -> torch.Tensor:
    """Compute softmax weights."""
    z = beta * y
    z = z - z.max()
    return torch.softmax(z, dim=0)


class ValleyAnalyzer:
    """
    Analyzer for bimodal valley sensitivity in soot PSD.
    
    This class computes gradients of valley characteristics (position, depth,
    prominence) with respect to kinetic parameters using automatic differentiation.
    
    Parameters
    ----------
    model : nn.Module
        Trained neural network model
    device : torch.device, optional
        Device for computation
    
    Examples
    --------
    >>> from nn4soot import SootMLP, ValleyAnalyzer
    >>> model = SootMLP.from_pretrained("model.pth")
    >>> analyzer = ValleyAnalyzer(model)
    >>> result = analyzer.analyze(baseline_params, d_bins, exp_peaks)
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        self.model.to(self.device)
        self.model.eval()
    
    def compute_valley_metrics(
        self,
        y_pred: torch.Tensor,
        d_bins_nm: torch.Tensor,
        window_inds: torch.Tensor,
        split_ind_in_window: int,
        beta_min: float = 20.0,
        beta_max: float = 20.0,
    ) -> ValleyMetrics:
        """
        Compute valley metrics from PSD prediction.
        
        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted log10 PSD (output_dim,)
        d_bins_nm : torch.Tensor
            Bin diameters in nm (output_dim,)
        window_inds : torch.Tensor
            Indices for valley window
        split_ind_in_window : int
            Index splitting left/right peaks
        beta_min : float
            Softmin temperature for valley
        beta_max : float
            Softmax temperature for peaks
        
        Returns
        -------
        ValleyMetrics
            Computed valley metrics
        """
        y_w = y_pred[window_inds]
        d_w = d_bins_nm[window_inds]
        
        # Valley position and depth (softmin)
        w_min = softmin_weights(y_w, beta=beta_min)
        y_valley = (w_min * y_w).sum()
        d_valley = (w_min * d_w).sum()
        
        # Left and right peaks (softmax)
        left = slice(0, split_ind_in_window + 1)
        right = slice(split_ind_in_window, y_w.numel())
        
        y_left = y_w[left]
        y_right = y_w[right]
        
        wL = softmax_weights(y_left, beta=beta_max)
        wR = softmax_weights(y_right, beta=beta_max)
        
        y_peak_left = (wL * y_left).sum()
        y_peak_right = (wR * y_right).sum()
        
        # Relative prominence
        y_prominence = y_valley - 0.5 * (y_peak_left + y_peak_right)
        
        return ValleyMetrics(
            y_valley=y_valley,
            d_valley=d_valley,
            y_peak_left=y_peak_left,
            y_peak_right=y_peak_right,
            y_prominence=y_prominence,
        )
    
    def find_double_peaks_and_valley(
        self,
        d_nm: np.ndarray,
        y_log10: np.ndarray,
        min_separation_in_logD: float = 0.15,
    ) -> Dict[str, float]:
        """
        Find double peaks and valley from experimental PSD.
        
        Parameters
        ----------
        d_nm : np.ndarray
            Diameter values (nm)
        y_log10 : np.ndarray
            log10 PSD values
        min_separation_in_logD : float
            Minimum peak separation in log10(D)
        
        Returns
        -------
        Dict[str, float]
            Peak and valley positions in nm
        """
        d_nm = np.asarray(d_nm, dtype=float)
        y_log10 = np.asarray(y_log10, dtype=float)
        mask = np.isfinite(d_nm) & np.isfinite(y_log10) & (d_nm > 0)
        d_nm = d_nm[mask]
        y_log10 = y_log10[mask]
        if d_nm.size < 7:
            raise ValueError("Experimental points too few for stable double-peak detection.")

        order = np.argsort(d_nm)
        d_nm = d_nm[order]
        y_log10 = y_log10[order]
        logD = np.log10(d_nm)

        idx = np.where((y_log10[1:-1] > y_log10[:-2]) & (y_log10[1:-1] > y_log10[2:]))[0] + 1
        end_idx: List[int] = []
        if y_log10.size >= 2:
            if y_log10[0] > y_log10[1]:
                end_idx.append(0)
            if y_log10[-1] > y_log10[-2]:
                end_idx.append(int(y_log10.size - 1))
        if end_idx:
            idx = np.unique(np.concatenate([idx, np.array(end_idx, dtype=int)]))

        if idx.size < 2:
            i_max = int(np.argmax(y_log10))
            left_candidates = np.arange(0, max(i_max, 1))
            right_candidates = np.arange(min(i_max + 1, y_log10.size - 1), y_log10.size)
            if left_candidates.size == 0 or right_candidates.size == 0:
                raise ValueError("Cannot separate left/right peaks from experimental data.")
            i_left = int(left_candidates[np.argmax(y_log10[left_candidates])])
            i_right = int(right_candidates[np.argmax(y_log10[right_candidates])])
            i1, i2 = sorted([i_left, i_right])
        else:
            cand = idx[np.argsort(y_log10[idx])[::-1]]
            chosen: List[int] = []
            for i in cand:
                if not chosen:
                    chosen.append(int(i))
                    continue
                if abs(logD[i] - logD[chosen[0]]) >= min_separation_in_logD and abs(int(i) - chosen[0]) >= 2:
                    chosen.append(int(i))
                    break
            if len(chosen) < 2:
                first = int(cand[0])
                far = int(cand[np.argmax(np.abs(logD[cand] - logD[first]))])
                chosen = [first, far]
            i1, i2 = sorted(chosen[:2])

        if i2 - i1 < 2:
            raise ValueError("Left/right peaks too close to define valley.")
        mid = slice(i1, i2 + 1)
        i_valley = int(i1 + np.argmin(y_log10[mid]))
        
        return {
            "left_peak_d_nm": float(d_nm[i1]),
            "right_peak_d_nm": float(d_nm[i2]),
            "valley_d_nm": float(d_nm[i_valley]),
        }
    
    def analyze(
        self,
        baseline_params: np.ndarray,
        d_bins_nm: np.ndarray,
        exp_peak_valley: Dict[str, float],
        beta_min: float = 20.0,
        beta_max: float = 20.0,
        compute_fd: bool = True,
        fd_eps: float = 1e-2,
        delta: float = 0.1,
    ) -> ValleySensitivityResult:
        """
        Analyze valley sensitivity at baseline parameters.
        
        Parameters
        ----------
        baseline_params : np.ndarray
            Baseline parameter values (normalized [0,1])
        d_bins_nm : np.ndarray
            Bin diameters in nm
        exp_peak_valley : Dict[str, float]
            Experimental peak and valley positions
        beta_min : float
            Softmin temperature
        beta_max : float
            Softmax temperature
        compute_fd : bool
            Whether to compute finite difference gradients
        fd_eps : float
            FD perturbation size
        delta : float
            Delta for linear effect computation
        
        Returns
        -------
        ValleySensitivityResult
            Sensitivity analysis results
        """
        baseline_params = np.asarray(baseline_params, dtype=float)
        input_dim = len(baseline_params)
        d_bins = torch.tensor(d_bins_nm, dtype=torch.float32, device=self.device)
        
        # Determine window from experimental peaks
        left_peak = exp_peak_valley["left_peak_d_nm"]
        right_peak = exp_peak_valley["right_peak_d_nm"]
        valley = exp_peak_valley["valley_d_nm"]
        
        left_bin = int(np.argmin(np.abs(d_bins_nm - left_peak)))
        right_bin = int(np.argmin(np.abs(d_bins_nm - right_peak)))
        
        lo, hi = (left_bin, right_bin) if left_bin < right_bin else (right_bin, left_bin)
        window_inds_np = np.arange(lo, hi + 1)
        
        split_global = int(np.argmin(np.abs(d_bins_nm - valley)))
        split_global = int(np.clip(split_global, lo, hi))
        split_in_window = split_global - lo
        
        window_inds = torch.tensor(window_inds_np, dtype=torch.long, device=self.device)
        
        # Compute gradients at baseline
        x = torch.tensor(
            baseline_params, dtype=torch.float32, 
            device=self.device, requires_grad=True
        ).unsqueeze(0)
        
        y_pred = self.model(x)[0]
        
        metrics = self.compute_valley_metrics(
            y_pred, d_bins, window_inds, split_in_window,
            beta_min, beta_max
        )
        
        # Compute gradients
        grads = {}
        for name, scalar in [
            ("y_valley", metrics.y_valley),
            ("d_valley_nm", metrics.d_valley),
            ("y_prominence", metrics.y_prominence),
        ]:
            g = torch.autograd.grad(scalar, x, retain_graph=True)[0][0]
            grads[name] = g.detach().cpu().numpy()
        
        # Optional: finite difference comparison
        fd_grads = {}
        if compute_fd:
            for metric_name in ["y_valley", "d_valley_nm", "y_prominence"]:
                fd_grads[metric_name] = np.zeros(input_dim)
            
            with torch.no_grad():
                for i in range(input_dim):
                    x_p = baseline_params.copy()
                    x_m = baseline_params.copy()
                    x_p[i] = np.clip(x_p[i] + fd_eps, 0.0, 1.0)
                    x_m[i] = np.clip(x_m[i] - fd_eps, 0.0, 1.0)
                    
                    for xp, xm, sign in [(x_p, x_m, 1)]:
                        yp = self.model(
                            torch.tensor(xp, dtype=torch.float32, device=self.device).unsqueeze(0)
                        )[0]
                        ym = self.model(
                            torch.tensor(xm, dtype=torch.float32, device=self.device).unsqueeze(0)
                        )[0]
                        
                        mp = self.compute_valley_metrics(
                            yp, d_bins, window_inds, split_in_window, beta_min, beta_max
                        )
                        mm = self.compute_valley_metrics(
                            ym, d_bins, window_inds, split_in_window, beta_min, beta_max
                        )
                        
                        fd_grads["y_valley"][i] = float(
                            (mp.y_valley - mm.y_valley).cpu().item()
                        ) / (2 * fd_eps)
                        fd_grads["d_valley_nm"][i] = float(
                            (mp.d_valley - mm.d_valley).cpu().item()
                        ) / (2 * fd_eps)
                        fd_grads["y_prominence"][i] = float(
                            (mp.y_prominence - mm.y_prominence).cpu().item()
                        ) / (2 * fd_eps)
        
        # Compute linear effects
        effects_y_valley = {
            f"+{delta}": grads["y_valley"] * delta,
            f"-{delta}": -grads["y_valley"] * delta,
        }
        effects_d_valley = {
            f"+{delta}": grads["d_valley_nm"] * delta,
            f"-{delta}": -grads["d_valley_nm"] * delta,
        }
        effects_y_prominence = {
            f"+{delta}": grads["y_prominence"] * delta,
            f"-{delta}": -grads["y_prominence"] * delta,
        }
        
        return ValleySensitivityResult(
            grad_y_valley=grads["y_valley"],
            grad_d_valley_nm=grads["d_valley_nm"],
            grad_y_prominence=grads["y_prominence"],
            y_valley=float(metrics.y_valley.cpu().item()),
            d_valley_nm=float(metrics.d_valley.cpu().item()),
            y_prominence=float(metrics.y_prominence.cpu().item()),
            x0=baseline_params.copy(),
            left_peak_d_nm=float(left_peak),
            right_peak_d_nm=float(right_peak),
            valley_d_nm_exp=float(valley),
            window_inds=window_inds_np.copy(),
            split_index_global=int(split_global),
            effects_y_valley=effects_y_valley,
            effects_d_valley=effects_d_valley,
            effects_y_prominence=effects_y_prominence,
            fd_grad_y_valley=fd_grads.get("y_valley") if compute_fd else None,
            fd_grad_d_valley_nm=fd_grads.get("d_valley_nm") if compute_fd else None,
            fd_grad_y_prominence=fd_grads.get("y_prominence") if compute_fd else None,
        )
    
    def plot_sensitivity(
        self,
        result: ValleySensitivityResult,
        param_names: Optional[List[str]] = None,
        title: str = "Valley Sensitivity",
        save_path: Optional[str] = None,
    ) -> str:
        """
        Plot valley sensitivity as bar charts.
        
        Parameters
        ----------
        result : ValleySensitivityResult
            Analysis result
        param_names : List[str], optional
            Parameter names
        title : str
            Plot title
        save_path : str, optional
            Path to save figure
        
        Returns
        -------
        str
            Path to saved figure
        """
        D = len(result.grad_y_valley)
        if param_names is None:
            param_names = [f"P{i+1}" for i in range(D)]
        
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        
        def bar(ax, values, title, ylabel):
            colors = ["#e74c3c" if v > 0 else "#3498db" for v in values]
            ax.bar(param_names, values, color=colors, alpha=0.85)
            ax.axhline(0.0, color="k", linewidth=1)
            ax.set_title(title, fontsize=11)
            ax.set_ylabel(ylabel)
            ax.grid(True, axis="y", alpha=0.3)
        
        bar(axes[0], result.grad_y_valley, 
            "Valley depth sensitivity  d(y_valley)/dPi   beta_min=20.0", 
            "d(log10 PSD)/dPi")
        bar(axes[1], result.grad_d_valley_nm, 
            "Valley location sensitivity  d(d_valley)/dPi", 
            "d(nm)/dPi")
        bar(axes[2], result.grad_y_prominence, 
            "Relative valley depth sensitivity  d(y_prominence)/dPi   beta_max=20.0", 
            "d(log10 PSD)/dPi")
        
        axes[2].set_xlabel("Parameter")
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
        return save_path or ""
    
    def save_results(
        self,
        result: ValleySensitivityResult,
        save_path: str,
        param_names: Optional[List[str]] = None,
    ) -> str:
        """Save valley sensitivity results to CSV."""
        D = len(result.grad_y_valley)
        if param_names is None:
            param_names = [f"P{i+1}" for i in range(D)]
        
        df = pd.DataFrame({
            "param": param_names,
            "x0": result.x0,
            "grad_y_valley": result.grad_y_valley,
            "grad_d_valley_nm": result.grad_d_valley_nm,
            "grad_y_prominence": result.grad_y_prominence,
        })
        if result.effects_y_valley is not None:
            for key, values in result.effects_y_valley.items():
                df[f"effect_({key})_y_valley"] = values
        if result.effects_d_valley is not None:
            for key, values in result.effects_d_valley.items():
                df[f"effect_({key})_d_valley_nm"] = values
        if result.effects_y_prominence is not None:
            for key, values in result.effects_y_prominence.items():
                df[f"effect_({key})_y_prominence"] = values
        
        if result.fd_grad_y_valley is not None:
            df["fd_grad_y_valley"] = result.fd_grad_y_valley
            df["fd_grad_d_valley_nm"] = result.fd_grad_d_valley_nm
            df["fd_grad_y_prominence"] = result.fd_grad_y_prominence
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        
        return save_path

    def plot_window_debug(
        self,
        exp_d_nm: np.ndarray,
        exp_psd_linear: np.ndarray,
        result: ValleySensitivityResult,
        save_path: str,
        exp_mode_label: str = "Exp",
    ) -> str:
        """Plot the experimental double-peak detection window debug figure."""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        ax.plot(exp_d_nm, exp_psd_linear, "ks", ms=4, label=exp_mode_label)
        ax.axvline(result.left_peak_d_nm, color="#2ecc71", linestyle="--", linewidth=1, label="left peak")
        ax.axvline(result.right_peak_d_nm, color="#2ecc71", linestyle="--", linewidth=1, label="right peak")
        ax.axvline(result.valley_d_nm_exp, color="#e67e22", linestyle="--", linewidth=1, label="valley")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Mobility diameter, $D_m$ (nm)")
        ax.set_ylabel("dN/dlog($D_m$) (a.u.)")
        ax.set_title("Experimental double-peak detection (Hp=0.70) + valley")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(frameon=False, fontsize=9)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return save_path
