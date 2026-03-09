"""
SootOptimizer: Gradient-Based Parameter Optimization for Soot Kinetic Models

This module provides optimization tools for fitting kinetic parameters to 
experimental soot PSD data using the neural network surrogate model.

Key Features:
- Gradient-based optimization with automatic differentiation
- Selective parameter optimization via gradient masking
- Softmin loss for targeting specific particle size bins
- Early stopping for robust convergence

Author: Feixue Cai
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


@dataclass
class OptimizerConfig:
    """Configuration for SootOptimizer."""
    
    # Optimization parameters
    num_epochs: int = 1500
    initial_lr: float = 0.01
    lr_schedule: Dict[int, float] = field(default_factory=lambda: {0: 0.01})
    
    # Parameter bounds
    param_min: float = 0.0
    param_max: float = 1.0
    
    # Early stopping
    early_stop_patience: int = 20
    early_stop_min_delta: float = 1e-6
    
    # Softmin loss
    softmin_beta: float = 1.0
    softmin_weight: float = 100.0
    
    # Output
    save_dir: str = "."


class Logger:
    """Logger for capturing optimization output to file."""
    
    def __init__(self, filename: str):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()


class SootOptimizer:
    """
    Gradient-based optimizer for soot kinetic parameters.
    
    This class optimizes kinetic parameters by fitting the neural network
    surrogate model predictions to experimental PSD data.
    
    Parameters
    ----------
    model : nn.Module
        Trained SootMLP model
    config : OptimizerConfig, optional
        Optimization configuration
    device : torch.device, optional
        Device for computation
    
    Examples
    --------
    >>> from nn4soot import SootMLP, SootOptimizer
    >>> model = SootMLP.from_pretrained("model.pth")
    >>> optimizer = SootOptimizer(model)
    >>> optimized_params, history = optimizer.optimize(
    ...     exp_data, optimize_indices=[0, 1]
    ... )
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[OptimizerConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.config = config or OptimizerConfig()
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        self.model.to(self.device)
        self.model.eval()
        
        # Optimization history
        self.param_history: List[np.ndarray] = []
        self.loss_history: List[float] = []
    
    def _mask_gradients(
        self,
        param_tensor: torch.Tensor,
        optimize_indices: List[int],
    ):
        """Mask gradients to only update selected parameters."""
        with torch.no_grad():
            grad = param_tensor.grad
            if grad is not None:
                mask = torch.zeros_like(grad)
                mask[0, optimize_indices] = 1
                grad *= mask
    
    def _softmin_loss(
        self,
        y: torch.Tensor,
        index_opt: int,
    ) -> torch.Tensor:
        """
        Compute softmin loss targeting a specific bin.
        
        The softmin loss encourages lower values at the target bin,
        useful for optimizing particle concentrations at specific sizes.
        """
        P = F.softmax(y, dim=1)
        loss = P[:, index_opt]
        return loss
    
    def interpolate_exp_data(
        self,
        exp_dia: np.ndarray,
        exp_psds: np.ndarray,
        model_dia: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolate experimental data to model bin positions.
        
        Parameters
        ----------
        exp_dia : np.ndarray
            Experimental diameter values
        exp_psds : np.ndarray
            Experimental PSD values
        model_dia : np.ndarray
            Model bin diameter values
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Filtered diameter and interpolated PSD values
        """
        # Filter to valid range
        max_exp_dia = np.max(exp_dia)
        min_exp_dia = np.min(exp_dia)
        
        valid_mask = (model_dia >= min_exp_dia) & (model_dia <= max_exp_dia)
        model_dia_filtered = model_dia[valid_mask]
        
        # Interpolate
        interp_psds = np.interp(model_dia_filtered, exp_dia, exp_psds)
        
        return model_dia_filtered, interp_psds
    
    def optimize(
        self,
        exp_data_dict: Dict[float, Dict],
        initial_params: np.ndarray,
        optimize_indices: List[int],
        bin_ranges: Optional[Dict[float, Tuple[int, int]]] = None,
        use_softmin: bool = False,
        softmin_config: Optional[Dict] = None,
        verbose: bool = True,
        models_by_hp: Optional[Dict[float, nn.Module]] = None,
    ) -> Tuple[np.ndarray, Dict[str, List]]:
        """
        Optimize parameters to fit experimental data.

        Each height position (Hp) uses its own dedicated surrogate model,
        matching the original training-loop design where a different checkpoint
        is loaded per Hp.

        Parameters
        ----------
        exp_data_dict : Dict[float, Dict]
            Experimental data keyed by hp.
            Format: {hp: {'dia': array, 'psd': array}}
        initial_params : np.ndarray
            Initial parameter values (normalized to [0, 1]).
        optimize_indices : List[int]
            Indices of parameters to optimize (others are frozen).
        bin_ranges : Dict[float, Tuple[int, int]], optional
            Model output slice for each hp: {hp: (lo, hi)} inclusive.
        use_softmin : bool
            Whether to add the softmax-based valley loss.
        softmin_config : Dict, optional
            {'hp': float, 'bin_index': int} — which Hp and output index.
        verbose : bool
            Print loss every 10 epochs.
        models_by_hp : Dict[float, nn.Module], optional
            Per-Hp surrogate models.  If *None*, ``self.model`` is used for
            every Hp (legacy fallback, not recommended).

        Returns
        -------
        Tuple[np.ndarray, Dict]
            Optimized parameters and history dict with keys 'params', 'loss'.
        """
        # Initialize parameters
        params = torch.tensor(
            initial_params, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        params.requires_grad = True

        # Setup optimizer
        adam = optim.Adam([params], lr=self.config.initial_lr)

        # Early stopping
        best_loss = float('inf')
        patience_counter = 0

        # History
        self.param_history = []
        self.loss_history = []

        for epoch in range(self.config.num_epochs):
            running_loss = 0.0

            for hp, exp_data in exp_data_dict.items():
                # Select the Hp-specific model (mirrors legacy per-Hp loading)
                model_hp = (models_by_hp or {}).get(hp, self.model)
                model_hp.to(self.device)
                model_hp.eval()

                exp_psd = torch.tensor(
                    np.log10(exp_data['psd']),
                    dtype=torch.float32,
                    device=self.device,
                )

                # Forward pass with this Hp's model
                pred = model_hp(params)

                # Apply bin range if specified
                if bin_ranges and hp in bin_ranges:
                    lo, hi = bin_ranges[hp]
                    pred_cut = pred[0, lo:hi + 1]
                else:
                    pred_cut = pred[0, :]

                # MSE loss against experimental data
                loss = F.mse_loss(pred_cut, exp_psd)

                # Optional softmin loss (only for the configured Hp)
                if use_softmin and softmin_config:
                    if abs(hp - softmin_config.get('hp', 0.70)) < 1e-6:
                        softmin_loss = self._softmin_loss(
                            pred, softmin_config['bin_index']
                        )
                        loss = loss + softmin_loss * self.config.softmin_weight

                loss.backward()
                self._mask_gradients(params, optimize_indices)

                adam.step()
                adam.zero_grad()

                with torch.no_grad():
                    params.clamp_(self.config.param_min, self.config.param_max)

                running_loss += loss.item()

            avg_loss = running_loss / len(exp_data_dict)
            self.loss_history.append(avg_loss)
            self.param_history.append(params.detach().cpu().numpy().copy())

            if verbose and (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{self.config.num_epochs}], Loss: {avg_loss:.6f}')

            if avg_loss < best_loss - self.config.early_stop_min_delta:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stop_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break

        optimized_params = params.detach().cpu().numpy().squeeze()
        return optimized_params, {
            'params': self.param_history,
            'loss': self.loss_history,
        }
    
    def plot_optimization_result(
        self,
        exp_data_dict: Dict[float, Dict],
        param_history: List[np.ndarray],
        model_dia: np.ndarray,
        save_path: str,
        hp_values: List[float] = [0.50, 0.70, 1.00],
        models_by_hp: Optional[Dict[float, "nn.Module"]] = None,
        plot_stride: int = 1,
    ) -> str:
        """
        Plot optimization trajectory, nominal baseline, and experimental data.

        For each height position, plots:
        - Every ``plot_stride``-th epoch prediction, coloured by the ``cool``
          colormap (oldest = blue, newest = red).
        - Nominal prediction at x = 0.5 (blue solid line).
        - Experimental data points (black squares).

        Parameters
        ----------
        exp_data_dict : Dict[float, Dict]
            Experimental data keyed by hp.  Values have keys ``'dia'`` and
            ``'psd'`` (linear scale).
        param_history : List[np.ndarray]
            List of parameter arrays saved at each optimisation epoch.
        model_dia : np.ndarray
            Bin diameter values (nm) – x-axis for model output.
        save_path : str
            File path to save the figure (dpi=600).
        hp_values : List[float]
            Height positions to plot (one subplot each).
        models_by_hp : Dict[float, nn.Module], optional
            Per-hp models.  If *None*, ``self.model`` is used for all hps.
        plot_stride : int
            Plot every ``plot_stride``-th epoch (default 1 = all epochs).

        Returns
        -------
        str
            Path to saved figure.
        """
        n_epochs = len(param_history)
        norm = Normalize(vmin=0, vmax=max(n_epochs - 1, 1))
        colors = plt.cm.cool(norm(range(n_epochs)))
        colors[-1] = np.array([1, 0, 0, 1])   # last epoch → red

        nominal_params = torch.zeros(1, param_history[0].squeeze().shape[0],
                                     dtype=torch.float32, device=self.device)
        nominal_params[:] = 0.5

        fig, axes = plt.subplots(1, len(hp_values), figsize=(7.2, 2.7),
                                 sharey=True,
                                 gridspec_kw={"wspace": 0})
        if len(hp_values) == 1:
            axes = [axes]

        for i, hp in enumerate(hp_values):
            ax = axes[i]
            model_hp = (models_by_hp or {}).get(hp, self.model)
            model_hp.to(self.device)
            model_hp.eval()

            # ── Nominal baseline ────────────────────────────────────────────
            with torch.no_grad():
                nominal_pred = model_hp(nominal_params).cpu().numpy()
            ax.plot(model_dia, 10 ** nominal_pred[0, :],
                    "b-", linewidth=1.5, alpha=0.8, label="Nominal")

            # ── Optimisation trajectory ─────────────────────────────────────
            epoch_indices = list(range(0, n_epochs, plot_stride))
            if (n_epochs - 1) not in epoch_indices:
                epoch_indices.append(n_epochs - 1)

            for epoch_idx in epoch_indices:
                params = np.asarray(param_history[epoch_idx]).squeeze()
                p_t = torch.tensor(params, dtype=torch.float32,
                                   device=self.device).unsqueeze(0)
                with torch.no_grad():
                    pred = model_hp(p_t).cpu().numpy()
                ax.plot(model_dia, 10 ** pred[0, :],
                        color=colors[epoch_idx], linestyle="-", alpha=0.8,
                        label="Optimized" if (epoch_idx == n_epochs - 1 and i == 1)
                        else None)

            # ── Experimental data ───────────────────────────────────────────
            if hp in exp_data_dict:
                ed = exp_data_dict[hp]
                ax.plot(ed["dia"], ed["psd"], "ks",
                        markerfacecolor="k", markersize=6,
                        linestyle="none", markeredgewidth=0,
                        alpha=0.8, label="Exp" if i == 1 else None)

            ax.set_xlabel("Mobility diameter, $D_m$ (nm)")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim([2, 100])
            ax.set_ylim([1e5, 1e12])
            ax.text(0.98, 0.98, f"Hp={hp}cm",
                    transform=ax.transAxes, ha="right", va="top", fontsize=11,
                    bbox=dict(facecolor="none", alpha=0.8, edgecolor="none"))

        axes[0].set_ylabel("d$N$/dlog($D_m$) (cm$^{-3}$)")
        axes[1 if len(hp_values) > 1 else 0].legend(frameon=False)

        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=600, bbox_inches="tight")
        plt.close()
        return save_path
    
    def save_results(
        self,
        optimized_params: np.ndarray,
        loss_history: List[float],
        param_history: List[np.ndarray],
        save_path: str,
    ) -> str:
        """Save optimization results to CSV."""
        param_history_arr = np.array(param_history).squeeze()
        
        df = pd.DataFrame(
            param_history_arr,
            columns=[f'P{i+1}' for i in range(param_history_arr.shape[1])]
        )
        df['Loss'] = loss_history
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index_label='Epoch')
        
        return save_path
