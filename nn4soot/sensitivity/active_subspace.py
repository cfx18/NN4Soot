"""
Active Subspace Sensitivity Analysis for Soot Kinetic Models

This module implements Active Subspace analysis for computing global sensitivity
indices from neural network surrogate models. The method uses gradient covariance
eigendecomposition to identify influential parameters.

Key Features:
- Gradient-based sensitivity computation
- Bootstrap confidence intervals
- Weighted sensitivity from eigenvalue decomposition

Reference:
    Paul G. Constantine. Active Subspaces: Emerging Ideas for Dimension
    Reduction in Parameter Studies. SIAM, 2015.

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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex


@dataclass
class ASResult:
    """Container for Active Subspace analysis results."""
    
    sensitivity: np.ndarray          # Weighted sensitivity (input_dim,)
    eigenvalues: np.ndarray          # Eigenvalues (input_dim,)
    eigenvectors: np.ndarray         # Eigenvectors (input_dim, input_dim)
    sensitivity_bootstrap_mean: Optional[np.ndarray] = None
    eigenvalues_bootstrap_mean: Optional[np.ndarray] = None
    active_subspace_vector_bootstrap_mean: Optional[np.ndarray] = None
    active_subspace_vector_ci_low: Optional[np.ndarray] = None
    active_subspace_vector_ci_high: Optional[np.ndarray] = None
    
    # Bootstrap confidence intervals
    sensitivity_ci_low: Optional[np.ndarray] = None
    sensitivity_ci_high: Optional[np.ndarray] = None
    eigenvalues_ci_low: Optional[np.ndarray] = None
    eigenvalues_ci_high: Optional[np.ndarray] = None
    
    # Active subspace components
    active_subspace_vector: Optional[np.ndarray] = None  # First eigenvector
    gradient_covariance: Optional[np.ndarray] = None     # C matrix
    projection_values: Optional[np.ndarray] = None       # QoI values for projection plot
    window_lo: Optional[int] = None
    window_hi: Optional[int] = None


class ActiveSubspaceAnalyzer:
    """
    Active Subspace sensitivity analyzer for neural network surrogate models.
    
    This class computes global sensitivity indices using the Active Subspace
    method, which analyzes the gradient covariance matrix to identify 
    influential parameters.
    
    Parameters
    ----------
    model : nn.Module
        Trained neural network model
    device : torch.device, optional
        Device for computation
    
    Examples
    --------
    >>> from nn4soot import SootMLP, ActiveSubspaceAnalyzer
    >>> model = SootMLP.from_pretrained("model.pth")
    >>> analyzer = ActiveSubspaceAnalyzer(model)
    >>> result = analyzer.analyze(inputs, bin_index=9)
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
    
    def compute_gradients(
        self,
        inputs: Union[np.ndarray, torch.Tensor],
        bin_index: int,
    ) -> np.ndarray:
        """
        Compute gradients of output w.r.t. inputs for a specific BIN.
        
        Parameters
        ----------
        inputs : array-like
            Input parameters (N, input_dim)
        bin_index : int
            BIN index (5-20, maps to output column bin_index-4)
        
        Returns
        -------
        np.ndarray
            Gradient array of shape (N, input_dim)
        """
        if isinstance(inputs, np.ndarray):
            x = torch.tensor(inputs, dtype=torch.float32, device=self.device)
        else:
            x = inputs.to(self.device)
        
        if not x.requires_grad:
            x = x.detach().clone().requires_grad_()
        
        # Forward pass
        preds = self.model(x)  # (N, output_dim)
        
        # Extract QoI (quantity of interest)
        # minus 4 since the Neural Networks outputs begin from BIN4
        qoi = preds[:, bin_index - 4]
        
        # Compute gradients
        gradients = torch.autograd.grad(
            qoi,
            x,
            grad_outputs=torch.ones_like(qoi),
            create_graph=False,
            retain_graph=False,
        )[0]
        
        return gradients.detach().cpu().numpy()
    
    def compute_gradient_covariance(
        self,
        gradients: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute gradient covariance matrix and its eigendecomposition.
        
        Parameters
        ----------
        gradients : np.ndarray
            Gradient array (N, input_dim)
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Covariance matrix, eigenvalues (descending), eigenvectors
        """
        G = np.asarray(gradients, dtype=np.float64)
        N = float(G.shape[0])
        
        # Gradient covariance (energy matrix)
        C = (G.T @ G) / N
        
        # Eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(C)  # Ascending order
        
        # Sort descending
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        return C, eigvals, eigvecs
    
    def compute_weighted_sensitivity(
        self,
        eigenvalues: np.ndarray,
        eigenvectors: np.ndarray,
    ) -> np.ndarray:
        """
        Compute weighted sensitivity from eigenvalue decomposition.
        
        The weighted sensitivity for parameter j is:
            s_j = sum_i (w_{j,i}^2 * lambda_i / sum_k lambda_k)
        
        This combines all eigenvalue contributions weighted by their
        relative importance.
        
        Parameters
        ----------
        eigenvalues : np.ndarray
            Eigenvalues in descending order
        eigenvectors : np.ndarray
            Corresponding eigenvectors as columns
        
        Returns
        -------
        np.ndarray
            Weighted sensitivity for each parameter
        """
        lam = np.asarray(eigenvalues, dtype=np.float64)
        W = np.asarray(eigenvectors, dtype=np.float64)
        
        lam_sum = float(lam.sum())
        if lam_sum <= 0 or not np.isfinite(lam_sum):
            return np.ones(W.shape[0]) / W.shape[0]
        
        sensitivity = np.sum((W ** 2) * (lam / lam_sum)[None, :], axis=1)
        return sensitivity
    
    def analyze_with_bootstrap(
        self,
        inputs: np.ndarray,
        bin_index: int,
        n_bootstrap: int = 1000,
        ci_level: float = 0.95,
        seed: int = 42,
    ) -> ASResult:
        """
        Perform Active Subspace analysis with bootstrap confidence intervals.
        
        Parameters
        ----------
        inputs : np.ndarray
            Input parameters (N, input_dim)
        bin_index : int
            BIN index to analyze
        n_bootstrap : int
            Number of bootstrap samples
        ci_level : float
            Confidence level (e.g., 0.95 for 95% CI)
        seed : int
            Random seed for reproducibility
        
        Returns
        -------
        ASResult
            Analysis results with sensitivity and confidence intervals
        """
        rng = np.random.default_rng(seed)
        N, D = inputs.shape
        
        # Compute gradients
        gradients = self.compute_gradients(inputs, bin_index)
        
        # Full analysis
        C, eigvals, eigvecs = self.compute_gradient_covariance(gradients)
        sensitivity = self.compute_weighted_sensitivity(eigvals, eigvecs)
        
        # Bootstrap
        boot_sens = np.zeros((n_bootstrap, D))
        boot_eigvals = np.zeros((n_bootstrap, D))
        boot_w1 = np.zeros((n_bootstrap, D))
        
        w1_ref = eigvecs[:, 0].copy()
        
        for b in range(n_bootstrap):
            idx = rng.integers(0, N, size=N)
            _, eigvals_b, eigvecs_b = self.compute_gradient_covariance(gradients[idx])
            
            boot_sens[b] = self.compute_weighted_sensitivity(eigvals_b, eigvecs_b)
            boot_eigvals[b] = eigvals_b
            
            w1_b = eigvecs_b[:, 0].copy()
            if np.dot(w1_b, w1_ref) < 0:
                w1_b *= -1
            boot_w1[b] = w1_b
        
        # Compute confidence intervals
        alpha = (1.0 - ci_level) / 2.0
        low_q = 100.0 * alpha
        high_q = 100.0 * (1.0 - alpha)
        
        sens_ci_low = np.percentile(boot_sens, low_q, axis=0)
        sens_ci_high = np.percentile(boot_sens, high_q, axis=0)
        eig_ci_low = np.percentile(boot_eigvals, low_q, axis=0)
        eig_ci_high = np.percentile(boot_eigvals, high_q, axis=0)
        
        return ASResult(
            sensitivity=sensitivity,
            eigenvalues=eigvals,
            eigenvectors=eigvecs,
            sensitivity_bootstrap_mean=boot_sens.mean(axis=0),
            eigenvalues_bootstrap_mean=boot_eigvals.mean(axis=0),
            active_subspace_vector_bootstrap_mean=boot_w1.mean(axis=0),
            active_subspace_vector_ci_low=np.percentile(boot_w1, low_q, axis=0),
            active_subspace_vector_ci_high=np.percentile(boot_w1, high_q, axis=0),
            sensitivity_ci_low=sens_ci_low,
            sensitivity_ci_high=sens_ci_high,
            eigenvalues_ci_low=eig_ci_low,
            eigenvalues_ci_high=eig_ci_high,
            active_subspace_vector=eigvecs[:, 0],
            gradient_covariance=C,
        )
    
    def analyze_multiple_bins(
        self,
        inputs: np.ndarray,
        bin_indices: List[int],
        n_bootstrap: int = 1000,
        ci_level: float = 0.95,
        seed: int = 42,
    ) -> Dict[int, ASResult]:
        """
        Analyze sensitivity for multiple BIN indices.
        
        Parameters
        ----------
        inputs : np.ndarray
            Input parameters
        bin_indices : List[int]
            List of BIN indices to analyze
        n_bootstrap : int
            Number of bootstrap samples
        ci_level : float
            Confidence level
        seed : int
            Random seed
        
        Returns
        -------
        Dict[int, ASResult]
            Results indexed by BIN index
        """
        results = {}
        for bin_idx in bin_indices:
            results[bin_idx] = self.analyze_with_bootstrap(
                inputs, bin_idx, n_bootstrap, ci_level, seed
            )
        return results

    @staticmethod
    def infer_window_bins(
        d_bins_nm: np.ndarray,
        dmin: float,
        dmax: float,
    ) -> Tuple[int, int]:
        """Infer bin window from diameter bounds."""
        mask = (d_bins_nm >= dmin) & (d_bins_nm <= dmax)
        idx = np.where(mask)[0]
        if idx.size < 2:
            raise ValueError(f"Window [{dmin}, {dmax}] nm covers too few bins: {idx.size}")
        return int(idx[0]), int(idx[-1])

    def compute_soft_valley_index_gradients(
        self,
        inputs: np.ndarray,
        window_lo: int,
        window_hi: int,
        beta_softmin: float = 1.0,
        batch_size: int = 256,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute soft valley index and its gradients with respect to inputs.
        """
        X = np.asarray(inputs, dtype=np.float32)
        N, D = X.shape
        M = int(window_hi - window_lo + 1)
        idx_local = torch.arange(M, dtype=torch.float32, device=self.device).view(1, M)

        qoi_all = np.zeros((N,), dtype=np.float32)
        grads_all = np.zeros((N, D), dtype=np.float32)

        pos = 0
        self.model.eval()
        for i in range(0, N, int(batch_size)):
            xb = torch.tensor(
                X[i : i + int(batch_size)],
                dtype=torch.float32,
                device=self.device,
                requires_grad=True,
            )
            yb = self.model(xb)
            yw = yb[:, int(window_lo) : int(window_hi) + 1]

            z = -float(beta_softmin) * yw
            z = z - z.max(dim=1, keepdim=True).values
            w = torch.softmax(z, dim=1)
            qoi = (w * idx_local).sum(dim=1) + float(window_lo)

            loss = qoi.sum()
            loss.backward()

            g = xb.grad.detach().cpu().numpy().astype(np.float32)
            q = qoi.detach().cpu().numpy().astype(np.float32)

            bsz = g.shape[0]
            qoi_all[pos : pos + bsz] = q
            grads_all[pos : pos + bsz] = g
            pos += bsz

        return qoi_all, grads_all

    def analyze_soft_valley_index_with_bootstrap(
        self,
        inputs: np.ndarray,
        d_bins_nm: np.ndarray,
        dmin: float = 2.248781333333,
        dmax: float = 13.04917,
        beta_softmin: float = 1.0,
        n_bootstrap: int = 1000,
        ci_level: float = 0.95,
        seed: int = 123,
        batch_size: int = 256,
    ) -> ASResult:
        """
        Active Subspace analysis for the differentiable soft valley index.
        """
        rng = np.random.default_rng(seed)
        lo, hi = self.infer_window_bins(np.asarray(d_bins_nm, dtype=float), dmin, dmax)

        qoi_values, gradients = self.compute_soft_valley_index_gradients(
            inputs=inputs,
            window_lo=lo,
            window_hi=hi,
            beta_softmin=beta_softmin,
            batch_size=batch_size,
        )

        N, D = gradients.shape
        C, eigvals, eigvecs = self.compute_gradient_covariance(gradients)
        sensitivity = self.compute_weighted_sensitivity(eigvals, eigvecs)

        boot_sens = np.zeros((n_bootstrap, D))
        boot_eigvals = np.zeros((n_bootstrap, D))
        boot_w1 = np.zeros((n_bootstrap, D))
        w1_ref = eigvecs[:, 0].copy()

        for b in range(n_bootstrap):
            idx = rng.integers(0, N, size=N)
            _, eigvals_b, eigvecs_b = self.compute_gradient_covariance(gradients[idx])
            boot_sens[b] = self.compute_weighted_sensitivity(eigvals_b, eigvecs_b)
            boot_eigvals[b] = eigvals_b
            w1_b = eigvecs_b[:, 0].copy()
            if np.dot(w1_b, w1_ref) < 0:
                w1_b *= -1
            boot_w1[b] = w1_b

        alpha = (1.0 - ci_level) / 2.0
        low_q = 100.0 * alpha
        high_q = 100.0 * (1.0 - alpha)

        return ASResult(
            sensitivity=sensitivity,
            eigenvalues=eigvals,
            eigenvectors=eigvecs,
            sensitivity_bootstrap_mean=boot_sens.mean(axis=0),
            eigenvalues_bootstrap_mean=boot_eigvals.mean(axis=0),
            active_subspace_vector_bootstrap_mean=boot_w1.mean(axis=0),
            active_subspace_vector_ci_low=np.percentile(boot_w1, low_q, axis=0),
            active_subspace_vector_ci_high=np.percentile(boot_w1, high_q, axis=0),
            sensitivity_ci_low=np.percentile(boot_sens, low_q, axis=0),
            sensitivity_ci_high=np.percentile(boot_sens, high_q, axis=0),
            eigenvalues_ci_low=np.percentile(boot_eigvals, low_q, axis=0),
            eigenvalues_ci_high=np.percentile(boot_eigvals, high_q, axis=0),
            active_subspace_vector=eigvecs[:, 0],
            gradient_covariance=C,
            projection_values=qoi_values,
            window_lo=lo,
            window_hi=hi,
        )
    
    def plot_sensitivity(
        self,
        result: ASResult,
        param_names: Optional[List[str]] = None,
        title: str = "Active Subspace Sensitivity",
        save_path: Optional[str] = None,
    ) -> str:
        """
        Plot sensitivity bar chart with confidence intervals.
        
        Parameters
        ----------
        result : ASResult
            Analysis result
        param_names : List[str], optional
            Parameter names (default: P1, P2, ...)
        title : str
            Plot title
        save_path : str, optional
            Path to save figure
        
        Returns
        -------
        str
            Path to saved figure
        """
        D = len(result.sensitivity)
        
        if param_names is None:
            param_names = [f"P{i+1}" for i in range(D)]
        
        x = np.arange(D)
        y = (
            result.sensitivity_bootstrap_mean
            if result.sensitivity_bootstrap_mean is not None
            else result.sensitivity
        )
        
        if result.sensitivity_ci_low is not None:
            yerr = np.vstack([
                y - result.sensitivity_ci_low,
                result.sensitivity_ci_high - y,
            ])
            yerr = np.clip(np.nan_to_num(yerr, nan=0.0), 0.0, None)
        else:
            yerr = None
        
        fig, ax = plt.subplots(figsize=(6, 2.6))
        ax.bar(x, y, yerr=yerr, capsize=3, color="#34495e", alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(param_names)
        ax.set_ylabel("AS sensitivity")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
        return save_path or ""
    
    def plot_eigenvalue_spectrum(
        self,
        result: ASResult,
        title: str = "Eigenvalue Spectrum",
        save_path: Optional[str] = None,
    ) -> str:
        """
        Plot eigenvalue spectrum with confidence intervals.
        
        Parameters
        ----------
        result : ASResult
            Analysis result
        title : str
            Plot title
        save_path : str, optional
            Path to save figure
        
        Returns
        -------
        str
            Path to saved figure
        """
        D = len(result.eigenvalues)
        idx = np.arange(1, D + 1)
        
        y = (
            result.eigenvalues_bootstrap_mean
            if result.eigenvalues_bootstrap_mean is not None
            else result.eigenvalues
        )
        
        fig, ax = plt.subplots(figsize=(4.8, 2.8))
        ax.plot(idx, y, "ko-", ms=4)
        
        if result.eigenvalues_ci_low is not None:
            ax.fill_between(
                idx, result.eigenvalues_ci_low, result.eigenvalues_ci_high,
                color="grey", alpha=0.4
            )
        
        ax.set_yscale("log")
        ax.set_xlabel("Eigenvalue Index")
        ax.set_ylabel("Eigenvalue")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
        return save_path or ""

    def plot_original_style_bin_figure(
        self,
        result: ASResult,
        inputs: np.ndarray,
        outputs: np.ndarray,
        bin_index: int,
        hp: float,
        save_path: Optional[str] = None,
        keep_indices: Optional[np.ndarray] = None,
    ) -> str:
        """
        Plot the original-style 2-row figure used in the legacy script:
        eigenvalue spectrum + active-variable projection scatter.
        """
        if keep_indices is None:
            keep_indices = np.arange(len(result.sensitivity))

        eig_m = (
            result.eigenvalues_bootstrap_mean
            if result.eigenvalues_bootstrap_mean is not None
            else result.eigenvalues
        )
        eig_l = result.eigenvalues_ci_low
        eig_u = result.eigenvalues_ci_high
        w1_m = (
            result.active_subspace_vector_bootstrap_mean
            if result.active_subspace_vector_bootstrap_mean is not None
            else result.active_subspace_vector
        )

        fig, axes = plt.subplots(2, 1, figsize=(4, 5))

        eig_m_filtered = eig_m[keep_indices]
        eig_l_filtered = eig_l[keep_indices] if eig_l is not None else eig_m_filtered
        eig_u_filtered = eig_u[keep_indices] if eig_u is not None else eig_m_filtered
        idx = np.arange(1, len(eig_m_filtered) + 1)

        axes[0].plot(idx, eig_m_filtered, "ko-")
        axes[0].fill_between(idx, eig_l_filtered, eig_u_filtered, color="grey", alpha=0.5)
        axes[0].set_yscale("log")
        axes[0].set_xlabel("Eigenvalue idx")
        axes[0].set_ylabel("Eigenvalue")
        axes[0].set_title(f"Hp={hp:.2f} cm  BIN{bin_index}")

        proj = np.asarray(inputs) @ np.asarray(w1_m)
        qoi_idx = bin_index - 5
        axes[1].scatter(proj, np.asarray(outputs)[:, qoi_idx], c="k", s=4)
        axes[1].set_xlabel(r"$w_1^T x$")
        axes[1].set_ylabel(r"log(d$N$/dlog$D_m$)")

        fig.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=600, bbox_inches="tight")
        plt.close(fig)
        return save_path or ""

    def plot_soft_valley_index_bundle(
        self,
        result: ASResult,
        inputs: np.ndarray,
        hp: float,
        save_dir: str,
        prefix: str = "AS_valley_index_soft_hp0.70",
    ) -> Dict[str, str]:
        """
        Save the original trio of plots for soft valley index AS analysis.
        """
        save_dir = str(save_dir)
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        paths = {}
        paths["sensitivity"] = self.plot_sensitivity(
            result,
            title=f"Active Subspace sensitivity of soft valley index (Hp={hp:.2f}, beta=1)",
            save_path=os.path.join(save_dir, f"{prefix}.png"),
        )
        paths["eigs"] = self.plot_eigenvalue_spectrum(
            result,
            title="AS eigen spectrum (bootstrap CI)",
            save_path=os.path.join(save_dir, f"{prefix}_eigs.png"),
        )

        w1 = (
            result.active_subspace_vector_bootstrap_mean
            if result.active_subspace_vector_bootstrap_mean is not None
            else result.active_subspace_vector
        )
        proj = np.asarray(inputs) @ np.asarray(w1)
        qoi = result.projection_values

        fig, ax = plt.subplots(1, 1, figsize=(4.8, 3.2))
        ax.scatter(proj, qoi, c="k", s=8, alpha=0.7)
        ax.set_xlabel(r"$w_1^\top x$")
        ax.set_ylabel(r"$k_{\mathrm{soft}}(x)$")
        ax.set_title("1D active variable projection")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        proj_path = os.path.join(save_dir, f"{prefix}_proj.png")
        fig.savefig(proj_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        paths["proj"] = proj_path
        return paths

    def plot_original_style_hp_summary(
        self,
        results_by_bin: Dict[int, ASResult],
        hp: float,
        save_path: Optional[str] = None,
        keep_indices: Optional[np.ndarray] = None,
    ) -> str:
        """
        Plot the original-style per-height sensitivity summary with all BIN bars.
        """
        if not results_by_bin:
            return save_path or ""

        first_result = next(iter(results_by_bin.values()))
        if keep_indices is None:
            keep_indices = np.arange(len(first_result.sensitivity))

        fig, ax = plt.subplots(figsize=(8, 2))
        offset, bar_w = 0.06, 0.055
        colors = plt.cm.viridis(np.linspace(0, 1, len(results_by_bin)))

        for j, (bin_index, result) in enumerate(results_by_bin.items()):
            sens_m = (
                result.sensitivity_bootstrap_mean
                if result.sensitivity_bootstrap_mean is not None
                else result.sensitivity
            )
            sens_l = result.sensitivity_ci_low
            sens_u = result.sensitivity_ci_high

            sens_m_filtered = sens_m[keep_indices]
            sens_l_filtered = sens_l[keep_indices] if sens_l is not None else sens_m_filtered
            sens_u_filtered = sens_u[keep_indices] if sens_u is not None else sens_m_filtered
            x = np.arange(len(sens_m_filtered)) + j * offset + 0.1
            color_hex = rgb2hex(colors[j])
            ax.bar(
                x,
                sens_m_filtered,
                yerr=[sens_m_filtered - sens_l_filtered, sens_u_filtered - sens_m_filtered],
                width=bar_w,
                color=color_hex,
                alpha=0.8,
                capsize=1,
                label=f"BIN{bin_index}",
                error_kw={"elinewidth": 1, "ecolor": "black", "capthick": 1},
            )

        ax.set_title(f"Hp={hp:.2f}")
        ax.set_ylabel("Sensitivity")
        ax.set_xlabel("Parameter Index")
        ax.set_xlim(0, len(keep_indices))
        fig.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=600, bbox_inches="tight")
        plt.close(fig)
        return save_path or ""

    @staticmethod
    def plot_original_style_cross_hp_comparison(
        results_by_hp: Dict[float, Dict[int, ASResult]],
        save_dir: str,
        selected_bins: Optional[List[int]] = None,
    ) -> List[str]:
        """
        Plot the legacy-style cross-height comparison figures by BIN.
        """
        if not results_by_hp:
            return []

        hp_values = sorted(results_by_hp.keys())
        common_bins = sorted(set.intersection(*[set(v.keys()) for v in results_by_hp.values()]))
        if not common_bins:
            return []

        out_dir = Path(save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        hp_colors = {0.50: "#3498db", 0.70: "#e74c3c", 1.00: "#2ecc71"}
        hp_labels = {0.50: "Hp=0.50 cm", 0.70: "Hp=0.70 cm", 1.00: "Hp=1.00 cm"}
        saved_paths = []

        for bin_index in common_bins:
            fig, ax = plt.subplots(figsize=(10, 5))
            n_params = len(next(iter(results_by_hp[hp_values[0]].values())).sensitivity)
            n_hps = len(hp_values)
            bar_width = 0.25
            x_positions = np.arange(n_params)

            for i, hp in enumerate(hp_values):
                result = results_by_hp[hp][bin_index]
                sens_m = (
                    result.sensitivity_bootstrap_mean
                    if result.sensitivity_bootstrap_mean is not None
                    else result.sensitivity
                )
                sens_l = result.sensitivity_ci_low
                sens_u = result.sensitivity_ci_high
                offset = (i - n_hps / 2 + 0.5) * bar_width
                ax.bar(
                    x_positions + offset,
                    sens_m,
                    bar_width,
                    yerr=[sens_m - sens_l, sens_u - sens_m],
                    color=hp_colors.get(hp, f"C{i}"),
                    alpha=0.8,
                    capsize=3,
                    label=hp_labels.get(hp, f"Hp={hp:.2f} cm"),
                    error_kw={"linewidth": 1, "elinewidth": 1},
                )

            ax.set_xlabel("Parameter Index", fontsize=12, fontweight="bold")
            ax.set_ylabel("Sensitivity", fontsize=12, fontweight="bold")
            ax.set_title(f"Sensitivity Analysis for BIN {bin_index}", fontsize=14, fontweight="bold")
            ax.set_xticks(x_positions)
            ax.set_xticklabels([f"P{i+1}" for i in range(n_params)])
            ax.legend(fontsize=10, loc="best", framealpha=0.9)
            ax.grid(True, alpha=0.3, axis="y")
            fig.tight_layout()
            path = out_dir / f"Sensitivity_BIN={bin_index}.png"
            fig.savefig(path, dpi=600, bbox_inches="tight")
            plt.close(fig)
            saved_paths.append(str(path))

        if selected_bins is None:
            selected_bins = [7, 12, 17]
        selected_bins = [b for b in selected_bins if b in common_bins][:3]
        if len(selected_bins) == 3:
            fig_combined, axes = plt.subplots(3, 1, figsize=(10, 12))
            n_params = len(next(iter(results_by_hp[hp_values[0]].values())).sensitivity)
            n_hps = len(hp_values)
            bar_width = 0.25
            x_positions = np.arange(n_params)

            for row_idx, bin_index in enumerate(selected_bins):
                ax = axes[row_idx]
                for i, hp in enumerate(hp_values):
                    result = results_by_hp[hp][bin_index]
                    sens_m = (
                        result.sensitivity_bootstrap_mean
                        if result.sensitivity_bootstrap_mean is not None
                        else result.sensitivity
                    )
                    sens_l = result.sensitivity_ci_low
                    sens_u = result.sensitivity_ci_high
                    offset = (i - n_hps / 2 + 0.5) * bar_width
                    ax.bar(
                        x_positions + offset,
                        sens_m,
                        bar_width,
                        yerr=[sens_m - sens_l, sens_u - sens_m],
                        color=hp_colors.get(hp, f"C{i}"),
                        alpha=0.8,
                        capsize=3,
                        label=hp_labels.get(hp, f"Hp={hp:.2f} cm"),
                        error_kw={"linewidth": 1, "elinewidth": 1},
                    )

                ax.set_ylabel("Sensitivity", fontsize=11, fontweight="bold")
                ax.set_title(f"BIN {bin_index}", fontsize=12, fontweight="bold", loc="left")
                ax.set_xticks(x_positions)
                ax.set_xticklabels([f"P{i+1}" for i in range(n_params)])
                ax.legend(fontsize=9, loc="upper right", framealpha=0.9, ncol=3)
                ax.grid(True, alpha=0.3, axis="y")
                if row_idx < len(selected_bins) - 1:
                    ax.set_xticklabels([])
                else:
                    ax.set_xlabel("Parameter Index", fontsize=12, fontweight="bold")

            fig_combined.suptitle("Sensitivity Comparison Across Heights", fontsize=14, fontweight="bold", y=0.995)
            fig_combined.tight_layout(rect=[0, 0, 1, 0.99])
            combined_path = out_dir / "Sensitivity_Combined_BINs.png"
            fig_combined.savefig(combined_path, dpi=600, bbox_inches="tight")
            plt.close(fig_combined)
            saved_paths.append(str(combined_path))

        return saved_paths
    
    def save_results(
        self,
        result: ASResult,
        save_dir: str,
        prefix: str = "AS",
        param_names: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """
        Save analysis results to files.
        
        Parameters
        ----------
        result : ASResult
            Analysis result
        save_dir : str
            Directory to save
        prefix : str
            Filename prefix
        param_names : List[str], optional
            Parameter names
        
        Returns
        -------
        Dict[str, str]
            Paths to saved files
        """
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        D = len(result.sensitivity)
        
        if param_names is None:
            param_names = [f"P{i+1}" for i in range(D)]
        
        paths = {}
        
        # Sensitivity CSV
        sens_df = pd.DataFrame({
            "param": param_names,
            "sensitivity": result.sensitivity,
            "boot_mean": result.sensitivity_bootstrap_mean,
            "ci_low": result.sensitivity_ci_low,
            "ci_high": result.sensitivity_ci_high,
        })
        sens_path = os.path.join(save_dir, f"{prefix}_sensitivity.csv")
        sens_df.to_csv(sens_path, index=False)
        paths["sensitivity"] = sens_path
        
        # Eigenvalues CSV
        eig_df = pd.DataFrame({
            "eig_idx": np.arange(1, D + 1),
            "eigenvalue": result.eigenvalues,
            "boot_mean": result.eigenvalues_bootstrap_mean,
            "ci_low": result.eigenvalues_ci_low,
            "ci_high": result.eigenvalues_ci_high,
        })
        eig_path = os.path.join(save_dir, f"{prefix}_eigenvalues.csv")
        eig_df.to_csv(eig_path, index=False)
        paths["eigenvalues"] = eig_path
        
        # First active-subspace direction
        if result.active_subspace_vector is not None:
            w1_df = pd.DataFrame({
                "param": param_names,
                "w1": result.active_subspace_vector,
                "w1_mean": result.active_subspace_vector_bootstrap_mean,
                "ci_low": result.active_subspace_vector_ci_low,
                "ci_high": result.active_subspace_vector_ci_high,
            })
            w1_path = os.path.join(save_dir, f"{prefix}_w1.csv")
            w1_df.to_csv(w1_path, index=False)
            paths["w1"] = w1_path
        
        # Covariance matrix
        if result.gradient_covariance is not None:
            C_path = os.path.join(save_dir, f"{prefix}_covariance.txt")
            np.savetxt(C_path, result.gradient_covariance, fmt="%.8e")
            paths["covariance"] = C_path
        
        return paths
