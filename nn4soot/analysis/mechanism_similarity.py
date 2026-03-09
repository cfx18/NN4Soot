"""
Mechanism Similarity Analysis Across UF Sets

This module analyzes the similarity of sensitivity patterns across different
UF (unified factor) parameter sets, demonstrating mechanism invariance.

Key Features:
- Compute sensitivity similarity using L2 distance
- Pre-computed sensitivity map with caching support
- Bootstrap confidence intervals (GPU-accelerated)
- Top-k parameter identification with parameter labels
- Visualization matching publication style

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


@dataclass
class SimilarityResult:
    """Container for similarity analysis results."""
    
    l2_distance: float
    l2_similarity: float  # Scaled to [0, 1]
    uf_a: int
    uf_b: int
    hp: float
    bin_index: int


class MechanismSimilarityAnalyzer:
    """
    Analyzer for mechanism similarity across UF sets.

    Compares Active-Subspace sensitivity vectors computed from models trained
    on different UF parameter sets to assess mechanism invariance.

    The primary workflow uses a pre-computed sensitivity map so that models
    do not need to be held in memory during the comparison/plotting steps:

        1. ``compute_sensitivity_map``  – run all models once, store results
        2. ``compare_from_sens_map``    – pairwise L2 similarity → DataFrame
        3. ``plot_similarity_barh``     – compact horizontal-bar publication figure
        4. ``compute_top_k_from_sens_map`` – top-k params with bootstrap CI
        5. ``plot_top_k_params``        – labelled bar chart per UF set

    Parameters
    ----------
    models : Dict[int, nn.Module], optional
        Legacy API: mapping of UF set ID → model (single model per UF).
        For multi-hp analysis use ``compute_sensitivity_map`` directly.
    device : torch.device, optional
        Computation device.
    """

    def __init__(
        self,
        models: Optional[Dict[int, nn.Module]] = None,
        device: Optional[torch.device] = None,
    ):
        self.models = models or {}
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        for model in self.models.values():
            model.to(self.device)
            model.eval()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_active_subspace_sensitivity(
        self,
        model: nn.Module,
        inputs: np.ndarray,
        bin_index: int,
    ) -> np.ndarray:
        """Weighted sensitivity via Active-Subspace gradient-covariance eig.

        BIN index mapping: ``bin_index - 4`` → output column
        (BIN 5 → col 1, BIN 6 → col 2, …, BIN 20 → col 16).
        """
        x = torch.tensor(inputs, dtype=torch.float32, device=self.device)
        x = x.detach().clone().requires_grad_()

        preds = model(x)
        qoi = preds[:, bin_index - 4]

        grads = torch.autograd.grad(
            qoi, x,
            grad_outputs=torch.ones_like(qoi),
            create_graph=False,
            retain_graph=False,
        )[0]

        g = grads.detach().cpu().numpy()
        c = np.dot(g.T, g) / g.shape[0]

        eigvals, eigvecs = np.linalg.eigh(c)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        lam_sum = float(np.sum(eigvals))
        if lam_sum <= 0 or not np.isfinite(lam_sum):
            return np.ones(eigvecs.shape[0]) / eigvecs.shape[0]

        return np.sum((eigvecs ** 2) * (eigvals / lam_sum), axis=1)

    # ------------------------------------------------------------------
    # Metric helpers
    # ------------------------------------------------------------------

    def l2_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """L2 distance ||a - b||_2."""
        return float(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float)))

    def l2_similarity_01(self, a: np.ndarray, b: np.ndarray) -> float:
        """Scaled L2 similarity in [0, 1]: ``1 - ||a-b||_2 / sqrt(2)``."""
        return float(1.0 - self.l2_distance(a, b) / np.sqrt(2.0))

    # ------------------------------------------------------------------
    # Sensitivity map
    # ------------------------------------------------------------------

    def compute_sensitivity_map(
        self,
        models_by_uf_hp: Dict[Tuple[int, float], nn.Module],
        inputs_by_uf: Dict[int, np.ndarray],
        bin_indices: List[int],
    ) -> Dict[Tuple[int, float, int], np.ndarray]:
        """Compute Active-Subspace sensitivity for all (UF, Hp, BIN) triples.

        Parameters
        ----------
        models_by_uf_hp : Dict[(uf, hp), nn.Module]
            One model per (UF set, height position) combination.
        inputs_by_uf : Dict[int, np.ndarray]
            Parameter samples for each UF set; shared across hp values.
        bin_indices : List[int]
            BIN indices to analyse (e.g. ``list(range(5, 21))``).

        Returns
        -------
        Dict[(uf, hp, bin_index), np.ndarray]
            Sensitivity vector (input_dim,) for every triple.
        """
        sens_map: Dict[Tuple[int, float, int], np.ndarray] = {}
        for (uf, hp), model in models_by_uf_hp.items():
            model.to(self.device)
            model.eval()
            inputs = inputs_by_uf[uf]
            for b in bin_indices:
                sens_map[(uf, hp, b)] = self._compute_active_subspace_sensitivity(
                    model, inputs, b
                )
        return sens_map

    # ------------------------------------------------------------------
    # Comparison from pre-computed map
    # ------------------------------------------------------------------

    def compare_from_sens_map(
        self,
        sens_map: Dict[Tuple[int, float, int], np.ndarray],
        uf_pairs: List[Tuple[int, int]],
        hp_values: List[float],
        bin_indices: List[int],
    ) -> pd.DataFrame:
        """Compute pairwise L2 similarity from a pre-computed sensitivity map.

        Parameters
        ----------
        sens_map : Dict[(uf, hp, bin), np.ndarray]
            Output of :meth:`compute_sensitivity_map`.
        uf_pairs : List[(uf_a, uf_b)]
            Pairs to compare, e.g. ``[(2, 10), (5, 10), (2, 5)]``.
        hp_values, bin_indices : List
            Height positions and BIN indices to iterate over.

        Returns
        -------
        pd.DataFrame
            Columns: Hp, BIN, UF_A, UF_B, Pair, L2Dist, L2Sim01.
        """
        rows = []
        for hp in hp_values:
            for b in bin_indices:
                for uf_a, uf_b in uf_pairs:
                    s1 = sens_map[(uf_a, hp, b)]
                    s2 = sens_map[(uf_b, hp, b)]
                    rows.append({
                        "Hp": hp,
                        "BIN": b,
                        "UF_A": uf_a,
                        "UF_B": uf_b,
                        "Pair": f"UF{uf_a}-UF{uf_b}",
                        "L2Dist": self.l2_distance(s1, s2),
                        "L2Sim01": self.l2_similarity_01(s1, s2),
                    })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Plots – publication style matching source
    # ------------------------------------------------------------------

    def plot_similarity_barh(
        self,
        df: pd.DataFrame,
        uf_pairs: List[Tuple[int, int]],
        hp_values: List[float],
        pair_labels: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[float, float] = (2.8, 5.5),
        metric_col: str = "L2Sim01",
    ) -> str:
        """Compact horizontal-bar similarity figure (matches source style).

        Pairs on the y-axis; three hp groups with fixed offsets per pair row.
        Error bars show 2.5/97.5 percentiles across BIN values.

        Parameters
        ----------
        df : pd.DataFrame
            Output of :meth:`compare_from_sens_map`.
        uf_pairs : List[(uf_a, uf_b)]
            Same order as used during comparison.
        hp_values : List[float]
            Height positions (determines bar colours).
        pair_labels : List[str], optional
            Human-readable labels for y-axis ticks, e.g.
            ``["UF set1-3", "UF set2-3", "UF set1-2"]``.
        save_path : str, optional
            File path to save figure (dpi=600).
        figsize : tuple, optional
            Figure size in inches.
        metric_col : str
            Column in *df* used as the similarity metric.

        Returns
        -------
        str
            Path to saved figure, or empty string if not saved.
        """
        x_labels = [f"UF{a}-UF{b}" for a, b in uf_pairs]
        if pair_labels is None:
            pair_labels = x_labels

        y_base = [0.1, 0.5, 0.9]
        hp_colors = {0.50: "C0", 0.70: "C1", 1.00: "C2"}
        hp_label_map = {0.50: "Hp=0.5 cm", 0.70: "Hp=0.7 cm", 1.00: "Hp=1.0 cm"}
        offsets = {0.50: -0.11, 0.70: 0.0, 1.00: 0.11}
        bar_h = 0.1

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        for hp in hp_values:
            means, xerr_low, xerr_high, ypos = [], [], [], []
            for i, pair_key in enumerate(x_labels):
                vals = df[(df["Hp"] == hp) & (df["Pair"] == pair_key)][metric_col].values.astype(float)
                mu = float(np.nanmean(vals))
                lo = float(np.nanpercentile(vals, 2.5))
                hi = float(np.nanpercentile(vals, 97.5))
                means.append(mu)
                xerr_low.append(mu - lo)
                xerr_high.append(hi - mu)
                ypos.append(y_base[i] + offsets.get(hp, 0.0))

            ax.barh(ypos, means, height=bar_h,
                    color=hp_colors.get(hp, "C0"), alpha=0.85, linewidth=0.8,
                    label=hp_label_map.get(hp, f"Hp={hp}"))
            ax.errorbar(means, ypos, xerr=[xerr_low, xerr_high],
                        fmt="none", ecolor="black", elinewidth=1.2,
                        capsize=3, capthick=1.2, zorder=5)

        ax.set_yticks([0.2, 0.6, 1.0])
        ax.set_yticklabels(pair_labels, fontsize=12, rotation=90)
        ax.set_xlabel("Scaled L2 similarity", fontsize=12)
        ax.set_xlim([0, 1])
        ax.legend(frameon=True, ncol=1, loc="upper left", fontsize=11)

        fig.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=600, bbox_inches="tight")
        plt.close(fig)
        return save_path or ""

    # ------------------------------------------------------------------
    # Top-k parameters
    # ------------------------------------------------------------------

    def compute_top_k_from_sens_map(
        self,
        sens_map: Dict[Tuple[int, float, int], np.ndarray],
        uf_values: List[int],
        hp_values: List[float],
        bin_indices: List[int],
        top_k: int = 4,
        n_bootstrap: int = 1000,
        ci_q: Tuple[float, float] = (0.025, 0.975),
    ) -> pd.DataFrame:
        """Identify top-k parameters per UF set with GPU-accelerated bootstrap CI.

        Sensitivity vectors for all (hp × BIN) combinations are treated as
        samples; bootstrap resampling is performed on-device for speed.

        Parameters
        ----------
        sens_map : Dict[(uf, hp, bin), np.ndarray]
            Pre-computed sensitivity map.
        uf_values : List[int]
            UF set IDs to process.
        hp_values, bin_indices : List
            Must match the keys present in *sens_map*.
        top_k : int
            Number of top-ranked parameters to return.
        n_bootstrap : int
            Number of bootstrap resamples.
        ci_q : (float, float)
            Lower/upper quantiles for the confidence interval.

        Returns
        -------
        pd.DataFrame
            Columns: UF, Rank, ParamIndex_1based, MeanSensitivity, CI2p5, CI97p5.
            Sorted by (UF, Rank).
        """
        rows = []
        for uf in uf_values:
            vecs = [sens_map[(uf, hp, b)] for hp in hp_values for b in bin_indices]
            s_np = np.stack(vecs, axis=0).astype(np.float32)
            S = torch.from_numpy(s_np).to(self.device)
            K = S.shape[0]

            mu_vec = S.mean(dim=0)
            top_idx = torch.argsort(mu_vec, descending=True)[:top_k]

            idx = torch.randint(0, K, size=(n_bootstrap, K), device=self.device)
            boot_means = S[idx].mean(dim=1)  # (n_bootstrap, D)

            for rank, j in enumerate(top_idx.tolist(), start=1):
                boot_j = boot_means[:, j]
                lo, hi = torch.quantile(
                    boot_j,
                    torch.tensor(list(ci_q), device=self.device, dtype=boot_j.dtype),
                ).tolist()
                rows.append({
                    "UF": uf,
                    "Rank": rank,
                    "ParamIndex_1based": int(j + 1),
                    "MeanSensitivity": float(mu_vec[j].item()),
                    "CI2p5": float(lo),
                    "CI97p5": float(hi),
                })

        return pd.DataFrame(rows).sort_values(["UF", "Rank"]).reset_index(drop=True)

    def plot_top_k_params(
        self,
        df_topk: pd.DataFrame,
        uf_order: List[int],
        uf_labels: Optional[Dict[int, str]] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[float, float] = (2.8, 5.5),
        top_k: int = 4,
    ) -> str:
        """Horizontal-bar chart of top-k parameters per UF set.

        Bars are colour-coded by rank and annotated with parameter numbers.

        Parameters
        ----------
        df_topk : pd.DataFrame
            Output of :meth:`compute_top_k_from_sens_map`.
        uf_order : List[int]
            Display order of UF sets (bottom → top).
        uf_labels : Dict[int, str], optional
            Y-axis tick labels, e.g. ``{2: "UF set1", 5: "UF set2", 10: "UF set3"}``.
        save_path : str, optional
            File path to save figure (dpi=600).
        figsize : tuple, optional
            Figure size in inches.
        top_k : int
            Maximum rank to display.

        Returns
        -------
        str
            Path to saved figure, or empty string if not saved.
        """
        if uf_labels is None:
            uf_labels = {uf: f"UF={uf}" for uf in uf_order}

        y_base = [0.1, 0.5, 0.9]
        rank_offsets = {1: -0.125, 2: -0.045, 3: 0.035, 4: 0.115}
        rank_colors = {1: "C0", 2: "C1", 3: "C2", 4: "C3"}
        bar_h = 0.08

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        max_val = 0.0

        for i, uf in enumerate(uf_order):
            sub = df_topk[df_topk["UF"] == uf].sort_values("Rank")
            for row in sub.itertuples(index=False):
                r = int(row.Rank)
                if r > top_k:
                    continue
                y = y_base[i] + rank_offsets.get(r, 0.0)
                val = float(row.MeanSensitivity)
                lo = float(row.CI2p5)
                hi = float(row.CI97p5)
                max_val = max(max_val, val)

                ax.barh([y], [val], height=bar_h,
                        color=rank_colors.get(r, f"C{r}"), alpha=0.85, linewidth=0.8,
                        label=f"Top{r}" if i == 0 else None)
                ax.errorbar([val], [y],
                            xerr=[[max(0.0, val - lo)], [max(0.0, hi - val)]],
                            fmt="none", ecolor="black", elinewidth=1.0,
                            capsize=2.5, capthick=1.0, zorder=5)
                ax.text(val, y, f"      P{int(row.ParamIndex_1based)}",
                        va="center", ha="left", fontsize=10, color="black")

        ax.set_yticks(y_base)
        ax.set_yticklabels([uf_labels.get(u, str(u)) for u in uf_order],
                           fontsize=12, rotation=90)
        ax.set_xlabel(f"Top {top_k} mean global sensitivity", fontsize=12)
        ax.set_xlim(0, max(0.7, max_val * 1.15))
        # ax.legend(frameon=True, ncol=1, loc="upper right", fontsize=10)

        fig.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=600, bbox_inches="tight")
        plt.close(fig)
        return save_path or ""

    # ------------------------------------------------------------------
    # Legacy API (kept for backward compatibility)
    # ------------------------------------------------------------------

    def compare_pair(
        self,
        inputs_a: np.ndarray,
        inputs_b: np.ndarray,
        uf_a: int,
        uf_b: int,
        hp: float,
        bin_index: int,
    ) -> SimilarityResult:
        """Compare sensitivity between two UF sets (legacy, single-model-per-UF)."""
        sens_a = self._compute_active_subspace_sensitivity(
            self.models[uf_a], inputs_a, bin_index
        )
        sens_b = self._compute_active_subspace_sensitivity(
            self.models[uf_b], inputs_b, bin_index
        )
        return SimilarityResult(
            l2_distance=self.l2_distance(sens_a, sens_b),
            l2_similarity=self.l2_similarity_01(sens_a, sens_b),
            uf_a=uf_a, uf_b=uf_b, hp=hp, bin_index=bin_index,
        )

    def compare_all(
        self,
        inputs_dict: Dict[int, np.ndarray],
        hp_values: List[float],
        bin_indices: List[int],
        uf_pairs: Optional[List[Tuple[int, int]]] = None,
    ) -> List[SimilarityResult]:
        """Compare all UF pairs (legacy, single-model-per-UF).

        .. note::
            This method uses ``self.models[uf]`` regardless of *hp*, so the
            same model is used for every height position.  For full multi-hp
            analysis use :meth:`compute_sensitivity_map` instead.
        """
        uf_ids = sorted(inputs_dict.keys())
        if uf_pairs is None:
            from itertools import combinations
            uf_pairs = list(combinations(uf_ids, 2))

        results = []
        for hp in hp_values:
            for bin_idx in bin_indices:
                for uf_a, uf_b in uf_pairs:
                    results.append(self.compare_pair(
                        inputs_dict[uf_a], inputs_dict[uf_b],
                        uf_a, uf_b, hp, bin_idx,
                    ))
        return results

    def results_to_dataframe(self, results: List[SimilarityResult]) -> pd.DataFrame:
        """Convert list of :class:`SimilarityResult` to DataFrame."""
        return pd.DataFrame([{
            "Hp": r.hp, "BIN": r.bin_index,
            "UF_A": r.uf_a, "UF_B": r.uf_b,
            "Pair": f"UF{r.uf_a}-UF{r.uf_b}",
            "L2Dist": r.l2_distance, "L2Sim01": r.l2_similarity,
        } for r in results])

    def plot_similarity_summary(
        self,
        results: List[SimilarityResult],
        save_path: Optional[str] = None,
    ) -> str:
        """Simple grouped-bar summary plot (legacy style)."""
        df = self.results_to_dataframe(results)
        uf_pairs_uniq = df["Pair"].unique()
        hp_values = sorted(df["Hp"].unique())
        agg = df.groupby(["Pair", "Hp"])["L2Sim01"].agg(["mean", "std"]).reset_index()

        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(len(uf_pairs_uniq))
        width = 0.25
        colors = plt.cm.tab10.colors[:len(hp_values)]

        for i, hp in enumerate(hp_values):
            sub = agg[agg["Hp"] == hp]
            means = [sub[sub["Pair"] == p]["mean"].values[0] for p in uf_pairs_uniq]
            stds = [sub[sub["Pair"] == p]["std"].values[0] for p in uf_pairs_uniq]
            ax.bar(x + i * width, means, width, yerr=stds,
                   label=f"Hp={hp:.2f}", color=colors[i], alpha=0.85, capsize=3)

        ax.set_xticks(x + width * (len(hp_values) - 1) / 2)
        ax.set_xticklabels(uf_pairs_uniq)
        ax.set_ylabel("Scaled L2 Similarity")
        ax.set_xlabel("UF Pair")
        ax.set_ylim([0, 1.05])
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=600, bbox_inches="tight")
        plt.close()
        return save_path or ""

    def save_results(self, results: List[SimilarityResult], save_path: str) -> str:
        """Save similarity results to CSV."""
        df = self.results_to_dataframe(results)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        return save_path
