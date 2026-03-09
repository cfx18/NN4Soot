"""
Model Comparison Analysis for Soot Surrogate Models

This module provides comprehensive comparison between neural network models
and baseline regression models (Linear, Polynomial).

Key Features:
- MSE and R2 comparison across UF sets
- Overfitting analysis (train vs test gap)
- Visualization of model performance

Author: Feixue Cai
"""

import os
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class ModelComparisonResult:
    """Container for model comparison results."""
    
    model_name: str
    uf: int
    hp: float
    
    train_mse: float
    test_mse: float
    train_r2: float
    test_r2: float
    
    @property
    def mse_gap(self) -> float:
        """MSE gap (test - train)."""
        return self.test_mse - self.train_mse
    
    @property
    def r2_gap(self) -> float:
        """R2 gap (train - test), indicates overfitting."""
        return self.train_r2 - self.test_r2
    
    @property
    def mse_ratio(self) -> float:
        """MSE ratio (test / train)."""
        return self.test_mse / self.train_mse if self.train_mse > 0 else float('inf')


class ModelComparator:
    """
    Comparator for neural network vs baseline models.
    
    This class provides comprehensive model comparison including
    MSE, R2 metrics and overfitting analysis.
    
    Parameters
    ----------
    device : torch.device, optional
        Device for computation
    
    Examples
    --------
    >>> from nn4soot import SootMLP, ModelComparator
    >>> model = SootMLP.from_pretrained("model.pth")
    >>> comparator = ModelComparator()
    >>> results = comparator.compare_single_case(model, inputs, outputs, uf=10, hp=0.70)
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device
    
    def evaluate_model(
        self,
        model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str,
        uf: int,
        hp: float,
    ) -> ModelComparisonResult:
        """Evaluate a single model on train and test sets."""
        
        if hasattr(model, 'predict'):
            # Sklearn-style models
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
        else:
            # PyTorch models
            model.eval()
            with torch.no_grad():
                X_train_t = torch.tensor(X_train, dtype=torch.float32, device=self.device)
                X_test_t = torch.tensor(X_test, dtype=torch.float32, device=self.device)
                y_train_pred = model(X_train_t).cpu().numpy()
                y_test_pred = model(X_test_t).cpu().numpy()
        
        # Compute mean across bins
        y_train_mean = y_train.mean(axis=1)
        y_test_mean = y_test.mean(axis=1)
        y_train_pred_mean = y_train_pred.mean(axis=1)
        y_test_pred_mean = y_test_pred.mean(axis=1)
        
        return ModelComparisonResult(
            model_name=model_name,
            uf=uf,
            hp=hp,
            train_mse=mean_squared_error(y_train_mean, y_train_pred_mean),
            test_mse=mean_squared_error(y_test_mean, y_test_pred_mean),
            train_r2=r2_score(y_train_mean, y_train_pred_mean),
            test_r2=r2_score(y_test_mean, y_test_pred_mean),
        )
    
    def compare_single_case(
        self,
        nn_model: nn.Module,
        inputs: np.ndarray,
        outputs: np.ndarray,
        uf: int,
        hp: float,
        test_size: float = 0.2,
        random_seed: int = 20,
        include_baselines: bool = True,
    ) -> List[ModelComparisonResult]:
        """Compare neural network with baselines for a single case."""
        
        X_train, X_test, y_train, y_test = train_test_split(
            inputs, outputs,
            test_size=test_size,
            random_state=random_seed,
        )
        
        results = []
        
        # Neural network
        nn_model.to(self.device)
        results.append(self.evaluate_model(
            nn_model, X_train, y_train, X_test, y_test, "MLP", uf, hp
        ))
        
        if include_baselines:
            # Linear regression
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            results.append(self.evaluate_model(lr, X_train, y_train, X_test, y_test, "Linear", uf, hp))
            
            # Polynomial degree 2
            poly2 = make_pipeline(PolynomialFeatures(2), LinearRegression())
            poly2.fit(X_train, y_train)
            results.append(self.evaluate_model(poly2, X_train, y_train, X_test, y_test, "Poly-2", uf, hp))
            
            # Polynomial degree 3
            poly3 = make_pipeline(PolynomialFeatures(3), LinearRegression())
            poly3.fit(X_train, y_train)
            results.append(self.evaluate_model(poly3, X_train, y_train, X_test, y_test, "Poly-3", uf, hp))
        
        return results
    
    def compare_all(
        self,
        models_dict: Dict[int, Dict[float, nn.Module]],
        inputs_dict: Dict[int, np.ndarray],
        outputs_dict: Dict[int, Dict[float, np.ndarray]],
        hp_values: List[float] = [0.50, 0.70, 1.00],
        test_size: float = 0.2,
        random_seed: int = 20,
    ) -> List[ModelComparisonResult]:
        """Compare all models across all UF sets and height positions."""
        results = []
        
        for uf, hp_models in models_dict.items():
            inputs = inputs_dict[uf]
            
            for hp in hp_values:
                if hp not in hp_models:
                    continue
                
                model = hp_models[hp]
                outputs = outputs_dict[uf][hp]
                
                case_results = self.compare_single_case(
                    model, inputs, outputs, uf, hp, test_size, random_seed
                )
                results.extend(case_results)
        
        return results
    
    def results_to_dataframe(self, results: List[ModelComparisonResult]) -> pd.DataFrame:
        """Convert results to DataFrame."""
        data = []
        for r in results:
            data.append({
                "UF": r.uf, "Hp": r.hp, "Model": r.model_name,
                "Train_MSE": r.train_mse, "Test_MSE": r.test_mse, "MSE_Gap": r.mse_gap,
                "Train_R2": r.train_r2, "Test_R2": r.test_r2, "R2_Gap": r.r2_gap,
                "MSE_Ratio": r.mse_ratio,
            })
        return pd.DataFrame(data)
    
    def plot_mse_comparison(self, results: List[ModelComparisonResult], save_path: Optional[str] = None) -> str:
        """Plot MSE comparison across models."""
        df = self.results_to_dataframe(results)
        
        models = ["Linear", "Poly-2", "Poly-3", "MLP"]
        uf_values = sorted(df["UF"].unique())
        
        agg = df.groupby(["UF", "Model"])[["Test_MSE", "MSE_Gap"]].mean().reset_index()
        
        fig, ax1 = plt.subplots(figsize=(6, 3))
        ax2 = ax1.twinx()
        
        x = np.arange(len(uf_values))
        width = 0.18
        # colors = ['#c0392b', '#27ae60', '#2ecc71', '#f39c12']
        
        for i, model in enumerate(models):
            sub = agg[agg["Model"] == model].sort_values("UF")
            offset = (i - len(models) / 2 + 0.5) * width
            ax1.bar(x + offset, sub["Test_MSE"].values, width, label=model, alpha=0.85)
            ax2.plot(x + offset, np.abs(sub["MSE_Gap"].values) * 1000, marker="o", markeredgecolor = "black",
            markeredgewidth = 1.2, markersize = 5, linewidth=1.5, linestyle="-")
        
        ax1.set_ylabel("Mean test MSE")
        ax1.set_yscale("log")
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"UF set{i+1}" for i in range(len(uf_values))])
        
        ax2.set_ylabel("Mean MSE gap * 1e3")
        
        ax1.legend(loc="upper left", frameon=False, fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
        
        plt.close()
        return save_path or ""
    
    def save_results(self, results: List[ModelComparisonResult], save_path: str) -> str:
        """Save comparison results to Excel."""
        df = self.results_to_dataframe(results)
        df = df.sort_values(["Model", "Hp", "UF"]).reset_index(drop=True)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(save_path, index=False)
        return save_path
