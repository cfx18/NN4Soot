"""
SootEvaluator: Model Evaluation and Comparison for Soot Surrogate Models

This module provides comprehensive evaluation tools including R², MSE metrics,
comparison with baseline models (Linear, Polynomial regression), and 
overfitting analysis.

Author: Feixue Cai
"""

import os
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
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
class EvaluationResult:
    """Container for evaluation results."""
    
    model_name: str
    hp: float
    
    train_r2: float
    test_r2: float
    train_mse: float
    test_mse: float
    
    @property
    def r2_gap(self) -> float:
        """R² gap (train - test), indicates overfitting."""
        return self.train_r2 - self.test_r2
    
    @property
    def mse_ratio(self) -> float:
        """MSE ratio (test / train), close to 1 is ideal."""
        return self.test_mse / self.train_mse if self.train_mse > 0 else float('inf')


class SootEvaluator:
    """
    Evaluator for SootMLP and baseline models.
    
    This class provides comprehensive model evaluation including:
    - R² and MSE metrics on train/test sets
    - Comparison with Linear and Polynomial regression baselines
    - Overfitting analysis
    - Visualization of predictions
    
    Parameters
    ----------
    device : torch.device, optional
        Device for evaluation (CPU by default)
    
    Examples
    --------
    >>> from nn4soot import SootMLP, SootEvaluator
    >>> model = SootMLP.from_pretrained("model.pth")
    >>> evaluator = SootEvaluator()
    >>> results = evaluator.evaluate_all(model, inputs, outputs, hp=0.70)
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device
    
    def evaluate_single_model(
        self,
        model: nn.Module,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_test: torch.Tensor,
        y_test: torch.Tensor,
        model_name: str,
        hp: float,
    ) -> EvaluationResult:
        """
        Evaluate a single model on train and test sets.
        
        Parameters
        ----------
        model : nn.Module
            Trained model
        X_train, y_train : torch.Tensor
            Training data
        X_test, y_test : torch.Tensor
            Test data
        model_name : str
            Name of the model for reporting
        hp : float
            Height position
        
        Returns
        -------
        EvaluationResult
            Evaluation metrics
        """
        # Only call eval() for PyTorch models
        if hasattr(model, 'eval') and not hasattr(model, 'predict'):
            model.eval()
        
        with torch.no_grad():
            # Handle different model types
            if hasattr(model, 'predict'):
                # Sklearn-style models
                y_train_pred = model.predict(X_train.numpy())
                y_test_pred = model.predict(X_test.numpy())
            else:
                # PyTorch models
                y_train_pred = model(X_train).numpy()
                y_test_pred = model(X_test).numpy()
        
        # Compute mean across bins
        y_train_true_mean = y_train.numpy().mean(axis=1)
        y_test_true_mean = y_test.numpy().mean(axis=1)
        y_train_pred_mean = y_train_pred.mean(axis=1)
        y_test_pred_mean = y_test_pred.mean(axis=1)
        
        return EvaluationResult(
            model_name=model_name,
            hp=hp,
            train_r2=r2_score(y_train_true_mean, y_train_pred_mean),
            test_r2=r2_score(y_test_true_mean, y_test_pred_mean),
            train_mse=mean_squared_error(y_train_true_mean, y_train_pred_mean),
            test_mse=mean_squared_error(y_test_true_mean, y_test_pred_mean),
        )
    
    def evaluate_all(
        self,
        nn_model: nn.Module,
        inputs: np.ndarray,
        outputs: np.ndarray,
        hp: float,
        test_size: float = 0.2,
        random_seed: int = 20,
        include_baselines: bool = True,
    ) -> List[EvaluationResult]:
        """
        Evaluate neural network and baseline models.
        
        Parameters
        ----------
        nn_model : nn.Module
            Trained neural network model
        inputs : np.ndarray
            Input parameters (N, input_dim)
        outputs : np.ndarray
            Target outputs (N, output_dim), log10-transformed
        hp : float
            Height position
        test_size : float
            Test set proportion
        random_seed : int
            Random seed for reproducibility
        include_baselines : bool
            Whether to include Linear/Poly baselines
        
        Returns
        -------
        List[EvaluationResult]
            Evaluation results for all models
        """
        # Convert to tensors
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
        outputs_tensor = torch.tensor(outputs, dtype=torch.float32)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            inputs_tensor, outputs_tensor,
            test_size=test_size,
            random_state=random_seed,
        )
        
        X_train = X_train.to(self.device).float()
        X_test = X_test.to(self.device).float()
        y_train = y_train.to(self.device).float()
        y_test = y_test.to(self.device).float()
        
        results = []
        
        # Evaluate neural network
        nn_model.to(self.device)
        results.append(self.evaluate_single_model(
            nn_model, X_train, y_train, X_test, y_test, "MLP", hp
        ))
        
        if include_baselines:
            # Linear regression
            lr_model = LinearRegression()
            lr_model.fit(X_train.cpu().numpy(), y_train.cpu().numpy())
            results.append(self.evaluate_single_model(
                lr_model, X_train, y_train, X_test, y_test, "Linear", hp
            ))
            
            # Polynomial regression (degree 2)
            poly2_model = make_pipeline(PolynomialFeatures(2), LinearRegression())
            poly2_model.fit(X_train.cpu().numpy(), y_train.cpu().numpy())
            results.append(self.evaluate_single_model(
                poly2_model, X_train, y_train, X_test, y_test, "Poly-2", hp
            ))
            
            # Polynomial regression (degree 3)
            poly3_model = make_pipeline(PolynomialFeatures(3), LinearRegression())
            poly3_model.fit(X_train.cpu().numpy(), y_train.cpu().numpy())
            results.append(self.evaluate_single_model(
                poly3_model, X_train, y_train, X_test, y_test, "Poly-3", hp
            ))
        
        return results
    
    def results_to_dataframe(
        self,
        results: List[EvaluationResult],
    ) -> pd.DataFrame:
        """Convert evaluation results to DataFrame."""
        data = []
        for r in results:
            data.append({
                "Hp": r.hp,
                "Model": r.model_name,
                "Train_R2": r.train_r2,
                "Test_R2": r.test_r2,
                "R2_Gap": r.r2_gap,
                "Train_MSE": r.train_mse,
                "Test_MSE": r.test_mse,
                "MSE_Ratio": r.mse_ratio,
            })
        return pd.DataFrame(data)
    
    def plot_model_comparison(
        self,
        results: List[EvaluationResult],
        save_path: str,
    ) -> str:
        """
        Create comparison plots for all models.
        
        Parameters
        ----------
        results : List[EvaluationResult]
            Evaluation results
        save_path : str
            Path to save the figure
        
        Returns
        -------
        str
            Path to saved figure
        """
        df = self.results_to_dataframe(results)
        models = df["Model"].unique()
        hp_values = sorted(df["Hp"].unique())
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        
        # Test R² comparison
        ax = axes[0]
        x = np.arange(len(hp_values))
        width = 0.18
        
        for i, model in enumerate(models):
            vals = [df[(df["Hp"] == hp) & (df["Model"] == model)]["Test_R2"].values[0]
                   for hp in hp_values]
            offset = (i - len(models) / 2 + 0.5) * width
            ax.bar(x + offset, vals, width, label=model, color=colors[i], alpha=0.8)
        
        ax.set_xlabel('Height Position (cm)')
        ax.set_ylabel('Test R²')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{hp:.2f}' for hp in hp_values])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Test MSE comparison (log scale)
        ax = axes[1]
        for i, model in enumerate(models):
            vals = [df[(df["Hp"] == hp) & (df["Model"] == model)]["Test_MSE"].values[0]
                   for hp in hp_values]
            offset = (i - len(models) / 2 + 0.5) * width
            ax.bar(x + offset, vals, width, label=model, color=colors[i], alpha=0.8)
        
        ax.set_xlabel('Height Position (cm)')
        ax.set_ylabel('Test MSE')
        ax.set_yscale('log')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{hp:.2f}' for hp in hp_values])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_prediction_scatter(
        self,
        model: nn.Module,
        inputs: np.ndarray,
        outputs: np.ndarray,
        hp: float,
        save_path: str,
        test_size: float = 0.2,
        random_seed: int = 20,
    ) -> str:
        """
        Create scatter plot of predicted vs true values.
        
        Parameters
        ----------
        model : nn.Module
            Trained model
        inputs : np.ndarray
            Input parameters
        outputs : np.ndarray
            Target outputs
        hp : float
            Height position
        save_path : str
            Path to save the figure
        test_size : float
            Test set proportion
        random_seed : int
            Random seed
        
        Returns
        -------
        str
            Path to saved figure
        """
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32).to(self.device)
        outputs_tensor = torch.tensor(outputs, dtype=torch.float32)
        
        X_train, X_test, y_train, y_test = train_test_split(
            inputs_tensor, outputs_tensor,
            test_size=test_size,
            random_state=random_seed,
        )
        
        model.eval()
        model.to(self.device)
        
        with torch.no_grad():
            y_pred_test = model(X_test).cpu().numpy()
        
        y_true_mean = y_test.numpy().mean(axis=1)
        y_pred_mean = y_pred_test.mean(axis=1)
        
        r2 = r2_score(y_true_mean, y_pred_mean)
        
        plt.figure(figsize=(5, 4))
        plt.scatter(y_pred_mean, y_true_mean, alpha=0.7, 
                   label=f'MLP: R²={r2:.3f}')
        plt.xlabel('Predicted MEAN(log(PSD))')
        plt.ylabel('True MEAN(log(PSD))')
        plt.legend()
        plt.title(f'Hp={hp:.2f} cm')
        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
