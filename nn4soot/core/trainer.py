"""
SootTrainer: Neural Network Training for Soot Surrogate Models

This module provides a comprehensive training framework for the SootMLP model,
including early stopping, learning rate scheduling, and training visualization.

Author: Feixue Cai
"""

import os
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class TrainingConfig:
    """Configuration for SootTrainer."""
    
    # Training parameters
    num_epochs: int = 20000
    batch_size: int = 64
    test_size: float = 0.2
    random_seed: int = 20
    
    # Optimizer parameters
    initial_lr: float = 0.001
    weight_decay: float = 1e-4
    
    # Learning rate schedule
    lr_schedule: Dict[int, float] = field(default_factory=lambda: {
        0: 0.001,
        1000: 0.00075,
        2000: 0.0005,
        5000: 0.00025,
        7000: 0.0001,
        10000: 0.00009,
        15000: 0.00008,
    })
    
    # Early stopping
    early_stop_patience: int = 1000
    early_stop_min_delta: float = 1e-5
    
    # Output
    save_dir: str = "."
    checkpoint_name: str = "."


class SootTrainer:
    """
    Trainer for SootMLP surrogate models.
    
    This class handles the complete training pipeline including data splitting,
    model training, early stopping, and result visualization.
    
    Parameters
    ----------
    model : SootMLP
        The neural network model to train
    config : TrainingConfig, optional
        Training configuration
    device : torch.device, optional
        Device for training (auto-detected if not specified)
    
    Examples
    --------
    >>> from nn4soot import SootMLP, SootTrainer
    >>> model = SootMLP(input_dim=10)
    >>> trainer = SootTrainer(model)
    >>> history = trainer.train(inputs, outputs, hp=0.70)
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[TrainingConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.config = config or TrainingConfig()
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        self.model.to(self.device)
        
        # Training history
        self.train_losses: List[float] = []
        self.test_losses: List[float] = []
    
    def _adjust_learning_rate(self, optimizer: optim.Optimizer, epoch: int):
        """Adjust learning rate according to schedule."""
        new_lr = self.config.lr_schedule.get(epoch, None)
        if new_lr is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
    
    def train(
        self,
        inputs: np.ndarray,
        outputs: np.ndarray,
        hp: float,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train the model on the given data.
        
        Parameters
        ----------
        inputs : np.ndarray
            Input parameters of shape (N, input_dim)
        outputs : np.ndarray
            Target outputs of shape (N, output_dim)
            Should be log10-transformed PSD values
        hp : float
            Height position for checkpoint naming
        verbose : bool
            Whether to print training progress
        
        Returns
        -------
        Dict[str, List[float]]
            Training history with 'train_loss' and 'test_loss' keys
        """
        # Convert to tensors
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
        outputs_tensor = torch.tensor(outputs, dtype=torch.float32)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            inputs_tensor, outputs_tensor,
            test_size=self.config.test_size,
            random_state=self.config.random_seed,
        )
        
        X_train = X_train.to(self.device).float()
        X_test = X_test.to(self.device).float()
        y_train = y_train.to(self.device).float()
        y_test = y_test.to(self.device).float()
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.config.batch_size, shuffle=False
        )
        
        # Setup optimizer
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.initial_lr,
            weight_decay=self.config.weight_decay,
        )
        
        criterion = nn.MSELoss()
        
        # Early stopping variables
        best_loss = float('inf')
        epochs_no_improve = 0
        
        # Training loop
        self.train_losses = []
        self.test_losses = []
        
        for epoch in range(self.config.num_epochs):
            self._adjust_learning_rate(optimizer, epoch)
            
            # Training phase
            self.model.train()
            running_train_loss = 0.0
            
            for inputs_batch, labels_batch in train_loader:
                optimizer.zero_grad()
                outputs_batch = self.model(inputs_batch)
                loss = criterion(outputs_batch, labels_batch)
                loss.backward()
                optimizer.step()
                running_train_loss += loss.item()
            
            train_loss = running_train_loss / len(train_loader)
            self.train_losses.append(train_loss)
            
            # Evaluation phase
            self.model.eval()
            running_test_loss = 0.0
            
            with torch.no_grad():
                for inputs_batch, labels_batch in test_loader:
                    outputs_batch = self.model(inputs_batch)
                    loss = criterion(outputs_batch, labels_batch)
                    running_test_loss += loss.item()
            
            test_loss = running_test_loss / len(test_loader)
            self.test_losses.append(test_loss)
            
            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{self.config.num_epochs}], "
                      f"Train Loss: {train_loss:.8f}, Test Loss: {test_loss:.8f}")
            
            # Early stopping check
            if test_loss < best_loss - self.config.early_stop_min_delta:
                best_loss = test_loss
                epochs_no_improve = 0
                # Save best model
                checkpoint_path = os.path.join(
                    self.config.save_dir,
                    self.config.checkpoint_name.format(hp=hp),
                )
                torch.save(self.model.state_dict(), checkpoint_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.config.early_stop_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1} "
                              f"(no improvement for {self.config.early_stop_patience} epochs)")
                    break
        
        return {
            "train_loss": self.train_losses,
            "test_loss": self.test_losses,
        }
    
    def save_training_history(
        self,
        hp: float,
        n_samples: int,
        save_dir: Optional[str] = None,
    ) -> str:
        """
        Save training history to Excel file.
        
        Parameters
        ----------
        hp : float
            Height position
        n_samples : int
            Number of training samples
        save_dir : str, optional
            Directory to save (uses config.save_dir if not specified)
        
        Returns
        -------
        str
            Path to saved file
        """
        save_dir = save_dir or self.config.save_dir
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        losses_df = pd.DataFrame({
            "Epoch": list(range(1, len(self.train_losses) + 1)),
            "Train Loss": self.train_losses,
            "Test Loss": self.test_losses,
        })
        
        file_path = os.path.join(
            save_dir, f"losses_N{n_samples}_Hp={hp:.2f}.xlsx"
        )
        losses_df.to_excel(file_path, index=False)
        
        return file_path
    
    def plot_training_curves(
        self,
        hp: float,
        n_samples: int,
        save_dir: Optional[str] = None,
    ) -> str:
        """
        Plot and save training curves.
        
        Parameters
        ----------
        hp : float
            Height position
        n_samples : int
            Number of training samples
        save_dir : str, optional
            Directory to save (uses config.save_dir if not specified)
        
        Returns
        -------
        str
            Path to saved figure
        """
        save_dir = save_dir or self.config.save_dir
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        plt.figure(figsize=(5, 4))
        plt.plot(self.train_losses, label="Train Loss")
        plt.plot(self.test_losses, label="Test Loss")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.legend()
        plt.title(f'Training History (Hp={hp:.2f} cm, N={n_samples})')
        plt.tight_layout()
        
        file_path = os.path.join(
            save_dir, f"loss_N{n_samples}_Hp={hp:.2f}.png"
        )
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return file_path
