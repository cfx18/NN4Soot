"""
SootMLP: Multi-Layer Perceptron for Soot PSD Prediction

This module defines the neural network architecture used as a surrogate model
for sectional soot kinetic modeling. The network maps normalized kinetic 
parameters to log10-transformed particle size distributions (PSD).

Architecture:
- Input: 10 normalized kinetic parameters (P1-P10)
- Hidden layers: 64 -> 128 -> 256 -> 256 -> 128 -> 64
- Output: 20-bin log10(PSD) values (BIN 4-23)
- Activation: ReLU

Author: Feixue Cai
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Any


class SootMLP(nn.Module):
    """
    Multi-Layer Perceptron for soot particle size distribution prediction.
    
    This network serves as a surrogate model for sectional soot kinetic simulations,
    enabling efficient forward prediction and gradient-based analysis.
    
    Parameters
    ----------
    input_dim : int
        Number of input parameters (default: 10 for kinetic parameters P1-P10)
    output_dim : int
        Number of output bins (default: 20 for BIN 5-24)
    hidden_dims : List[int], optional
        Hidden layer dimensions. Default: [64, 128, 256, 256, 128, 64]
    
    Examples
    --------
    >>> model = SootMLP(input_dim=10)
    >>> x = torch.randn(32, 10)  # batch of 32 samples
    >>> y = model(x)  # shape: (32, 20)
    """
    
    def __init__(
        self,
        input_dim: int = 10,
        output_dim: int = 20,
        hidden_dims: Optional[List[int]] = None,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        if hidden_dims is None:
            hidden_dims = [64, 128, 256, 256, 128, 64]
        
        self.hidden_dims = hidden_dims
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        
        self.hidden_layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(prev_dim, output_dim)

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim)
            Values should be normalized to [0, 1] range
        
        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, output_dim)
            Values are log10-transformed PSD values
        """
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        
        x = self.output_layer(x)
        return x
    
    def get_num_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_architecture_info(self) -> Dict[str, Any]:
        """Return architecture information as a dictionary."""
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_dims": self.hidden_dims,
            "total_parameters": self.get_num_parameters(),
        }
    
    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        input_dim: int = 10,
        output_dim: int = 20,
        device: torch.device = None,
    ) -> "SootMLP":
        """
        Load a pretrained model from checkpoint.
        
        Parameters
        ----------
        checkpoint_path : str
            Path to the checkpoint file (.pth)
        input_dim : int
            Input dimension (must match the saved model)
        output_dim : int
            Output dimension (must match the saved model)
        device : torch.device, optional
            Device to load the model on
        
        Returns
        -------
        SootMLP
            Loaded model in evaluation mode
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model = cls(input_dim=input_dim, output_dim=output_dim)
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        return model


# Alias for backward compatibility with original code
CNN_Model = SootMLP
