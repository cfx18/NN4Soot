"""
Tests for the SootMLP model.
"""

import pytest
import torch
import numpy as np

from nn4soot.models import SootMLP


class TestSootMLP:
    """Tests for the SootMLP neural network model."""
    
    def test_default_initialization(self):
        """Test model initialization with default parameters."""
        model = SootMLP()
        
        assert model.input_dim == 10
        assert model.output_dim == 20
        assert model.hidden_dims == [64, 128, 256, 256, 128, 64]
    
    def test_custom_initialization(self):
        """Test model initialization with custom parameters."""
        model = SootMLP(
            input_dim=5,
            output_dim=10,
            hidden_dims=[32, 64, 32]
        )
        
        assert model.input_dim == 5
        assert model.output_dim == 10
        assert model.hidden_dims == [32, 64, 32]
    
    def test_forward_pass_shape(self):
        """Test that forward pass produces correct output shape."""
        model = SootMLP(input_dim=10, output_dim=20)
        batch_size = 32
        
        x = torch.randn(batch_size, 10)
        output = model(x)
        
        assert output.shape == (batch_size, 20)
    
    def test_forward_pass_single_sample(self):
        """Test forward pass with a single sample."""
        model = SootMLP()
        
        x = torch.randn(1, 10)
        output = model(x)
        
        assert output.shape == (1, 20)
    
    def test_get_num_parameters(self):
        """Test parameter counting."""
        model = SootMLP(input_dim=10, output_dim=20, hidden_dims=[64, 64])
        
        n_params = model.get_num_parameters()
        
        # 10*64 + 64 + 64*64 + 64 + 64*20 + 20 = 640 + 64 + 4096 + 64 + 1280 + 20 = 6164
        assert n_params > 0
        assert isinstance(n_params, int)
    
    def test_get_architecture_info(self):
        """Test architecture info retrieval."""
        model = SootMLP(input_dim=10, output_dim=20)
        
        info = model.get_architecture_info()
        
        assert "input_dim" in info
        assert "output_dim" in info
        assert "hidden_dims" in info
        assert "total_parameters" in info
        assert info["input_dim"] == 10
        assert info["output_dim"] == 20
    
    def test_gradient_flow(self):
        """Test that gradients can be computed through the model."""
        model = SootMLP()
        x = torch.randn(4, 10, requires_grad=True)
        
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.shape == (4, 10)
    
    def test_deterministic_output(self):
        """Test that the model produces deterministic outputs."""
        model = SootMLP()
        model.eval()
        
        x = torch.randn(4, 10)
        
        output1 = model(x)
        output2 = model(x)
        
        assert torch.allclose(output1, output2)
    
    def test_batch_independence(self):
        """Test that batch processing gives same results as individual samples."""
        model = SootMLP()
        model.eval()
        
        x_batch = torch.randn(4, 10)
        
        output_batch = model(x_batch)
        
        for i in range(4):
            output_single = model(x_batch[i:i+1])
            assert torch.allclose(output_batch[i], output_single.squeeze(0), atol=1e-6)
    
    def test_cnn_model_alias(self):
        """Test that CNN_Model is an alias for SootMLP."""
        from nn4soot.models.mlp import CNN_Model
        
        assert CNN_Model is SootMLP


class TestSootMLPDeviceHandling:
    """Tests for device handling in SootMLP."""
    
    def test_cpu_inference(self):
        """Test inference on CPU."""
        model = SootMLP()
        x = torch.randn(4, 10)
        
        output = model(x)
        
        assert output.device == x.device
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_inference(self):
        """Test inference on CUDA if available."""
        model = SootMLP().cuda()
        x = torch.randn(4, 10).cuda()
        
        output = model(x)
        
        assert output.is_cuda
