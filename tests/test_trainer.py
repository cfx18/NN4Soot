"""
Tests for the SootTrainer training utilities.
"""

import pytest
import torch
import numpy as np
import tempfile
import os

from nn4soot.models import SootMLP
from nn4soot.core.trainer import SootTrainer, TrainingConfig


class TestTrainingConfig:
    """Tests for the TrainingConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = TrainingConfig()
        
        assert config.num_epochs == 20000
        assert config.batch_size == 64
        assert config.test_size == 0.2
        assert config.random_seed == 20
        assert config.initial_lr == 0.001
        assert config.early_stop_patience == 1000
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = TrainingConfig(
            num_epochs=100,
            batch_size=32,
            initial_lr=0.0005
        )
        
        assert config.num_epochs == 100
        assert config.batch_size == 32
        assert config.initial_lr == 0.0005
    
    def test_lr_schedule_default(self):
        """Test that default LR schedule is defined."""
        config = TrainingConfig()
        
        assert 0 in config.lr_schedule
        assert 1000 in config.lr_schedule
        assert config.lr_schedule[0] == 0.001


class TestSootTrainer:
    """Tests for the SootTrainer class."""
    
    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        return SootMLP(input_dim=10, output_dim=20, hidden_dims=[32, 32])
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        inputs = np.random.rand(100, 10)
        outputs = np.random.rand(100, 20)
        return inputs, outputs
    
    def test_initialization(self, simple_model):
        """Test trainer initialization."""
        trainer = SootTrainer(simple_model)
        
        assert trainer.model is simple_model
        assert trainer.config is not None
        assert trainer.device is not None
    
    def test_initialization_with_config(self, simple_model):
        """Test trainer initialization with custom config."""
        config = TrainingConfig(num_epochs=50)
        trainer = SootTrainer(simple_model, config=config)
        
        assert trainer.config.num_epochs == 50
    
    def test_short_training_run(self, simple_model, sample_data):
        """Test a short training run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                num_epochs=10,
                batch_size=16,
                early_stop_patience=5,
                save_dir=tmpdir,
                checkpoint_name="model_hp{hp:.2f}.pth"
            )
            trainer = SootTrainer(simple_model, config=config)
            
            inputs, outputs = sample_data
            history = trainer.train(inputs, outputs, hp=0.50, verbose=False)
            
            assert "train_loss" in history
            assert "test_loss" in history
            assert len(history["train_loss"]) > 0
            assert len(history["test_loss"]) > 0
    
    def test_training_reduces_loss(self, simple_model, sample_data):
        """Test that training reduces the loss."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                num_epochs=100,
                batch_size=16,
                early_stop_patience=50,
                save_dir=tmpdir,
                checkpoint_name="model_hp{hp:.2f}.pth"
            )
            trainer = SootTrainer(simple_model, config=config)
            
            inputs, outputs = sample_data
            history = trainer.train(inputs, outputs, hp=0.50, verbose=False)
            
            # Loss should generally decrease
            initial_train_loss = history["train_loss"][0]
            final_train_loss = history["train_loss"][-1]
            
            # Final loss should be lower than initial (with high probability)
            assert final_train_loss < initial_train_loss or len(history["train_loss"]) < 100
    
    def test_model_state_changes(self, simple_model, sample_data):
        """Test that model parameters change during training."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                num_epochs=10,
                batch_size=16,
                early_stop_patience=5,
                save_dir=tmpdir,
                checkpoint_name="model_hp{hp:.2f}.pth"
            )
            trainer = SootTrainer(simple_model, config=config)
            
            # Store initial parameters
            initial_params = {
                name: param.clone() 
                for name, param in simple_model.named_parameters()
            }
            
            inputs, outputs = sample_data
            trainer.train(inputs, outputs, hp=0.50, verbose=False)
            
            # Check that at least some parameters changed
            params_changed = False
            for name, param in simple_model.named_parameters():
                if not torch.allclose(initial_params[name], param):
                    params_changed = True
                    break
            
            assert params_changed


class TestSootTrainerFileOperations:
    """Tests for trainer file operations."""
    
    @pytest.fixture
    def trained_model(self):
        """Create and briefly train a model."""
        model = SootMLP(input_dim=10, output_dim=20, hidden_dims=[16, 16])
        return model
    
    def test_save_training_history(self, trained_model):
        """Test saving training history."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                num_epochs=5,
                save_dir=tmpdir,
                checkpoint_name="model_hp{hp:.2f}.pth"
            )
            trainer = SootTrainer(trained_model, config=config)
            
            inputs = np.random.rand(50, 10)
            outputs = np.random.rand(50, 20)
            trainer.train(inputs, outputs, hp=0.50, verbose=False)
            
            history_path = trainer.save_training_history(hp=0.50, n_samples=50)
            
            assert os.path.exists(history_path)
            assert history_path.endswith(".xlsx")
    
    def test_plot_training_curves(self, trained_model):
        """Test plotting training curves."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                num_epochs=5,
                save_dir=tmpdir,
                checkpoint_name="model_hp{hp:.2f}.pth"
            )
            trainer = SootTrainer(trained_model, config=config)
            
            inputs = np.random.rand(50, 10)
            outputs = np.random.rand(50, 20)
            trainer.train(inputs, outputs, hp=0.50, verbose=False)
            
            plot_path = trainer.plot_training_curves(hp=0.50, n_samples=50)
            
            assert os.path.exists(plot_path)
            assert plot_path.endswith(".png")


class TestSootTrainerEarlyStopping:
    """Tests for early stopping functionality."""
    
    def test_early_stopping_patience(self):
        """Test that early stopping works with small patience."""
        model = SootMLP(input_dim=10, output_dim=20, hidden_dims=[16, 16])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                num_epochs=1000,
                early_stop_patience=3,
                batch_size=16,
                save_dir=tmpdir,
                checkpoint_name="model_hp{hp:.2f}.pth"
            )
            trainer = SootTrainer(model, config=config)
            
            # Create simple linear data that's easy to learn
            inputs = np.random.rand(50, 10)
            outputs = np.random.rand(50, 20) * 0.01  # Small values
            
            history = trainer.train(inputs, outputs, hp=0.50, verbose=False)
            
            # Should stop early due to patience
            assert len(history["train_loss"]) < 1000
