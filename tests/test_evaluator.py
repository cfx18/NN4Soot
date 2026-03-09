"""
Tests for the SootEvaluator evaluation utilities.
"""

import pytest
import torch
import numpy as np
import tempfile
import os

from nn4soot.models import SootMLP
from nn4soot.core.evaluator import SootEvaluator, EvaluationResult


class TestEvaluationResult:
    """Tests for the EvaluationResult dataclass."""
    
    def test_evaluation_result_properties(self):
        """Test EvaluationResult properties."""
        result = EvaluationResult(
            model_name="TestModel",
            hp=0.70,
            train_r2=0.95,
            test_r2=0.90,
            train_mse=0.01,
            test_mse=0.02
        )
        
        # Use approx for floating point comparison
        assert result.r2_gap == pytest.approx(0.05, rel=1e-6)  # train_r2 - test_r2
        assert result.mse_ratio == pytest.approx(2.0, rel=1e-6)  # test_mse / train_mse
    
    def test_evaluation_result_mse_ratio_edge_case(self):
        """Test MSE ratio when train_mse is zero."""
        result = EvaluationResult(
            model_name="TestModel",
            hp=0.70,
            train_r2=1.0,
            test_r2=1.0,
            train_mse=0.0,
            test_mse=0.01
        )
        
        assert result.mse_ratio == float('inf')


class TestSootEvaluator:
    """Tests for the SootEvaluator class."""
    
    @pytest.fixture
    def model(self):
        """Create a simple trained model for testing."""
        model = SootMLP(input_dim=10, output_dim=20, hidden_dims=[32, 32])
        return model
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for evaluation."""
        np.random.seed(42)
        inputs = np.random.rand(100, 10)
        outputs = np.random.rand(100, 20)
        return inputs, outputs
    
    def test_initialization(self):
        """Test evaluator initialization."""
        evaluator = SootEvaluator()
        
        assert evaluator.device is not None
    
    def test_evaluate_all_with_baselines(self, model, sample_data):
        """Test evaluation with baseline models."""
        evaluator = SootEvaluator()
        inputs, outputs = sample_data
        
        results = evaluator.evaluate_all(
            model, inputs, outputs, hp=0.70,
            include_baselines=True
        )
        
        assert len(results) == 4  # MLP, Linear, Poly-2, Poly-3
        
        model_names = [r.model_name for r in results]
        assert "MLP" in model_names
        assert "Linear" in model_names
        assert "Poly-2" in model_names
        assert "Poly-3" in model_names
    
    def test_evaluate_all_without_baselines(self, model, sample_data):
        """Test evaluation without baseline models."""
        evaluator = SootEvaluator()
        inputs, outputs = sample_data
        
        results = evaluator.evaluate_all(
            model, inputs, outputs, hp=0.70,
            include_baselines=False
        )
        
        assert len(results) == 1
        assert results[0].model_name == "MLP"
    
    def test_results_to_dataframe(self, model, sample_data):
        """Test conversion of results to DataFrame."""
        evaluator = SootEvaluator()
        inputs, outputs = sample_data
        
        results = evaluator.evaluate_all(
            model, inputs, outputs, hp=0.70,
            include_baselines=False
        )
        
        df = evaluator.results_to_dataframe(results)
        
        assert "Hp" in df.columns
        assert "Model" in df.columns
        assert "Train_R2" in df.columns
        assert "Test_R2" in df.columns
        assert "R2_Gap" in df.columns
        assert len(df) == 1
    
    def test_r2_in_valid_range(self, model, sample_data):
        """Test that R² values are in valid range (or close to it for random data)."""
        evaluator = SootEvaluator()
        inputs, outputs = sample_data
        
        results = evaluator.evaluate_all(
            model, inputs, outputs, hp=0.70,
            include_baselines=False
        )
        
        # R² can be negative for poor fits, but should be finite
        for r in results:
            assert np.isfinite(r.train_r2)
            assert np.isfinite(r.test_r2)
            assert np.isfinite(r.train_mse)
            assert np.isfinite(r.test_mse)


class TestSootEvaluatorPlotting:
    """Tests for evaluator plotting functions."""
    
    @pytest.fixture
    def model(self):
        """Create a simple model for testing."""
        return SootMLP(input_dim=10, output_dim=20, hidden_dims=[16, 16])
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for evaluation."""
        np.random.seed(42)
        inputs = np.random.rand(50, 10)
        outputs = np.random.rand(50, 20)
        return inputs, outputs
    
    def test_plot_model_comparison(self, model, sample_data):
        """Test model comparison plotting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = SootEvaluator()
            inputs, outputs = sample_data
            
            results = evaluator.evaluate_all(
                model, inputs, outputs, hp=0.70,
                include_baselines=True
            )
            
            plot_path = os.path.join(tmpdir, "comparison.png")
            saved_path = evaluator.plot_model_comparison(results, plot_path)
            
            assert os.path.exists(saved_path)
    
    def test_plot_prediction_scatter(self, model, sample_data):
        """Test prediction scatter plotting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = SootEvaluator()
            inputs, outputs = sample_data
            
            plot_path = os.path.join(tmpdir, "scatter.png")
            saved_path = evaluator.plot_prediction_scatter(
                model, inputs, outputs, hp=0.70, save_path=plot_path
            )
            
            assert os.path.exists(saved_path)


class TestSootEvaluatorMultipleHeights:
    """Tests for evaluation across multiple heights."""
    
    def test_multiple_heights(self):
        """Test evaluation at multiple height positions."""
        model = SootMLP(input_dim=10, output_dim=20, hidden_dims=[16, 16])
        evaluator = SootEvaluator()
        
        np.random.seed(42)
        inputs = np.random.rand(50, 10)
        outputs = np.random.rand(50, 20)
        
        all_results = []
        for hp in [0.50, 0.70, 1.00]:
            results = evaluator.evaluate_all(
                model, inputs, outputs, hp=hp,
                include_baselines=False
            )
            all_results.extend(results)
        
        df = evaluator.results_to_dataframe(all_results)
        
        assert len(df) == 3
        assert set(df["Hp"].unique()) == {0.50, 0.70, 1.00}
