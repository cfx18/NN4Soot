"""
Tests for the Active Subspace sensitivity analysis.
"""

import pytest
import torch
import numpy as np
import tempfile
import os

from nn4soot.models import SootMLP
from nn4soot.sensitivity.active_subspace import (
    ActiveSubspaceAnalyzer,
    ASResult
)


class TestASResult:
    """Tests for the ASResult dataclass."""
    
    def test_as_result_creation(self):
        """Test ASResult creation with minimal fields."""
        result = ASResult(
            sensitivity=np.array([0.1, 0.2, 0.3]),
            eigenvalues=np.array([1.0, 0.5, 0.1]),
            eigenvectors=np.eye(3)
        )
        
        assert len(result.sensitivity) == 3
        assert len(result.eigenvalues) == 3
        assert result.sensitivity_bootstrap_mean is None
        assert result.sensitivity_ci_low is None
    
    def test_as_result_with_bootstrap(self):
        """Test ASResult with bootstrap confidence intervals."""
        result = ASResult(
            sensitivity=np.array([0.1, 0.2, 0.3]),
            eigenvalues=np.array([1.0, 0.5, 0.1]),
            eigenvectors=np.eye(3),
            sensitivity_ci_low=np.array([0.08, 0.18, 0.28]),
            sensitivity_ci_high=np.array([0.12, 0.22, 0.32])
        )
        
        assert result.sensitivity_ci_low is not None
        assert result.sensitivity_ci_high is not None


class TestActiveSubspaceAnalyzer:
    """Tests for the ActiveSubspaceAnalyzer class."""
    
    @pytest.fixture
    def model(self):
        """Create a simple model for testing."""
        model = SootMLP(input_dim=10, output_dim=20, hidden_dims=[32, 32])
        model.eval()
        return model
    
    @pytest.fixture
    def sample_inputs(self):
        """Create sample input data."""
        np.random.seed(42)
        return np.random.rand(50, 10).astype(np.float32)
    
    def test_initialization(self, model):
        """Test analyzer initialization."""
        analyzer = ActiveSubspaceAnalyzer(model)
        
        assert analyzer.model is model
        assert analyzer.device is not None
    
    def test_compute_gradients(self, model, sample_inputs):
        """Test gradient computation."""
        analyzer = ActiveSubspaceAnalyzer(model)
        
        gradients = analyzer.compute_gradients(sample_inputs, bin_index=9)
        
        assert gradients.shape == sample_inputs.shape
        assert not np.any(np.isnan(gradients))
    
    def test_compute_gradient_covariance(self, model, sample_inputs):
        """Test gradient covariance computation."""
        analyzer = ActiveSubspaceAnalyzer(model)
        
        gradients = analyzer.compute_gradients(sample_inputs, bin_index=9)
        C, eigvals, eigvecs = analyzer.compute_gradient_covariance(gradients)
        
        assert C.shape == (10, 10)
        assert len(eigvals) == 10
        assert eigvecs.shape == (10, 10)
        
        # Eigenvalues should be in descending order
        assert np.all(np.diff(eigvals) <= 0)
    
    def test_compute_weighted_sensitivity(self, model, sample_inputs):
        """Test weighted sensitivity computation."""
        analyzer = ActiveSubspaceAnalyzer(model)
        
        gradients = analyzer.compute_gradients(sample_inputs, bin_index=9)
        _, eigvals, eigvecs = analyzer.compute_gradient_covariance(gradients)
        sensitivity = analyzer.compute_weighted_sensitivity(eigvals, eigvecs)
        
        assert len(sensitivity) == 10
        assert np.all(sensitivity >= 0)
        # Sensitivities should sum to approximately 1
        assert np.abs(sensitivity.sum() - 1.0) < 0.1
    
    def test_analyze_with_bootstrap(self, model, sample_inputs):
        """Test full analysis with bootstrap."""
        analyzer = ActiveSubspaceAnalyzer(model)
        
        result = analyzer.analyze_with_bootstrap(
            sample_inputs,
            bin_index=9,
            n_bootstrap=10,  # Small number for speed
            ci_level=0.95,
            seed=42
        )
        
        assert len(result.sensitivity) == 10
        assert len(result.eigenvalues) == 10
        assert result.sensitivity_bootstrap_mean is not None
        assert result.sensitivity_ci_low is not None
        assert result.sensitivity_ci_high is not None
        assert result.active_subspace_vector is not None
    
    def test_analyze_multiple_bins(self, model, sample_inputs):
        """Test analysis for multiple BIN indices."""
        analyzer = ActiveSubspaceAnalyzer(model)
        
        results = analyzer.analyze_multiple_bins(
            sample_inputs,
            bin_indices=[7, 9, 12],
            n_bootstrap=10,
            seed=42
        )
        
        assert len(results) == 3
        assert 7 in results
        assert 9 in results
        assert 12 in results
    
    def test_infer_window_bins(self, model):
        """Test bin window inference."""
        d_bins = np.array([1.78, 2.25, 2.80, 3.46, 4.32, 5.38, 6.71, 8.38, 
                          10.45, 13.05, 16.29, 20.29, 25.35, 31.68, 39.59, 
                          49.50, 61.76, 77.24, 96.61, 120.86])
        
        lo, hi = ActiveSubspaceAnalyzer.infer_window_bins(
            d_bins, dmin=2.0, dmax=15.0
        )
        
        assert lo >= 0
        assert hi < len(d_bins)
        assert d_bins[lo] >= 2.0
        assert d_bins[hi] <= 15.0
    
    def test_infer_window_bins_error(self, model):
        """Test bin window inference with invalid range."""
        d_bins = np.array([1.0, 2.0, 3.0])
        
        with pytest.raises(ValueError):
            ActiveSubspaceAnalyzer.infer_window_bins(
                d_bins, dmin=100.0, dmax=200.0
            )


class TestActiveSubspaceAnalyzerPlotting:
    """Tests for analyzer plotting functions."""
    
    @pytest.fixture
    def model(self):
        """Create a simple model for testing."""
        model = SootMLP(input_dim=10, output_dim=20, hidden_dims=[16, 16])
        model.eval()
        return model
    
    @pytest.fixture
    def sample_inputs(self):
        """Create sample input data."""
        np.random.seed(42)
        return np.random.rand(30, 10).astype(np.float32)
    
    @pytest.fixture
    def analysis_result(self, model, sample_inputs):
        """Get analysis result for plotting tests."""
        analyzer = ActiveSubspaceAnalyzer(model)
        return analyzer.analyze_with_bootstrap(
            sample_inputs,
            bin_index=9,
            n_bootstrap=10,
            seed=42
        )
    
    def test_plot_sensitivity(self, analysis_result):
        """Test sensitivity plotting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = ActiveSubspaceAnalyzer(SootMLP())
            save_path = os.path.join(tmpdir, "sensitivity.png")
            
            result_path = analyzer.plot_sensitivity(
                analysis_result,
                save_path=save_path
            )
            
            assert os.path.exists(result_path)
    
    def test_plot_eigenvalue_spectrum(self, analysis_result):
        """Test eigenvalue spectrum plotting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = ActiveSubspaceAnalyzer(SootMLP())
            save_path = os.path.join(tmpdir, "eigenvalues.png")
            
            result_path = analyzer.plot_eigenvalue_spectrum(
                analysis_result,
                save_path=save_path
            )
            
            assert os.path.exists(result_path)
    
    def test_save_results(self, analysis_result):
        """Test saving results to files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = ActiveSubspaceAnalyzer(SootMLP())
            
            paths = analyzer.save_results(
                analysis_result,
                save_dir=tmpdir,
                prefix="test"
            )
            
            assert "sensitivity" in paths
            assert "eigenvalues" in paths
            assert os.path.exists(paths["sensitivity"])
            assert os.path.exists(paths["eigenvalues"])


class TestActiveSubspaceGradients:
    """Tests for gradient computation properties."""
    
    def test_gradient_different_bins(self):
        """Test that different BINs produce different gradients."""
        model = SootMLP(input_dim=10, output_dim=20, hidden_dims=[16, 16])
        model.eval()
        analyzer = ActiveSubspaceAnalyzer(model)
        
        inputs = np.random.rand(20, 10).astype(np.float32)
        
        grad_7 = analyzer.compute_gradients(inputs, bin_index=7)
        grad_12 = analyzer.compute_gradients(inputs, bin_index=12)
        
        # Gradients for different bins should be different
        assert not np.allclose(grad_7, grad_12)
    
    def test_gradient_sensitivity(self):
        """Test that gradients respond to input changes."""
        model = SootMLP(input_dim=10, output_dim=20, hidden_dims=[16, 16])
        model.eval()
        analyzer = ActiveSubspaceAnalyzer(model)
        
        inputs1 = np.random.rand(20, 10).astype(np.float32)
        inputs2 = inputs1 + 0.1 * np.random.randn(20, 10).astype(np.float32)
        
        grad1 = analyzer.compute_gradients(inputs1, bin_index=9)
        grad2 = analyzer.compute_gradients(inputs2, bin_index=9)
        
        # Different inputs should produce different gradients
        assert not np.allclose(grad1, grad2)
