"""
Tests for the data loading utilities.
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
import pandas as pd

from nn4soot.utils.data_loader import DataLoader, SootDataset


class TestSootDataset:
    """Tests for the SootDataset dataclass."""
    
    def test_soot_dataset_properties(self):
        """Test SootDataset properties."""
        dataset = SootDataset(
            uf_set=1,
            factors=np.random.rand(100, 10),
            psd_outputs={0.50: np.random.rand(100, 20)},
            diameters=np.linspace(1, 100, 20)
        )
        
        assert dataset.n_samples == 100
        assert dataset.n_params == 10


class TestDataLoader:
    """Tests for the DataLoader class."""
    
    def test_initialization(self):
        """Test DataLoader initialization."""
        loader = DataLoader("data/UF_set1")
        
        assert str(loader.data_dir) == "data/UF_set1"
    
    def test_get_default_diameters(self):
        """Test default diameter retrieval."""
        dia = DataLoader.get_default_diameters()
        
        assert len(dia) == 20
        assert dia[0] < dia[-1]  # Ascending order
        assert np.all(dia > 0)  # All positive
    
    def test_interpolate_exp_to_model_bins(self):
        """Test interpolation of experimental data to model bins."""
        loader = DataLoader(".")
        
        # Create synthetic experimental data
        exp_dia = np.array([2, 5, 10, 20, 50, 100])
        exp_psd = np.log10(exp_dia)  # Some arbitrary relationship
        
        # Model bins
        model_dia = np.array([1.78, 2.25, 2.80, 3.46, 4.32, 5.38, 6.71, 8.38, 
                              10.45, 13.05, 16.29, 20.29, 25.35, 31.68, 39.59, 
                              49.50, 61.76, 77.24, 96.61, 120.86])
        
        filtered_dia, interp_psd = loader.interpolate_exp_to_model_bins(
            exp_dia, exp_psd, model_dia
        )
        
        # All filtered diameters should be within experimental range
        assert np.all(filtered_dia >= exp_dia.min())
        assert np.all(filtered_dia <= exp_dia.max())
        assert len(filtered_dia) == len(interp_psd)
    
    def test_interpolate_exp_to_model_bins_edge_cases(self):
        """Test interpolation edge cases."""
        loader = DataLoader(".")
        
        # Test with minimal overlapping data
        exp_dia = np.array([5, 10])
        exp_psd = np.array([1, 2])
        model_dia = np.array([3, 5, 7, 10, 15])
        
        filtered_dia, interp_psd = loader.interpolate_exp_to_model_bins(
            exp_dia, exp_psd, model_dia
        )
        
        # Should only include bins within [5, 10]
        assert len(filtered_dia) == 3  # 5, 7, 10
        assert filtered_dia[0] == 5
        assert filtered_dia[-1] == 10


class TestDataLoaderWithFiles:
    """Tests requiring actual file I/O."""
    
    def test_load_factors_from_file(self):
        """Test loading factors from a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test factors file
            factors = np.random.rand(10, 5)
            factors_path = os.path.join(tmpdir, "test_factors.txt")
            np.savetxt(factors_path, factors)
            
            loader = DataLoader(tmpdir)
            loaded = loader.load_factors(factors_file=factors_path)
            
            assert loaded.shape == (10, 5)
            np.testing.assert_array_almost_equal(loaded, factors)
    
    def test_load_psd_outputs_from_file(self):
        """Test loading PSD outputs from a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test PSD file with proper format (DataFrame with header)
            psd_data = np.random.rand(10, 26)
            psd_path = os.path.join(tmpdir, "Soot_PSDs_Hp=0.50.csv")
            # Create DataFrame with column names
            columns = [f"col_{i}" for i in range(26)]
            df = pd.DataFrame(psd_data, columns=columns)
            df.to_csv(psd_path, index=False)
            
            loader = DataLoader(tmpdir)
            loaded = loader.load_psd_outputs(hp=0.50, psd_file=psd_path, log_transform=False)
            
            # Should slice columns 3:23 (20 columns)
            assert loaded.shape == (10, 20)
    
    def test_load_diameters_from_file(self):
        """Test loading diameters from a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test diameter file with proper format
            test_dia = np.linspace(1, 100, 20)
            # Create a DataFrame with proper structure
            # Row 0: header row (or any data)
            # Row 1: the diameter values in columns 3:23
            data = np.zeros((2, 26))
            data[1, 3:23] = test_dia
            columns = [f"col_{i}" for i in range(26)]
            df = pd.DataFrame(data, columns=columns)
            dia_path = os.path.join(tmpdir, "Soot_dia_Hp=0.50.csv")
            df.to_csv(dia_path, index=False)
            
            loader = DataLoader(tmpdir)
            loaded = loader.load_diameters(hp=0.50, dia_file=dia_path)
            
            assert len(loaded) == 20
            np.testing.assert_array_almost_equal(loaded, test_dia)
