"""
Data Loading Utilities for NN4Soot

This module provides utilities for loading and preprocessing soot 
simulation data and experimental measurements.

Author: Feixue Cai
"""

import os
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SootDataset:
    """Container for a complete soot dataset."""
    
    uf_set: int
    factors: np.ndarray
    psd_outputs: Dict[float, np.ndarray]  # {hp: psd_array}
    diameters: np.ndarray
    
    @property
    def n_samples(self) -> int:
        return self.factors.shape[0]
    
    @property
    def n_params(self) -> int:
        return self.factors.shape[1]


class DataLoader:
    """
    Data loader for soot simulation results.
    
    This class handles loading of:
    - Kinetic parameter samples (factors)
    - Soot PSD outputs at different heights
    - Diameter bins
    - Experimental reference data
    
    Parameters
    ----------
    data_dir : str
        Base directory for data files
    
    Examples
    --------
    >>> from nn4soot import DataLoader
    >>> loader = DataLoader("data/UF_set1")
    >>> dataset = loader.load_dataset(uf_set=1, hp_values=[0.50, 0.70, 1.00])
    """
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
    
    def load_factors(
        self,
        factors_file: Optional[str] = None,
        uf: int = 10,
        n_samples: int = 768,
    ) -> np.ndarray:
        """
        Load kinetic parameter samples.
        
        Parameters
        ----------
        factors_file : str, optional
            Path to factors file (auto-generated if not provided)
        uf : int
            UF set ID
        n_samples : int
            Number of samples
        
        Returns
        -------
        np.ndarray
            Factor array of shape (N, D)
        """
        if factors_file is None:
            factors_file = self.data_dir / f"SplitFactors_UF={uf}_N={n_samples}_noNH.txt"
        else:
            factors_file = Path(factors_file)
        
        factors = np.loadtxt(factors_file)
        return factors
    
    def load_psd_outputs(
        self,
        hp: float,
        psd_file: Optional[str] = None,
        log_transform: bool = True,
        bin_slice: slice = slice(3, 23),
    ) -> np.ndarray:
        """
        Load PSD output data for a specific height position.
        
        Parameters
        ----------
        hp : float
            Height position (cm)
        psd_file : str, optional
            Path to PSD file
        log_transform : bool
            Whether to apply log10 transform
        bin_slice : slice
            Slice for selecting BIN columns
        
        Returns
        -------
        np.ndarray
            PSD array of shape (N, n_bins)
        """
        if psd_file is None:
            psd_file = self.data_dir / f"Soot_PSDs_Hp={hp:.2f}.csv"
        else:
            psd_file = Path(psd_file)
        
        df = pd.read_csv(psd_file)
        outputs = df.values[:, bin_slice]
        
        if log_transform:
            outputs = np.log10(outputs)
        
        return outputs
    
    def load_diameters(
        self,
        hp: float,
        dia_file: Optional[str] = None,
        bin_slice: slice = slice(3, 23),
    ) -> np.ndarray:
        """
        Load diameter bins.
        
        Parameters
        ----------
        hp : float
            Height position
        dia_file : str, optional
            Path to diameter file
        bin_slice : slice
            Slice for selecting BIN columns
        
        Returns
        -------
        np.ndarray
            Diameter array in nm
        """
        if dia_file is None:
            dia_file = self.data_dir / f"Soot_dia_Hp={hp:.2f}.csv"
        else:
            dia_file = Path(dia_file)
        
        df = pd.read_csv(dia_file)
        dia = df.values[1, bin_slice].astype(float)
        
        return dia
    
    def load_dataset(
        self,
        uf_set: int,
        hp_values: List[float] = [0.50, 0.70, 1.00],
        n_samples: int = 768,
    ) -> SootDataset:
        """
        Load complete dataset for a UF set.
        
        Parameters
        ----------
        uf_set : int
            UF set ID (1, 2, or 3)
        hp_values : List[float]
            Height positions to load
        n_samples : int
            Number of samples
        
        Returns
        -------
        SootDataset
            Complete dataset
        """
        uf_map = {1: 2, 2: 5, 3: 10}
        uf = uf_map.get(uf_set, uf_set)
        
        factors = self.load_factors(uf=uf, n_samples=n_samples)
        
        psd_outputs = {}
        for hp in hp_values:
            psd_outputs[hp] = self.load_psd_outputs(hp)
        
        diameters = self.load_diameters(hp_values[0])
        
        return SootDataset(
            uf_set=uf_set,
            factors=factors,
            psd_outputs=psd_outputs,
            diameters=diameters,
        )
    
    def load_experimental_data(
        self,
        refs_file: str,
        hp: float,
        sheet_exp: str = "Exp",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load experimental reference data.
        
        Parameters
        ----------
        refs_file : str
            Path to Excel file with experimental data
        hp : float
            Height position
        sheet_exp : str
            Sheet name for experimental data
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Diameters and PSD values
        """
        hp_values = [0.50, 0.70, 1.00]
        i = int(np.argmin(np.abs(np.array(hp_values) - hp)))
        
        exp = pd.read_excel(refs_file, sheet_name=sheet_exp)
        d = exp.iloc[:, 2 * i].to_numpy(dtype=float)
        psd = exp.iloc[:, 2 * i + 1].to_numpy(dtype=float)
        
        m = np.isfinite(d) & np.isfinite(psd)
        return d[m], psd[m]
    
    def interpolate_exp_to_model_bins(
        self,
        exp_dia: np.ndarray,
        exp_psd: np.ndarray,
        model_dia: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolate experimental data to model bin positions.
        
        Parameters
        ----------
        exp_dia : np.ndarray
            Experimental diameters
        exp_psd : np.ndarray
            Experimental PSD values
        model_dia : np.ndarray
            Model bin diameters
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Filtered model diameters and interpolated PSD
        """
        max_exp_dia = np.max(exp_dia)
        min_exp_dia = np.min(exp_dia)
        
        mask = (model_dia >= min_exp_dia) & (model_dia <= max_exp_dia)
        model_dia_filtered = model_dia[mask]
        
        interp_psd = np.interp(model_dia_filtered, exp_dia, exp_psd)
        
        return model_dia_filtered, interp_psd
    
    @staticmethod
    def get_default_diameters() -> np.ndarray:
        """Get default diameter bins (nm)."""
        return np.array([
            1.785207, 2.248781333, 2.801218667, 3.463145333,
            4.316847333, 5.382750333, 6.713855667, 8.376489667,
            10.45364667, 13.04917, 16.29302667, 20.28754,
            25.34768, 31.67643, 39.59304, 49.49739,
            61.76184, 77.23861, 96.60935, 120.8569
        ])
