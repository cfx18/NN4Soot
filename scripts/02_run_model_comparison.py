"""
Run Model Comparison Script

This script compares neural network models with baseline regression models
(Linear, Polynomial) across all UF sets and height positions.

Usage:
    python scripts/run_model_comparison.py --data_dir data/ --model_dir pretrained/

Author: Feixue Cai
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from nn4soot import SootMLP, ModelComparator
from nn4soot.utils import DataLoader


def main():
    parser = argparse.ArgumentParser(description="Run model comparison")
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--model_dir", type=str, default="pretrained/")
    parser.add_argument("--output_dir", type=str, default="results/")
    parser.add_argument("--uf_sets", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--hp_values", type=float, nargs="+", default=[0.50, 0.70, 1.00])
    args = parser.parse_args()
    
    uf_dir_map = {1: "UF_set1", 2: "UF_set2", 3: "UF_set3"}
    uf_values = {1: 2, 2: 5, 3: 10}
    
    comparator = ModelComparator()
    all_results = []
    
    for uf in args.uf_sets:
        data_path = Path(args.data_dir) / uf_dir_map[uf]
        model_dir = Path(args.model_dir) / f"UF_set{uf}"
        
        print(f"\n--- UF Set {uf} ---")
        
        loader = DataLoader(str(data_path))
        inputs = loader.load_factors(uf=uf_values[uf])
        
        for hp in args.hp_values:
            model_path = model_dir / f"best_MLP_Hp={hp:.2f}.pth"
            
            if not model_path.exists():
                print(f"  Model not found: {model_path}")
                continue
            
            outputs = loader.load_psd_outputs(hp)
            model = SootMLP.from_pretrained(str(model_path))
            
            results = comparator.compare_single_case(model, inputs, outputs, uf, hp)
            all_results.extend(results)
            
            for r in results:
                print(f"  Hp={hp:.2f}, {r.model_name}: Test R2={r.test_r2:.4f}, Test MSE={r.test_mse:.6f}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    comparator.save_results(all_results, str(output_dir / "model_comparison.xlsx"))
    comparator.plot_mse_comparison(all_results, str(output_dir / "model_comparison.png"))
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()