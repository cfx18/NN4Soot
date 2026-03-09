"""
Run Sensitivity Analysis Script

This script performs comprehensive sensitivity analysis including:
- Active Subspace analysis for all BINs
- Valley sensitivity analysis
- Soft-valley global sensitivity analysis

Usage:
    python scripts/run_sensitivity_analysis.py --config configs/sensitivity_analysis_config.yaml
    python scripts/run_sensitivity_analysis.py --config configs/sensitivity_analysis_config.yaml --save_csv false

Author: Feixue Cai
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from nn4soot import (
    SootMLP,
    ActiveSubspaceAnalyzer,
    ValleyAnalyzer,
    CombinedPlotter,
)
from nn4soot.utils import DataLoader


def run_active_subspace_analysis(model, inputs, outputs, bin_indices, output_dir, hp, n_bootstrap=1000, save_csv=True):
    """Run Active Subspace analysis for multiple BINs."""
    print("\n--- Active Subspace Analysis ---")
    
    analyzer = ActiveSubspaceAnalyzer(model)
    results_dir = Path(output_dir) / "active_subspace"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_by_bin = {}
    
    for bin_idx in bin_indices:
        print(f"  Analyzing BIN {bin_idx}...")
        result = analyzer.analyze_with_bootstrap(inputs, bin_idx, n_bootstrap)
        results_by_bin[bin_idx] = result
        if save_csv:
            analyzer.save_results(result, str(results_dir), prefix=f"AS_BIN{bin_idx}")
        analyzer.plot_original_style_bin_figure(
            result,
            inputs=inputs,
            outputs=outputs,
            bin_index=bin_idx,
            hp=hp,
            save_path=str(results_dir / f"Eigen_Hp={hp:.2f}_BIN={bin_idx}.png"),
        )

    analyzer.plot_original_style_hp_summary(
        results_by_bin,
        hp=hp,
        save_path=str(results_dir / f"Sensitivity_Hp={hp:.2f}.png"),
    )


def run_valley_analysis(model, baseline_params, d_bins, exp_data, output_dir, save_csv=True):
    """Run valley sensitivity analysis."""
    print("\n--- Valley Sensitivity Analysis ---")
    
    analyzer = ValleyAnalyzer(model)
    results_dir = Path(output_dir) / "valley"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    peak_valley = analyzer.find_double_peaks_and_valley(exp_data['dia'], np.log10(exp_data['psd']))
    print(f"  Left peak: {peak_valley['left_peak_d_nm']:.2f} nm")
    print(f"  Right peak: {peak_valley['right_peak_d_nm']:.2f} nm")
    print(f"  Valley: {peak_valley['valley_d_nm']:.2f} nm")
    
    result = analyzer.analyze(baseline_params, d_bins, peak_valley)
    # Always save valley CSV for combined plot
    analyzer.save_results(result, str(results_dir / "valley_sensitivity_hp0.70_softmin.csv"))
    analyzer.plot_sensitivity(
        result,
        title="Valley Sensitivity",
        save_path=str(results_dir / "valley_sensitivity_hp0.70_softmin.png"),
    )


def run_soft_valley_as_analysis(model, inputs, d_bins, hp, output_dir, n_bootstrap=1000, save_csv=True):
    """Run original-style AS analysis for the soft valley index."""
    print("\n--- AS of Soft Valley Index ---")

    analyzer = ActiveSubspaceAnalyzer(model)
    results_dir = Path(output_dir) / "valley" / "Global"
    results_dir.mkdir(parents=True, exist_ok=True)

    result = analyzer.analyze_soft_valley_index_with_bootstrap(
        inputs=inputs,
        d_bins_nm=d_bins,
        beta_softmin=1.0,
        n_bootstrap=max(int(n_bootstrap), 2000),
        ci_level=0.95,
        seed=123,
        batch_size=256,
    )
    prefix = f"AS_valley_index_soft_hp{hp:.2f}"
    # Always save AS CSV for combined plot
    analyzer.save_results(result, str(results_dir), prefix=prefix)
    analyzer.plot_soft_valley_index_bundle(
        result=result,
        inputs=inputs,
        hp=hp,
        save_dir=str(results_dir),
        prefix=prefix,
    )
    print(f"  Saved soft-valley AS results to: {results_dir}")


def run_combined_sensitivity_plots(output_dir):
    """Create legacy-style combined plot using AS global sensitivity and valley local sensitivity."""
    print("\n--- Combined Global + Local Plot ---")

    output_dir = Path(output_dir)
    valley_dir = output_dir / "valley"
    valley_csv = valley_dir / "valley_sensitivity_hp0.70_softmin.csv"
    as_csv = valley_dir / "Global" / "AS_valley_index_soft_hp0.70_sensitivity.csv"

    if not valley_csv.exists():
        print("  Skipping combined plots: valley result is missing.")
        return

    valley_df = pd.read_csv(valley_csv)
    plotter = CombinedPlotter()

    if as_csv.exists():
        as_df = pd.read_csv(as_csv)
        as_path = valley_dir / "AS_and_dvalley_hp0.70.png"
        plotter.plot_global_and_dvalley(
            global_df=as_df,
            valley_df=valley_df,
            save_path=str(as_path),
            global_col="boot_mean",
            global_low_col="ci_low",
            global_high_col="ci_high",
        )
        print(f"  Saved: {as_path}")


def main():
    parser = argparse.ArgumentParser(description="Run sensitivity analysis")
    parser.add_argument("--config", type=str, default="configs/sensitivity_analysis_config.yaml",
                        help="Path to YAML configuration file")
    parser.add_argument("--save_csv", type=str, default=None,
                        help="Override save_csv from config (true/false)")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    data_dir = config.get('data_dir', 'data/')
    model_dir = config.get('model_dir', 'pretrained/')
    output_dir = config.get('output_dir', 'results/')
    
    uf_sets_config = config.get('uf_sets', {})
    analysis_config = config.get('analysis', {})
    output_config = config.get('output', {})
    
    uf_sets = analysis_config.get('uf_sets', [1, 2, 3])
    hp_values = analysis_config.get('hp_values', [0.50, 0.70, 1.00])
    n_bootstrap = analysis_config.get('n_bootstrap', 1000)
    bin_indices = analysis_config.get('bin_indices', list(range(6, 21)))
    
    save_csv = output_config.get('save_csv', True)
    if args.save_csv is not None:
        save_csv = args.save_csv.lower() == 'true'
    
    uf_dir_map = {1: "UF_set1", 2: "UF_set2", 3: "UF_set3"}
    for uf in uf_sets:
        
        if uf not in uf_dir_map:
            print(f"Skipping unknown UF set: {uf}")
            continue
        
        uf_key = f"UF_set{uf}"
        uf_config = uf_sets_config.get(uf_key, {})
        if not uf_config.get('enabled', True):
            print(f"Skipping disabled UF set: {uf_key}")
            continue
        
        uf_value = uf_config.get('uf_value', {1: 2, 2: 5, 3: 10}.get(uf, 2))
        run_valley = uf_config.get('run_valley', False)
        
        print(f"\n--- UF Set {uf} ---")

        data_path = Path(data_dir) / uf_dir_map[uf]

        for hp in hp_values:
            print(f"\n--- Hp={hp:.2f} cm ---")
            model_path = Path(model_dir) / f"UF_set{uf}" / f"best_MLP_Hp={hp:.2f}.pth"
            
            print(f"Loading data from {data_path}")
            loader = DataLoader(str(data_path))
            
            inputs = loader.load_factors(uf=uf_value)
            outputs = loader.load_psd_outputs(hp)
            d_bins = loader.load_diameters(hp)
            
            print(f"Loading model from {model_path}")
            model = SootMLP.from_pretrained(str(model_path))
            
            output_dir_path = Path(output_dir) / f"UF_set{uf}" / f"Hp{hp:.2f}"
            output_dir_path.mkdir(parents=True, exist_ok=True)
            
            run_active_subspace_analysis(model, inputs, outputs, bin_indices, str(output_dir_path), hp, n_bootstrap, save_csv)
            
            if run_valley:
                if abs(hp - 0.70) < 1e-6:
                    refs_file = data_path / "Refs.xlsx"
                    exp_data = None
                    if refs_file.exists():
                        exp_dia, exp_psd = loader.load_experimental_data(str(refs_file), hp)
                        exp_data = {'dia': exp_dia, 'psd': exp_psd}
                    else:
                        interp_file = data_path / f"Soot_Exp_interpolated_psds_Hp={hp:.2f}.csv"
                        if interp_file.exists():
                            interp_df = np.asarray(np.genfromtxt(interp_file, delimiter=",", skip_header=1))
                            if interp_df.ndim == 2 and interp_df.shape[1] >= 2:
                                exp_data = {'dia': interp_df[:, 0], 'psd': interp_df[:, 1]}
                    if exp_data is not None:
                        baseline = np.full(inputs.shape[1], 0.5)
                        run_valley_analysis(model, baseline, d_bins, exp_data, str(output_dir_path), save_csv)
                
                if abs(hp - 0.70) < 1e-6:
                    run_soft_valley_as_analysis(model, inputs, d_bins, hp, str(output_dir_path), n_bootstrap, save_csv)
                    run_combined_sensitivity_plots(str(output_dir_path))
            
            print(f"\nSensitivity analysis complete! Results saved to: {output_dir_path}")


if __name__ == "__main__":
    main()