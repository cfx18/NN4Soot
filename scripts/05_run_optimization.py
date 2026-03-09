"""
Run Optimization Script

Gradient-based optimization of kinetic parameters to fit experimental soot
PSD data, followed by inverse-transform of the normalised result back to
physical Arrhenius values and generation of solver-ready input files.

Usage
-----
    python scripts/run_optimization.py
    python scripts/run_optimization.py --config configs/optimization_config.yaml

Author: Feixue Cai
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from nn4soot import (
    SootMLP,
    SootOptimizer,
    OptimizerConfig,
    ParameterRecovery,
    ParamRecoveryConfig,
    KineticRunner,
    KineticRunnerConfig,
)
from nn4soot.utils import DataLoader


# ──────────────────────────────────────────────────────────────────────────────
# Config loading
# ──────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────────────────────────────────────
# Parameter recovery helper
# ──────────────────────────────────────────────────────────────────────────────

def run_parameter_recovery(
    optimized_params: np.ndarray,
    recovery_cfg: dict,
    uf_set_key: str,
    output_dir: Path,
) -> None:
    """Inverse-transform optimised params and write solver input files."""
    set_cfg = recovery_cfg.get(uf_set_key, {})
    if not set_cfg:
        print(f"  [recovery] No recovery config found for '{uf_set_key}', skipping.")
        return

    global_file = Path(set_cfg["global_file"])
    input_file = Path(set_cfg["input_file"])

    if not global_file.exists():
        print(f"  [recovery] global_file not found: {global_file}  — skipping.")
        return
    if not input_file.exists():
        print(f"  [recovery] input_file not found: {input_file}  — skipping.")
        return

    param_cfg = ParamRecoveryConfig.from_dict(set_cfg)
    recovery = ParameterRecovery(param_cfg)

    nominal_dict, global_lines = recovery.load_nominal_dict(global_file)
    input_template_lines = input_file.read_text(encoding="utf-8").splitlines(keepends=True)

    D = len(nominal_dict)
    if optimized_params.shape[0] != D:
        print(
            f"  [recovery] Parameter dimension mismatch: "
            f"optimized_params has {optimized_params.shape[0]} entries, "
            f"but nominal_dict has {D}.  Skipping."
        )
        return

    samples_recovered = recovery.recover(
        optimized_params.reshape(1, -1), nominal_dict
    )

    uf_val = int(set_cfg["uf_value"])
    recovery.write_input_files(
        samples_recovered,
        nominal_dict,
        global_lines,
        input_template_lines,
        output_dir=output_dir,
        uf=uf_val,
        output_subdir_template=set_cfg.get("output_subdir_template", "Opt_UF={uf}_N={N}"),
        global_filename_template=set_cfg.get(
            "global_filename_template", "Input_ArrheniusGlobal_{idx_str}.txt"
        ),
        input_filename_template=set_cfg.get(
            "input_filename_template", "input_{idx_str}.txt"
        ),
        soot_output_filename_template=set_cfg.get(
            "soot_output_filename_template", "A_Soot_{idx_str}.txt"
        ),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Kinetics runner helper
# ──────────────────────────────────────────────────────────────────────────────

def run_kinetics(
    sootgen_cfg: dict,
    recovery_out_dir: Path,
) -> None:
    """
    Run SootGEN (and optionally gen_kinetics) inside *recovery_out_dir*.

    Parameters
    ----------
    sootgen_cfg : dict
        The ``sootgen`` sub-section from the YAML config.
    recovery_out_dir : Path
        Directory where ``input_*.txt`` / ``Input_ArrheniusGlobal_*.txt``
        were written by parameter recovery (e.g. ``opt_UF=2_N=1/``).
    """
    if not sootgen_cfg.get("enabled", False):
        return

    runner_config = KineticRunnerConfig.from_dict(sootgen_cfg)
    runner = KineticRunner(runner_config)

    print("\n--- SootGEN ---")
    produced = runner.run_sootgen(recovery_out_dir)

    if runner_config.gen_kinetics.enabled:
        print("\n--- gen_kinetics ---")
        runner.run_gen_kinetics(recovery_out_dir, soot_files=produced)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def run_single_uf(cfg: dict, uf_set: int) -> None:
    """Run the full optimization pipeline for one UF set."""
    uf_dir_map = {1: "UF_set1", 2: "UF_set2", 3: "UF_set3"}
    uf_values_map = {1: 2, 2: 5, 3: 10}

    uf_set_key = uf_dir_map[uf_set]
    uf_val = uf_values_map[uf_set]

    data_dir = Path(cfg.get("data_dir", "data/"))
    model_dir = Path(cfg.get("model_dir", "pretrained/"))
    output_dir = Path(cfg.get("output_dir", "results/"))

    data_path = data_dir / uf_set_key
    model_subdir = model_dir / uf_set_key

    # ── Optimization config ────────────────────────────────────────────────
    opt_cfg = cfg.get("optimization", {})
    hp_values = list(opt_cfg.get("hp_values", [0.50, 0.70, 1.00]))
    optimize_params = list(opt_cfg.get("optimize_params", [0, 1]))
    num_epochs = int(opt_cfg.get("num_epochs", 1500))
    initial_lr = float(opt_cfg.get("initial_lr", 0.01))
    early_stop_patience = int(opt_cfg.get("early_stop_patience", 20))
    early_stop_min_delta = float(opt_cfg.get("early_stop_min_delta", 1e-6))
    original_setting = bool(opt_cfg.get("original_setting", False))
    use_softmin = bool(opt_cfg.get("use_softmin", False))
    softmin_hp = float(opt_cfg.get("softmin_hp", 0.70))
    softmin_bin_index = int(opt_cfg.get("softmin_bin_index", 5))
    softmin_weight = float(opt_cfg.get("softmin_weight", 100.0))

    print(f"\n{'='*60}")
    print(f"  UF set {uf_set}  ({uf_set_key}, UF={uf_val})")
    print(f"{'='*60}")

    # ── Load data ──────────────────────────────────────────────────────────
    print(f"Loading data from {data_path}")
    loader = DataLoader(str(data_path))
    inputs = loader.load_factors(uf=uf_val)
    d_bins = loader.load_diameters(hp_values[0])

    exp_data_dict = {}
    for hp in hp_values:
        interp_file = data_path / f"Soot_Exp_interpolated_psds_Hp={hp:.2f}.csv"
        if interp_file.exists():
            raw = np.genfromtxt(str(interp_file), delimiter=",", skip_header=1)
            if raw.ndim == 2 and raw.shape[1] >= 2:
                exp_data_dict[hp] = {"dia": raw[:, 0], "psd": raw[:, 1]}

    if not exp_data_dict:
        print("  Error: no experimental data found — skipping this UF set.")
        return

    # ── Load per-hp models (each Hp uses its own checkpoint) ──────────────
    models_by_hp: dict = {}
    for hp in hp_values:
        p = model_subdir / f"best_MLP_Hp={hp:.2f}.pth"
        if p.exists():
            models_by_hp[hp] = SootMLP.from_pretrained(str(p))
            print(f"  Loaded model for Hp={hp:.2f}: {p}")
        else:
            print(f"  Warning: model not found for Hp={hp:.2f} ({p}), skipping.")

    if not models_by_hp:
        print("  Error: no per-Hp models found — skipping this UF set.")
        return

    # SootOptimizer requires a model arg; use first loaded as placeholder (actual training uses models_by_hp)
    _fallback_model = next(iter(models_by_hp.values()))

    # ── Output directory ───────────────────────────────────────────────────
    opt_out_dir = output_dir / uf_set_key
    opt_out_dir.mkdir(parents=True, exist_ok=True)

    # ── Interpolate experimental data to model bins ────────────────────────
    interp_exp_data: dict = {}
    bin_ranges: dict = {}

    for hp, exp_data in exp_data_dict.items():
        d_filt, psd_interp = loader.interpolate_exp_to_model_bins(
            exp_data["dia"], exp_data["psd"], d_bins
        )
        interp_exp_data[hp] = {"dia": d_filt, "psd": psd_interp}

        if original_setting:
            _defaults = {0.50: (1, 12), 0.70: (2, 14), 1.00: (2, 16)}
            _cfg_ranges = opt_cfg.get("bin_ranges", {})
            _key = round(hp, 2)
            if _key in _cfg_ranges:
                lo_hi = _cfg_ranges[_key]
                bin_ranges[hp] = (int(lo_hi[0]), int(lo_hi[1]))
            else:
                bin_ranges[hp] = _defaults.get(_key, (0, len(d_bins)))
        else:
            lo = int(np.argmin(np.abs(d_bins - d_filt[0])))
            hi = int(np.argmin(np.abs(d_bins - d_filt[-1])))
            bin_ranges[hp] = (lo, hi)

    # ── Run optimization ───────────────────────────────────────────────────
    initial_params = np.full(inputs.shape[1], 0.5)

    opt_config = OptimizerConfig(
        num_epochs=num_epochs,
        initial_lr=initial_lr,
        early_stop_patience=early_stop_patience,
        early_stop_min_delta=early_stop_min_delta,
        softmin_weight=softmin_weight,
        save_dir=str(opt_out_dir),
    )
    optimizer = SootOptimizer(_fallback_model, opt_config)

    softmin_config = None
    if use_softmin:
        softmin_config = {"hp": softmin_hp, "bin_index": softmin_bin_index}

    print(f"\nOptimizing parameters: {[f'P{i+1}' for i in optimize_params]}")
    optimized_params, history = optimizer.optimize(
        interp_exp_data,
        initial_params,
        optimize_params,
        bin_ranges=bin_ranges,
        use_softmin=use_softmin,
        softmin_config=softmin_config,
        verbose=True,
        models_by_hp=models_by_hp,
    )

    # ── Save optimization CSV ──────────────────────────────────────────────
    results_csv = opt_out_dir / "optimization_results.csv"
    optimizer.save_results(
        optimized_params, history["loss"], history["params"], str(results_csv)
    )
    print(f"\nOptimization results saved to: {results_csv}")

    # ── Print summary ──────────────────────────────────────────────────────
    print("\nOptimized parameters (normalised [0,1]):")
    for i, p in enumerate(optimized_params):
        marker = " *" if i in optimize_params else ""
        print(f"  P{i+1}: {p:.6f}{marker}")

    # ── Plot optimisation result ───────────────────────────────────────────
    print("\n--- Plotting optimisation result ---")
    plot_exp = {hp: {"dia": d["dia"], "psd": d["psd"]}
                for hp, d in exp_data_dict.items()}
    opt_fig = opt_out_dir / "optimization_result.png"
    optimizer.plot_optimization_result(
        exp_data_dict=plot_exp,
        param_history=history["params"],
        model_dia=d_bins,
        save_path=str(opt_fig),
        hp_values=hp_values,
        models_by_hp=models_by_hp,
    )
    print(f"  Saved: {opt_fig}")

    # ── Parameter recovery → write solver input files ──────────────────────
    recovery_cfg = cfg.get("recovery", {})
    recovery_out_dir: Optional[Path] = None
    if recovery_cfg and uf_set_key in recovery_cfg:
        print("\n--- Parameter Recovery ---")
        set_cfg = recovery_cfg[uf_set_key]
        subdir_template = set_cfg.get("output_subdir_template", "Opt_UF={uf}_N={N}")
        uf_val_rec = int(set_cfg.get("uf_value", uf_val))
        recovery_out_dir = opt_out_dir / subdir_template.format(uf=uf_val_rec, N=1)
        run_parameter_recovery(
            optimized_params=optimized_params,
            recovery_cfg=recovery_cfg,
            uf_set_key=uf_set_key,
            output_dir=opt_out_dir,
        )
    else:
        print(
            "\n[recovery] No recovery config provided for this UF set — "
            "physical input files not generated."
        )

    # ── SootGEN + gen_kinetics (optional) ─────────────────────────────────
    sootgen_cfg = cfg.get("sootgen", {})
    if sootgen_cfg.get("enabled", False) and recovery_out_dir is not None:
        run_kinetics(sootgen_cfg, recovery_out_dir)
    elif sootgen_cfg.get("enabled", False) and recovery_out_dir is None:
        print("\n[sootgen] Skipped: parameter recovery was not run for this UF set.")

    print(f"\nAll outputs saved to: {opt_out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run parameter optimization")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/optimization_config.yaml",
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    # uf_set can be a single int or list, e.g. 3 or [1, 2, 3]
    raw_uf = cfg.get("uf_set", 3)
    uf_sets = raw_uf if isinstance(raw_uf, list) else [int(raw_uf)]

    for uf_set in uf_sets:
        run_single_uf(cfg, int(uf_set))


if __name__ == "__main__":
    main()
