"""
Run the full NN4Soot reproduction pipeline.

This orchestrator follows the same order as the numbered scripts in
`scripts/`:
1. `01_train_from_config.py`
2. `02_run_model_comparison.py`
3. `03_run_sensitivity_analysis.py`
4. `04_run_mechanism_similarity.py`
5. `05_run_optimization.py`

Usage:
    python reproduce_all.py
    python reproduce_all.py --skip_pretraining
    python reproduce_all.py --optimization_configs \
        configs/optimization_no_softmin_config.yaml \
        configs/optimization_softmin_config.yaml
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT_DIR / "scripts"


def _root_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return ROOT_DIR / path


def _run_step(step_idx: int, total_steps: int, label: str, command: list[str]) -> None:
    print(f"\n[Step {step_idx}/{total_steps}] {label}")
    print("Command:", " ".join(command))
    subprocess.run(command, check=True, cwd=ROOT_DIR)


def build_commands(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    commands: list[tuple[str, list[str]]] = []

    if not args.skip_pretraining:
        commands.append(
            (
                "Pretraining",
                [
                    sys.executable,
                    str(SCRIPTS_DIR / "01_train_from_config.py"),
                    "--config",
                    str(_root_path(args.train_config)),
                ],
            )
        )

    commands.append(
        (
            "Model comparison",
            [
                sys.executable,
                str(SCRIPTS_DIR / "02_run_model_comparison.py"),
                "--data_dir",
                args.data_dir,
                "--model_dir",
                args.model_dir,
                "--output_dir",
                args.output_dir,
            ],
        )
    )

    commands.append(
        (
            "Sensitivity analysis",
            [
                sys.executable,
                str(SCRIPTS_DIR / "03_run_sensitivity_analysis.py"),
                "--config",
                str(_root_path(args.sensitivity_config)),
            ],
        )
    )

    commands.append(
        (
            "Mechanism similarity analysis",
            [
                sys.executable,
                str(SCRIPTS_DIR / "04_run_mechanism_similarity.py"),
                "--config",
                str(_root_path(args.mechanism_config)),
            ],
        )
    )

    for config_path in args.optimization_configs:
        config_name = Path(config_path).name
        commands.append(
            (
                f"Optimization ({config_name})",
                [
                    sys.executable,
                    str(SCRIPTS_DIR / "05_run_optimization.py"),
                    "--config",
                    str(_root_path(config_path)),
                ],
            )
        )

    return commands


def main(default_skip_pretraining: bool = False) -> None:
    parser = argparse.ArgumentParser(
        description="Run the full NN4Soot reproduction pipeline."
    )
    parser.add_argument(
        "--train_config",
        type=str,
        default="configs/train_config.yaml",
        help="Config used by scripts/01_train_from_config.py",
    )
    parser.add_argument(
        "--sensitivity_config",
        type=str,
        default="configs/sensitivity_analysis_config.yaml",
        help="Config used by scripts/03_run_sensitivity_analysis.py",
    )
    parser.add_argument(
        "--mechanism_config",
        type=str,
        default="configs/mechanism_similarity_config.yaml",
        help="Config used by scripts/04_run_mechanism_similarity.py",
    )
    parser.add_argument(
        "--optimization_configs",
        type=str,
        nargs="+",
        default=[
            "configs/optimization_no_softmin_config.yaml",
            "configs/optimization_softmin_config.yaml",
        ],
        help="One or more configs used by scripts/05_run_optimization.py",
    )
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--model_dir", type=str, default="pretrained/")
    parser.add_argument("--output_dir", type=str, default="results/")

    skip_group = parser.add_mutually_exclusive_group()
    parser.set_defaults(skip_pretraining=default_skip_pretraining)
    skip_group.add_argument(
        "--skip_pretraining",
        dest="skip_pretraining",
        action="store_true",
        help="Skip scripts/01_train_from_config.py",
    )
    skip_group.add_argument(
        "--run_pretraining",
        dest="skip_pretraining",
        action="store_false",
        help="Force running scripts/01_train_from_config.py",
    )

    args = parser.parse_args()

    print("=" * 72)
    print("NN4Soot: Reproducing Results")
    print("=" * 72)
    print(f"Project root       : {ROOT_DIR}")
    print(f"Skip pretraining   : {args.skip_pretraining}")
    print(f"Optimization runs  : {len(args.optimization_configs)}")

    commands = build_commands(args)
    total_steps = len(commands)

    for idx, (label, command) in enumerate(commands, start=1):
        _run_step(idx, total_steps, label, command)

    print("\nAll requested steps completed.")


if __name__ == "__main__":
    main()
