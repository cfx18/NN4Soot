"""
Train Models from YAML Config

Usage:
    python scripts/train_from_config.py --config configs/train_config.yaml

    # Run only specific UF set(s) (overrides enabled in yaml):
    python scripts/train_from_config.py --config configs/train_config.yaml --uf_sets UF_set1 UF_set2

    # Run only specific heights:
    python scripts/train_from_config.py --config configs/train_config.yaml --hp_values 0.50 1.00

Author: Feixue Cai
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from nn4soot import SootMLP, SootTrainer, SootEvaluator, TrainingConfig
from nn4soot.utils import DataLoader


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_training_config(global_cfg: dict, uf_cfg: dict, save_dir: str) -> TrainingConfig:
    lr_schedule = {int(k): float(v) for k, v in uf_cfg["lr_schedule"].items()}
    return TrainingConfig(
        save_dir=save_dir,
        num_epochs=global_cfg["num_epochs"],
        batch_size=global_cfg["batch_size"],
        test_size=global_cfg["test_size"],
        random_seed=global_cfg["random_seed"],
        initial_lr=lr_schedule[0],
        lr_schedule=lr_schedule,
        weight_decay=global_cfg["weight_decay"],
        early_stop_patience=global_cfg["early_stop_patience"],
        early_stop_min_delta=global_cfg["early_stop_min_delta"],
        checkpoint_name="best_MLP_Hp={hp:.2f}.pth",
    )



def main():
    parser = argparse.ArgumentParser(description="Train NN4Soot models from YAML config")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--hp_values", type=float, nargs="+", default=None)
    parser.add_argument("--uf_sets", type=str, nargs="+", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    global_cfg = cfg["global"]
    data_root = Path(cfg["data_dir"])
    output_root = Path(cfg["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)

    hp_values = args.hp_values if args.hp_values is not None else global_cfg["hp_values"]
    n_samples = global_cfg["n_samples"]

    # bin_slice: yaml stores [start, stop], convert to Python slice
    bin_start, bin_stop = global_cfg.get("bin_slice", [3, 23])
    bin_slice = slice(bin_start, bin_stop)

    uf_sets_to_run = (
        {k: v for k, v in cfg["uf_sets"].items() if k in args.uf_sets}
        if args.uf_sets
        else {k: v for k, v in cfg["uf_sets"].items() if v.get("enabled", True)}
    )

    if not uf_sets_to_run:
        print("No UF set to train (check enabled field or --uf_sets argument).")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluator = SootEvaluator(device=device)

    for uf_set_name, uf_cfg in uf_sets_to_run.items():
        uf_value = uf_cfg["uf_value"]
        data_dir = data_root / uf_set_name
        save_dir = output_root / uf_set_name
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"Training {uf_set_name}  (UF={uf_value})")
        print(f"  Data dir  : {data_dir}")
        print(f"  Output dir: {save_dir}")
        print(f"  BIN range : [{bin_start}, {bin_stop})  ({bin_stop - bin_start} BINs)")
        print(f"  LR schedule: {uf_cfg['lr_schedule']}")
        print(f"{'=' * 60}")

        loader = DataLoader(str(data_dir))

        for hp in hp_values:
            print(f"\n--- Hp={hp:.2f} cm ---")
            factors = loader.load_factors(uf=uf_value, n_samples=n_samples)
            outputs = loader.load_psd_outputs(hp, bin_slice=bin_slice)

            config = build_training_config(global_cfg, uf_cfg, str(save_dir))
            model = SootMLP(input_dim=factors.shape[1])
            trainer = SootTrainer(model, config)

            trainer.train(factors, outputs, hp, verbose=True)
            trainer.save_training_history(hp, n_samples, str(save_dir))
            trainer.plot_training_curves(hp, n_samples, str(save_dir))

            scatter_path = str(save_dir / f"scatter_Hp={hp:.2f}.png")
            evaluator.plot_prediction_scatter(
                model, factors, outputs, hp,
                save_path=scatter_path,
                test_size=config.test_size,
                random_seed=config.random_seed,
            )
            print(f"Model saved   : {save_dir / f'best_MLP_Hp={hp:.2f}.pth'}")
            print(f"Scatter saved: {scatter_path}")

    print("\nAll training completed!")


if __name__ == "__main__":
    main()
