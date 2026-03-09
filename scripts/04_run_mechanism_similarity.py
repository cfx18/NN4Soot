"""
Run Mechanism Similarity Analysis Script

Compares Active-Subspace sensitivity patterns across different UF parameter
sets to assess mechanism invariance, replicating the analysis in
compare_UF_mechanism_similarity.py using the nn4soot module interface.

Outputs
-------
- UF_Mechanism_Similarity_L2Sim01_byBIN.xlsx   pairwise similarity per BIN
- UF_Mechanism_Similarity_OneFigure_L2Sim01.png compact barh publication figure
- UF_Top{k}_Params_MeanSensitivity_overBINxHp.xlsx  top-k params per UF set
- UF_Top{k}_Params_MeanSensitivity_overBINxHp.png   top-k params bar chart
- UF_mechanism_similarity_cache.npz            sensitivity map cache (optional)

Usage
-----
    python scripts/run_mechanism_similarity.py
    python scripts/run_mechanism_similarity.py --config configs/mechanism_similarity_config.yaml
    python scripts/run_mechanism_similarity.py --no_cache

Author: Feixue Cai
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from nn4soot import SootMLP, MechanismSimilarityAnalyzer
from nn4soot.utils import DataLoader


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_models(model_dir: Path, uf_sets_cfg: dict, hp_values: list, device: torch.device):
    """Load one SootMLP per (UF set, hp) combination.

    Returns
    -------
    models_by_uf_hp : Dict[(uf_value, hp), SootMLP]
    """
    models_by_uf_hp = {}
    for set_key, set_cfg in uf_sets_cfg.items():
        uf_val = set_cfg["uf_value"]
        for hp in hp_values:
            ckpt = model_dir / set_key / f"best_MLP_Hp={hp:.2f}.pth"
            if not ckpt.exists():
                raise FileNotFoundError(f"Model checkpoint not found: {ckpt}")
            model = SootMLP.from_pretrained(str(ckpt))
            model.to(device)
            model.eval()
            models_by_uf_hp[(uf_val, hp)] = model
            print(f"  Loaded model: {ckpt.name}  (UF={uf_val}, Hp={hp:.2f})")
    return models_by_uf_hp


def load_inputs(data_dir: Path, uf_sets_cfg: dict) -> dict:
    """Load parameter samples for each UF set.

    Returns
    -------
    inputs_by_uf : Dict[uf_value, np.ndarray]
    """
    inputs_by_uf = {}
    for set_key, set_cfg in uf_sets_cfg.items():
        uf_val = set_cfg["uf_value"]
        loader = DataLoader(str(data_dir / set_key))
        inputs = loader.load_factors(uf=uf_val)
        inputs_by_uf[uf_val] = inputs
        print(f"  Loaded inputs: {set_key}  shape={inputs.shape}")
    return inputs_by_uf


def main():
    parser = argparse.ArgumentParser(description="Run mechanism similarity analysis")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/mechanism_similarity_config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--no_cache",
        action="store_true",
        help="Ignore existing cache and recompute sensitivity map",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    data_dir = Path(cfg.get("data_dir", "data/"))
    model_dir = Path(cfg.get("model_dir", "pretrained/"))
    output_dir = Path(cfg.get("output_dir", "results/UF_Comparison_Results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis_cfg = cfg.get("analysis", {})
    hp_values = analysis_cfg.get("hp_values", [0.50, 0.70, 1.00])
    bin_indices = analysis_cfg.get("bin_indices", list(range(5, 21)))
    n_bootstrap = analysis_cfg.get("n_bootstrap", 1000)
    top_k = analysis_cfg.get("top_k", 4)
    uf_pairs = [tuple(p) for p in analysis_cfg.get("uf_pairs", [[2, 10], [5, 10], [2, 5]])]
    pair_labels = analysis_cfg.get("pair_labels", [f"UF{a}-UF{b}" for a, b in uf_pairs])

    uf_sets_cfg = cfg.get("uf_sets", {
        "UF_set1": {"uf_value": 2},
        "UF_set2": {"uf_value": 5},
        "UF_set3": {"uf_value": 10},
    })
    uf_values = [v["uf_value"] for v in uf_sets_cfg.values()]
    uf_labels = {
        cfg_val["uf_value"]: set_key.replace("_", " ")
        for set_key, cfg_val in uf_sets_cfg.items()
    }

    cache_cfg = cfg.get("cache", {})
    use_cache = cache_cfg.get("use_cache", True) and not args.no_cache
    cache_file = Path(cache_cfg.get("cache_file", str(output_dir / "UF_mechanism_similarity_cache.npz")))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    analyzer = MechanismSimilarityAnalyzer(device=device)

    # ── 1. Sensitivity map ────────────────────────────────────────────────────
    print("\n--- Computing / loading sensitivity map ---")
    if use_cache and cache_file.exists():
        cached = np.load(cache_file, allow_pickle=True)
        sens_map = cached["sens_map"].item()
        print(f"  Loaded cache: {cache_file}")
    else:
        print("  Loading models...")
        models_by_uf_hp = build_models(model_dir, uf_sets_cfg, hp_values, device)
        print("  Loading inputs...")
        inputs_by_uf = load_inputs(data_dir, uf_sets_cfg)

        print("  Computing sensitivity vectors (this may take a while)...")
        sens_map = analyzer.compute_sensitivity_map(
            models_by_uf_hp=models_by_uf_hp,
            inputs_by_uf=inputs_by_uf,
            bin_indices=bin_indices,
        )
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_file, sens_map=sens_map)
        print(f"  Saved cache: {cache_file}")

    # ── 2. Pairwise similarity ────────────────────────────────────────────────
    print("\n--- Pairwise L2 similarity ---")
    sim_df = analyzer.compare_from_sens_map(
        sens_map=sens_map,
        uf_pairs=uf_pairs,
        hp_values=hp_values,
        bin_indices=bin_indices,
    )

    sim_xlsx = output_dir / "UF_Mechanism_Similarity_L2Sim01_byBIN.xlsx"
    sim_df.to_excel(str(sim_xlsx), index=False)
    print(f"  Saved: {sim_xlsx}")

    # ── 3. Similarity figure ──────────────────────────────────────────────────
    print("\n--- Similarity figure ---")
    sim_fig = output_dir / "UF_Mechanism_Similarity_OneFigure_L2Sim01.png"
    analyzer.plot_similarity_barh(
        df=sim_df,
        uf_pairs=uf_pairs,
        hp_values=hp_values,
        pair_labels=pair_labels,
        save_path=str(sim_fig),
    )
    print(f"  Saved: {sim_fig}")

    # ── 4. Top-k parameters ───────────────────────────────────────────────────
    print("\n--- Top-k parameter analysis ---")
    df_topk = analyzer.compute_top_k_from_sens_map(
        sens_map=sens_map,
        uf_values=uf_values,
        hp_values=hp_values,
        bin_indices=bin_indices,
        top_k=top_k,
        n_bootstrap=n_bootstrap,
    )

    topk_xlsx = output_dir / f"UF_Top{top_k}_Params_MeanSensitivity_overBINxHp.xlsx"
    df_topk.to_excel(str(topk_xlsx), index=False)
    print(f"  Saved: {topk_xlsx}")

    topk_fig = output_dir / f"UF_Top{top_k}_Params_MeanSensitivity_overBINxHp.png"
    analyzer.plot_top_k_params(
        df_topk=df_topk,
        uf_order=uf_values,
        uf_labels=uf_labels,
        save_path=str(topk_fig),
        top_k=top_k,
    )
    print(f"  Saved: {topk_fig}")

    print("\nDone. All outputs saved to:", output_dir)


if __name__ == "__main__":
    main()
