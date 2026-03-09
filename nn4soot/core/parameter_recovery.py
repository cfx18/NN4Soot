"""
Parameter Recovery for Soot Kinetic Optimization

Converts normalized parameter vectors (output of SootOptimizer, range [0, 1])
back to physical Arrhenius values, and writes the CFD/kinetic solver input
files (``Input_ArrheniusGlobal_*.txt`` + ``input_*.txt``) required for
follow-up simulations.

The inverse-transform logic mirrors the sampling scheme in
``a_gen_samples_split_newEaNew_noNH_UF10.py``:

  - **A parameters** (pre-exponential factor):
      ``sampled_data = normed * (A_upper - A_lower) + A_lower``
      ``true_A = A_base ** sampled_data * nominal``
  - **n parameters** (temperature exponent):
      ``true_n = (normed * (n_upper - n_lower) + n_lower) + nominal``
  - **Ea parameters** (activation energy):
      ``true_Ea = (normed * (Ea_upper - Ea_lower) + Ea_lower) * nominal``

Author: Feixue Cai
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


class ParamRecoveryConfig:
    """
    Per-UF parameter-type configuration for inverse normalisation.

    Parameters
    ----------
    A_indices : List[int]
        Column indices that correspond to pre-exponential (A) parameters.
    n_indices : List[int]
        Column indices for temperature-exponent (n) parameters.
    Ea_indices : List[int]
        Column indices for activation-energy (Ea) parameters.
    A_base : float
        Base for the A-factor log-scale: ``true_A = A_base**x * nominal``.
        Typically ``10.0`` (UF=10), ``2.0`` (UF=2), or ``5.0`` (UF=5).
    A_lower, A_upper : float
        Unnormalised range for A sampling (default ``-1, 1``).
    n_lower, n_upper : float
        Unnormalised range for n sampling (default ``-0.3, 0.3``).
    Ea_lower, Ea_upper : float
        Unnormalised range for Ea sampling (default ``0.8, 1.2``).
    """

    def __init__(
        self,
        A_indices: List[int],
        n_indices: List[int],
        Ea_indices: List[int],
        A_base: float = 10.0,
        A_lower: float = -1.0,
        A_upper: float = 1.0,
        n_lower: float = -0.3,
        n_upper: float = 0.3,
        Ea_lower: float = 0.8,
        Ea_upper: float = 1.2,
    ):
        self.A_indices = A_indices
        self.n_indices = n_indices
        self.Ea_indices = Ea_indices
        self.A_base = A_base
        self.A_lower = A_lower
        self.A_upper = A_upper
        self.n_lower = n_lower
        self.n_upper = n_upper
        self.Ea_lower = Ea_lower
        self.Ea_upper = Ea_upper

    @classmethod
    def from_dict(cls, d: dict) -> "ParamRecoveryConfig":
        """Construct from a plain dictionary (e.g. loaded from YAML)."""
        return cls(
            A_indices=list(d["A_indices"]),
            n_indices=list(d["n_indices"]),
            Ea_indices=list(d["Ea_indices"]),
            A_base=float(d.get("A_base", 10.0)),
            A_lower=float(d.get("A_lower", -1.0)),
            A_upper=float(d.get("A_upper", 1.0)),
            n_lower=float(d.get("n_lower", -0.3)),
            n_upper=float(d.get("n_upper", 0.3)),
            Ea_lower=float(d.get("Ea_lower", 0.8)),
            Ea_upper=float(d.get("Ea_upper", 1.2)),
        )


class ParameterRecovery:
    """
    Inverse-transform normalised optimisation results to physical values
    and write solver-ready input files.

    Examples
    --------
    >>> from nn4soot import ParameterRecovery, ParamRecoveryConfig
    >>> cfg = ParamRecoveryConfig(
    ...     A_indices=[0, 1, 2, 3, 6, 8],
    ...     n_indices=[7],
    ...     Ea_indices=[4, 5, 9],
    ...     A_base=10.0,
    ... )
    >>> recovery = ParameterRecovery(cfg)
    >>> nominal_dict, global_lines = recovery.load_nominal_dict("Input_ArrheniusGlobal_noNH.txt")
    >>> samples_recovered = recovery.recover(optimized_params.reshape(1, -1), nominal_dict)
    >>> recovery.write_input_files(
    ...     samples_recovered, nominal_dict, global_lines,
    ...     input_template_lines, output_dir=Path("results/opt_inputs"), uf=10
    ... )
    """

    def __init__(self, config: ParamRecoveryConfig):
        self.config = config

    # ------------------------------------------------------------------
    # Static helpers (inverse-transform formulae)
    # ------------------------------------------------------------------

    @staticmethod
    def _unnormalize(data_norm: np.ndarray, lower: float, upper: float) -> np.ndarray:
        """Map [0, 1] → [lower, upper]."""
        return data_norm * (upper - lower) + lower

    def _true_A(self, normed: np.ndarray, nominal: float) -> np.ndarray:
        sampled = self._unnormalize(normed, self.config.A_lower, self.config.A_upper)
        return (self.config.A_base ** sampled) * nominal

    def _true_n(self, normed: np.ndarray, nominal: float) -> np.ndarray:
        sampled = self._unnormalize(normed, self.config.n_lower, self.config.n_upper)
        return sampled + nominal

    def _true_Ea(self, normed: np.ndarray, nominal: float) -> np.ndarray:
        sampled = self._unnormalize(normed, self.config.Ea_lower, self.config.Ea_upper)
        return sampled * nominal

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load_nominal_dict(
        self, global_file: str | Path
    ) -> Tuple[Dict[str, float], List[str]]:
        """
        Parse an ``Input_ArrheniusGlobal_noNH.txt`` file.

        Returns
        -------
        nominal_dict : Dict[str, float]
            Ordered mapping of parameter name → nominal value.
        raw_lines : List[str]
            Original file lines (used as template when writing output files).
        """
        global_file = Path(global_file)
        nominal_dict: Dict[str, float] = {}
        with global_file.open("r", encoding="utf-8") as f:
            lines = f.readlines()
            in_block = False
            for line in lines:
                stripped = line.strip()
                if stripped == "@GlobalValue {":
                    in_block = True
                    continue
                if stripped == "}":
                    in_block = False
                    continue
                if in_block and not stripped.startswith("//") and len(stripped) > 2:
                    key, val = stripped.split(";")[0].split()
                    nominal_dict[key] = float(val)
        return nominal_dict, lines

    def recover(
        self,
        samples_norm: np.ndarray,
        nominal_dict: Dict[str, float],
    ) -> np.ndarray:
        """
        Convert normalised parameter matrix to physical (true) values.

        Parameters
        ----------
        samples_norm : np.ndarray, shape (N, D)
            Normalised samples in [0, 1].  Single sample should be passed as
            shape (1, D) or (D,) — 1-D arrays are automatically reshaped.
        nominal_dict : Dict[str, float]
            Ordered nominal values as returned by :meth:`load_nominal_dict`.

        Returns
        -------
        np.ndarray, shape (N, D)
            Physical parameter values.
        """
        samples_norm = np.asarray(samples_norm, dtype=float)
        if samples_norm.ndim == 1:
            samples_norm = samples_norm.reshape(1, -1)

        D = len(nominal_dict)
        if samples_norm.shape[1] != D:
            raise ValueError(
                f"samples_norm has {samples_norm.shape[1]} columns but "
                f"nominal_dict has {D} entries."
            )

        keys = list(nominal_dict.keys())
        N = samples_norm.shape[0]
        recovered = np.empty((N, D), dtype=float)

        for idx, key in enumerate(keys):
            nominal = nominal_dict[key]
            col = samples_norm[:, idx]
            if idx in self.config.A_indices:
                recovered[:, idx] = self._true_A(col, nominal)
            elif idx in self.config.n_indices:
                recovered[:, idx] = self._true_n(col, nominal)
            elif idx in self.config.Ea_indices:
                recovered[:, idx] = self._true_Ea(col, nominal)
            else:
                recovered[:, idx] = nominal  # fixed parameter → keep nominal

        return recovered

    def write_input_files(
        self,
        samples_recovered: np.ndarray,
        nominal_dict: Dict[str, float],
        global_template_lines: List[str],
        input_template_lines: List[str],
        output_dir: str | Path,
        uf: int,
        output_subdir_template: str = "Opt_UF={uf}_N={N}",
        global_filename_template: str = "Input_ArrheniusGlobal_{idx_str}.txt",
        input_filename_template: str = "input_{idx_str}.txt",
        soot_output_filename_template: str = "A_Soot_{idx_str}.txt",
    ) -> Path:
        """
        Write per-sample ``Input_ArrheniusGlobal_*.txt`` and ``input_*.txt``.

        The output directory is named
        ``Opt_UF={uf}_N={N}`` inside *output_dir*.

        Parameters
        ----------
        samples_recovered : np.ndarray, shape (N, D)
            Physical parameter values (output of :meth:`recover`).
        nominal_dict : Dict[str, float]
            Ordered parameter names (same object used to call :meth:`recover`).
        global_template_lines : List[str]
            Raw lines of the Arrhenius global template file.
        input_template_lines : List[str]
            Raw lines of the solver ``input.txt`` template file.
        output_dir : str or Path
            Root directory in which the sample sub-folder is created.
        uf : int
            UF value used in the folder and file naming.

        Returns
        -------
        Path
            Path to the created sample sub-folder.
        """
        samples_recovered = np.atleast_2d(samples_recovered)
        N = samples_recovered.shape[0]
        keys = list(nominal_dict.keys())

        outdir_name = output_subdir_template.format(uf=uf, N=N)
        outdir = Path(output_dir) / outdir_name
        outdir.mkdir(parents=True, exist_ok=True)

        for i, sample in enumerate(samples_recovered):
            idx_str = f"{i:03d}"
            fmt_ctx = {"uf": uf, "N": N, "idx": i, "idx_str": idx_str}
            global_out_name = global_filename_template.format(**fmt_ctx)
            input_out_name = input_filename_template.format(**fmt_ctx)
            soot_out_name = soot_output_filename_template.format(**fmt_ctx)
            global_out_path = outdir / global_out_name

            # -- Write Arrhenius global file --
            with global_out_path.open("w", encoding="utf-8") as f:
                in_block = False
                ivar = 0
                for line in global_template_lines:
                    stripped = line.strip()
                    if stripped == "@GlobalValue {":
                        in_block = True
                        f.write(line)
                        continue
                    if stripped == "}":
                        in_block = False
                        f.write(line)
                        continue
                    if in_block:
                        if stripped.startswith("//") or len(stripped) < 2:
                            continue
                        f.write(f"\t{keys[ivar]} \t\t {sample[ivar]:e};\n")
                        ivar += 1
                    else:
                        f.write(line)

            # -- Write input.txt (replace filenames) --
            new_input_lines: List[str] = []
            for line in input_template_lines:
                if "@OutputFileName" in line and "A_Soot.txt" in line:
                    new_input_lines.append(
                        line.replace("A_Soot.txt", soot_out_name)
                    )
                elif (
                    "#InputArrheniusParameter" in line
                    and "Input_ArrheniusGlobal.txt" in line
                ):
                    new_input_lines.append(
                        line.replace("Input_ArrheniusGlobal.txt", global_out_name)
                    )
                else:
                    new_input_lines.append(line)

            (outdir / input_out_name).write_text(
                "".join(new_input_lines), encoding="utf-8"
            )

        print(f"  Wrote {N} sample set(s) to: {outdir}")
        return outdir
