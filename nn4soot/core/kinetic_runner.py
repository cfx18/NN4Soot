"""
KineticRunner: Run SootGEN and generate kinetics input files.

After parameter recovery writes ``input_*.txt`` / ``Input_ArrheniusGlobal_*.txt``
into the output directory, this module runs two optional post-processing steps:

  1. **SootGEN** — for every ``input_{idx}.txt`` in the output folder, rename it
     to ``input.txt`` (what SootGEN expects), execute the SootGEN binary, then
     rename it back.  SootGEN reads the ``@OutputFileName`` directive inside the
     file and therefore writes ``A_Soot_{idx}.txt`` directly.

  2. **gen_kinetics** — for every ``A_Soot_{idx}.txt`` produced in step 1,
     patch its kinetics coefficients into a copy of a CKI template folder and
     optionally update the ``input.dic`` output label.

Author: Feixue Cai
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Config dataclasses
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GenKineticsConfig:
    """Configuration for the CKI-patching step."""

    enabled: bool = False
    template_folder: str = ""
    target_file_name: str = "kinetics.CHEMKIN_base3pp.CKI"
    input_dic_file_name: str = "input.dic"
    output_folder_pattern: str = "Kinetics_inputs_{index}"
    output_label_prefix: str = "kinetics-base3"
    read_start_line: int = 197
    write_start_line: int = 18088

    @classmethod
    def from_dict(cls, d: dict) -> "GenKineticsConfig":
        return cls(
            enabled=bool(d.get("enabled", False)),
            template_folder=str(d.get("template_folder", "")),
            target_file_name=str(d.get("target_file_name", "kinetics.CHEMKIN_base3pp.CKI")),
            input_dic_file_name=str(d.get("input_dic_file_name", "input.dic")),
            output_folder_pattern=str(d.get("output_folder_pattern", "Kinetics_inputs_{index}")),
            output_label_prefix=str(d.get("output_label_prefix", "kinetics-base3")),
            read_start_line=int(d.get("read_start_line", 197)),
            write_start_line=int(d.get("write_start_line", 18088)),
        )


@dataclass
class KineticRunnerConfig:
    """Top-level configuration for KineticRunner."""

    sootgen_bin_dir: str = ""
    sootgen_lib_dir: str = ""
    cleanup_work_dir: bool = True   # delete per-sample tmp dir after SootGEN
    gen_kinetics: GenKineticsConfig = field(default_factory=GenKineticsConfig)

    @classmethod
    def from_dict(cls, d: dict) -> "KineticRunnerConfig":
        return cls(
            sootgen_bin_dir=str(d.get("sootgen_bin_dir", "")),
            sootgen_lib_dir=str(d.get("sootgen_lib_dir", "")),
            cleanup_work_dir=bool(d.get("cleanup_work_dir", True)),
            gen_kinetics=GenKineticsConfig.from_dict(d.get("gen_kinetics", {})),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────────────

class KineticRunner:
    """
    Orchestrate SootGEN execution and optional CKI patching.

    Parameters
    ----------
    config : KineticRunnerConfig
    """

    def __init__(self, config: KineticRunnerConfig):
        self.config = config

    # ------------------------------------------------------------------
    # Step 1: SootGEN
    # ------------------------------------------------------------------

    def run_sootgen(
        self,
        work_dir: "str | Path",
        indices: Optional[List[int]] = None,
    ) -> List[Path]:
        """
        Execute SootGEN for every ``input_{idx:03d}.txt`` in *work_dir*.

        Each sample is processed in an isolated temporary subdirectory
        (``sootgen_tmp_{idx}/``) so that SootGEN's intermediate output files
        (``BinProperties.txt``, ``*.CKT``, ``*.TRC``, etc.) never pollute
        *work_dir*.  Only ``A_Soot_{idx}.txt`` is moved back to *work_dir*
        when done; the temporary directory is then removed (unless
        ``cleanup_work_dir`` is *False*).

        SootGEN reads ``input.txt`` and ``Input_ArrheniusGlobal_*.txt`` from
        its working directory, and writes the soot output to the file named
        in the ``@OutputFileName`` directive.

        Parameters
        ----------
        work_dir : path-like
            Directory containing the recovered ``input_*.txt`` and
            ``Input_ArrheniusGlobal_*.txt`` files.
        indices : list of int, optional
            Specific sample indices to process.  *None* → all
            ``input_???.txt`` files in *work_dir* in sorted order.

        Returns
        -------
        list of Path
            Paths to ``A_Soot_*.txt`` files moved into *work_dir*.
        """
        work_dir = Path(work_dir).resolve()
        bin_dir = Path(self.config.sootgen_bin_dir).resolve()
        sootgen_exe = bin_dir / "SootGEN.sh"

        if not sootgen_exe.exists():
            raise FileNotFoundError(
                f"SootGEN.sh not found at {sootgen_exe}. "
                "Check 'sootgen_bin_dir' in your config."
            )

        # Build environment
        env = os.environ.copy()
        if self.config.sootgen_lib_dir:
            lib_dir = str(Path(self.config.sootgen_lib_dir).resolve())
            prev = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = f"{lib_dir}:{prev}" if prev else lib_dir
        env["PATH"] = f"{str(bin_dir)}:{env.get('PATH', '')}"

        # Determine files to process
        if indices is not None:
            candidates = [work_dir / f"input_{i:03d}.txt" for i in indices]
        else:
            candidates = sorted(work_dir.glob("input_???.txt"))

        produced: List[Path] = []
        for inp_path in candidates:
            if not inp_path.exists():
                print(f"  [sootgen] Warning: {inp_path.name} not found, skipping.")
                continue

            idx_str = inp_path.stem.rsplit("_", 1)[-1]   # e.g. "000"

            # ── Create an isolated tmp dir for this sample ──────────────────
            tmp_dir = work_dir / f"sootgen_tmp_{idx_str}"
            tmp_dir.mkdir(exist_ok=True)

            # Copy input_*.txt → input.txt (SootGEN expects this name)
            shutil.copy2(inp_path, tmp_dir / "input.txt")

            # Copy companion Input_ArrheniusGlobal_*.txt if present
            global_src = work_dir / f"Input_ArrheniusGlobal_{idx_str}.txt"
            if global_src.exists():
                shutil.copy2(global_src, tmp_dir / global_src.name)

            # ── Run SootGEN inside the tmp dir ──────────────────────────────
            print(f"  [sootgen] Running SootGEN for sample {idx_str} ...")
            try:
                result = subprocess.run(
                    [str(sootgen_exe)],
                    cwd=str(tmp_dir),
                    env=env,
                    input="\n",          # mirrors `echo |` in process_SOOTGEN.sh
                    text=True,
                    capture_output=True,
                )
                if result.returncode != 0:
                    print(f"  [sootgen] stderr:\n{result.stderr}")
                    raise RuntimeError(
                        f"SootGEN failed for sample {idx_str} "
                        f"(exit code {result.returncode})."
                    )
                if result.stdout.strip():
                    print(f"  [sootgen] stdout: {result.stdout.strip()}")
            except Exception:
                if self.config.cleanup_work_dir:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                raise

            # ── Move A_Soot_*.txt back to work_dir ─────────────────────────
            a_soot_tmp = tmp_dir / f"A_Soot_{idx_str}.txt"
            if a_soot_tmp.exists():
                a_soot_out = work_dir / f"A_Soot_{idx_str}.txt"
                shutil.move(str(a_soot_tmp), str(a_soot_out))
                produced.append(a_soot_out)
                print(f"  [sootgen] Created: {a_soot_out.name}")
            else:
                print(f"  [sootgen] Warning: expected A_Soot_{idx_str}.txt not found after run.")

            # ── Clean up tmp dir ────────────────────────────────────────────
            if self.config.cleanup_work_dir:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            else:
                print(f"  [sootgen] Intermediate files kept in: {tmp_dir.name}/")

        return produced

    # ------------------------------------------------------------------
    # Step 2: gen_kinetics (patch A_Soot data into CKI template)
    # ------------------------------------------------------------------

    def run_gen_kinetics(
        self,
        work_dir: "str | Path",
        soot_files: Optional[List[Path]] = None,
    ) -> List[Path]:
        """
        Patch each ``A_Soot_{idx}.txt`` into a copy of the CKI template folder.

        Patches kinetics data from ``A_Soot_{idx}.txt`` into a CKI template folder.

        Parameters
        ----------
        work_dir : path-like
            Directory where ``Kinetics_inputs_{idx}/`` folders will be created.
        soot_files : list of Path, optional
            Explicit list of A_Soot files.  *None* → all ``A_Soot_???.txt``
            files in *work_dir* sorted by name.

        Returns
        -------
        list of Path
            Paths to the produced ``Kinetics_inputs_{idx}`` folders.
        """
        gk = self.config.gen_kinetics
        if not gk.enabled:
            return []

        work_dir = Path(work_dir).resolve()
        template = Path(gk.template_folder).resolve()
        if not template.exists():
            raise FileNotFoundError(
                f"Kinetics template folder not found: {template}. "
                "Set 'gen_kinetics.template_folder' in your config."
            )

        if soot_files is None:
            soot_files = sorted(work_dir.glob("A_Soot_???.txt"))

        output_dirs: List[Path] = []
        for soot_path in soot_files:
            if not soot_path.exists():
                print(f"  [gen_kinetics] {soot_path.name} not found, skipping.")
                continue

            idx_str = soot_path.stem.rsplit("_", 1)[-1]   # e.g. "000"

            # Read A_Soot lines starting at read_start_line (1-based)
            with soot_path.open("r") as f:
                soot_lines = f.readlines()[gk.read_start_line - 1:]

            # Copy template folder
            out_folder = work_dir / gk.output_folder_pattern.format(index=idx_str)
            if out_folder.exists():
                shutil.rmtree(out_folder)
            shutil.copytree(str(template), str(out_folder))

            # Patch CKI file
            cki_path = out_folder / gk.target_file_name
            if not cki_path.exists():
                print(f"  [gen_kinetics] CKI file not found: {cki_path}, skipping.")
                continue

            with cki_path.open("r") as f:
                cki_lines = f.readlines()

            ws = gk.write_start_line - 1   # convert to 0-based
            with cki_path.open("w") as f:
                f.writelines(cki_lines[:ws] + soot_lines + cki_lines[ws:])

            # Patch input.dic
            dic_path = out_folder / gk.input_dic_file_name
            if dic_path.exists():
                with dic_path.open("r") as f:
                    dic_lines = f.readlines()
                patched = []
                for line in dic_lines:
                    if line.strip().startswith("@Output") and "kinetics-test;" in line:
                        line = line.replace(
                            "kinetics-test",
                            f"{gk.output_label_prefix}_{idx_str}",
                        )
                    patched.append(line)
                with dic_path.open("w") as f:
                    f.writelines(patched)

            output_dirs.append(out_folder)
            print(f"  [gen_kinetics] Produced: {out_folder.name}")

        return output_dirs
