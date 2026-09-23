"""
Sweep 14: GLS activity weighting on the paper-equation datasets.

For every (dataset x hyperparameter combination), writes one config.json per
repeat and submits a single Slurm GPU job that trains all repeats in
parallel.

    python scripts/sweeps/sweep_14_gls_activity_weight.py [--dry-run]

Run directories are created relative to the submission directory:
    paper_equations/14_gls_activity_weight/<dataset>/binn_eql_<settings>repeat_<k>/
"""
import argparse
import itertools
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from modules.simulation.reaction_registry import library_size

TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_binn_eql.py"
DATA_DIR = REPO_ROOT / "data" / "paper_equations"
OUT_DIR = "paper_equations/14_gls_activity_weight"


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Dataset:
    file: str
    reaction: str               # ground-truth reaction (REACTION_REGISTRY key)
    params: tuple               # ground-truth reaction parameters
    diff_coeffs: tuple = (0.01, 1.0)  # () to learn D
    mcas: bool = True           # False for non-conserving systems (Gray-Scott, Brusselator, ...)


DATASETS = [
    Dataset("hill_hill_du_0.01_dv_1_a_2_b_2_k1_0.1_n1_3_k2_0.01_n2_2.pt", "hill_hill", (2, 2, 0.1, 3, 0.01, 2)),
    Dataset("hill_hill_du_0.01_dv_1_a_4_b_2_k1_0.1_n1_3_k2_0.5_n2_2.pt",  "hill_hill", (4, 2, 0.1, 3, 0.5, 2)),
    Dataset("hill_hill_du_0.01_dv_1_a_2_b_4_k1_0.01_n1_3_k2_0.5_n2_2.pt", "hill_hill", (2, 4, 0.01, 3, 0.5, 2)),
    Dataset("hill_hill_du_0.01_dv_1_a_4_b_2_k1_0.1_n1_3_k2_0.01_n2_2.pt", "hill_hill", (4, 2, 0.1, 3, 0.01, 2)),
    Dataset("hill_hill_du_0.01_dv_1_a_4_b_4_k1_0.1_n1_3_k2_0.5_n2_3.pt",  "hill_hill", (4, 4, 0.1, 3, 0.5, 3)),
    Dataset("poly_poly_du_0.01_dv_1_a_4_b_2.pt",                          "poly_poly", (4, 2)),
    Dataset("hill_poly_du_0.01_dv_1_a_4_b_2_k_0.5_n_3.pt",                "hill_poly", (4, 2, 0.5, 3)),
    Dataset("poly_hill_du_0.01_dv_1_a_1_b_1_k_0.1_n_2.pt",                "poly_hill", (1, 1, 0.1, 2)),
]

# ---------------------------------------------------------------------------
# Hyperparameters: every combination is run for every dataset
# ---------------------------------------------------------------------------
SWEEP = {
    "batch_size": [50_000],
    "species": [2],
    "dimensions": [2],
    "epsilon": [0],
    "points": [50],
    "duplicates": [3],
    "degree": [3],
    "l0_weight": [100],
    "param_bounds": [10],
    "include_poly": [True],
    "include_increasing_hill": [True],
    "include_decreasing_hill": [True],
}
REPEATS = range(1, 6)  # all repeats share one GPU job

# Fixed library size that l0_weight is priced against (see calibration.l0_scale).
L0_REFERENCE_GATES = library_size(
    species=2, degree=3, duplicates=3, mcas=True,
    include_poly=True, include_increasing_hill=True, include_decreasing_hill=True)

SLURM_ARGS = ["-p", "gpu-hp", "-N", "1", "-n", "1", "--mem=32g", "--gres=gpu:1",
              "--qos=unc_h200_hp", "-t", "2-00:00:00", "--output=/dev/null"]
# Standard partition:
# SLURM_ARGS = ["-p", "gpu", "-N", "1", "-n", "1", "--mem=32g", "--gres=gpu:1",
#               "-t", "1-00:00:00", "--output=/dev/null"]


# ---------------------------------------------------------------------------
def run_prefix(dataset, hp):
    """Run directory without the repeat suffix. Settings at their defaults are omitted."""
    name = "binn_eql_"
    if hp["batch_size"] != 0:          name += f"batch_{hp['batch_size']}_"
    if hp["l0_weight"] != 1:           name += f"l0_{hp['l0_weight']}_"
    if hp["epsilon"] != 0.0:           name += f"epsilon_{hp['epsilon']}_"
    if hp["points"] != 0:              name += f"points_{hp['points']}_"
    if hp["duplicates"] != 5:          name += f"duplicates_{hp['duplicates']}_"
    if not hp["include_poly"]:             name += "nopoly_"
    if not hp["include_increasing_hill"]:  name += "noinchill_"
    if not hp["include_decreasing_hill"]:  name += "nodechill_"
    return f"{OUT_DIR}/{Path(dataset.file).stem}/{name}"


def make_config(dataset, hp):
    return {
        "training_data_path": str(DATA_DIR / dataset.file),
        "reaction": dataset.reaction,
        "params": list(dataset.params),
        "diff_coeffs": list(dataset.diff_coeffs),
        "mcas": dataset.mcas,
        "l0_reference_gates": L0_REFERENCE_GATES,
        **hp,
    }


def gate_count(dataset, hp):
    return library_size(
        species=hp["species"], degree=hp["degree"], duplicates=hp["duplicates"],
        mcas=dataset.mcas, include_poly=hp["include_poly"],
        include_increasing_hill=hp["include_increasing_hill"],
        include_decreasing_hill=hp["include_decreasing_hill"])


def sbatch_command(run_dirs):
    """One job that trains every run in run_dirs concurrently, each logging to its own directory."""
    jobs = " ".join(f"python {TRAIN_SCRIPT} {d} >> {d}/training.log 2>&1 &" for d in run_dirs)
    script = f'eval "$(conda shell.bash hook)" && conda activate binns; {jobs} wait'
    return ["sbatch", *SLURM_ARGS, f"--wrap=bash -c '{script}'"]


def main(dry_run):
    seen = set()
    for dataset, values in itertools.product(DATASETS, itertools.product(*SWEEP.values())):
        hp = dict(zip(SWEEP.keys(), values))
        prefix = run_prefix(dataset, hp)

        # species, dimensions, degree and param_bounds are not encoded in the
        # directory name, so sweeping them would overwrite earlier runs.
        if prefix in seen:
            raise ValueError(f"Two combinations map to {prefix}; encode the swept "
                             f"setting in run_prefix().")
        seen.add(prefix)

        run_dirs = [f"{prefix}repeat_{r}" for r in REPEATS]
        if not dry_run:
            for d in run_dirs:
                os.makedirs(d, exist_ok=True)
                with open(os.path.join(d, "config.json"), "w") as f:
                    json.dump(make_config(dataset, hp), f, indent=4)

        gates = gate_count(dataset, hp)
        print(f"{prefix}: {len(run_dirs)} repeats, {gates} gates "
              f"({gates / L0_REFERENCE_GATES:.2f}x reference)")

        command = sbatch_command(run_dirs)
        if dry_run:
            print("  " + " ".join(command[:-1]) + " --wrap=...")
        else:
            subprocess.run(command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--dry-run", action="store_true",
                        help="print what would be submitted without writing configs or submitting")
    main(parser.parse_args().dry_run)