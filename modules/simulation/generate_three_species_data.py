import sys, os
file_dir = os.path.dirname(os.path.realpath(__file__))
repo_start = f'{file_dir}/../../'
sys.path.append(repo_start)

import json
import subprocess

from modules.simulation.reaction_registry import REACTION_SPECS

save_path = "/hpc/home/nsmyers1/projects/kinetic_binns/data/three_species/"
os.makedirs(save_path, exist_ok=True)

python_script = "/hpc/home/nsmyers1/projects/kinetic_binns/modules/simulation/simulation.py"

# ---------------------------------------------------------
# Three-species mass-conserving polarity models: Sostar & Dzeroski, Table 2,
# models 1-3. Each entry is ONE validated parameter set (rate constants,
# total concentration u_total, and diffusion coefficients [D1, D2, D3]).
#
# The paper does not print parameters for these models (its optimized sets
# are in its supplement, and are for a 1D domain of length 15). These were
# chosen for THIS simulator -- 2D, L = 10, periodic -- by linear stability
# analysis:
#   * the homogeneous steady state is stable without diffusion, and
#     Turing-unstable with it (a diffusion-driven pattern)
#   * fastest-growing wavelength ~2.2-2.5, i.e. ~4 spots across the domain
#   * still Turing-unstable for >= 90% of random +-10% perturbations of the
#     four rate constants, so small errors in a learned model do not flip
#     the behaviour (the Brusselator lesson)
# and then checked by direct 2D simulation: spots appear by t ~ 30-45 and
# then coarsen slowly; every species stays positive.
#
# u2 is the fast (cytosolic) species in models 1-3. Model 1 needs u1 to be
# the less slow of the two slow species; models 2 and 3 the opposite.
# ---------------------------------------------------------
datasets = {
    "mcrd3_m1": {
        "params": {"k1": 3.2, "k15": 3.2, "k16": 4.7, "k20": 0.77, "u_total": 7.7},
        "diff_coeffs": [0.04, 1.0, 0.01],
        # steady state (2.83, 0.434, 4.44); max growth 0.17 at wavelength 2.3
    },
    "mcrd3_m2": {
        "params": {"k1": 2.2, "k6": 3.3, "k9": 5.7, "k23": 1.1, "u_total": 3.75},
        "diff_coeffs": [0.01, 1.0, 0.04],
        # steady state (1.10, 1.17, 1.48); max growth 0.37 at wavelength 2.2
    },
    "mcrd3_m3": {
        "params": {"k1": 1.85, "k6": 1.13, "k9": 1.44, "k11": 1.2, "u_total": 5.5},
        "diff_coeffs": [0.01, 1.0, 0.04],
        # steady state (2.10, 1.86, 1.54); max growth 0.31 at wavelength 2.5
    },
}

# Per-reaction T / early_stop / dt_cap / num_frames overrides. Omit an entry
# to fall back to REACTION_SPECS's default for that reaction (T = 100,
# 101 frames). With D_max = 1 the diffusion CFL limit sets dt = 2.5e-4, so
# T = 100 is 400,000 steps.
sim_overrides = {
    # e.g. "mcrd3_m1": {"T": 150},
}

for reaction, d in datasets.items():
    if reaction not in REACTION_SPECS:
        raise ValueError(f"'{reaction}' not found in REACTION_SPECS -- "
                         f"check spelling or add it to reaction_registry.py first.")

    # Fail fast on a typo'd or missing param key rather than discovering it
    # after a job has already been submitted.
    expected_keys = set(REACTION_SPECS[reaction]["param_keys"])
    if set(d["params"]) != expected_keys:
        raise ValueError(f"'{reaction}' param keys {set(d['params'])} don't match "
                         f"REACTION_SPECS param_keys {expected_keys}")
    species = REACTION_SPECS[reaction].get("species", 2)
    if len(d["diff_coeffs"]) != species:
        raise ValueError(f"'{reaction}' needs {species} diffusion coefficients, "
                         f"got {d['diff_coeffs']}")

    config_dict = {
        "save_path": save_path,
        "reaction": reaction,
        "diff_coeffs": d["diff_coeffs"],
    }
    config_dict.update(d["params"])
    config_dict.update(sim_overrides.get(reaction, {}))

    d_string = "_".join(str(x) for x in d["diff_coeffs"])
    param_string = "_".join(f"{k}_{v}" for k, v in d["params"].items())
    config_filename = f"{reaction}_D_{d_string}_{param_string}.json"
    config_filepath = os.path.join(save_path, config_filename)

    with open(config_filepath, "w") as f:
        json.dump(config_dict, f, indent=4)

    cmd_string = f'eval "$(conda shell.bash hook)" && conda activate binns && python {python_script} {config_filepath}'
    wrap_command = f"bash -c '{cmd_string}'"
    out_filename = config_filename.replace('.json', '.out')
    out_file = os.path.join(save_path, out_filename)

    sbatch_cmd = [
        "sbatch", "-p", "common", "-N", "1", "-n", "1",
        "--cpus-per-task=4", "--mem=16g", "-t", "12:00:00",
        f"--output={out_file}",
        f"--wrap={wrap_command}"
    ]

    subprocess.run(sbatch_cmd)