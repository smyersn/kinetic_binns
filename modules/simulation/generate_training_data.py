import sys, os
file_dir = os.path.dirname(os.path.realpath(__file__))
repo_start = f'{file_dir}/../../'
sys.path.append(repo_start)

import json
import subprocess
import itertools

from modules.simulation.reaction_registry import REACTION_SPECS

save_path = "/hpc/home/nsmyers1/projects/kinetic_binns/data/nonconserved/two"
os.makedirs(save_path, exist_ok=True)

python_script = "/hpc/home/nsmyers1/projects/kinetic_binns/modules/simulation/simulation.py"

# ---------------------------------------------------------
# Parameter + diffusion-coefficient grids, one entry per reaction to sweep.
# ---------------------------------------------------------
diff_coeffs_pool = {
    # Paper (mass-conserving) reactions share one Du/Dv range
    # "hill_poly": {"du": [0.01], "dv": [1.0]},
    # "poly_poly": {"du": [0.01], "dv": [1.0]},
    # "hill_hill": {"du": [0.01], "dv": [1.0]},
    # "poly_hill": {"du": [0.01], "dv": [1.0]},

    # --- Non-conserving ---
    # These were originally set to the textbook dimensionless values
    # (Du~1, Dv~8-10). Those produce a Turing wavelength COMPARABLE TO the
    # L=10 domain -- 2*pi/k_c works out to 5.6 for Brusselator and 10.6 for
    # Schnakenberg, i.e. only 1.8 and 0.9 features fit across the whole box.
    # That is why they looked like single polarity sites rather than
    # patterns. Wavelength scales as sqrt(D), so these are scaled down to
    # pack roughly 10 wavelengths across the domain. Feature counts in the
    # comments are from direct simulation at N=128 (connected above-mean
    # regions in u at the final frame).
    "fitzhugh_nagumo": {"du": [0.02], "dv": [0.0]},   # only the activator diffuses
    "gray_scott": {"du": [0.0004], "dv": [0.0002]},   # 271 features (was: total extinction)
                                                       # classic GS values (0.16/0.08) assume
                                                       # dx=1 on a 200-wide grid -> domain 200.
                                                       # L=10 here, so scale by (10/200)^2 = 1/400.
    "brusselator": {"du": [0.02], "dv": [0.16]},      # 132 features (was 5)
    "schnakenberg": {"du": [0.01], "dv": [0.1]},      # 92 features (was 1)
}

param_pool = {
    # "hill_poly": {"a": [1, 2, 4], "b": [1, 2, 4], "k": [0.01, 0.1, 0.5], "n": [2, 3]},
    # "poly_poly": {"a": [1, 2, 4], "b": [1, 2, 4]},
    # "hill_hill": {"a": [1, 2, 4], "b": [1, 2, 4], "k1": [0.01, 0.1, 0.5], "n1": [2, 3],
    #               "k2": [0.01, 0.1, 0.5], "n2": [2, 3]},
    # "poly_hill": {"a": [1, 2, 4], "b": [1, 2, 4], "k": [0.01, 0.1, 0.5], "n": [2, 3]},

    # FHN rest state is stable (EXCITABLE) only when u*^2 > (1 - eps*b)/3.
    # b=0.5 fails that for a=0.5 and a=0.7, making the entire domain a bulk
    # oscillator -- every cell flashing in unison, which is the "weird
    # oscillatory behavior" seen in the first sweep. b=0.8 is excitable for
    # every a/eps below.
    # CAVEAT: excitable is necessary but not sufficient for spiral waves.
    # Sustained spirals also need a broken-wavefront initial condition that
    # ic_pulse_stimulus does not provide, so expect decaying pulses from
    # these runs until that IC is written.
    "fitzhugh_nagumo": {"a": [0.5, 0.7, 0.9], "b": [0.8], "eps": [0.05, 0.1]},
    "brusselator": {"a": [1.0, 2.0], "b": [2.0, 3.0, 4.0]},
    "schnakenberg": {"a": [0.1, 0.2], "b": [0.8, 0.9, 1.2]},
    # gray_scott is intentionally absent here -- see paired_params below.
    # Gray-Scott's autocatalytic term is quadratic in v, so feed/kill can't
    # be gridded independently without risking sub-threshold combinations
    # that just decay back to the trivial state (feed=0.055, kill=0.065 did
    # exactly this). Use validated (feed, kill) pairs instead.
}

# Parameter sets that must be swept as validated tuples rather than an
# independent cartesian grid (see gray_scott note above).
paired_params = {
    "gray_scott": {
        "keys": ["feed", "kill"],
        # (feed, kill) pairs from Pearson's (1993) classification, each a
        # known distinct pattern regime.
        "combos": [
            (0.014, 0.054),  # gliders
            (0.029, 0.057),  # maze
            (0.030, 0.062),  # spots
            (0.058, 0.065),  # worms
            (0.018, 0.051),  # spirals
        ],
    },
}

# Per-reaction T / early_stop / dt_cap overrides. Omit an entry to fall
# back to REACTION_SPECS's default for that reaction.
sim_overrides = {
    # e.g. "gray_scott": {"T": 12000},
}

# T and dt_cap defaults in REACTION_SPECS were verified by direct simulation
# against the diffusion coefficients above -- and ONLY against those. If you
# change diff_coeffs_pool, re-check that T still reaches the pattern and that
# the step count stays tractable; simulation.py prints the chosen dt, which
# limit is binding, and the total step count at startup.

all_reactions = list(param_pool) + list(paired_params)

for reaction in all_reactions:
    if reaction not in REACTION_SPECS:
        raise ValueError(f"'{reaction}' not found in REACTION_SPECS -- "
                          f"check spelling or add it to reaction_registry.py first.")

    if reaction in paired_params:
        param_keys = paired_params[reaction]["keys"]
        param_combos = paired_params[reaction]["combos"]
    else:
        param_keys = list(param_pool[reaction].keys())
        param_combos = list(itertools.product(*param_pool[reaction].values()))

    # Fail fast on a typo'd or missing param key rather than discovering it
    # after a job has already been submitted.
    expected_keys = set(REACTION_SPECS[reaction]["param_keys"])
    if set(param_keys) != expected_keys:
        raise ValueError(f"'{reaction}' param keys {set(param_keys)} don't match "
                          f"REACTION_SPECS param_keys {expected_keys}")

    diff_keys = list(diff_coeffs_pool[reaction].keys())
    diff_values = list(diff_coeffs_pool[reaction].values())

    for diff_combo in itertools.product(*diff_values):
        for param_combo in param_combos:
            c = dict(zip(diff_keys, diff_combo))
            c.update(dict(zip(param_keys, param_combo)))

            config_dict = {
                "save_path": save_path,
                "reaction": reaction,
                "du": c["du"],
                "dv": c["dv"],
            }
            config_dict.update({k: v for k, v in c.items() if k not in ("du", "dv")})
            config_dict.update(sim_overrides.get(reaction, {}))

            param_string = "_".join(f"{k}_{v}" for k, v in c.items() if k not in ("du", "dv"))
            config_filename = f"{reaction}_du_{c['du']}_dv_{c['dv']}_{param_string}.json"
            config_filepath = os.path.join(save_path, config_filename)

            with open(config_filepath, "w") as f:
                json.dump(config_dict, f, indent=4)

            # Matches the pattern in your BINN-training submission script:
            # eval "$(conda shell.bash hook)" initializes conda directly rather
            # than sourcing ~/.bashrc (which often early-returns in non-
            # interactive shells before reaching conda setup), and wrapping in
            # `bash -c` ensures bash -- not Slurm's default /bin/sh, which
            # doesn't have `source`/conda-activate-as-a-function support --
            # actually interprets the command.
            cmd_string = f'eval "$(conda shell.bash hook)" && conda activate binns && python {python_script} {config_filepath}'
            wrap_command = f"bash -c '{cmd_string}'"
            out_filename = config_filename.replace('.json', '.out')
            out_file = os.path.join(save_path, out_filename)

            sbatch_cmd = [
                "sbatch", "-p", "common", "-N", "1", "-n", "1",
                "--cpus-per-task=4", "--mem=16g", "-t", "4:00:00",
                f"--output={out_file}",
                f"--wrap={wrap_command}"
            ]

            subprocess.run(sbatch_cmd)