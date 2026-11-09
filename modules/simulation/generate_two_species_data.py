import sys, os
file_dir = os.path.dirname(os.path.realpath(__file__))
repo_start = f'{file_dir}/../../'
sys.path.append(repo_start)

import json
import subprocess
import itertools

from modules.simulation.reaction_registry import REACTION_SPECS

save_path = "/hpc/home/nsmyers1/projects/kinetic_binns/data/nonconserved/waves"
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
    # "fitzhugh_nagumo": {"du": [0.02], "dv": [0.0]},   # only the activator diffuses
    # "gray_scott": {"du": [0.0004], "dv": [0.0002]},   # 271 features (was: total extinction)
                                                       # classic GS values (0.16/0.08) assume
                                                       # dx=1 on a 200-wide grid -> domain 200.
                                                       # L=10 here, so scale by (10/200)^2 = 1/400.
    # "brusselator": {"du": [0.02], "dv": [0.16]},      # 132 features (was 5)
    # "schnakenberg": {"du": [0.01], "dv": [0.1]},      # 92 features (was 1)

    # --- Waves (spirals) instead of spots ---
    # With Dv >> Du both systems are Turing-unstable almost everywhere, and the
    # stationary Turing mode outgrows any oscillation, so they only make spots.
    # EQUAL diffusion removes the Turing instability; with (a, b) past the Hopf
    # bifurcation (oscillatory steady state) the noise in the initial condition
    # breaks the domain into regions out of phase, which organize into rotating
    # spirals. Verified by direct simulation (N=200, registry IC, 3 seeds each):
    # spirals persist through the saved window for every seed.
    # (Oscillatory media only make wide, loosely wound spirals -- about one arm
    # turn across the box -- so these are off; uncomment to make them.)
    # "brusselator": {"du": [0.02], "dv": [0.02]},    # Hopf at b = 1 + a^2 = 2; period ~7
    # "schnakenberg": {"du": [0.01], "dv": [0.01]},   # period ~11-13

    # --- Big, tightly wound spirals: excitable FitzHugh-Nagumo ---
    # Only the fast variable diffuses. Small Du makes thin, closely spaced arms
    # (several turns across the box). Fronts are ~6 grid points wide at N=200:
    # train at points 200 (100 at the very least) -- 50 cannot resolve them.
    # (FitzHugh-Nagumo's u goes negative -- not a concentration -- so it is off.)
    # "fitzhugh_nagumo": {"du": [0.002], "dv": [0.0]},

    # --- Big BZ spiral waves with positive concentrations: Oregonator ---
    # Only the activator u diffuses (in gel BZ experiments the catalyst, v, is
    # immobilized). Du = 0.01 gives bands ~7 grid points wide at N=200 with
    # ~2 spiral turns across the box. Train at points 200.
    # "oregonator": {"du": [0.01], "dv": [0.0]},   # needs a Hill term; off

    # --- Big spiral waves, positive AND purely polynomial: Barkley model ---
    # Only u diffuses. Du = 0.02 (at k = 1; 0.002 at k = 0.1, see below) gives
    # bands ~8 grid points wide at N=200 with several spiral turns across the
    # box. Train at points 200.
    "barkley": {"du": [0.002], "dv": [0.0]},
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
    # "fitzhugh_nagumo": {"a": [0.5, 0.7, 0.9], "b": [0.8], "eps": [0.05, 0.1]},
    # "brusselator": {"a": [2.0], "b": [3.5]},
    # "schnakenberg": {"a": [0.05], "b": [1.0]},
    # Waves: just past the Hopf bifurcation, so the fronts stay smooth enough for
    # the surface fit (b = 3 also gives spirals, but with much sharper fronts).
    "brusselator": {"a": [1.0], "b": [2.6]},
    "schnakenberg": {"a": [0.1], "b": [0.5]},
    # Excitable (stable rest state). eps = 0.05 curls the broken wave into a
    # tight spiral pair; eps = 0.08 is too weak (the wave passes without curling).
    "fitzhugh_nagumo": {"a": [0.5], "b": [0.8], "eps": [0.05]},
    # Excitable (f > ~2.41 at these eps, q): eps = 0.05 widens the pulses enough
    # to learn (eps = 0.02 gives 1-2 point wide bands); eps = 0.08 is too weak
    # and the wave dies out.
    "oregonator": {"eps": [0.05], "f": [2.5], "q": [0.01]},
    # Standard Barkley parameters. eps = 0.02 curls the broken wave into a
    # spiral pair; eps = 0.05 is too weak (a flat pulse just crosses the box).
    # k = 0.1 slows the clock 10x: the patterns are identical to k = 1, but
    # every coefficient is 10x smaller (largest 6.7 instead of 66.7, so the
    # default param_bounds of 10 suffices). Du, T, t_start and dt_cap are
    # scaled to match (x k, / k, / k, / k).
    "barkley": {"a": [0.75], "b": [0.06], "eps": [0.02], "k": [0.1]},
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
    "gray_scott": {"T": 2000, "num_frames": 101},
    # Waves: "ic": "spiral_pair" starts every point on the ODE limit cycle with
    # the phase winding once around each of two cores, giving ONE clean,
    # symmetric, counter-rotating spiral pair (the default steady state + noise
    # seeds a random number of cores inside a flashing bulk oscillation, which
    # looks messy). t_start is a burn-in that is simulated but not saved, so the
    # spiral waves have already taken over the domain at the first saved frame
    # (t = 0), which is also where the learned PDE is later simulated from.
    # T gives ~10 frames per oscillation period with 101 frames:
    #   brusselator : 0.6 per frame, period ~6.7 -> ~9 periods in 60
    #   schnakenberg: 1.5 per frame, period ~12.5 -> ~12 periods in 150
    "brusselator": {"T": 60, "t_start": 30, "num_frames": 101, "ic": "spiral_pair"},
    "schnakenberg": {"T": 150, "t_start": 60, "num_frames": 101, "ic": "spiral_pair"},
    # Big spirals: "ic": "broken_wavefront" excites a strip across the middle half
    # of the domain with a refractory band behind it; both free ends curl into a
    # counter-rotating spiral pair. The spirals are fully wound by t ~ 100 (the
    # burn-in) and then rotate steadily: each point fires every ~62 time units,
    # so T = 200 is ~3 rotations at ~30 frames per rotation.
    "fitzhugh_nagumo": {"T": 200, "t_start": 100, "num_frames": 101, "ic": "broken_wavefront"},
    # Oregonator: spirals are fully formed by t ~ 30 (burn-in 40) and rotate with
    # period ~4.9; each pulse passes a point in only ~0.3-0.6 time units, so
    # frames must be ~0.1 apart: T = 15 with 151 frames = 3 rotations, ~32
    # frames per rotation.
    "oregonator": {"T": 15, "t_start": 40, "num_frames": 151, "ic": "broken_wavefront"},
    # Barkley (k = 0.1): spirals formed by t ~ 160 (burn-in 200); rotation
    # period ~39, each pulse keeps a point excited ~10 time units: T = 150 with
    # 101 frames (1.5 apart) = ~4 rotations, ~26 frames per rotation, ~6 frames
    # per pulse. (At k = 1 all of these times are 10x shorter.)
    "barkley": {"T": 150, "t_start": 200, "num_frames": 101, "ic": "broken_wavefront",
                "dt_cap": 0.02},
}

# Random seeds for the initial-condition noise, per reaction. Each seed is a
# separate dataset, named ..._seed_<s>. Reactions not listed here get one
# unseeded run, as before. The spiral-pair IC is not random, so seeds do not
# change it; to make distinct spiral datasets, vary "spiral_separation"
# (fraction of L between the two cores, default 0.5) in sim_overrides instead.
seeds = {
    # "brusselator": [1, 2, 3],
    # "schnakenberg": [1, 2, 3],
}

# T and dt_cap defaults in REACTION_SPECS were verified by direct simulation
# against the diffusion coefficients above -- and ONLY against those. If you
# change diff_coeffs_pool, re-check that T still reaches the pattern and that
# the step count stays tractable; simulation.py prints the chosen dt, which
# limit is binding, and the total step count at startup.

# Only reactions with diffusion coefficients above are run (comment one out there to skip it).
all_reactions = [r for r in list(param_pool) + list(paired_params) if r in diff_coeffs_pool]

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

    for diff_combo, param_combo, seed in itertools.product(
            itertools.product(*diff_values), param_combos, seeds.get(reaction, [None])):
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
        if seed is not None:
            config_dict["seed"] = seed

        param_string = "_".join(f"{k}_{v}" for k, v in c.items() if k not in ("du", "dv"))
        config_filename = f"{reaction}_du_{c['du']}_dv_{c['dv']}_{param_string}"
        if config_dict.get("ic", "default") != "default":
            config_filename += f"_{config_dict['ic']}"
            if "spiral_separation" in config_dict:
                config_filename += f"_sep_{config_dict['spiral_separation']}"
        if config_dict.get("t_start"):
            config_filename += f"_tstart_{config_dict['t_start']}"
        if seed is not None:
            config_filename += f"_seed_{seed}"
        config_filename += ".json"
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