import os
import sys
import json
import math
import subprocess

# Species written to the .xyz (molpos). Cdc42 pools:
#   active   (biosensor NR channel): Cdc42T, BemGEF42, complex_Ra_Cdc42T
#   inactive (membrane, GFP - NR):   Cdc42Dm (+ the Dm-containing complexes)
#   cytosol  (not seen in TIRF):     Cdc42Dc
# The complexes last a single time step (catalysis probability 1), so they are
# tiny, but writing them makes the Cdc42 total exact. BemGEF* are ground truth
# for checking the reduced (quasi-steady GEF) model.
OUTPUT_SPECIES = [
    'Cdc42T', 'Cdc42Dm', 'Cdc42Dc', 'BemGEF42',
    'complex_Cdc42Dm_BemGEF42', 'complex_Cdc42Dm_BemGEFm',
    'complex_Ra_Cdc42T', 'complex_Cdc42D_Ra',
    'BemGEFm', 'BemGEFc',
]

def smoldyn_cfg_variable_size(fileprefix, directory, tstop, samplingrate, n_Cdc42, n_BemGEF, n_FarGEF, length, random_seed,
                              initial_counts=None):
    """
    Generates a Smoldyn configuration file.
    Adapted from "make_smoldyn_cfg.m"

    initial_counts: {species: number} placed uniformly at t = 0 for Cdc42 and
        Bem1-GEF (see initial_counts_at_steady_state). None = the original start,
        all Cdc42 as Cdc42Dc and all Bem1-GEF as BemGEFc.
    """
    # Define file paths
    # Note: .cfg is created inside the directory
    cfg_name = os.path.join(directory, f"{fileprefix}.cfg")
    # Output file name for the simulation to write to (relative path)
    xyz_name = f"{fileprefix}.xyz"

    with open(cfg_name, 'w') as fid:
        # Parameters for the simulation
        rho = 0.05
        dt = 0.0001
        k1a = 10
        k1b = 40
        P2a = 5.3 * dt
        k2b = 0.35
        P3 = 180 * dt
        P2c = P3
        P4a = 9.6 * dt
        k4b = 40
        k5a = 36
        k5b = 13
        P7 = 256 * dt
        P8a = 300 * dt
        k8b = 0.11
        P9 = 0.025 * dt
        k10 = 0.002
        Dm = 0.0025
        Dc = 15

        # Define a random number generator
        fid.write(f'random_seed {int(random_seed)}\n')
        # Reaction radius rho
        fid.write(f'variable rho = {rho:g}\n')
        # A really small number which helps separate molecules
        fid.write('variable rho_eps = 0.00001\n')
        # Domain length
        fid.write(f'variable L = {length:g}\n')

        # Reaction rates
        fid.write(f'variable k1a = {k1a:g}\n')
        fid.write(f'variable k1b = {k1b:g}\n')
        fid.write(f'variable P2a = {P2a:g}\n')
        fid.write(f'variable P2c = {P2c:g}\n')
        fid.write(f'variable k2b = {k2b:g}\n')
        fid.write(f'variable P3 = {P3:g}\n')
        fid.write(f'variable P4a = {P4a:g}\n')
        fid.write(f'variable k4b = {k4b:g}\n')
        fid.write(f'variable k5a = {k5a:g}\n')
        fid.write(f'variable k5b = {k5b:g}\n')
        fid.write(f'variable P7  = {P7:g}\n')
        fid.write(f'variable P8a = {P8a:g}\n')
        fid.write(f'variable P9 = {P9:g}\n')
        fid.write(f'variable k8b = {k8b:g}\n')
        fid.write(f'variable k10 = {k10:g}\n')

        # Define domain boundaries
        fid.write('dim 2\n')
        fid.write('boundaries x -0.2 L+0.2\n')
        fid.write('boundaries y -0.2 L+0.2\n')

        # Define species
        fid.write('species Cdc42T Cdc42Dc Cdc42Dm BemGEFc BemGEFm BemGEF42 complex_Cdc42Dm_BemGEF42 complex_Cdc42Dm_BemGEFm\n')
        fid.write('species RaGEF Ram Ric FarGEF complex_Cdc42D_Ra complex_Ra_Cdc42T\n')

        # Define diffusion rates.
        fid.write(f'difc Cdc42T {Dm:g}\n')
        fid.write(f'difc Cdc42Dc {Dc:g}\n')
        fid.write(f'difc Cdc42Dm {Dm:g}\n')
        fid.write(f'difc BemGEF42 {Dm:g}\n')
        fid.write(f'difc BemGEFc {Dc:g}\n')
        fid.write(f'difc BemGEFm {Dm:g}\n')
        fid.write(f'difc complex_Cdc42Dm_BemGEF42 {Dm:g}\n')
        fid.write(f'difc complex_Cdc42Dm_BemGEFm {Dm:g}\n')
        fid.write(f'difc RaGEF {0.0001:g}\n')
        fid.write(f'difc Ram {0.0001:g}\n')
        fid.write(f'difc Ric {Dc:g}\n')
        fid.write(f'difc FarGEF {Dc:g}\n')
        fid.write(f'difc complex_Cdc42D_Ra {0.0001:g}\n')
        fid.write(f'difc complex_Ra_Cdc42T {0.0001:g}\n')

        # Set up lists to store molecular coordinates
        fid.write('molecule_lists list1 list2 list3 list4 list5 list6 list7 list8 list9 list10\n')
        fid.write('mol_list Cdc42T list1\n')
        fid.write('mol_list BemGEF42 list2\n')
        fid.write('mol_list Cdc42Dm list3\n')
        fid.write('mol_list Cdc42Dc list4\n')
        fid.write('mol_list BemGEFm list5\n')
        fid.write('mol_list BemGEFc list6\n')
        fid.write('mol_list Ram list7\n')
        fid.write('mol_list Ric list8\n')
        fid.write('mol_list RaGEF list9\n')
        fid.write('mol_list FarGEF list10\n')

        # Define a domain. The boundary condition is periodic.
        fid.write('start_surface inner_walls\n')
        fid.write('action both all jump\n')
        fid.write('polygon both edge\n')
        fid.write('panel rect +x 0 0 L r1\n')
        fid.write('panel rect -x L 0 L r2\n')
        fid.write('panel rect +y 0 0 L r3\n')
        fid.write('panel rect -y 0 L L r4\n')
        fid.write('jump r1 front <-> r2 front\n')
        fid.write('jump r3 front <-> r4 front\n')
        fid.write('end_surface\n')

        # Define the compartment which is required by Smoldyn to input molecules.
        fid.write('start_compartment full_domain\n')
        fid.write('surface inner_walls\n')
        fid.write('point L/2 L/2\n')
        fid.write('end_compartment\n')

        # Define reactions
        # Bem1-GEF associates and dissociates from the membrane
        fid.write('reaction BemGEF_cmtransition BemGEFc <-> BemGEFm k1a k1b\n')
        # Cdc42-GDP associates and dissociates from the membrane
        fid.write('reaction Cdc42_cmtransition Cdc42Dc <-> Cdc42Dm k5a k5b\n')
        # Receptors dissociates from the membrane
        fid.write('reaction Ra_cmtransition Ram -> Ric k10\n')

        # Bem1-GEFm + Cdc42Dm -> Bem1-GEFm + Cdc42T
        fid.write('reaction Cdc42Dm_2_T_bindTo_BemGEFm Cdc42Dm + BemGEFm -> complex_Cdc42Dm_BemGEFm\n')
        fid.write('reaction Cdc42Dm_2_T_catBy_BemGEFm complex_Cdc42Dm_BemGEFm -> Cdc42T + BemGEFm\n')
        fid.write('reaction_probability Cdc42Dm_2_T_bindTo_BemGEFm P2a\n')
        fid.write('binding_radius Cdc42Dm_2_T_bindTo_BemGEFm rho\n')
        fid.write('reaction_probability Cdc42Dm_2_T_catBy_BemGEFm 1\n')
        # Place molecules beyond their reactive radius
        fid.write('product_placement Cdc42Dm_2_T_catBy_BemGEFm unbindrad rho+rho_eps\n')

        # Cdc42T -> Cdc42Dm
        fid.write('reaction Cdc42T_2_Cdc42Dm Cdc42T -> Cdc42Dm k2b\n')

        # Cdc42T-Bem1-GEF + Cdc42Dm -> Cdc42T-Bem1-GEF + Cdc42T
        fid.write('reaction Cdc42Dm_2_T_bindTo_BemGEF42 Cdc42Dm + BemGEF42 -> complex_Cdc42Dm_BemGEF42\n')
        fid.write('reaction Cdc42Dm_2_T_catBy_BemGEF42 complex_Cdc42Dm_BemGEF42 -> Cdc42T + BemGEF42\n')
        fid.write('reaction_probability Cdc42Dm_2_T_bindTo_BemGEF42 P3\n')
        fid.write('binding_radius Cdc42Dm_2_T_bindTo_BemGEF42 rho\n')
        fid.write('reaction_probability Cdc42Dm_2_T_catBy_BemGEF42 1\n')
        fid.write('product_placement Cdc42Dm_2_T_catBy_BemGEF42 unbindrad rho+rho_eps\n')

        # Cdc42T + Bem1-GEFm -> Cdc42T-Bem1-GEF
        fid.write('reaction make_BemGEF42_fromm BemGEFm + Cdc42T <-> BemGEF42\n')
        fid.write('reaction_probability make_BemGEF42_frommfwd P4a\n')
        fid.write('binding_radius make_BemGEF42_frommfwd rho\n')
        fid.write('reaction_rate make_BemGEF42_frommrev k4b\n')
        fid.write('product_placement make_BemGEF42_frommrev unbindrad rho+rho_eps\n')

        # Cdc42T + BemGEFc -> Cdc42T-Bem1-GEF
        fid.write('reaction make_BemGEF42_fromc BemGEFc + Cdc42T -> BemGEF42\n')
        fid.write('reaction_probability make_BemGEF42_fromc P7\n')
        fid.write('binding_radius make_BemGEF42_fromc rho\n')

        # Ric + Cdc42T -> Ram + Cdc42T
        fid.write('reaction Ric_recruitBy_Cdc42T Ric + Cdc42T -> complex_Ra_Cdc42T\n')
        fid.write('reaction Ric_2_Ram complex_Ra_Cdc42T -> Ram + Cdc42T\n')
        fid.write('reaction_probability Ric_recruitBy_Cdc42T P9\n')
        fid.write('reaction_probability Ric_2_Ram 1\n')
        fid.write('binding_radius Ric_recruitBy_Cdc42T rho\n')
        fid.write('product_placement Ric_2_Ram unbindrad rho+rho_eps\n')

        # Far1-GEF + Ram -> RaGEF
        fid.write('reaction FarGEF_bindTo_Ra Ram + FarGEF -> RaGEF\n')
        fid.write('reaction_probability FarGEF_bindTo_Ra P8a\n')
        fid.write('binding_radius FarGEF_bindTo_Ra rho\n')

        # RaGEF -> Far1-GEF + Ram
        fid.write('reaction RaGEF_dissociation RaGEF -> Ram + FarGEF k8b\n')
        fid.write('product_placement RaGEF_dissociation unbindrad rho+rho_eps\n')

        # RaGEF -> Ric + Far1-GEF
        fid.write('reaction RaGEF_endocytosis RaGEF -> Ric + FarGEF k10\n')
        fid.write('product_placement RaGEF_endocytosis unbindrad rho+rho_eps\n')

        # Cdc42Dm + RaGEF -> Cdc42T + RaGEF
        fid.write('reaction Cdc42D_2_T_bindTo_RaGEF  Cdc42Dm + RaGEF -> complex_Cdc42D_Ra\n')
        fid.write('reaction Cdc42D_2_T_catBy_RaGEF complex_Cdc42D_Ra -> Cdc42T + RaGEF\n')
        fid.write('reaction_probability Cdc42D_2_T_bindTo_RaGEF P2c\n')
        fid.write('binding_radius Cdc42D_2_T_bindTo_RaGEF rho\n')
        fid.write('reaction_probability Cdc42D_2_T_catBy_RaGEF 1\n')
        fid.write('product_placement Cdc42D_2_T_catBy_RaGEF unbindrad rho+rho_eps\n')

        # Define the starting and ending point and time step
        fid.write('time_start 0\n')
        fid.write(f'time_stop {int(tstop)}\n')
        fid.write(f'time_step {dt:g}\n')

        # Input initial conditions (uniform random positions)
        if initial_counts is None:
            fid.write(f'compartment_mol {int(n_Cdc42)} Cdc42Dc full_domain\n')
            fid.write(f'compartment_mol {int(n_BemGEF)} BemGEFc full_domain\n')
        else:
            for species, n in initial_counts.items():
                if n > 0:
                    fid.write(f'compartment_mol {int(n)} {species} full_domain\n')
        fid.write(f'compartment_mol {int(n_FarGEF)} FarGEF full_domain\n')
        fid.write('compartment_mol 500 Ric full_domain\n')
        fid.write('compartment_mol 2000 Ram full_domain\n')

        # Output coordinates of molecules.
        # Every species that carries Cdc42, plus Bem1-GEF, so any observation
        # model (biosensor channels, TIRF footprint) can be built afterwards from
        # the ground truth without rerunning. One line per species per output
        # time, in the order of OUTPUT_SPECIES; pbs_to_training_data.py reads the
        # order from these cmd lines, so it can be changed freely.
        fid.write(f'output_files {xyz_name}\n')
        for species in OUTPUT_SPECIES:
            fid.write(f'cmd N {int(samplingrate)} molpos {species} {xyz_name}\n')

# --- Steady-state initial conditions ---
# Starting with all Cdc42 as cytosolic GDP makes ~90% of it activate within the
# first ~3 s (inactive Cdc42 drops ~114 -> ~14 per um^2 at scale 3), faster than
# the frame spacing. Instead, each run starts from the composition the system
# relaxes to, placed uniformly: no unresolved transient, and the polarity sites
# still form from the noise of the uniform start.
# The composition is MEASURED with a short Smoldyn burn-in per scale (small
# domain, same densities), because Smoldyn's effective rates differ from
# mass-action ones. Burn-in results are cached in <run dir>/burn_in/.
STEADY_SPECIES = ['Cdc42T', 'BemGEF42', 'Cdc42Dm', 'BemGEFm']   # Cdc42Dc, BemGEFc = remainders
BURN_IN_REPORT = ['Cdc42T', 'BemGEF42', 'Cdc42Dm', 'Cdc42Dc', 'BemGEFm', 'BemGEFc']
BURN_IN_LENGTH = 5       # um (cost grows with area; one polarity site fits)
BURN_IN_TIME = 20        # s simulated (scale 3 levels off by ~3 s, scale 1 needs >10 s)
BURN_IN_AVERAGE = 4.0    # average the composition over the last this-many seconds
BURN_IN_SEED = 12345


def steady_state_densities(scales, burn_dir):
    """
    Burn-in per scale, all scales in parallel (each Smoldyn run uses one core).
    Returns {scale: {species: molecules / um^2}} for STEADY_SPECIES.
    Results are cached as burn_in_scale_<s>.json; delete a file to redo it.
    """
    os.makedirs(burn_dir, exist_ok=True)
    L = BURN_IN_LENGTH
    result, running = {}, {}
    for scale in scales:
        prefix = f'burn_in_scale_{scale}'
        cache = os.path.join(burn_dir, f'{prefix}.json')
        if os.path.exists(cache):
            with open(cache) as f:
                result[scale] = json.load(f)['densities']
            print(f'  scale {scale}: using cached burn-in {cache}')
            continue
        n_Cdc42 = round(c_cdc42 * L**2) * scale
        n_BemGEF = round(c_BemGEF * L**2) * scale
        n_FarGEF = round(c_FarGEF * L**2) * scale
        every = int(round(0.5 / 0.0001))             # a frame every 0.5 s
        smoldyn_cfg_variable_size(prefix, burn_dir, BURN_IN_TIME, every,
                                  n_Cdc42, n_BemGEF, n_FarGEF, L, BURN_IN_SEED)
        log = open(os.path.join(burn_dir, f'{prefix}.log'), 'w')
        running[scale] = (subprocess.Popen([smoldyn_exec, f'{prefix}.cfg'], cwd=burn_dir,
                                           stdout=log, stderr=subprocess.STDOUT), log, prefix, cache)
        print(f'  scale {scale}: burn-in started ({L} um domain, {BURN_IN_TIME} s, {n_Cdc42} Cdc42)', flush=True)
    if running:
        print(f'  waiting for {len(running)} burn-in(s); expect a few minutes ...', flush=True)

    for scale, (proc, log, prefix, cache) in running.items():
        code = proc.wait()
        log.close()
        if code != 0:
            sys.exit(f'ERROR: burn-in for scale {scale} failed (exit {code}); see {burn_dir}/{prefix}.log')
        cfg = parse_cfg(os.path.join(burn_dir, f'{prefix}.cfg'))
        frames = list(read_frames(os.path.join(burn_dir, f'{prefix}.xyz'), cfg['molpos'], cfg['L']))
        print(f'  scale {scale} burn-in (molecules / um^2):')
        print('      t (s)' + ''.join(f'{s:>10s}' for s in BURN_IN_REPORT))
        for t, block in frames:
            print(f'     {t:6.2f}' + ''.join(f'{len(block[s]) / L**2:10.2f}' for s in BURN_IN_REPORT))
        late = [(t, b) for t, b in frames if t >= BURN_IN_TIME - BURN_IN_AVERAGE - 1e-9]
        densities = {s: sum(len(b[s]) for _, b in late) / len(late) / L**2 for s in STEADY_SPECIES}
        # Has the inactive pool levelled off? Compare the two halves of the averaging window.
        inactive = [(len(b['Cdc42Dm']) + len(b['Cdc42Dc'])) / L**2 for _, b in late]
        half = len(inactive) // 2
        first, second = sum(inactive[:half]) / half, sum(inactive[half:]) / (len(inactive) - half)
        drift = abs(second - first) / max(first, 1e-12)
        if drift > 0.05:
            print(f'  WARNING: scale {scale}: inactive Cdc42 still changing by {drift:.0%} over the '
                  f'last {BURN_IN_AVERAGE:g} s; consider raising BURN_IN_TIME')
        with open(cache, 'w') as f:
            json.dump(dict(scale=scale, length=L, time=BURN_IN_TIME, average_over=BURN_IN_AVERAGE,
                           seed=BURN_IN_SEED, densities=densities, inactive_drift=drift,
                           trajectory={f'{t:g}': {s: len(b[s]) / L**2 for s in BURN_IN_REPORT}
                                       for t, b in frames}), f, indent=2)
        result[scale] = densities
    return result


def initial_counts_at_steady_state(densities, length, n_Cdc42, n_BemGEF):
    """
    Steady-state counts for a length x length domain. Cdc42Dc and BemGEFc take
    the remainder, so the Cdc42 and Bem1-GEF totals are exactly n_Cdc42 and
    n_BemGEF (BemGEF42 holds one of each).
    """
    area = length ** 2
    counts = {s: int(round(densities[s] * area)) for s in STEADY_SPECIES}
    counts['Cdc42Dc'] = n_Cdc42 - counts['Cdc42T'] - counts['BemGEF42'] - counts['Cdc42Dm']
    counts['BemGEFc'] = n_BemGEF - counts['BemGEF42'] - counts['BemGEFm']
    if counts['Cdc42Dc'] < 0 or counts['BemGEFc'] < 0:
        raise ValueError(f'steady-state counts exceed the totals for L={length}: {counts}')
    return counts


# --- Main Script ---

# --- Cluster settings (edit these) ---
# Smoldyn executable (built with cmake inside the source folder; same build for every job)
smoldyn_exec = "/hpc/home/nsmyers1/smoldyn-2.75/build/smoldyn"
# CPU partition and time limit (check the partition's maximum with: sinfo -o "%P %l")
partition = 'common'
time_limit = '48:00:00'
# Where runs are written. The .xyz files are large (see disk note below), so point
# this at a scratch/project filesystem if your home directory has a quota.
runs_root = 'runs'
# After Smoldyn finishes, each job converts its .xyz into training data (.pt, .json)
# and an animation (.gif) with pbs_to_training_data.py (kept next to this script),
# using the Python that runs this script -- so run it with the env active:
#   conda activate binns && python generate_cfg.py
python_exec = sys.executable
converter = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pbs_to_training_data.py')
if not os.path.exists(converter):
    sys.exit(f'ERROR: {converter} not found; put pbs_to_training_data.py next to generate_cfg.py')
try:
    import torch, numpy, matplotlib  # noqa: F401  (only checking the env)
except ImportError:
    sys.exit(f"ERROR: {python_exec} cannot import torch/numpy/matplotlib; run 'conda activate binns' first")
if not os.access(smoldyn_exec, os.X_OK):
    sys.exit(f'ERROR: Smoldyn not found at {smoldyn_exec}')
# The burn-in is read with the converter's own parser.
sys.path.insert(0, os.path.dirname(converter))
from pbs_to_training_data import parse_cfg, read_frames

# --- Initial condition ---
# True : start from the steady-state composition measured by a short burn-in
#        (no fast transient; the data then sit near steady state).
# False: start with ALL Cdc42 as cytosolic GDP and all Bem1-GEF cytosolic (the
#        original start). Most Cdc42 activates within the first few seconds and
#        the polarity sites form over the next ~minute, so the data sweep a wide
#        range of active AND inactive Cdc42 -- what the network needs to tell
#        how the reaction depends on each. Sample it densely (see below).
START_FROM_STEADY_STATE = False

# Create the directory that stores the simulations (a new folder per start and
# sampling, so runs with different settings never mix)
dir_name = 'smart_ic' if START_FROM_STEADY_STATE else 'all_gdp_dense'
directory = os.path.join(runs_root, dir_name)
if not os.path.exists(directory):
    os.makedirs(directory)

# Simulation parameters
# Output every samplingrate * dt = 50000 * 1e-4 = 5 s -> 101 frames over 500 s.
# (tstop = 300 with samplingrate = 30000 also gives 101 frames, 3 s apart, all in the
# window where the spot forms and coarsens.)
# Disk: ~17 bytes per molecule per frame, so length 50 / scale 3 (~330k molecules)
# is ~5.5 MB per frame, ~0.6 GB per run. Check quota before submitting.
# all_gdp_dense: a frame every samplingrate * dt = 5000 * 1e-4 = 0.5 s for 100 s
# -> 201 frames (the converter keeps them all; use its --stride 2 for 101).
# Even sampling throughout, as a microscope would record.
tstop = 100
samplingrate = 5000
k_l = 8.8623
c_cdc42 = 3000 / (k_l**2)
c_BemGEF = 400 / (k_l**2)
c_FarGEF = 30 / (k_l**2)

# Range of seeds and lengths
# random_seeds = range(1, 11)  # 1 to 10
# lengths = range(25, 51, 5)    # 5 to 50
# scales = range(1, 4, 1)
random_seeds = range(1, 4)
lengths = [10]
# Scale 1 keeps active and inactive Cdc42 within ~2x of each other (~60% active
# at steady state). At scale 3-5 ~90% is active, the inactive pool is nearly used
# up everywhere, and the network cannot tell how the reaction depends on it.
scales = [1]

# Steady-state composition, measured once per scale (shared by all seeds and lengths)
if START_FROM_STEADY_STATE:
    print('Burn-in: measuring the steady-state composition for each scale')
    burn_dir = os.path.join(directory, 'burn_in')
    steady = steady_state_densities(scales, burn_dir)

# Create the shell script for SLURM
run_script_path = os.path.join(directory, 'run.sh')
with open(run_script_path, 'w') as fid:
    fid.write('#!/bin/bash\n\n')

    for j in random_seeds:
        for l in lengths:
            for scale in scales:
                curr_seed = j
                curr_length = l
                curr_fileprefix = f'length_{curr_length}_scale_{scale}-seeds_{curr_seed}'

                # Rounding to match MATLAB's integer molecule counts
                n_Cdc42 = round(c_cdc42 * (l**2)) * scale
                n_BemGEF = round(c_BemGEF * (l**2)) * scale
                n_FarGEF = round(c_FarGEF * (l**2)) * scale
                initial_counts = (initial_counts_at_steady_state(steady[scale], l, n_Cdc42, n_BemGEF)
                                  if START_FROM_STEADY_STATE else None)   # None = all cytosolic

                # Call the function to generate .cfg
                smoldyn_cfg_variable_size(
                    curr_fileprefix,
                    directory,
                    tstop,
                    samplingrate,
                    n_Cdc42,
                    n_BemGEF,
                    n_FarGEF,
                    l,
                    curr_seed,
                    initial_counts=initial_counts
                )

                # Define the config file name
                cfg_name = f'{curr_fileprefix}.cfg'

                # Write the sbatch command. -D runs the job inside the run directory,
                # so Smoldyn finds the .cfg and writes the .xyz next to it no matter
                # where run.sh is launched from. Using --wrap requires the command
                # to be a single string. The conversion (.pt, .json, .gif) runs only
                # if Smoldyn finished successfully (&&).
                cmd = f'sbatch -p {partition} -N 1 -J Smoldyn -t {time_limit} --mem=5g -D {os.path.abspath(directory)} --wrap="{smoldyn_exec} {cfg_name} && {python_exec} {converter} {curr_fileprefix}.xyz --gif"\n'
                fid.write(cmd)