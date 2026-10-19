import sys, os, json
from pathlib import Path
file_dir = os.path.dirname(os.path.realpath(__file__))
repo_start = f'{file_dir}/../../'
sys.path.append(repo_start)

import subprocess
import itertools

# Baseline sweep: PDE-FIND, weak SINDy and DeepMoD on the same datasets, noise
# levels and grids as the BINN sweeps. Each run directory gets a config.json
# that run_baseline.py reads; results land in <run dir>/result.json.
# Collect everything (baselines and BINN runs) with collect_benchmark.py.
#
# Directories that already hold a complete result.json are skipped, so the
# script can be rerun after failures without redoing finished runs.

# --- Datasets (These iterate in lockstep) ---
dir_path = "benchmark/1_baselines"

training_data_paths = [
    "/hpc/home/nsmyers1/projects/kinetic_binns/data/nonconserved/brusselator_du_0.02_dv_0.16_a_1.0_b_3.0.pt",
    "/hpc/home/nsmyers1/projects/kinetic_binns/data/nonconserved/schnakenberg_du_0.01_dv_0.1_a_0.1_b_0.8.pt",
    "/hpc/home/nsmyers1/projects/kinetic_binns/data/nonconserved/gray_scott_du_0.0004_dv_0.0002_feed_0.029_kill_0.057.pt",
]
reactions = ["brusselator", "schnakenberg", "gray_scott"]
params_list = [[1.0, 3.0], [0.1, 0.8], [0.029, 0.057]]
diff_coeffs_list = [[0.02, 0.16], [0.01, 0.1], [0.0004, 0.0002]]   # true D per dataset
mcas_list = [False] * len(training_data_paths)

dataset_configs = list(zip(training_data_paths, reactions, params_list, diff_coeffs_list, mcas_list))

# --- Methods: name -> (method, options). The name goes into the directory name. ---
# Every option not given takes the default documented at the top of
# modules/benchmarks/<method>.py. Keep options FIXED across datasets and noise
# levels; tuning them per dataset is the comparison reviewers will distrust.
methods = {
    "pdefind_fd":      ("pdefind",    {}),                                     # finite differences
    "pdefind_sg":      ("pdefind",    {"sg_window_t": 7, "sg_window_x": 7}),  # Savitzky-Golay
    "weak_sindy":      ("weak_sindy", {}),
    "deepmod_nn":      ("deepmod",    {"network": "nn"}),        # DeepMoD as published
    "deepmod_fourier": ("deepmod",    {"network": "fourier"}),   # DeepMoD with the BINN surface
}

# --- SWEEP HYPERPARAMETERS (These compute every combination) ---
repeats = 3                 # noise realisations; repeat k uses seed k
epsilons = [0.0, 0.01, 0.05, 0.1]
points = [50, 100]
degrees = [3]
include_constant_list = [True]
learn_diffusion_list = [False]          # [False, True] to also learn D
selection_tols = [0.01]                 # SINDy methods: term must hold > tol of the explainable range
simulate = True                         # integrate each learned PDE (pattern check)

keys = ['dataset_config', 'method_name', 'eps', 'pt', 'deg', 'const', 'learn_D', 'tol']
iterables = [dataset_configs, list(methods), epsilons, points, degrees,
             include_constant_list, learn_diffusion_list, selection_tols]

python_script = "/hpc/home/nsmyers1/projects/kinetic_binns/modules/scripts/run_baseline.py"


def is_complete(dir_name):
    """True if result.json exists and has its simulation (or simulation is off)."""
    path = os.path.join(dir_name, "result.json")
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            result = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    return "simulation" in result or not simulate


for combo in itertools.product(*iterables):
    c = dict(zip(keys, combo))

    data_path, reaction, params, true_D, mcas = c['dataset_config']
    method, options = methods[c['method_name']]
    options = dict(options)
    if method in ("pdefind", "weak_sindy"):
        options["selection_tol"] = c['tol']

    base = Path(data_path).stem
    base_model_name = f"{dir_path}/{base}/{c['method_name']}_"
    if c['eps'] != 0.0:      base_model_name += f"epsilon_{c['eps']}_"
    if c['pt'] != 0:         base_model_name += f"points_{c['pt']}_"
    base_model_name += f"degree_{c['deg']}_"
    if c['const']:           base_model_name += "const_"
    if c['learn_D']:         base_model_name += "learnD_"
    if method in ("pdefind", "weak_sindy"):
        base_model_name += f"tol_{c['tol']}_"

    background_commands = []
    for repeat in list(range(1, repeats + 1)):
        dir_name = f"{base_model_name}repeat_{repeat}"
        if is_complete(dir_name):
            continue
        os.makedirs(dir_name, exist_ok=True)

        config_dict = {
            "method": method,
            "training_data_path": data_path,
            "reaction": reaction,
            "params": params,
            "epsilon": c['eps'],
            "points": c['pt'],
            "diff_coeffs": [] if c['learn_D'] else true_D,
            "true_diff_coeffs": true_D,
            "degree": c['deg'],
            "include_constant": c['const'],
            "mcas": mcas,
            "seed": repeat,
            "simulate": simulate,
            "options": options,
        }
        with open(os.path.join(dir_name, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=4)

        # Log to run.log in the run folder, plus the exit code
        # (0 ok, 1 Python error, 137 out-of-memory kill, 139 segfault, 143 cancelled).
        background_commands.append(
            f'(python {python_script} {dir_name} >> {dir_name}/run.log 2>&1; '
            f'echo "=== process exited with code $? ===" >> {dir_name}/run.log) &')

    if not background_commands:
        continue

    cmd_string = (f'eval "$(conda shell.bash hook)" && conda activate binns; '
                  f'export PYTHONUNBUFFERED=1; export TORCHINDUCTOR_COMPILE_THREADS=1; '
                  f'{" ".join(background_commands)} wait')
    wrap_command = f"bash -c '{cmd_string}'"
    slurm_log_path = f"{base_model_name.rstrip('_')}_slurm-%j.out"

    if method == "deepmod":
        # Network training: GPU. The repeats share the GPU, as in the BINN sweeps.
        sbatch_cmd = [
            "sbatch", "-p", "gpu", "-N", "1", "-n", "1",
            "-c", "4", "--mem=48g", "--gres=gpu:1",
            "-t", "1-00:00:00",
            f"--output={slurm_log_path}",
            f"--wrap={wrap_command}"
        ]
    else:
        # SINDy methods are CPU-only least squares; the learned-PDE simulation is
        # the slow part on CPU. Weak SINDy needs a few GB per repeat at 200 points.
        sbatch_cmd = [
            "sbatch", "-p", "common", "-N", "1", "-n", "1",
            "-c", str(2 * len(background_commands)), "--mem=48g",
            "-t", "12:00:00",
            f"--output={slurm_log_path}",
            f"--wrap={wrap_command}"
        ]

    subprocess.run(sbatch_cmd)
