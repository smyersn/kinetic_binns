import os
import json
import subprocess
import itertools
from pathlib import Path

# ==========================================
# --- CONFIGURATION OPTION ---
# "combinations": Tests every possible combination of the lists below.
# "lockstep": Pairs parameters directly (index 0 with index 0, etc.).
#             *Note: In lockstep mode, ALL lists must be the exact same length!
ITERATION_MODE = "combinations" 
# ==========================================

# # 1. Define parameter space
# save_path = "/work/users/s/m/smyersn/elston/projects/kinetics_binns/data/custom_equation"

# reactions = ["custom_equation", "custom_equation", "custom_equation", "custom_equation"]
# dus = [0.01, 0.01, 0.01, 0.01]
# dvs = [1, 1, 1, 1]
# a_params = [1, 4, 8, 8] 
# b_params = [1, 4, 1, 4]
# k_params = [0.1, 0.05, 0.5, 0.05]

# 1. Define parameter space
save_path = "/hpc/home/nsmyers1/projects/kinetic_binns/data/wave_pinning/diff_coeffs_du_sweep"

reactions = ["wave_pinning"]
dus = [0.01, 0.025, 0.05, 0.1]
dvs = [0.1, 1, 10]
a_params = [1] 
b_params = [1]
k_params = [0.01]

# Make sure the target data directory exists
os.makedirs(save_path, exist_ok=True)

# 2. Group lists
keys = ['reaction', 'du', 'dv', 'a', 'b', 'k']
iterables = [reactions, dus, dvs, a_params, b_params, k_params]

# 3. Generate Iteration Strategy
if ITERATION_MODE == "combinations":
    combo_generator = itertools.product(*iterables)

elif ITERATION_MODE == "lockstep":
    # Safety Check: zip() silently truncates lists to the shortest one. 
    # We raise an error if you accidentally mismatch your list lengths.
    lengths = [len(lst) for lst in iterables]
    if len(set(lengths)) > 1:
        raise ValueError(f"Lockstep Error: All lists must be the exact same length! Lengths found: {dict(zip(keys, lengths))}")
    
    combo_generator = zip(*iterables)

else:
    raise ValueError("ITERATION_MODE must be either 'combinations' or 'lockstep'")


# 4. Iterate and Submit
for combo in combo_generator:
    # Map combo back to dictionary for easy access
    c = dict(zip(keys, combo))
    
    reaction, du, dv, a, b, k = c['reaction'], c['du'], c['dv'], c['a'], c['b'], c['k']
    
    # --- Build and save the configuration JSON ---
    config_dict = {
        "save_path": save_path,
        "reaction": reaction,
        "du": du,
        "dv": dv,
        "a": a,
        "b": b,
        "k": k
    }
    
    config_filename = f"{reaction}_du_{du}_dv_{dv}_a_{a}_b_{b}_k_{k}.json"
    config_filepath = os.path.join(save_path, config_filename)
    
    with open(config_filepath, "w") as f:
        json.dump(config_dict, f, indent=4)
    
    # --- Construct and run Slurm submission ---
    python_script = "/work/users/s/m/smyersn/elston/projects/kinetics_binns/modules/simulation/simulation.py"
    
    wrap_command = f"source ~/.bashrc && conda activate binns && python {python_script} {config_filepath}"
    
    out_filename = f"{reaction}_du_{du}_dv_{dv}_a_{a}_b_{b}_k_{k}.out"
    out_file = os.path.join(save_path, out_filename)
    
    # sbatch_cmd = [
    #     "sbatch", "-p", "volta-gpu", "-N", "1", "-n", "1", 
    #     "--mem=32g", "--qos", "gpu_access", "--gres=gpu:1", 
    #     "-t", "1:00:00", 
    #     f"--output={out_file}",
    #     f"--wrap={wrap_command}"
    # ]    
    
    # CPU Alternative
    sbatch_cmd = [
        "sbatch", "-p", "common", "-N", "1", "-n", "1", 
        "--cpus-per-task=4", "--mem=16g", "-t", "12:00:00", 
        f"--output={out_file}",
        f"--wrap={wrap_command}"
    ]

    subprocess.run(sbatch_cmd)