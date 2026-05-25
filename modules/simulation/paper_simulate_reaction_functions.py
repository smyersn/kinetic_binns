import os
import json
import subprocess
import itertools

# 1. Define your universal parameter grids
save_path = "/work/users/s/m/smyersn/elston/projects/kinetics_binns/data/paper_functions"
os.makedirs(save_path, exist_ok=True)

dus = [0.01]
dvs = [1]

# Sweeping the same values for all polynomial and hill terms
a_params = [1, 2, 4] 
b_params = [1, 2, 4]
k_params = [0.01, 0.1, 0.5] 
n_params = [2, 3]

# Map all available parameters to a master dictionary
# Notice how k1/k2 and n1/n2 point to the same master lists
param_pool = {
    'du': dus, 'dv': dvs, 
    'a': a_params, 'b': b_params, 
    'k': k_params, 'n': n_params,
    'k1': k_params, 'k2': k_params,
    'n1': n_params, 'n2': n_params
}

# 2. Define which parameters belong to which equation (matching your functions)
equation_requirements = {
    "hill_poly": ['du', 'dv', 'a', 'b', 'k', 'n'],
    "poly_poly": ['du', 'dv', 'a', 'b'],
    "hill_hill": ['du', 'dv', 'a', 'b', 'k1', 'n1', 'k2', 'n2'],
    "poly_hill": ['du', 'dv', 'a', 'b', 'k', 'n']
}

# 3. Iterate per equation to avoid redundant jobs
for reaction, required_keys in equation_requirements.items():
    
    # Extract only the lists needed for this specific equation
    iterables = [param_pool[key] for key in required_keys]
    
    # Generate combinations strictly for the parameters this equation actually uses
    for combo in itertools.product(*iterables):
        
        # Map combo back to a dictionary (e.g., {'du': 0.01, 'a': 5, 'k1': 0.1, ...})
        c = dict(zip(required_keys, combo))
        
        # --- Build and save the configuration JSON ---
        config_dict = {
            "save_path": save_path,
            "reaction": reaction,
            
            # Standard PDE params
            "du": c.get('du'),
            "dv": c.get('dv'),
            
            # Equation params. .get() safely defaults to 0 if the equation 
            # doesn't use that specific parameter.
            "a": c.get('a'),
            "b": c.get('b'),
            "k": c.get('k', 0), 
            "n": c.get('n', 0),
            "k1": c.get('k1', 0),
            "k2": c.get('k2', 0),
            "n1": c.get('n1', 0),
            "n2": c.get('n2', 0)
        }
        
        # Create a clean filename dynamically based ONLY on used parameters
        # This keeps the filenames short and accurate to the specific function
        param_string = "_".join([f"{key}_{val}" for key, val in c.items() if key not in ['du', 'dv']])
        config_filename = f"{reaction}_du_{c['du']}_dv_{c['dv']}_{param_string}.json"        
        config_filepath = os.path.join(save_path, config_filename)
        
        with open(config_filepath, "w") as f:
            json.dump(config_dict, f, indent=4)
            
        # --- Construct and run Slurm submission ---
        python_script = "/work/users/s/m/smyersn/elston/projects/kinetics_binns/modules/simulation/paper_simulation.py"
        wrap_command = f"source ~/.bashrc && conda activate binns && python {python_script} {config_filepath}"
        
        out_filename = config_filename.replace('.json', '.out')
        out_file = os.path.join(save_path, out_filename)
        
        sbatch_cmd = [
            "sbatch", "-p", "general", "-N", "1", "-n", "1", 
            "--cpus-per-task=4", "--mem=16g", "-t", "2:00:00", 
            f"--output={out_file}",
            f"--wrap={wrap_command}"
        ]

        subprocess.run(sbatch_cmd)