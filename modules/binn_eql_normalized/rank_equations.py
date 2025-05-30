import sys, os
file_dir = os.path.dirname(os.path.realpath(__file__))
repo_start = f'{file_dir}/../../'
sys.path.append(repo_start)

from modules.utils.imports import *
from modules.loaders.format_data import format_data_general
from modules.generate_data.simulate_system import wave_pinning
from modules.binn_symbolic_net.individual import individual
from modules.genetic_algorithm.genetic_algorithm_base.custom_deap_functions import (
    calculate_poly_terms, calculate_hill_terms)


# load params from configuration file
config = {}
exec(Path(f'{sys.argv[1]}/config.cfg').read_text(encoding="utf8"), {}, config)

# Training data params
training_data_path = config['training_data_path']
species = int(config['species'])
dimensions = int(config['dimensions'])
epsilon = float(config['epsilon'])
points = int(config['points'])

# BINN params
diff_coeffs = [float(x) for x in config['diff_coeffs'].strip("()").split()]

# Symbolic Net params
degree = int(config['degree'])
l05_reg = float(config['l05_reg'])
param_bounds = float(config['param_bounds'])

dir_name = sys.argv[1]

# Set training hyperparameters
device = 'cpu'

# Load training data (columns: x*dimensions, t, species concentrations)
training_data = format_data_general(dimensions, species, file=training_data_path)

# Get all uv vals seen during training
uv_training_data = training_data[:, -2:]

# Create triangle mesh from min and max uv vals seen in training data
u_triangle_mesh, v_triangle_mesh = lltriangle(uv_training_data[:, 0], 
                                                uv_training_data[:, 1])
# Create 1d arrays from meshes
u_triangle, v_triangle = np.ravel(u_triangle_mesh), np.ravel(v_triangle_mesh)

# Create separate variables for arrays containing and not containing nans
uv_nans = np.stack((u_triangle, v_triangle), axis=1)
mask = ~np.isnan(uv_nans).any(axis=1)
uv = uv_nans[mask]

# Generate true surface
params = 1, 1, 0.01
F_true = torch.from_numpy(wave_pinning(u_triangle, v_triangle, params))

### ANALYSIS

individuals = []

# Calculate AIC and BIC for all learned equations
parent_dir = '/'.join(dir_name.rstrip('/').split('/')[:-1])

child_dirs = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d)) and 'binn_best_val_model' in os.listdir(os.path.join(parent_dir, d))]

for child_dir in child_dirs:
    
    model = f'{parent_dir}/{child_dir}/binn_best_val_model'

    if os.path.exists(model): 
        # Load density weight for directory
        config = {}
        exec(Path(f'{parent_dir}/{child_dir}/config.cfg').read_text(
            encoding="utf8"), {}, config)
    
        # Load model    
        weights = torch.load(model, map_location=device)

        # Generate individual and fix terms
        poly_terms = calculate_poly_terms(species, degree)
        hill_terms = calculate_hill_terms(species, degree)

        ind = individual(weights['reaction.params'], poly_terms, hill_terms)
        
        ind.fix_insignificant_terms(torch.tensor(training_data))
        ind.fix_cheating_hill_functions(torch.tensor(training_data))

        preds = ind.predict_f(torch.from_numpy(uv_nans))
        
        # Calculate AIC
        F_true_formatted = F_true[~torch.isnan(F_true)]
        preds_formatted = preds[~torch.isnan(preds)]
        aic, bic = ind.abic(F_true_formatted, preds_formatted)

        individuals.append((child_dir, ind, aic, bic))

# Rank learned equations from least (best) to greatest (worst) AIC
individuals_sorted = sorted(individuals, key=lambda x: x[2])

# Write equations to file
fn = f'{parent_dir}/equations_ranked.txt'
file = open(fn, 'w')

for i in range(len(individuals_sorted)):
    # load model   
    file.write(f'Directory: {individuals_sorted[i][0]}\n')
    file.write(f'AIC: {individuals_sorted[i][2]}\n')
    terms = individuals_sorted[i][1].write_terms()
    for term in terms:
        file.write(f'{term}\n')
    file.write('\n')