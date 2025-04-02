import sys, os
file_dir = os.path.dirname(os.path.realpath(__file__))
repo_start = f'{file_dir}/../../'
sys.path.append(repo_start)

from modules.utils.imports import *
from modules.utils.histogram import create_data_histogram
from modules.symbolic_net.model_wrapper import model_wrapper
from modules.symbolic_net.build_symbolic_net import symbolic_net
from modules.utils.training_test_split import training_test_split
from modules.analysis.generate_loss_curves import generate_loss_curves
from modules.generate_data.simulate_system import wave_pinning_reaction
from modules.symbolic_net.write_terms import write_terms
from modules.symbolic_net.visualize_surface import visualize_surface
from modules.symbolic_net.individual import individual
from modules.binn.build_binns_2d_diffusion import BINN
from modules.loaders.format_data import format_data_general

# Load BINN parameters from config file
binn_path = '/work/users/s/m/smyersn/elston/projects/kinetics_binns/development/binn_testing/debugged_development/comprehensive/runs/2d_repeats/2d_alpha_0.5_repeat_5'
config = {}
exec(Path(f'{binn_path}/config.cfg').read_text(encoding="utf8"), {}, config)

dimensions = int(config['dimensions'])
species = int(config['species'])

density_weight = int(config['density_weight'])

uv_layers = int(config['uv_layers'])
uv_neurons = int(config['uv_neurons'])
f_layers = int(config['f_layers'])
f_neurons = int(config['f_neurons'])

diffusion = bool(config['diffusion'])
alpha = float(config['alpha'])

uv_arch = uv_layers * [uv_neurons] + [2]
f_arch = f_layers * [f_neurons] + [1]


# Load symbolic net parameters from config file
config = {}
exec(Path(f'{sys.argv[1]}/config.cfg').read_text(encoding="utf8"), {}, config)

training_data_path = config['training_data_path']
species = int(config['species'])
degree = int(config['degree'])
nonzero_term_reg = int(config['nonzero_term_reg'])
l1_reg = float(config['l1_reg'])
param_bounds = float(config['param_bounds'])
density_weight = float(config['density_weight'])
learning_rate = float(config['learning_rate'])

dir_name = sys.argv[1]

# Set training hyperparameters
device = 'cpu'

# Initialize BINN
binn = BINN(
    species=species, 
    dimensions=dimensions,
    uv_layers=uv_arch, 
    f_layers=f_arch,
    diff=diffusion)

binn.to(device)

opt = torch.optim.Adam(list(binn.parameters()), lr=1e-3)
model = model_wrapper(
    model=binn,
    optimizer=opt,
    loss=binn.loss,
    augmentation=None,
    save_name=f'{binn_path}/binn')

model.load(f"{binn_path}/binn_best_val_model", device=device)

# Generate training data with BINN
training_data = format_data_general(dimensions, species, file=training_data_path)

uv = training_data[:, -2:]

# Generate data density histogram from simulation if density weight is nonzero
hist = None
edges = None

if density_weight != 0:
    hist, edges = create_data_histogram(torch.from_numpy(uv), device)   

u_triangle_mesh, v_triangle_mesh = lltriangle(uv[:, 0], uv[:, 1])
u, v = np.ravel(u_triangle_mesh), np.ravel(v_triangle_mesh)

# Calculate surface to fit with symbolic net using BINN
uv = np.column_stack((u, v))
F_true = to_numpy(model.model.reaction(to_torch(uv)[:, None])).flatten()

training_data_nans = np.stack((u, v, F_true), axis=1)

# Remove nans from training data
mask = ~np.isnan(training_data_nans).any(axis=1)
training_data = training_data_nans[mask]

# initialize symbolic net
sym_net = symbolic_net(species, degree, param_bounds, device, l1_reg,
                       nonzero_term_reg)
sym_net.to(device)

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
        density_weight = float(config['density_weight'])
            
        # Load model    
        weights = torch.load(model, map_location=device)
        sym_net.load_state_dict(weights)

        # Generate individual and fix terms
        ind = individual(sym_net.params, species, degree)
        
        ind.fix_insignificant_terms(torch.tensor(training_data))
        ind.fix_cheating_hill_functions(torch.tensor(training_data))
        
        uv = torch.from_numpy(training_data[:, :-1])
        true_vals = torch.from_numpy(training_data[:, -1])
        predicted_vals = ind.predict_f(torch.from_numpy(training_data[:, :-1]))
        
        # Calculate AIC
        # aic, bic = ind.abic(uv, true_vals, predicted_vals, 
        #                     density_weight, hist, edges)
        aic, bic = ind.abic(uv, true_vals, predicted_vals, 
                            0, None, None)
        
        individuals.append((child_dir, ind.params, aic, bic))

# Rank learned equations from least (best) to greatest (worst) AIC
individuals_sorted = sorted(individuals, key=lambda x: x[2])

# Write equations to file
fn = f'{parent_dir}/equations_ranked.txt'
file = open(fn, 'w')

for i in range(len(individuals_sorted)):
    # load model   
    ind = individual(individuals_sorted[i][1], species, degree)
    file.write(f'Directory: {individuals_sorted[i][0]}\n')
    file.write(f'AIC: {individuals_sorted[i][2]}\n')
    terms = ind.write_terms()
    for term in terms:
        file.write(f'{term}\n')
    file.write('\n')