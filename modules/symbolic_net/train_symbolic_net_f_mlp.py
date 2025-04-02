import sys, os
file_dir = os.path.dirname(os.path.realpath(__file__))
repo_start = f'{file_dir}/../../'
sys.path.append(repo_start)

from modules.utils.imports import *
from modules.utils.histogram import create_data_histogram
from modules.symbolic_net.model_wrapper import model_wrapper
from modules.binn.build_binns_2d_diffusion import BINN
from modules.symbolic_net.build_symbolic_net import symbolic_net
from modules.utils.training_test_split import training_test_split
from modules.analysis.generate_loss_curves import generate_loss_curves
from modules.loaders.format_data import format_data_general
from modules.generate_data.simulate_system import wave_pinning_reaction
from modules.symbolic_net.write_terms import write_terms
from modules.symbolic_net.visualize_surface import visualize_surface
from modules.symbolic_net.individual import individual

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
epochs = int(1e6)
rel_save_thresh = 0.05
device = 'cuda'

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
training_data = format_data_general(2, 2, file=training_data_path)
uv = training_data[:, -2:]
u_triangle_mesh, v_triangle_mesh = lltriangle(uv[:, 0], uv[:, 1])
u, v = np.ravel(u_triangle_mesh), np.ravel(v_triangle_mesh)

# Generate data density histogram from simulation if density weight is nonzero
hist = None
edges = None

if density_weight != 0:
    hist, edges = create_data_histogram(torch.from_numpy(uv), device)   

# Calculate true surface to fit (F_mlp)
uv = np.column_stack((u, v))
F_true = to_numpy(model.model.reaction(to_torch(uv)[:, None])).flatten()

training_data_nans = np.stack((u, v, F_true), axis=1)

# Remove nans from training data
mask = ~np.isnan(training_data_nans).any(axis=1)
training_data = training_data_nans[mask]

# Split training data
x_train, y_train, x_val, y_val = training_test_split(training_data, 1, device)
    
# initialize model and compile
sym_net = symbolic_net(species, degree, param_bounds, device, l1_reg,
                       nonzero_term_reg, density_weight, hist, edges)

sym_net.to(device)

# initialize optimizer
parameters = sym_net.parameters()
opt = torch.optim.Adam(parameters, lr=learning_rate)

model = model_wrapper(
    model=sym_net,
    optimizer=opt,
    loss=sym_net.loss,
    augmentation=None,
    save_name=f'{dir_name}/binn')

# train jointly
train_loss_dict, val_loss_dict = model.fit(
    x=x_train,
    y=y_train,
    batch_size=int(0.1*len(training_data)),
    epochs=epochs,
    callbacks=None,
    verbose=1,
    validation_data=[x_val, y_val],
    early_stopping=5000,
    rel_save_thresh=rel_save_thresh,
    density_weight=density_weight,
    hist=hist,
    edges=edges)

generate_loss_curves(train_loss_dict, val_loss_dict, dir_name)

### ANALYSIS

# Load model for analaysis
model.load(f"{dir_name}/binn_best_val_model", device=device)

# Generate surface for unrefined equation
F_mlp_unformatted = model.model.individual.predict_f(torch.tensor(training_data_nans).to(device))
F_mlp = F_mlp_unformatted.cpu().detach().numpy().reshape(501, 501)
visualize_surface(dir_name, u_triangle_mesh, v_triangle_mesh,
                  F_true.reshape(501, 501), F_mlp, 'f_mlp_surface_unrefined')

# Refine and print equation
ind = individual(model.model.params, species, degree)
    
fn = f'{dir_name}/equation.txt'
file = open(fn, 'w')

file.write(f'Original Equation:\n')
terms = write_terms(ind)
for term in terms:
    file.write(f'{term}\n')

file.write(f'\nNo insignificant terms:\n')
ind.fix_insignificant_terms(torch.tensor(training_data).to(device))
terms = ind.write_terms()
for term in terms:
    file.write(f'{term}\n')

file.write(f'\nNo cheating Hill functions:\n')
ind.fix_cheating_hill_functions(torch.tensor(training_data).to(device))
terms = ind.write_terms()
for term in terms:
    file.write(f'{term}\n')

file.close()

# Generate surface for refined equation
F_mlp_unformatted = ind.predict_f(torch.tensor(training_data_nans).to(device))
F_mlp = F_mlp_unformatted.cpu().detach().numpy().reshape(501, 501)
visualize_surface(dir_name, u_triangle_mesh, v_triangle_mesh,
                  F_true.reshape(501, 501), F_mlp, 'f_mlp_surface_refined')