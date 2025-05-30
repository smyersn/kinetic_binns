import sys, os
import importlib
file_dir = os.path.dirname(os.path.realpath(__file__))
repo_start = f'{file_dir}/../../'
sys.path.append(repo_start)

from modules.utils.imports import *
from modules.binn_eql.model_wrapper_2d import model_wrapper
from modules.binn_eql.build_binn_eql_net import BINN
from modules.loaders.format_data import format_data_general
from modules.loaders.visualize_training_data import animate_data
from modules.utils.noise_and_interpolate import noise_and_interpolate
from modules.utils.training_test_split import training_test_split
from modules.analysis.generate_loss_curves import generate_loss_curves
from modules.binn_eql.simulate_surface import simulate_surface
from modules.binn_eql.visualize_surface import visualize_surface
from modules.generate_data.simulate_system import wave_pinning

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
duplicates = int(config['duplicates'])
degree = int(config['degree'])
gls_weight=float(config['gls_weight'])
pde_weight=float(config['pde_weight'])
l05_weight = float(config['l05_weight'])
l1_weight = float(config['l1_weight'])
param_bounds = float(config['param_bounds'])

dir_name = sys.argv[1]

# Set training hyperparameters
epochs = int(1e6)
rel_save_thresh = 0.01

# Get GPU
device = 'cuda'

# Load training data (columns: x*dimensions, t, species concentrations)
training_data = format_data_general(dimensions, species, file=training_data_path)

# Add noise to training data if specified in config file
if epsilon != 0 or points != 0:
    training_data = noise_and_interpolate(training_data, points, epsilon, dimensions, species)

animate_data(training_data, dimensions, species, name=f'{dir_name}/training_data')

# Create triangle mesh from min and max uv vals seen in training data
u_triangle_mesh, v_triangle_mesh = lltriangle(training_data[:, -2:], 
                                                training_data[:, -1:])
# Create 1d arrays from meshes
u_triangle, v_triangle = np.ravel(u_triangle_mesh), np.ravel(v_triangle_mesh)

# Create separate variables for arrays containing and not containing nans
uv_nans = np.stack((u_triangle, v_triangle), axis=1)
mask = ~np.isnan(uv_nans).any(axis=1)
uv = torch.from_numpy(uv_nans[mask]).to(device)

# Generate true surface
params = 1, 1, 0.01
F_true = wave_pinning(uv_nans, params).reshape(501, 501)

# Split training data
x_train, y_train, x_val, y_val = training_test_split(training_data, dimensions, device)

# initialize model and compile
binn = BINN(
    dimensions=dimensions,
    species=species, 
    duplicates=duplicates,
    data=x_train.cpu(), 
    diff_coeffs=diff_coeffs,
    degree=degree,
    gls_weight=gls_weight,
    pde_weight=pde_weight,
    l05_weight=l05_weight,
    l1_weight=l1_weight,
    param_bounds=param_bounds)

binn.to(device)

parameters = binn.parameters()

opt = torch.optim.Adam(parameters, lr=0.001)

model = model_wrapper(
    model=binn,
    optimizer=opt,
    loss=binn.loss,
    augmentation=None,
    save_name=f'{dir_name}/binn')

# train jointly
train_loss_dict, val_loss_dict = model.fit(
    x=x_train,
    y=y_train,
    batch_size=int(0.05*len(training_data)),
    epochs=epochs,
    validation_data=[x_val, y_val],
    early_stopping=500,
    rel_save_thresh=rel_save_thresh)

generate_loss_curves(train_loss_dict, val_loss_dict, dir_name, 20, 'training_loss_curves')

# Load model
model.load(f"{dir_name}/binn_best_val_model", device=device)

# Generate learned learned surface after initial training
F_mlp_unformatted = model.model.reaction(torch.tensor(uv_nans).float().to(device))
F_mlp = F_mlp_unformatted.cpu().detach().numpy().reshape(501, 501)

# Manual refinement and equation printing       
fn = f'{dir_name}/equation.txt'
file = open(fn, 'w')

if not diff_coeffs:
    file.write(f'Diff. coeffs. before fine tuning:\n')
    file.write(f'{[D.item() for D in model.model.diffusion_fitter()]}\n')

file.write(f'\nOriginal Equation:\n')
for term in model.model.generate_equation():
    file.write(f'{term}\n')

file.write(f'\nNo insignificant terms:\n')
model.model.remove_insignificant_terms(uv)
for term in model.model.generate_equation():
    file.write(f'{term}\n')

file.write(f'\nNo cheating Hill functions:\n')
model.model.fix_cheating_hill_functions(uv)
for term in model.model.generate_equation():
    file.write(f'{term}\n')

file.close()

# Generate learned surface after correction
F_mlp_corrected_unformatted = model.model.reaction(torch.tensor(uv_nans).float().to(device))
F_mlp_corrected = F_mlp_corrected_unformatted.cpu().detach().numpy().reshape(501, 501)

# Fine tune model
parameters = model.model.parameters()

opt = torch.optim.Adam(parameters, lr=0.001)

model = model_wrapper(
    model=binn,
    optimizer=opt,
    loss=binn.loss,
    augmentation=None,
    save_name=f'{dir_name}/binn')

# train jointly
train_loss_dict, val_loss_dict = model.fit(
    x=x_train,
    y=y_train,
    batch_size=int(0.05*len(training_data)),
    epochs=5000,
    validation_data=[x_val, y_val],
    early_stopping=500,
    rel_save_thresh=rel_save_thresh,
    fine_tune=True)

generate_loss_curves(train_loss_dict, val_loss_dict, dir_name, 2.5, 'fine_tuning_loss_curves')

model.load(f"{dir_name}/binn_best_val_fine_tuned_model", device=device)

# Print fine tuned equation
file = open(fn, 'a')

file.write(f'\nFine tuned equation:\n')
model.model.remove_insignificant_terms(uv)
model.model.fix_cheating_hill_functions(uv)
for term in model.model.generate_equation():
    file.write(f'{term}\n')

if not diff_coeffs:
    file.write(f'\nDiff. coeffs. after fine tuning:\n')
    file.write(f'{[D.item() for D in model.model.diffusion_fitter()]}\n')
        
file.close()

# Generate learned surface after correction
F_mlp_fine_tuned_unformatted = model.model.reaction(torch.tensor(uv_nans).float().to(device))
F_mlp_fine_tuned = F_mlp_fine_tuned_unformatted.cpu().detach().numpy().reshape(501, 501)

# Visualize surfaces
visualize_surface(dir_name, u_triangle_mesh, v_triangle_mesh,
                F_true, F_mlp, F_mlp_corrected, F_mlp_fine_tuned, 
                'f_mlp_surfaces')