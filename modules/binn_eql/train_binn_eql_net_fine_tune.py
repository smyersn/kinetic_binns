import sys, os, re
import importlib
file_dir = os.path.dirname(os.path.realpath(__file__))
repo_start = f'{file_dir}/../../'
sys.path.append(repo_start)

from modules.utils.imports import *
from modules.binn_eql.model_wrapper_2d import model_wrapper
from modules.binn_eql.build_binn_eql_net import BINN
from modules.binn_eql.plot_param_history import plot_param_history
from modules.loaders.format_data import format_data_torch
from modules.utils.noise_and_interpolate import noise_and_interpolate
from modules.utils.training_test_split import training_test_split
from modules.analysis.generate_loss_curves import generate_loss_curves
from modules.binn_eql.visualize_surface import visualize_surface
from modules.loaders.sims_and_animations import (simulate_uvmlp, simulate_feql,
                                                 format_training_data_for_animation,
                                                 animate_uarray, animate_residuals)
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
pde_weight=float(config['pde_weight'])
l0_weight = float(config['l0_weight'])
warm_up = float(config['warm_up'])
lux_tax = float(config['lux_tax'])
param_bounds = float(config['param_bounds'])

dir_name = sys.argv[1]

# Set training hyperparameters
# epochs = 10
epochs = 100_000
# rel_save_thresh = 0.01

# Get GPU
device = 'cuda'

# Load training data (columns: x*dimensions, t, species concentrations)
data = torch.load(training_data_path)
u_array, x_array, t_array = data['u_array'], data['x_array'], data['t_array']
training_data = format_data_torch(u_array, x_array, t_array)
training_data = training_data[training_data[:, dimensions] <= 60]

# Add noise to training data if specified in config file
if epsilon != 0 or points != 0:
    training_data = noise_and_interpolate(training_data, points, epsilon, 
                                          dimensions, species, 
                                          multiplicative_noise=False)

training_u_array, training_times = format_training_data_for_animation(training_data)
animate_uarray(training_u_array, training_times, f'{dir_name}/training_data_sim')


########
# DO TRIANGLE BS IN PYTORCH
########




# Create triangle mesh from min and max uv vals seen in training data
u_triangle_mesh, v_triangle_mesh = lltriangle(to_numpy(training_data[:, -2:]), 
                                                to_numpy(training_data[:, -1:]))
# Create 1d arrays from meshes
u_triangle, v_triangle = np.ravel(u_triangle_mesh), np.ravel(v_triangle_mesh)

# Create separate variables for arrays containing and not containing nans
uv_nans = np.stack((u_triangle, v_triangle), axis=1)
mask = ~np.isnan(uv_nans).any(axis=1)
uv = torch.from_numpy(uv_nans[mask]).to(device)

# Get params for calculating true surface
filename = os.path.basename(training_data_path)
numbers = re.findall(r'(?<=_)(?:\d*\.\d+|\d+)', filename)

if len(numbers) == 3:
    params = list(map(float, numbers))
else:
    params = 1, 1, 0.01
    
# Generate true surface
F_true = wave_pinning(uv_nans, params).reshape(501, 501)

# Split training data
batch_size=int(0.1*len(training_data))
# train_loader, val_loader = training_test_split(training_data, batch_size, species)
train_data, val_data = training_test_split(training_data, device)

# initialize model and compile
binn = BINN(
    dimensions=dimensions,
    species=species, 
    train_data=train_data, 
    duplicates=duplicates,
    diff_coeffs=diff_coeffs,
    degree=degree,
    param_bounds=param_bounds)

binn.to(device)

# Initialize optimizer
param_groups = [
    {'params': binn.surface_fitter.parameters(), 
     'lr': 1e-2, 
     'weight_decay': 1e-5,  
     'name': 'surface'},
    
    {'params': binn.reaction.parameters(), 
     'lr': 1e-3, 
     'weight_decay': 0.0,   
     'name': 'reaction'}]

if binn.diffusion_fitter:
    param_groups.append({
        'params': binn.diffusion_fitter.parameters(), 
        'lr': 1e-3,
        'weight_decay': 0.0,
        'name': 'diffusion'})

opt = torch.optim.AdamW(param_groups, weight_decay=0.0)

# Initialize Scheduler (OneCycleLR)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    opt,
    # Provide a list of max_lrs matching the order of param_groups
    max_lr=[group['lr'] for group in param_groups],
    total_steps=int(warm_up),  
    pct_start=0.3,        
    div_factor=25,               
    final_div_factor=1e4)

model = model_wrapper(
    model=binn,
    optimizer=opt,
    scheduler=scheduler,
    loss=binn.loss,
    dir_name=dir_name,
    save_name=f'{dir_name}/binn')

# train jointly
param_history, train_loss_dict, val_loss_dict = model.fit(
    train_data=train_data,
    val_data=val_data,
    pde_weight=pde_weight,
    l0_weight=l0_weight,
    warm_up=warm_up,
    lux_tax=lux_tax,
    batch_size=batch_size,
    epochs=epochs,
    early_stopping=5000)

generate_loss_curves(train_loss_dict, val_loss_dict, dir_name, 20, 'training_loss_curves')

plot_param_history(binn, param_history, f"{dir_name}/param_history.png")

# Load and prune final model
model.load(f"{dir_name}/binn_best_val_model", device=device)

# Print final equation
fn = f'{dir_name}/equation.txt'
file = open(fn, 'a')

file.write(f'Final equation:\n')
for term in model.model.generate_equation():
    file.write(f'{term}\n')

if not diff_coeffs:
    file.write(f'\nDiff. coeffs. after fine tuning:\n')
    
    file.write(f'{[D.item() for D in model.model.diffusion_fitter()]}\n')
        
file.close()

# Generate learned learned surface after initial training
# Prepare Scaled Inputs (Physical -> Dimensionless)
uv_nans_scaled = np.zeros_like(uv_nans)
s_u = model.model.max_scale[0, 0].cpu().detach().numpy()
s_v = model.model.max_scale[0, 1].cpu().detach().numpy()
uv_nans_scaled[:, 0] = uv_nans[:, 0] / s_u
uv_nans_scaled[:, 1] = uv_nans[:, 1] / s_v

# Generate learned learned surface after initial training
uv_nans_scaled = np.zeros_like(uv_nans)
uv_nans_scaled[:, 0] = uv_nans[:, 0] / model.model.max_scale[0, 0].cpu().detach().numpy()
uv_nans_scaled[:, 1] = uv_nans[:, 1] / model.model.max_scale[0, 1].cpu().detach().numpy()
F_mlp_unformatted = model.model.reaction(torch.tensor(uv_nans_scaled).float().to(device))
F_mlp = F_mlp_unformatted.cpu().detach().numpy().reshape(501, 501)

# Visualize surfaces
visualize_surface(dir_name, u_triangle_mesh, v_triangle_mesh,
                F_true, F_mlp, 'feql_surface')

# Simulate uvmlp
uvmlp_u_array, uvmlp_times = simulate_uvmlp(training_data, model)
animate_uarray(uvmlp_u_array, uvmlp_times, f'{dir_name}/uvmlp_sim')
animate_residuals(uvmlp_u_array, training_u_array, uvmlp_times, 
                  name=f'{dir_name}/uvmlp_residuals')

# Simulate feql
feql_u_array, feql_times = simulate_feql(training_data, model)
animate_uarray(feql_u_array, feql_times, f'{dir_name}/feql_sim')
animate_residuals(feql_u_array, training_u_array, feql_times, 
                  name=f'{dir_name}/feql_residuals')