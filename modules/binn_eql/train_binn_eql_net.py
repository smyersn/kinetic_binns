import sys, os, re, json
from pathlib import Path
file_dir = os.path.dirname(os.path.realpath(__file__))
repo_start = f'{file_dir}/../../'
sys.path.append(repo_start)

from modules.utils.imports import *
from modules.utils.format_data import format_training_data_to_u_array
from modules.utils.noise_and_interpolate import noise_and_interpolate
from modules.utils.training_test_split import training_test_split
from modules.binn_eql.model_wrapper_2d import model_wrapper
from modules.binn_eql.build_binn_eql_net import BINN
from modules.analysis.plot_param_history import plot_param_history
from modules.analysis.generate_loss_curves import generate_loss_curves
from modules.analysis.visualize_surface import compare_surfaces_over_training_domain
from modules.simulation.animation import (animate_u_array, animate_residuals)
from modules.simulation.simulation import (simulate_uvmlp, simulate_feql)
from modules.simulation.reaction_functions import (wave_pinning, turing_type,
                                                   custom_equation) 

# Load params from configuration file
dir_name = sys.argv[1]

# 1. Load JSON directly
config_path = Path(dir_name) / 'config.json'
with open(config_path, 'r') as f:
    config = json.load(f)

# 2. Load variable from JSON
training_data_path = config['training_data_path']
species = config['species']           
dimensions = config['dimensions']     
epsilon = config['epsilon']           
points = config['points']             
params = config['params']             
diff_coeffs = config['diff_coeffs']   

duplicates = config['duplicates']
degree = config['degree']
pde_weight = config['pde_weight']
l0_weight = config['l0_weight']
warm_up = config['warm_up']
lux_tax = config['lux_tax']
param_bounds = config['param_bounds']

# 3. Map reaction function
reaction_map = {
    'wave_pinning': wave_pinning,
    'turing_type': turing_type,
    'custom_equation': custom_equation
}
reaction = reaction_map[config['reaction']]

# Set training hyperparameters
# epochs = 10
epochs = 100_000

# Get GPU
# device = 'cpu'
device = 'cuda'

# Load training data (columns: x*dimensions, t, species concentrations)
training_data = torch.load(training_data_path)['training_data']

# Add noise to training data if specified in config file
if epsilon != 0 or points != 0:
    training_data = noise_and_interpolate(training_data, points, epsilon, 
                                          dimensions, species, 
                                          multiplicative_noise=False)
    
# Reformat and animate training data
u_array, x_array, t_array = format_training_data_to_u_array(training_data)
animate_u_array(u_array, t_array, f'{dir_name}/training_data.gif')
    
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
    early_stopping=10000)

generate_loss_curves(train_loss_dict, val_loss_dict, dir_name, 20, 'training_loss_curves.png')

plot_param_history(binn, param_history, f"{dir_name}/param_history.png")

# Load model and print equation
model.load(f"{dir_name}/binn_best_val_model", device=device)

fn = f'{dir_name}/equation.txt'
file = open(fn, 'a')

file.write(f'Final equation:\n')
for term in model.model.generate_equation():
    file.write(f'{term}\n')

if not diff_coeffs:
    file.write(f'\nDiff. coeffs:\n')
    
    file.write(f'{[D.item() for D in model.model.diffusion_fitter()]}\n')
        
file.close()

# Prune equation
model.model.fine_tune_eql(threshold=0.01, epsilon=0.1)

# Print pruned equation
fn = f'{dir_name}/equation.txt'
file = open(fn, 'a')

file.write(f'\nFine tuned final equation:\n')
for term in model.model.generate_equation():
    file.write(f'{term}\n')
        
file.close()

compare_surfaces_over_training_domain(training_data, model, device, 
                                          reaction, params, dir_name)

# Simulate uvmlp
uvmlp_u_array, uvmlp_x_array, uvmlp_times = simulate_uvmlp(training_data, model)
animate_u_array(uvmlp_u_array, uvmlp_times, f'{dir_name}/uvmlp_sim.gif')
animate_residuals(uvmlp_u_array, u_array, uvmlp_times, 
                  name=f'{dir_name}/uvmlp_residuals.gif')

# Simulate feql
feql_u_array, feql_x_array, feql_times = simulate_feql(training_data, model)
animate_u_array(feql_u_array, feql_times, f'{dir_name}/feql_sim.gif')
animate_residuals(feql_u_array, u_array, feql_times, 
                  name=f'{dir_name}/feql_residuals.gif')